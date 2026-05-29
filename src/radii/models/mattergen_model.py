"""
MatterGen: A Generative Model for Inorganic Materials Design

Corrected implementation based on Zeni et al., Nature 2025.
https://arxiv.org/abs/2312.03687
https://github.com/microsoft/mattergen

Key concepts:
1. Joint diffusion on THREE components:
   - Atom types A: Discrete, using masked diffusion (absorbing state)
   - Coordinates X: Continuous Cartesian coordinates
   - Lattice L: Symmetric 3x3 matrix via polar decomposition

2. GemNet-based equivariant score network that jointly denoises all three

3. Physically motivated base distributions:
   - Atom types: Uniform over elements
   - Coordinates: Uniform in unit cell
   - Lattice: Distribution of random lattices

4. Property conditioning via adapter modules (ControlNet-style)

Adapted for RADII benchmark Task 1: Forward Generation (unit cell → nanoparticle)
"""

import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torch_geometric.data import Batch
from torch_geometric.nn import radius_graph, global_mean_pool
from torch_geometric.nn.models import SchNet


# =============================================================================
# Diffusion Utilities
# =============================================================================


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule as proposed in "Improved DDPM".
    Returns beta values for each timestep.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(
    timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02
) -> torch.Tensor:
    """Linear beta schedule."""
    return torch.linspace(beta_start, beta_end, timesteps)


class DiffusionSchedule(nn.Module):
    """
    Precomputes diffusion schedule parameters.
    """

    def __init__(self, num_timesteps: int = 1000, schedule_type: str = "cosine"):
        super().__init__()
        self.num_timesteps = num_timesteps

        if schedule_type == "cosine":
            betas = cosine_beta_schedule(num_timesteps)
        else:
            betas = linear_beta_schedule(num_timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Precompute useful quantities
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        # For posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )


# =============================================================================
# Atom Type Diffusion (Masked/Absorbing State)
# =============================================================================


class MaskedAtomTypeDiffusion(nn.Module):
    """
    Discrete diffusion for atom types using absorbing (masked) state.

    Forward: Progressively mask atoms towards a "MASK" token
    Reverse: Predict original atom type from masked state

    This is simpler than full categorical diffusion and works well for
    materials where composition is important.
    """

    def __init__(self, num_atom_types: int = 100, mask_token_id: int = 0):
        super().__init__()
        self.num_atom_types = num_atom_types
        self.mask_token_id = mask_token_id

    def q_sample(
        self, atom_types: torch.Tensor, t: torch.Tensor, mask_rate: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: randomly mask atom types.

        Args:
            atom_types: [N] original atom types
            t: [B] timesteps (not directly used, mask_rate derived from it)
            mask_rate: [N] probability of masking each atom

        Returns:
            noisy_types: [N] atom types with some masked
            mask: [N] boolean mask of which atoms were masked
        """
        # Sample which atoms to mask
        mask = torch.rand_like(mask_rate) < mask_rate

        # Apply mask
        noisy_types = atom_types.clone()
        noisy_types[mask] = self.mask_token_id

        return noisy_types, mask

    def get_mask_rate(self, t: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Get mask rate for each atom based on timestep.
        Uses cosine schedule for smooth masking.
        """
        # t is in [0, 1], higher t = more masking
        t_atoms = t[batch]
        # Cosine schedule for mask rate
        mask_rate = 1.0 - torch.cos(t_atoms * math.pi / 2)
        return mask_rate


# =============================================================================
# Numerically stable eigendecomposition for ill-conditioned matrices
# =============================================================================


def _safe_sym_eigh(
    S: torch.Tensor, min_eig: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Eigendecomposition for symmetric matrices with fallback for ill-conditioned cases.
    Returns (eigenvalues, eigenvectors) with eigenvalues clamped to >= min_eig.
    """
    B = S.shape[0]
    device = S.device
    S = (S + S.transpose(-2, -1)) / 2
    S = S + torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1) * 1e-5
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(S)
        eigenvalues = torch.clamp(eigenvalues, min=min_eig)
        return eigenvalues, eigenvectors
    except Exception:
        U, s, Vh = torch.linalg.svd(S)
        eigenvalues = torch.clamp(s, min=min_eig)
        return eigenvalues, U


# =============================================================================
# Lattice Diffusion (Symmetric Matrix)
# =============================================================================


class SymmetricLatticeDiffusion(nn.Module):
    """
    Diffusion on lattice matrices using symmetric representation.

    MatterGen uses:
    1. Niggli reduction to get canonical lattice
    2. Polar decomposition: L = R @ S where R is rotation, S is symmetric
    3. Diffusion on the symmetric part S (6 unique values for 3x3)

    For simplicity here, we diffuse the full 3x3 matrix with symmetry constraint.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def to_symmetric(L: torch.Tensor) -> torch.Tensor:
        """Convert lattice to symmetric via L^T @ L (Gram matrix)."""
        return torch.bmm(L.transpose(-2, -1), L)

    @staticmethod
    def from_symmetric(S: torch.Tensor) -> torch.Tensor:
        """
        Recover lattice from symmetric Gram matrix via Cholesky.
        S = L^T @ L => L = cholesky(S)^T
        """
        # Add small diagonal for numerical stability
        S = S + torch.eye(3, device=S.device).unsqueeze(0) * 1e-6
        try:
            L_chol = torch.linalg.cholesky(S)
            return L_chol.transpose(-2, -1)
        except Exception:
            eigenvalues, eigenvectors = _safe_sym_eigh(S, min_eig=1e-6)
            return eigenvectors @ torch.diag_embed(torch.sqrt(eigenvalues))

    def q_sample(
        self,
        lattice: torch.Tensor,
        sqrt_alpha: torch.Tensor,
        sqrt_one_minus_alpha: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion on symmetric lattice representation.

        Args:
            lattice: [B, 3, 3] lattice matrices
            sqrt_alpha: [B, 1, 1] sqrt(alpha_bar_t)
            sqrt_one_minus_alpha: [B, 1, 1] sqrt(1 - alpha_bar_t)
            noise: Optional noise, sampled if None

        Returns:
            noisy_lattice: [B, 3, 3] noised lattice
            noise: [B, 3, 3] the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(lattice)
            # Make noise symmetric for symmetric diffusion
            noise = (noise + noise.transpose(-2, -1)) / 2

        # Convert to Gram matrix (symmetric)
        S = self.to_symmetric(lattice)

        # Add noise in Gram space
        noisy_S = sqrt_alpha * S + sqrt_one_minus_alpha * noise

        # Ensure positive semi-definite
        eigenvalues, eigenvectors = _safe_sym_eigh(noisy_S, min_eig=0.1)
        noisy_S = (
            eigenvectors
            @ torch.diag_embed(eigenvalues)
            @ eigenvectors.transpose(-2, -1)
        )

        # Convert back to lattice
        noisy_lattice = self.from_symmetric(noisy_S)

        return noisy_lattice, noise


# =============================================================================
# Coordinate Diffusion (Cartesian)
# =============================================================================


class CartesianCoordinateDiffusion(nn.Module):
    """
    Standard Gaussian diffusion on Cartesian coordinates.

    MatterGen uses Cartesian (not fractional) coordinates.
    Coordinates are kept centered (zero center of mass).
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def center_positions(pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Center positions to zero center of mass per structure."""
        # Compute center of mass per batch
        com = global_mean_pool(pos, batch)  # [B, 3]

        # Subtract CoM
        return pos - com[batch]

    def q_sample(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        sqrt_alpha: torch.Tensor,
        sqrt_one_minus_alpha: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion on Cartesian coordinates.

        Args:
            pos: [N, 3] atom positions
            batch: [N] batch indices
            sqrt_alpha: [N, 1] or broadcastable
            sqrt_one_minus_alpha: [N, 1] or broadcastable

        Returns:
            noisy_pos: [N, 3] noised positions
            noise: [N, 3] the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(pos)
            # Center noise to preserve zero CoM
            noise = self.center_positions(noise, batch)

        # Center original positions
        pos_centered = self.center_positions(pos, batch)

        # Standard diffusion
        noisy_pos = sqrt_alpha * pos_centered + sqrt_one_minus_alpha * noise

        return noisy_pos, noise


# =============================================================================
# Score Network (GemNet-inspired)
# =============================================================================


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)


class GaussianSmearing(nn.Module):
    """Gaussian radial basis functions."""

    def __init__(self, start: float, stop: float, num_gaussians: int):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.register_buffer("offset", offset)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * dist**2)


class EquivariantBlock(nn.Module):
    """
    Simplified equivariant message passing block.
    Inspired by GemNet but simplified for clarity.
    """

    def __init__(self, hidden_dim: int, num_gaussians: int = 50):
        super().__init__()

        # Message network
        self.message_net = nn.Sequential(
            nn.Linear(2 * hidden_dim + num_gaussians, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Update network
        self.update_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Coordinate update (scalar for equivariance)
        self.coord_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.distance_expansion = GaussianSmearing(0.0, 5.0, num_gaussians)

        # LayerNorm for stable training at large hidden dims (matches GemNet-T)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, h: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = edge_index

        # Edge features
        edge_vec = pos[col] - pos[row]
        edge_dist = torch.norm(edge_vec, dim=-1, keepdim=False)
        edge_attr = self.distance_expansion(edge_dist)

        # Messages
        msg_input = torch.cat([h[row], h[col], edge_attr], dim=-1)
        msg = self.message_net(msg_input)

        # Aggregate
        agg = torch.zeros_like(h)
        agg.scatter_add_(0, row.unsqueeze(-1).expand(-1, h.size(-1)), msg)

        # Update features
        h_new = self.update_net(torch.cat([h, agg], dim=-1))
        h_new = self.layer_norm(h + h_new)  # Residual + LayerNorm

        # Coordinate updates (equivariant)
        coord_weights = self.coord_net(msg).squeeze(-1)
        edge_dir = edge_vec / (edge_dist.unsqueeze(-1) + 1e-8)
        weighted_update = coord_weights.unsqueeze(-1) * edge_dir

        coord_update = torch.zeros_like(pos)
        coord_update.scatter_add_(0, row.unsqueeze(-1).expand(-1, 3), weighted_update)

        return h_new, coord_update


class MatterGenScoreNetwork(nn.Module):
    """
    Joint score network for MatterGen.

    Predicts:
    - Atom type logits (for masked atoms)
    - Coordinate noise/score
    - Lattice noise/score
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_gaussians: int = 50,
        cutoff: float = 5.0,
        time_dim: int = 64,
        num_atom_types: int = 100,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff
        self.num_atom_types = num_atom_types

        # Atom embedding (including mask token at index 0)
        self.atom_emb = nn.Embedding(num_atom_types + 1, hidden_dim)

        # Time embedding
        self.time_emb = TimeEmbedding(time_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)

        # Lattice embedding (6 unique values from symmetric matrix)
        self.lattice_emb = nn.Sequential(
            nn.Linear(9, hidden_dim),  # Flatten 3x3
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Message passing layers
        self.layers = nn.ModuleList(
            [EquivariantBlock(hidden_dim, num_gaussians) for _ in range(num_layers)]
        )

        # Output heads
        # 1. Atom type prediction
        self.atom_type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_atom_types),
        )

        # 2. Coordinate noise prediction
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

        # 3. Lattice noise prediction
        self.lattice_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 9),  # 3x3 flattened
        )

    def forward(
        self,
        atom_types: torch.Tensor,  # [N] (with mask tokens)
        pos: torch.Tensor,  # [N, 3]
        lattice: torch.Tensor,  # [B, 3, 3]
        t: torch.Tensor,  # [B] timestep in [0, 1]
        batch: torch.Tensor,  # [N]
        cond: Optional[torch.Tensor] = None,  # [B, hidden_dim] conditioning
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass predicting scores for all three components.
        """
        # Build graph
        edge_index = radius_graph(pos, self.cutoff, batch=batch)

        # Initial features
        h = self.atom_emb(atom_types)  # [N, hidden_dim]

        # Add time embedding
        t_emb = self.time_proj(self.time_emb(t))  # [B, hidden_dim]
        h = h + t_emb[batch]

        # Add lattice embedding
        l_emb = self.lattice_emb(lattice.view(-1, 9))  # [B, hidden_dim]
        h = h + l_emb[batch]

        # Add conditioning if provided
        if cond is not None:
            h = h + cond[batch]

        # Message passing
        coord_updates = torch.zeros_like(pos)
        for layer in self.layers:
            h, coord_update = layer(h, pos, edge_index)
            coord_updates = coord_updates + coord_update

        # Output predictions
        # 1. Atom type logits (for masked atoms)
        atom_type_logits = self.atom_type_head(h)  # [N, num_atom_types]

        # 2. Coordinate noise
        coord_noise = self.coord_head(h) + coord_updates  # [N, 3]

        # 3. Lattice noise (aggregate to graph level)
        h_graph = global_mean_pool(h, batch)  # [B, hidden_dim]
        lattice_noise = self.lattice_head(h_graph).view(-1, 3, 3)  # [B, 3, 3]
        # Make symmetric
        lattice_noise = (lattice_noise + lattice_noise.transpose(-2, -1)) / 2

        return {
            "atom_type_logits": atom_type_logits,
            "coord_noise": coord_noise,
            "lattice_noise": lattice_noise,
        }


# =============================================================================
# MatterGen Model
# =============================================================================


class MatterGen(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="radii",
    repo_url="https://github.com/KurbanIntelligenceLab/RADII",
    pipeline_tag="other",
    license="mit",
    tags=["materials-science", "crystal-structures", "generative-models", "kdd-2026"],
):
    """
    MatterGen: Joint Diffusion Model for Crystal Generation

    Jointly diffuses:
    - Atom types (masked diffusion)
    - Coordinates (Cartesian Gaussian diffusion)
    - Lattice (symmetric matrix diffusion)

    Adapted for RADII benchmark: generates nanoparticle structures
    conditioned on unit cell and target radius.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_gaussians: int = 50,
        cutoff_radius: float = 5.0,
        time_dim: int = 64,
        num_atom_types: int = 100,
        num_diffusion_steps: int = 1000,
        cond_hidden_dim: int = 128,
        cond_num_layers: int = 3,
    ):
        super().__init__()
        self.num_diffusion_steps = num_diffusion_steps
        self.num_atom_types = num_atom_types

        # Diffusion components
        self.schedule = DiffusionSchedule(num_diffusion_steps, "cosine")
        self.atom_diffusion = MaskedAtomTypeDiffusion(num_atom_types, mask_token_id=0)
        self.coord_diffusion = CartesianCoordinateDiffusion()
        self.lattice_diffusion = SymmetricLatticeDiffusion()

        # Score network
        self.score_net = MatterGenScoreNetwork(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_gaussians=num_gaussians,
            cutoff=cutoff_radius,
            time_dim=time_dim,
            num_atom_types=num_atom_types,
        )

        # Unit cell encoder (conditioning)
        # SchNet outputs [B, 1] (property prediction); project to cond_hidden_dim for conditioning
        self.cell_encoder = SchNet(
            hidden_channels=cond_hidden_dim,
            num_filters=cond_hidden_dim,
            num_interactions=cond_num_layers,
            num_gaussians=num_gaussians,
            cutoff=cutoff_radius,
            readout="add",
        )
        self.cell_proj = nn.Linear(1, cond_hidden_dim)

        # Radius embedding
        self.r_emb = nn.Sequential(
            nn.Linear(1, cond_hidden_dim),
            nn.SiLU(),
            nn.Linear(cond_hidden_dim, cond_hidden_dim),
        )

        # Condition projection
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def _batch_from_ptr(ptr: torch.Tensor) -> torch.Tensor:
        diffs = ptr[1:] - ptr[:-1]
        return torch.repeat_interleave(
            torch.arange(diffs.size(0), device=ptr.device), diffs
        )

    def _process_cell_ptr(self, cell_ptr: torch.Tensor) -> torch.Tensor:
        if cell_ptr.numel() > 2 and cell_ptr.numel() % 2 == 0:
            cp2d = cell_ptr.view(-1, 2)
            lens = cp2d[:, 1]
            return torch.cat([lens.new_zeros(1), lens.cumsum(dim=0)], dim=0)
        return cell_ptr

    def encode_condition(self, data: Batch) -> torch.Tensor:
        """Encode unit cell + radius into conditioning vector."""
        cell_ptr = self._process_cell_ptr(data.cell_ptr)
        cell_batch = self._batch_from_ptr(cell_ptr)

        cell_raw = self.cell_encoder(
            data.cell_z, data.cell_pos, cell_batch
        )  # [B, 1] from SchNet
        cell_emb = self.cell_proj(cell_raw)  # [B, cond_hidden_dim]
        r_emb = self.r_emb(data.radius.view(-1, 1))

        return self.cond_proj(torch.cat([cell_emb, r_emb], dim=-1))

    def _get_lattice_from_data(
        self, data: Batch, batch: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Extract or compute lattice matrix from data."""
        if hasattr(data, "lattice") and data.lattice is not None:
            return data.lattice

        # Create bounding box lattice
        lattices = []
        for b in range(batch_size):
            mask = batch == b
            pos_b = data.pos[mask]
            box_size = pos_b.max(dim=0).values - pos_b.min(dim=0).values + 2.0
            lattice = torch.diag(box_size)
            lattices.append(lattice)
        return torch.stack(lattices, dim=0)

    def forward(self, data: Batch) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.

        Jointly diffuses atom types, coordinates, and lattice,
        then predicts the noise/original for each.
        """
        device = data.pos.device
        batch = self._batch_from_ptr(data.ptr)
        batch_size = int(batch.max().item()) + 1

        # Get conditioning
        cond = self.encode_condition(data)

        # Get target data
        atom_types = data.z  # [N]
        pos = data.pos  # [N, 3]
        lattice = self._get_lattice_from_data(data, batch, batch_size)  # [B, 3, 3]

        # Sample timestep
        t_idx = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=device)
        t = t_idx.float() / self.num_diffusion_steps

        # Get schedule parameters
        sqrt_alpha = self.schedule.sqrt_alphas_cumprod[t_idx]
        sqrt_one_minus_alpha = self.schedule.sqrt_one_minus_alphas_cumprod[t_idx]

        # === Diffuse each component ===

        # 1. Atom types (masked diffusion)
        mask_rate = self.atom_diffusion.get_mask_rate(t, batch)
        noisy_types, atom_mask = self.atom_diffusion.q_sample(atom_types, t, mask_rate)

        # 2. Coordinates (Gaussian diffusion)
        sqrt_alpha_coords = sqrt_alpha[batch].view(-1, 1)
        sqrt_one_minus_alpha_coords = sqrt_one_minus_alpha[batch].view(-1, 1)
        noisy_pos, coord_noise = self.coord_diffusion.q_sample(
            pos, batch, sqrt_alpha_coords, sqrt_one_minus_alpha_coords
        )

        # 3. Lattice (symmetric diffusion)
        sqrt_alpha_lat = sqrt_alpha.view(-1, 1, 1)
        sqrt_one_minus_alpha_lat = sqrt_one_minus_alpha.view(-1, 1, 1)
        noisy_lattice, lattice_noise = self.lattice_diffusion.q_sample(
            lattice, sqrt_alpha_lat, sqrt_one_minus_alpha_lat
        )

        # === Predict scores ===
        preds = self.score_net(noisy_types, noisy_pos, noisy_lattice, t, batch, cond)

        return {
            "pred_atom_logits": preds["atom_type_logits"],
            "pred_coord_noise": preds["coord_noise"],
            "pred_lattice_noise": preds["lattice_noise"],
            "target_atom_types": atom_types,
            "target_coord_noise": coord_noise,
            "target_lattice_noise": lattice_noise,
            "atom_mask": atom_mask,
            "t": t,
        }

    def loss(self, output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Combined loss for all three diffusion targets.
        """
        # 1. Atom type loss (cross-entropy on masked atoms)
        atom_logits = output["pred_atom_logits"]  # [N, num_types]
        atom_targets = output["target_atom_types"]  # [N]
        atom_mask = output["atom_mask"]  # [N]

        if atom_mask.any():
            atom_loss = F.cross_entropy(atom_logits[atom_mask], atom_targets[atom_mask])
        else:
            atom_loss = torch.tensor(0.0, device=atom_logits.device)

        # 2. Coordinate noise loss (MSE)
        coord_loss = F.mse_loss(
            output["pred_coord_noise"], output["target_coord_noise"]
        )

        # 3. Lattice noise loss (MSE)
        lattice_loss = F.mse_loss(
            output["pred_lattice_noise"], output["target_lattice_noise"]
        )

        # Combine with weights
        total_loss = atom_loss + coord_loss + lattice_loss

        return total_loss

    @torch.no_grad()
    def sample(
        self,
        data: Batch,
        num_steps: Optional[int] = None,
        return_trajectory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate samples via reverse diffusion.

        Jointly denoises atom types, coordinates, and lattice.
        """
        device = next(self.parameters()).device
        num_steps = num_steps or self.num_diffusion_steps

        batch = self._batch_from_ptr(data.ptr)
        batch_size = int(batch.max().item()) + 1
        num_atoms = int(data.ptr[-1].item())

        # Get conditioning
        cond = self.encode_condition(data)

        # Initialize from base distributions
        # 1. Atom types: all masked
        atom_types = torch.zeros(num_atoms, dtype=torch.long, device=device)

        # 2. Coordinates: random in scaled box
        pos = torch.randn(num_atoms, 3, device=device)
        radius = torch.clamp(
            data.radius[batch].unsqueeze(-1).expand(-1, 3),
            min=0.1,
            max=1e4,
        )
        pos = pos * radius
        pos = self.coord_diffusion.center_positions(pos, batch)
        pos = torch.nan_to_num(pos, nan=0.0, posinf=1e4, neginf=-1e4)
        pos = torch.clamp(pos, -1e4, 1e4)

        # 3. Lattice: random symmetric positive definite
        lattice = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        r = torch.clamp(data.radius.view(-1, 1, 1), min=0.1, max=1e4)
        lattice = lattice * r * 2
        noise = torch.randn_like(lattice) * 0.1
        noise = (noise + noise.transpose(-2, -1)) / 2
        lattice = lattice + noise
        lattice = torch.nan_to_num(lattice, nan=0.0, posinf=1e4, neginf=-1e4)

        trajectory = [] if return_trajectory else None

        # Reverse diffusion
        for i in reversed(range(num_steps)):
            t_idx = torch.full((batch_size,), i, device=device, dtype=torch.long)
            t = t_idx.float() / num_steps

            # Predict scores
            preds = self.score_net(atom_types, pos, lattice, t, batch, cond)

            # Sanitize predictions (model can output nan/inf)
            preds["coord_noise"] = torch.nan_to_num(
                preds["coord_noise"], nan=0.0, posinf=1e4, neginf=-1e4
            )
            preds["lattice_noise"] = torch.nan_to_num(
                preds["lattice_noise"], nan=0.0, posinf=1e4, neginf=-1e4
            )
            preds["coord_noise"] = torch.clamp(preds["coord_noise"], -1e4, 1e4)
            preds["lattice_noise"] = torch.clamp(preds["lattice_noise"], -1e4, 1e4)

            # === Update atom types ===
            # For masked atoms, sample from predicted distribution
            is_masked = atom_types == 0
            if is_masked.any():
                logits = preds["atom_type_logits"][is_masked]
                logits = torch.clamp(logits, min=-50.0, max=50.0)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
                probs = F.softmax(logits, dim=-1)
                probs = torch.clamp(probs, min=1e-8)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                # Gradually unmask based on timestep
                unmask_prob = 1.0 - t[batch[is_masked]]
                should_unmask = (
                    torch.rand(is_masked.sum(), device=device) < unmask_prob * 0.1
                )
                if should_unmask.any():
                    # Sample from predicted distribution
                    p = probs[should_unmask]
                    p = torch.clamp(p, min=1e-8)
                    p = p / p.sum(dim=-1, keepdim=True)
                    new_types = torch.multinomial(p, 1).squeeze(-1)
                    # Shift by 1 since 0 is mask token
                    new_types = new_types + 1
                    # Find positions to update
                    masked_indices = torch.where(is_masked)[0]
                    update_indices = masked_indices[should_unmask]
                    atom_types[update_indices] = new_types

            # === Update coordinates ===
            sqrt_one_minus_alpha_bar = self.schedule.sqrt_one_minus_alphas_cumprod[i]
            sqrt_alpha_bar = torch.clamp(self.schedule.sqrt_alphas_cumprod[i], min=1e-8)

            if i > 0:
                # Predicted x_0
                pred_x0 = (
                    pos - sqrt_one_minus_alpha_bar * preds["coord_noise"]
                ) / sqrt_alpha_bar
                pred_x0 = torch.nan_to_num(pred_x0, nan=0.0, posinf=1e4, neginf=-1e4)
                pred_x0 = torch.clamp(pred_x0, -1e4, 1e4)

                # Posterior mean
                coef1 = self.schedule.posterior_mean_coef1[i]
                coef2 = self.schedule.posterior_mean_coef2[i]
                mean = coef1 * pred_x0 + coef2 * pos

                # Add noise
                noise = torch.randn_like(pos)
                noise = self.coord_diffusion.center_positions(noise, batch)
                var = torch.clamp(self.schedule.posterior_variance[i], min=1e-20)
                pos = mean + torch.sqrt(var) * noise
                pos = self.coord_diffusion.center_positions(pos, batch)
                pos = torch.nan_to_num(pos, nan=0.0, posinf=1e4, neginf=-1e4)
                pos = torch.clamp(pos, -1e4, 1e4)
            else:
                # Final step: just use predicted x_0
                pos = (
                    pos - sqrt_one_minus_alpha_bar * preds["coord_noise"]
                ) / sqrt_alpha_bar
                pos = torch.nan_to_num(pos, nan=0.0, posinf=1e4, neginf=-1e4)
                pos = torch.clamp(pos, -1e4, 1e4)

            # === Update lattice ===
            if i > 0:
                sqrt_alpha_bar_lat = torch.clamp(
                    self.schedule.sqrt_alphas_cumprod[i].view(-1, 1, 1), min=1e-8
                )
                sqrt_one_minus_alpha_bar_lat = (
                    self.schedule.sqrt_one_minus_alphas_cumprod[i].view(-1, 1, 1)
                )

                # Predicted clean lattice (Gram matrix)
                pred_lattice = (
                    lattice - sqrt_one_minus_alpha_bar_lat * preds["lattice_noise"]
                ) / sqrt_alpha_bar_lat
                pred_lattice = torch.nan_to_num(
                    pred_lattice, nan=0.0, posinf=1e4, neginf=-1e4
                )

                # Ensure positive definite
                S = self.lattice_diffusion.to_symmetric(pred_lattice)
                eigenvalues, eigenvectors = _safe_sym_eigh(S, min_eig=0.5)
                S = (
                    eigenvectors
                    @ torch.diag_embed(eigenvalues)
                    @ eigenvectors.transpose(-2, -1)
                )
                pred_lattice = self.lattice_diffusion.from_symmetric(S)

                # Posterior with noise
                coef1 = self.schedule.posterior_mean_coef1[i]
                coef2 = self.schedule.posterior_mean_coef2[i]
                mean = coef1 * pred_lattice + coef2 * lattice

                noise = torch.randn_like(lattice)
                noise = (noise + noise.transpose(-2, -1)) / 2
                var = torch.clamp(self.schedule.posterior_variance[i], min=1e-20)
                lattice = mean + torch.sqrt(var) * noise
                lattice = torch.nan_to_num(lattice, nan=0.0, posinf=1e4, neginf=-1e4)

            if return_trajectory:
                trajectory.append(
                    {
                        "atom_types": atom_types.clone(),
                        "pos": pos.clone(),
                        "lattice": lattice.clone(),
                    }
                )

        result = {
            "atom_types": atom_types,
            "pos": pos,
            "lattice": lattice,
            "z": atom_types,  # Alias for compatibility
        }

        if return_trajectory:
            result["trajectory"] = trajectory

        return result


# =============================================================================
# Training utilities
# =============================================================================


def train_step(
    model: MatterGen, data: Batch, optimizer: torch.optim.Optimizer
) -> float:
    """Single training step."""
    model.train()
    optimizer.zero_grad()

    output = model(data)
    loss = model.loss(output)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item()


if __name__ == "__main__":
    # Test the model
    print("Testing MatterGen implementation...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dummy data
    batch_size = 2
    num_atoms_per_sample = [8, 10]

    all_pos = []
    all_z = []
    ptr = [0]

    for n in num_atoms_per_sample:
        pos = torch.randn(n, 3) * 3
        z = torch.randint(1, 20, (n,))
        all_pos.append(pos)
        all_z.append(z)
        ptr.append(ptr[-1] + n)

    data = Batch(
        pos=torch.cat(all_pos),
        z=torch.cat(all_z),
        ptr=torch.tensor(ptr),
        cell_pos=torch.randn(4, 3),
        cell_z=torch.tensor([14, 14, 14, 14]),
        cell_ptr=torch.tensor([0, 2, 4]),
        radius=torch.tensor([5.0, 6.0]),
    ).to(device)

    # Create model
    model = MatterGen(
        hidden_dim=128,
        num_layers=2,
        num_diffusion_steps=100,
        cond_hidden_dim=64,
        cond_num_layers=2,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    output = model(data)
    loss = model.loss(output)
    print(f"Training loss: {loss.item():.6f}")

    # Test sampling
    result = model.sample(data, num_steps=10)
    print(f"Generated positions shape: {result['pos'].shape}")
    print(f"Generated atom types shape: {result['atom_types'].shape}")
    print(f"Generated lattice shape: {result['lattice'].shape}")

    print("\nMatterGen model test passed!")
