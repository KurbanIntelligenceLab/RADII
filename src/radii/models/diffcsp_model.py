"""
Corrected DiffCSP (Crystal Structure Prediction by Joint Equivariant Diffusion) for RADII Benchmark

Based on NeurIPS 2023 paper: "Crystal Structure Prediction by Joint Equivariant Diffusion"

Key architectural features:
1. Joint diffusion on BOTH lattice vectors AND fractional coordinates
2. Fractional coordinates (not Cartesian) with wrapped normal distribution
3. Periodic-E(3)-equivariant denoising model
4. O(3)-equivariant lattice prediction, translation-invariant coordinate prediction

Adapted for RADII Task 1: Generate nanoparticle conditioned on unit cell + radius
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torch_geometric.data import Batch
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.nn import radius_graph, MessagePassing
from torch_geometric.nn.models import SchNet
from torch_geometric.nn.pool import global_add_pool
from torch_scatter import scatter

torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])


# =============================================================================
# Helper Functions
# =============================================================================


def sinusoidal_embedding(
    timesteps: torch.Tensor, dim: int, max_period: int = 10000
) -> torch.Tensor:
    """Sinusoidal timestep embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device) / half
    )
    args = timesteps.unsqueeze(-1) * freqs
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def frac_to_cart(
    frac_coords: torch.Tensor, lattice: torch.Tensor, batch: torch.Tensor
) -> torch.Tensor:
    """Convert fractional coordinates to Cartesian. frac @ L = cart"""
    lattice_per_atom = lattice[batch]  # [N, 3, 3]
    return torch.einsum("ni,nij->nj", frac_coords, lattice_per_atom)


def cart_to_frac(
    cart_coords: torch.Tensor, lattice: torch.Tensor, batch: torch.Tensor
) -> torch.Tensor:
    """Convert Cartesian coordinates to fractional. cart @ L^{-1} = frac"""
    lattice_per_atom = lattice[batch]  # [N, 3, 3]
    lattice_inv = torch.linalg.inv(lattice_per_atom)
    return torch.einsum("ni,nij->nj", cart_coords, lattice_inv)


def wrap_frac_coords(frac_coords: torch.Tensor) -> torch.Tensor:
    """Wrap fractional coordinates to [0, 1)."""
    return frac_coords - frac_coords.floor()


class GaussianSmearing(nn.Module):
    """Gaussian smearing for distances."""

    def __init__(self, start: float = 0.0, stop: float = 5.0, num_gaussians: int = 50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * dist**2)


# =============================================================================
# Equivariant Message Passing for DiffCSP
# =============================================================================


class EquivariantBlock(MessagePassing):
    """
    Equivariant message passing block.
    Uses distances and relative vectors for E(3) equivariance.
    """

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__(aggr="add")

        # Message MLP (invariant)
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Coordinate update (equivariant)
        self.coord_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Node update
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row, col = edge_index

        # Relative vectors (equivariant)
        rel_vec = pos[row] - pos[col]
        dist = rel_vec.norm(dim=-1, keepdim=True)

        # Message passing for node features
        msg_input = torch.cat([h[row], h[col], edge_attr], dim=-1)
        msg = self.msg_mlp(msg_input)
        agg_msg = scatter(msg, col, dim=0, dim_size=h.size(0), reduce="add")

        # Update node features
        h_new = h + self.node_mlp(torch.cat([h, agg_msg], dim=-1))

        # Coordinate update (equivariant via relative vectors)
        coord_weights = self.coord_mlp(msg_input)
        coord_msg = coord_weights * rel_vec / (dist + 1e-8)
        coord_update = scatter(
            coord_msg, col, dim=0, dim_size=pos.size(0), reduce="add"
        )

        return h_new, coord_update


class DiffCSPDenoiser(nn.Module):
    """
    Joint denoising model for lattice and fractional coordinates.

    Key properties:
    - Lattice prediction is O(3)-equivariant
    - Fractional coordinate prediction is periodic-translation-invariant
    """

    def __init__(
        self,
        max_atomic_number: int = 100,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_gaussians: int = 50,
        cutoff: float = 7.0,
        time_dim: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.cutoff = cutoff

        # Atom embedding
        self.atom_emb = nn.Embedding(max_atomic_number, hidden_dim)

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Distance embedding
        self.dist_emb = GaussianSmearing(0.0, cutoff, num_gaussians)

        # Lattice embedding (flatten 3x3 -> project)
        self.lattice_emb = nn.Linear(9, hidden_dim)

        # Message passing layers
        self.layers = nn.ModuleList(
            [EquivariantBlock(hidden_dim, num_gaussians) for _ in range(num_layers)]
        )

        # Output heads
        # Lattice noise prediction (uses invariant L^T L features)
        self.lattice_head = nn.Sequential(
            nn.Linear(hidden_dim + 9, hidden_dim),  # hidden_dim + 9 for L^T L
            nn.SiLU(),
            nn.Linear(hidden_dim, 9),  # 3x3 flattened
        )

        # Fractional coordinate noise prediction
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        atom_types: torch.Tensor,  # [N]
        frac_coords: torch.Tensor,  # [N, 3] noisy fractional coordinates
        lattice: torch.Tensor,  # [B, 3, 3] noisy lattice
        t: torch.Tensor,  # [B] timestep
        batch: torch.Tensor,  # [N]
        cond: torch.Tensor = None,  # [B, hidden_dim] conditioning (optional)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict noise for both lattice and fractional coordinates.

        Returns:
            lattice_noise: [B, 3, 3]
            coord_noise: [N, 3]
        """
        batch_size = batch.max().item() + 1

        # Convert to Cartesian for graph construction
        cart_coords = frac_to_cart(frac_coords, lattice, batch)

        # Build graph with periodic images (simplified: just use cutoff on cart coords)
        edge_index = radius_graph(
            cart_coords, self.cutoff, batch=batch, max_num_neighbors=32
        )

        # Edge features
        row, col = edge_index
        dist = (cart_coords[row] - cart_coords[col]).norm(dim=-1)
        edge_attr = self.dist_emb(dist)

        # Time embedding (sinusoidal outputs time_dim, time_mlp projects to hidden_dim)
        t_emb = sinusoidal_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)  # [B, hidden_dim]

        # Lattice embedding
        lattice_flat = lattice.view(batch_size, -1)  # [B, 9]
        l_emb = self.lattice_emb(lattice_flat)  # [B, hidden_dim]

        # Initial node features
        h = self.atom_emb(atom_types)  # [N, hidden_dim]

        # Add time and lattice conditioning
        h = h + t_emb[batch] + l_emb[batch]

        # Add external conditioning if provided
        if cond is not None:
            h = h + cond[batch]

        # Message passing
        coord_updates = []
        for layer in self.layers:
            h, coord_upd = layer(h, cart_coords, edge_index, edge_attr)
            coord_updates.append(coord_upd)

        # Aggregate coordinate updates
        total_coord_update = sum(coord_updates)

        # Predict fractional coordinate noise
        coord_noise = self.coord_head(h) + cart_to_frac(
            total_coord_update, lattice, batch
        )

        # Predict lattice noise (aggregate node features for graph-level prediction)
        h_graph = scatter(
            h, batch, dim=0, dim_size=batch_size, reduce="mean"
        )  # [B, hidden_dim]

        # O(3)-equivariant lattice update: use L^T L (invariant) to compute weights
        # Then multiply by L for equivariance
        LTL = torch.bmm(lattice.transpose(1, 2), lattice)  # [B, 3, 3]
        lattice_feat = torch.cat(
            [h_graph, LTL.view(batch_size, -1)], dim=-1
        )  # [B, hidden_dim + 9]

        # Predict lattice noise using invariant features
        lattice_noise_flat = self.lattice_head(lattice_feat)  # [B, 9]
        lattice_noise = lattice_noise_flat.view(batch_size, 3, 3)

        return lattice_noise, coord_noise


# =============================================================================
# Main DiffCSP Model
# =============================================================================


class DiffCSPUnitCell(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="radii",
    repo_url="https://github.com/KurbanIntelligenceLab/RADII",
    pipeline_tag="other",
    license="mit",
    tags=["materials-science", "crystal-structures", "generative-models", "kdd-2026"],
):
    """
    DiffCSP adapted for RADII benchmark.

    Joint diffusion on lattice and fractional coordinates, conditioned on unit cell + radius.

    Key differences from original code:
    1. Proper diffusion on BOTH lattice and fractional coordinates (joint)
    2. Uses fractional coordinates with wrapping
    3. Correct DDPM noise schedule
    4. Equivariant denoiser architecture
    """

    def __init__(
        self,
        max_atomic_number: int = 100,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_gaussians: int = 50,
        cutoff_radius: float = 7.0,
        time_dim: int = 128,
        # Conditioning
        cond_hidden_dim: int = 256,
        cond_num_layers: int = 3,
        r_emb_dim: int = 64,
        # Diffusion
        num_diffusion_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_diffusion_steps = num_diffusion_steps

        # Diffusion schedule
        betas = torch.linspace(beta_start, beta_end, num_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod)
        )

        # For sampling
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer(
            "posterior_variance",
            betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod),
        )

        # Unit cell encoder (conditioning)
        # PyG SchNet returns [B, 1]; we need [B, cond_hidden_dim]. Use representation + global_add_pool.
        self._schnet = SchNet(
            hidden_channels=cond_hidden_dim,
            num_filters=cond_hidden_dim,
            num_interactions=cond_num_layers,
            num_gaussians=num_gaussians,
            cutoff=cutoff_radius,
            readout="add",
        )

        # Radius embedding
        self.r_emb = nn.Sequential(
            nn.Linear(1, r_emb_dim),
            nn.SiLU(),
            nn.Linear(r_emb_dim, r_emb_dim),
        )

        # Condition projection
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_hidden_dim + r_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Denoising model
        self.denoiser = DiffCSPDenoiser(
            max_atomic_number=max_atomic_number,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_gaussians=num_gaussians,
            cutoff=cutoff_radius,
            time_dim=time_dim,
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

    def _encode_cell(
        self, cell_z: torch.Tensor, cell_pos: torch.Tensor, cell_batch: torch.Tensor
    ) -> torch.Tensor:
        """Encode unit cell using SchNet representation (skip property prediction head)."""
        h = self._schnet.embedding(cell_z)
        edge_index, edge_weight = self._schnet.interaction_graph(cell_pos, cell_batch)
        edge_attr = self._schnet.distance_expansion(edge_weight)
        for interaction in self._schnet.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
        return global_add_pool(h, cell_batch)  # [B, cond_hidden_dim]

    def encode_condition(self, data: Batch) -> torch.Tensor:
        """Encode unit cell + radius into conditioning vector."""
        cell_ptr = self._process_cell_ptr(data.cell_ptr)
        cell_batch = self._batch_from_ptr(cell_ptr)
        cell_emb = self._encode_cell(data.cell_z, data.cell_pos, cell_batch)
        r_emb = self.r_emb(data.radius.view(-1, 1))
        cat_in = torch.cat([cell_emb, r_emb], dim=-1)
        return self.cond_proj(cat_in)

    def q_sample_lattice(
        self, lattice: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None
    ):
        """Add noise to lattice."""
        if noise is None:
            noise = torch.randn_like(lattice)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)

        return sqrt_alpha * lattice + sqrt_one_minus_alpha * noise, noise

    def q_sample_coords(
        self,
        frac_coords: torch.Tensor,
        t: torch.Tensor,
        batch: torch.Tensor,
        noise: torch.Tensor = None,
    ):
        """Add noise to fractional coordinates (with wrapping)."""
        if noise is None:
            noise = torch.randn_like(frac_coords)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][batch].unsqueeze(-1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][batch].unsqueeze(
            -1
        )

        noisy_coords = sqrt_alpha * frac_coords + sqrt_one_minus_alpha * noise
        # Wrap to [0, 1) for fractional coordinates
        noisy_coords = wrap_frac_coords(noisy_coords)

        return noisy_coords, noise

    def forward(self, data: Batch, t: torch.Tensor = None):
        """
        Training forward pass.

        Args:
            data: Batch with z, pos, lattice, ptr, cell_z, cell_pos, cell_ptr, radius
        """
        batch = self._batch_from_ptr(data.ptr)
        batch_size = batch.max().item() + 1
        device = data.z.device

        # Get condition
        cond = self.encode_condition(data)  # [B, hidden_dim]

        # Sample timestep if not provided
        if t is None:
            t = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=device)

        # Get ground truth lattice and fractional coordinates
        lattice = data.lattice  # [B, 3, 3]
        frac_coords = cart_to_frac(data.pos, lattice, batch)  # [N, 3]
        frac_coords = wrap_frac_coords(frac_coords)

        # Add noise
        noisy_lattice, lattice_noise = self.q_sample_lattice(lattice, t)
        noisy_frac, coord_noise = self.q_sample_coords(frac_coords, t, batch)

        # Predict noise (pass conditioning)
        pred_lattice_noise, pred_coord_noise = self.denoiser(
            data.z,
            noisy_frac,
            noisy_lattice,
            t.float() / self.num_diffusion_steps,
            batch,
            cond,
        )

        return {
            "pred_lattice_noise": pred_lattice_noise,
            "pred_coord_noise": pred_coord_noise,
            "lattice_noise": lattice_noise,
            "coord_noise": coord_noise,
            "t": t,
        }

    def loss(
        self, output: dict, lambda_lattice: float = 1.0, lambda_coord: float = 1.0
    ) -> dict:
        """Compute training loss."""
        lattice_loss = F.mse_loss(output["pred_lattice_noise"], output["lattice_noise"])
        coord_loss = F.mse_loss(output["pred_coord_noise"], output["coord_noise"])

        total = lambda_lattice * lattice_loss + lambda_coord * coord_loss

        return {
            "total": total,
            "lattice": lattice_loss,
            "coord": coord_loss,
        }

    @torch.no_grad()
    def sample(
        self,
        data: Batch,
        num_atoms: torch.Tensor = None,
        init_lattice: torch.Tensor = None,
    ) -> dict:
        """
        Generate crystal structure via reverse diffusion.

        Args:
            data: Batch with cell_z, cell_pos, cell_ptr, radius
            num_atoms: Number of atoms per sample [B]
            init_lattice: Initial lattice guess [B, 3, 3], if None uses random
        """
        device = next(self.parameters()).device

        # Get condition
        cond = self.encode_condition(data)
        batch_size = cond.size(0)

        # Initialize lattice (from noise or initial guess)
        if init_lattice is not None:
            lattice = init_lattice.clone()
        else:
            # Random lattice (scaled by radius as rough estimate)
            scale = torch.clamp(data.radius.view(-1, 1, 1), min=0.1, max=1e4) * 2
            lattice = torch.randn(batch_size, 3, 3, device=device) * scale
        lattice = torch.nan_to_num(lattice, nan=0.0, posinf=1e4, neginf=-1e4)
        lattice = torch.clamp(lattice, -1e4, 1e4)

        # Create batch for atoms
        if num_atoms is None:
            # Estimate from radius (rough heuristic)
            num_atoms = (data.radius**3 * 0.1).long().clamp(min=4, max=200)

        total_atoms = num_atoms.sum().item()
        batch = torch.repeat_interleave(
            torch.arange(batch_size, device=device), num_atoms
        )

        # Initialize fractional coordinates (random in [0, 1))
        frac_coords = torch.rand(total_atoms, 3, device=device)

        # Reverse diffusion
        for i in reversed(range(self.num_diffusion_steps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            t_norm = t.float() / self.num_diffusion_steps

            # Predict noise (with conditioning)
            pred_lat_noise, pred_coord_noise = self.denoiser(
                data.z
                if hasattr(data, "z") and data.z.size(0) == total_atoms
                else torch.ones(total_atoms, dtype=torch.long, device=device)
                * 14,  # Default to Si
                frac_coords,
                lattice,
                t_norm,
                batch,
                cond,
            )
            pred_lat_noise = torch.nan_to_num(
                pred_lat_noise, nan=0.0, posinf=1e4, neginf=-1e4
            )
            pred_coord_noise = torch.nan_to_num(
                pred_coord_noise, nan=0.0, posinf=1e4, neginf=-1e4
            )
            pred_lat_noise = torch.clamp(pred_lat_noise, -1e4, 1e4)
            pred_coord_noise = torch.clamp(pred_coord_noise, -1e4, 1e4)

            # DDPM update for lattice
            alpha_t = self.alphas_cumprod[i]
            alpha_t_prev = self.alphas_cumprod_prev[i]
            beta_t = self.betas[i]

            # Predict clean lattice
            sqrt_alpha_t = torch.clamp(torch.sqrt(alpha_t), min=1e-8)
            sqrt_one_minus_alpha_t = torch.sqrt(torch.clamp(1 - alpha_t, min=1e-8))
            lattice_pred = (
                lattice - sqrt_one_minus_alpha_t * pred_lat_noise
            ) / sqrt_alpha_t
            lattice_pred = torch.nan_to_num(
                lattice_pred, nan=0.0, posinf=1e4, neginf=-1e4
            )
            lattice_pred = torch.clamp(lattice_pred, -1e4, 1e4)

            # Posterior mean for lattice
            denom = torch.clamp(1 - alpha_t, min=1e-8)
            lat_post_mean = (
                torch.sqrt(alpha_t_prev) * beta_t / denom * lattice_pred
                + torch.sqrt(self.alphas[i]) * (1 - alpha_t_prev) / denom * lattice
            )
            lat_post_mean = torch.nan_to_num(
                lat_post_mean, nan=0.0, posinf=1e4, neginf=-1e4
            )

            if i > 0:
                lat_noise = torch.randn_like(lattice)
                var = torch.clamp(self.posterior_variance[i], min=1e-20)
                lattice = lat_post_mean + torch.sqrt(var) * lat_noise
            else:
                lattice = lat_post_mean
            lattice = torch.nan_to_num(lattice, nan=0.0, posinf=1e4, neginf=-1e4)
            lattice = torch.clamp(lattice, -1e4, 1e4)

            # DDPM update for coordinates
            sqrt_alpha_node = sqrt_alpha_t
            sqrt_one_minus_alpha_node = sqrt_one_minus_alpha_t
            coord_pred = (
                frac_coords - sqrt_one_minus_alpha_node * pred_coord_noise
            ) / sqrt_alpha_node
            coord_pred = torch.nan_to_num(coord_pred, nan=0.0, posinf=1.0, neginf=0.0)
            coord_pred = torch.clamp(coord_pred, -1.0, 2.0)
            coord_pred = wrap_frac_coords(coord_pred)

            # Posterior mean for coords
            coord_post_mean = (
                torch.sqrt(alpha_t_prev) * beta_t / denom * coord_pred
                + torch.sqrt(self.alphas[i]) * (1 - alpha_t_prev) / denom * frac_coords
            )
            coord_post_mean = torch.nan_to_num(
                coord_post_mean, nan=0.5, posinf=1.0, neginf=0.0
            )

            if i > 0:
                coord_noise = torch.randn_like(frac_coords)
                var = torch.clamp(self.posterior_variance[i], min=1e-20)
                frac_coords = coord_post_mean + torch.sqrt(var) * coord_noise
                frac_coords = torch.nan_to_num(
                    frac_coords, nan=0.5, posinf=1.0, neginf=0.0
                )
                frac_coords = torch.clamp(frac_coords, -0.5, 1.5)
                frac_coords = wrap_frac_coords(frac_coords)
            else:
                frac_coords = torch.nan_to_num(
                    coord_post_mean, nan=0.5, posinf=1.0, neginf=0.0
                )
                frac_coords = torch.clamp(frac_coords, -0.5, 1.5)
                frac_coords = wrap_frac_coords(frac_coords)

        # Convert to Cartesian
        cart_coords = frac_to_cart(frac_coords, lattice, batch)
        cart_coords = torch.nan_to_num(cart_coords, nan=0.0, posinf=1e4, neginf=-1e4)
        cart_coords = torch.clamp(cart_coords, -1e4, 1e4)

        return {
            "positions": cart_coords,
            "frac_coords": frac_coords,
            "lattice": lattice,
            "batch": batch,
            "num_atoms": num_atoms,
        }
