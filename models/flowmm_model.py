"""
FlowMM: Generating Materials with Riemannian Flow Matching

Corrected implementation based on Miller et al., ICML 2024.
https://arxiv.org/abs/2406.04713

Key concepts:
1. Flow Matching learns a vector field v_θ(x, t) that transports samples from
   a base distribution p_0 to data distribution p_1 via an ODE: dx/dt = v_θ(x, t)
2. Training: regress v_θ to match conditional vector field u_t(x|x_1) = (x_1 - x_0)/(1-t)
   along the geodesic interpolation x_t = (1-t)*x_0 + t*x_1
3. For crystals, we use Riemannian manifolds:
   - Fractional coordinates: flat torus T^(3n) with geodesic wrapping
   - Lattice parameters: (a,b,c,α,β,γ) with constrained base distribution
   - Atom types (DNG): analog bits representation

Adapted for RADII benchmark Task 1: Forward Generation (unit cell → nanoparticle)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import radius_graph, global_mean_pool
from torch_geometric.nn.models import SchNet


# =============================================================================
# Torus Geometry Utilities (for fractional coordinates)
# =============================================================================


def wrap_to_unit_interval(x: torch.Tensor) -> torch.Tensor:
    """Wrap values to [0, 1) - the flat torus."""
    return x - torch.floor(x)


def torus_log_map(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Logarithmic map on the flat torus: log_x(y).
    Returns the shortest tangent vector from x to y on T^d.

    On the torus, the geodesic distance is the minimum of distances
    through the boundary. The log map gives the direction.
    """
    diff = y - x
    # Wrap to [-0.5, 0.5) to get shortest path
    diff = diff - torch.round(diff)
    return diff


def torus_exp_map(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Exponential map on the flat torus: exp_x(v).
    Moves from x in direction v, wrapping around the torus.
    """
    return wrap_to_unit_interval(x + v)


def torus_geodesic(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Geodesic interpolation on the flat torus.
    x_t = exp_{x0}(t * log_{x0}(x1))

    This gives the shortest path from x0 to x1, wrapping around if needed.
    """
    # t should be broadcastable: [B, 1] or [N, 1]
    log_vec = torus_log_map(x0, x1)  # Direction from x0 to x1
    return torus_exp_map(x0, t * log_vec)


def torus_conditional_vector_field(
    x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """
    Conditional vector field for flow matching on torus.
    u_t(x | x1) = log_{x_t}(x1) / (1 - t)

    For the linear interpolation case, this simplifies to:
    u_t = (x1 - x0) wrapped to shortest path
    """
    # The conditional VF is constant along the geodesic for linear interpolation
    return torus_log_map(x0, x1)


# =============================================================================
# Lattice Parameter Utilities
# =============================================================================


def lattice_params_to_matrix(params: torch.Tensor) -> torch.Tensor:
    """
    Convert lattice parameters (a, b, c, α, β, γ) to 3x3 lattice matrix.

    Args:
        params: [B, 6] tensor with (a, b, c, alpha, beta, gamma)
                angles in radians

    Returns:
        lattice: [B, 3, 3] lattice vectors as rows
    """
    a, b, c = params[:, 0], params[:, 1], params[:, 2]
    alpha, beta, gamma = params[:, 3], params[:, 4], params[:, 5]

    # Standard crystallographic convention
    cos_alpha = torch.cos(alpha)
    cos_beta = torch.cos(beta)
    cos_gamma = torch.cos(gamma)
    sin_gamma = torch.sin(gamma)

    # Lattice vector a along x-axis
    # Lattice vector b in xy-plane
    # Lattice vector c general

    v1 = torch.stack([a, torch.zeros_like(a), torch.zeros_like(a)], dim=-1)

    v2 = torch.stack([b * cos_gamma, b * sin_gamma, torch.zeros_like(b)], dim=-1)

    cx = c * cos_beta
    cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cz = torch.sqrt(torch.clamp(c**2 - cx**2 - cy**2, min=1e-8))
    v3 = torch.stack([cx, cy, cz], dim=-1)

    return torch.stack([v1, v2, v3], dim=1)  # [B, 3, 3]


def lattice_matrix_to_params(lattice: torch.Tensor) -> torch.Tensor:
    """
    Convert 3x3 lattice matrix to parameters (a, b, c, α, β, γ).

    Args:
        lattice: [B, 3, 3] lattice vectors as rows

    Returns:
        params: [B, 6] tensor with (a, b, c, alpha, beta, gamma) in radians
    """
    v1, v2, v3 = lattice[:, 0], lattice[:, 1], lattice[:, 2]

    a = torch.norm(v1, dim=-1)
    b = torch.norm(v2, dim=-1)
    c = torch.norm(v3, dim=-1)

    # Angles between lattice vectors
    cos_alpha = torch.sum(v2 * v3, dim=-1) / (b * c + 1e-8)
    cos_beta = torch.sum(v1 * v3, dim=-1) / (a * c + 1e-8)
    cos_gamma = torch.sum(v1 * v2, dim=-1) / (a * b + 1e-8)

    # Clamp for numerical stability
    cos_alpha = torch.clamp(cos_alpha, -1 + 1e-7, 1 - 1e-7)
    cos_beta = torch.clamp(cos_beta, -1 + 1e-7, 1 - 1e-7)
    cos_gamma = torch.clamp(cos_gamma, -1 + 1e-7, 1 - 1e-7)

    alpha = torch.acos(cos_alpha)
    beta = torch.acos(cos_beta)
    gamma = torch.acos(cos_gamma)

    return torch.stack([a, b, c, alpha, beta, gamma], dim=-1)


# =============================================================================
# Base Distributions
# =============================================================================


class LatticeBaseDistribution(nn.Module):
    """
    Base distribution for lattice parameters that produces plausible unit cells.

    FlowMM key insight: using a natural base distribution that already samples
    plausible lattices drastically simplifies learning compared to Gaussian noise.

    We sample:
    - lengths (a, b, c) from log-normal centered on training data statistics
    - angles (α, β, γ) from truncated normal around 90° (most crystals are ~orthogonal)
    """

    def __init__(
        self,
        length_mean: float = 1.0,  # Will be scaled by conditioning
        length_std: float = 0.3,
        angle_mean: float = math.pi / 2,  # 90 degrees
        angle_std: float = 0.2,
    ):
        super().__init__()
        self.length_mean = length_mean
        self.length_std = length_std
        self.angle_mean = angle_mean
        self.angle_std = angle_std

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample lattice parameters from base distribution.

        Args:
            batch_size: Number of samples
            device: Device to create tensors on
            scale: Optional [B] tensor to scale lengths (e.g., from radius)

        Returns:
            params: [B, 6] tensor (a, b, c, α, β, γ)
        """
        # Sample lengths from log-normal
        log_lengths = torch.randn(batch_size, 3, device=device) * self.length_std
        lengths = torch.exp(log_lengths) * self.length_mean

        if scale is not None:
            lengths = lengths * scale.view(-1, 1)

        # Sample angles from truncated normal (keep in valid range)
        angles = (
            torch.randn(batch_size, 3, device=device) * self.angle_std + self.angle_mean
        )
        # Clamp to valid crystallographic range
        angles = torch.clamp(angles, 0.3, math.pi - 0.3)  # ~17° to ~163°

        return torch.cat([lengths, angles], dim=-1)


class FracCoordsBaseDistribution(nn.Module):
    """
    Base distribution for fractional coordinates on the flat torus.

    For conditional generation (CSP), we use uniform on [0,1)^(3n).
    This is the natural "uninformative" prior on the torus.
    """

    def sample(self, num_atoms: int, device: torch.device) -> torch.Tensor:
        """Sample uniform fractional coordinates."""
        return torch.rand(num_atoms, 3, device=device)


# =============================================================================
# Vector Field Network
# =============================================================================


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] or [B, 1] timesteps in [0, 1]
        Returns:
            emb: [B, dim] time embeddings
        """
        t = t.view(-1)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class EquivariantMessagePassing(nn.Module):
    """
    E(3)-equivariant message passing layer for crystal structures.

    Updates both scalar features and coordinate velocities.
    """

    def __init__(self, hidden_dim: int, num_gaussians: int = 50, cutoff: float = 5.0):
        super().__init__()
        self.cutoff = cutoff

        # Distance expansion
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        # Message MLP
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + num_gaussians, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Update MLP
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Coordinate update (outputs scalar weight for direction)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h: torch.Tensor,  # [N, hidden_dim] node features
        pos: torch.Tensor,  # [N, 3] positions (Cartesian)
        edge_index: torch.Tensor,  # [2, E] edges
        edge_vec: torch.Tensor,  # [E, 3] edge vectors
        edge_dist: torch.Tensor,  # [E] edge distances
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            h_new: [N, hidden_dim] updated features
            vel: [N, 3] coordinate velocities
        """
        row, col = edge_index

        # Expand distances
        edge_attr = self.distance_expansion(edge_dist)

        # Compute messages
        msg_input = torch.cat([h[row], h[col], edge_attr], dim=-1)
        msg = self.message_mlp(msg_input)

        # Aggregate messages
        agg = torch.zeros_like(h)
        agg.scatter_add_(0, row.unsqueeze(-1).expand(-1, h.size(-1)), msg)

        # Update features
        h_new = self.update_mlp(torch.cat([h, agg], dim=-1))
        h_new = h + h_new  # Residual

        # Compute coordinate velocities (equivariant)
        coord_weights = self.coord_mlp(msg).squeeze(-1)  # [E]
        # Normalize edge vectors
        edge_dir = edge_vec / (edge_dist.unsqueeze(-1) + 1e-8)
        weighted_dir = coord_weights.unsqueeze(-1) * edge_dir

        vel = torch.zeros_like(pos)
        vel.scatter_add_(0, row.unsqueeze(-1).expand(-1, 3), weighted_dir)

        return h_new, vel


class GaussianSmearing(nn.Module):
    """Gaussian radial basis functions for distance expansion."""

    def __init__(self, start: float, stop: float, num_gaussians: int):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.register_buffer("offset", offset)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * dist**2)


class FlowMMVectorField(nn.Module):
    """
    Neural network that predicts the vector field for flow matching.

    Outputs:
    - v_frac: [N, 3] velocity for fractional coordinates (on torus tangent space)
    - v_lattice: [B, 6] velocity for lattice parameters
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_gaussians: int = 50,
        cutoff: float = 5.0,
        time_dim: int = 64,
        max_atomic_number: int = 100,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff

        # Embeddings
        self.atom_emb = nn.Embedding(max_atomic_number, hidden_dim)
        self.time_emb = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Lattice parameter embedding
        self.lattice_emb = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Message passing layers
        self.mp_layers = nn.ModuleList(
            [
                EquivariantMessagePassing(hidden_dim, num_gaussians, cutoff)
                for _ in range(num_layers)
            ]
        )

        # Output heads
        self.frac_vel_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

        self.lattice_vel_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6),
        )

    def forward(
        self,
        atom_types: torch.Tensor,  # [N] atomic numbers
        frac_coords: torch.Tensor,  # [N, 3] fractional coordinates
        lattice_params: torch.Tensor,  # [B, 6] lattice parameters
        t: torch.Tensor,  # [B] time in [0, 1]
        batch: torch.Tensor,  # [N] batch indices
        cond: Optional[torch.Tensor] = None,  # [B, hidden_dim] conditioning
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict vector field at time t.

        Returns:
            v_frac: [N, 3] fractional coordinate velocity
            v_lattice: [B, 6] lattice parameter velocity
        """

        # Convert to Cartesian for message passing
        lattice_matrix = lattice_params_to_matrix(lattice_params)  # [B, 3, 3]
        cart_coords = torch.bmm(
            frac_coords.unsqueeze(1), lattice_matrix[batch]
        ).squeeze(1)  # [N, 3]

        # Build graph
        edge_index = radius_graph(cart_coords, self.cutoff, batch=batch)
        row, col = edge_index
        edge_vec = cart_coords[col] - cart_coords[row]
        edge_dist = torch.norm(edge_vec, dim=-1)

        # Initial node features
        h = self.atom_emb(atom_types)

        # Add time embedding
        t_emb = self.time_emb(t)  # [B, hidden_dim]
        h = h + t_emb[batch]

        # Add lattice embedding
        l_emb = self.lattice_emb(lattice_params)  # [B, hidden_dim]
        h = h + l_emb[batch]

        # Add conditioning if provided
        if cond is not None:
            h = h + cond[batch]

        # Message passing
        total_vel = torch.zeros_like(cart_coords)
        for mp_layer in self.mp_layers:
            h, vel = mp_layer(h, cart_coords, edge_index, edge_vec, edge_dist)
            total_vel = total_vel + vel

        # Output fractional coordinate velocity
        # Convert Cartesian velocity to fractional
        v_frac = self.frac_vel_head(h) + total_vel

        # Transform to fractional coordinates (multiply by inverse lattice)
        lattice_inv = torch.linalg.inv(lattice_matrix)  # [B, 3, 3]
        v_frac = torch.bmm(v_frac.unsqueeze(1), lattice_inv[batch]).squeeze(1)

        # Output lattice parameter velocity
        h_graph = global_mean_pool(h, batch)  # [B, hidden_dim]
        v_lattice = self.lattice_vel_head(h_graph)

        return v_frac, v_lattice


# =============================================================================
# FlowMM Model
# =============================================================================


class FlowMM(nn.Module):
    """
    FlowMM: Riemannian Flow Matching for Crystal Generation

    Adapted for RADII benchmark: generates nanoparticle structures
    conditioned on unit cell and target radius.

    Key differences from diffusion:
    - Learns vector field v_θ, not noise prediction
    - Uses geodesic interpolation on manifolds (torus for coords)
    - Samples via ODE integration, not iterative denoising
    - Natural base distributions for lattice parameters
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_gaussians: int = 50,
        cutoff_radius: float = 5.0,
        time_dim: int = 64,
        max_atomic_number: int = 100,
        cond_hidden_dim: int = 128,
        cond_num_layers: int = 3,
        sigma_min: float = 1e-4,  # Small noise for stability
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sigma_min = sigma_min

        # Base distributions
        self.lattice_base = LatticeBaseDistribution()
        self.frac_base = FracCoordsBaseDistribution()

        # Vector field network
        self.vector_field = FlowMMVectorField(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_gaussians=num_gaussians,
            cutoff=cutoff_radius,
            time_dim=time_dim,
            max_atomic_number=max_atomic_number,
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
        r_emb_out = self.r_emb(data.radius.view(-1, 1))

        return self.cond_proj(torch.cat([cell_emb, r_emb_out], dim=-1))

    def forward(self, data: Batch) -> dict:
        """
        Training forward pass using Conditional Flow Matching.

        Loss: E_t,x0,x1 [ ||v_θ(x_t, t) - u_t(x|x_1)||² ]

        where:
        - t ~ U[0,1]
        - x0 ~ base distribution
        - x1 = data (target)
        - x_t = geodesic interpolation
        - u_t = conditional vector field (direction from x0 to x1)
        """
        device = data.pos.device
        batch = self._batch_from_ptr(data.ptr)
        batch_size = int(batch.max().item()) + 1

        # Get conditioning
        cond = self.encode_condition(data)  # [B, hidden_dim]

        # === Target data (x1) ===
        # Convert Cartesian to fractional coordinates
        # For nanoparticles, we use a bounding box as "lattice"
        # Or use the provided lattice if available
        if hasattr(data, "lattice") and data.lattice is not None:
            lattice_1 = data.lattice  # [B, 3, 3]
            lattice_params_1 = lattice_matrix_to_params(lattice_1)
        else:
            # Create bounding box lattice from positions
            lattice_params_1 = self._positions_to_lattice_params(
                data.pos, batch, batch_size
            )
            lattice_1 = lattice_params_to_matrix(lattice_params_1)

        # Convert to fractional
        lattice_inv = torch.linalg.inv(lattice_1)
        frac_coords_1 = torch.bmm(data.pos.unsqueeze(1), lattice_inv[batch]).squeeze(1)
        frac_coords_1 = wrap_to_unit_interval(frac_coords_1)

        # === Base distribution samples (x0) ===
        num_atoms = data.pos.size(0)
        frac_coords_0 = self.frac_base.sample(num_atoms, device)

        # Sample lattice params scaled by radius
        lattice_params_0 = self.lattice_base.sample(
            batch_size, device, scale=data.radius * 2
        )

        # === Sample time ===
        t = torch.rand(batch_size, device=device)
        t_atoms = t[batch]  # Expand to per-atom

        # === Geodesic interpolation ===
        # Fractional coords: geodesic on torus
        frac_coords_t = torus_geodesic(
            frac_coords_0, frac_coords_1, t_atoms.unsqueeze(-1)
        )

        # Lattice params: linear interpolation (Euclidean for now)
        # More sophisticated: use SPD manifold for metric tensor
        lattice_params_t = (1 - t.unsqueeze(-1)) * lattice_params_0 + t.unsqueeze(
            -1
        ) * lattice_params_1

        # === Compute target vector fields ===
        # Torus: shortest path direction
        target_v_frac = torus_conditional_vector_field(
            frac_coords_0, frac_coords_1, t_atoms
        )

        # Lattice: simple linear
        target_v_lattice = lattice_params_1 - lattice_params_0

        # === Predict vector field ===
        pred_v_frac, pred_v_lattice = self.vector_field(
            data.z, frac_coords_t, lattice_params_t, t, batch, cond
        )

        return {
            "pred_v_frac": pred_v_frac,
            "pred_v_lattice": pred_v_lattice,
            "target_v_frac": target_v_frac,
            "target_v_lattice": target_v_lattice,
            "t": t,
        }

    def _positions_to_lattice_params(
        self, pos: torch.Tensor, batch: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Create lattice parameters from bounding box of positions."""
        device = pos.device
        params_list = []

        for b in range(batch_size):
            mask = batch == b
            pos_b = pos[mask]

            # Bounding box with padding
            min_pos = pos_b.min(dim=0).values
            max_pos = pos_b.max(dim=0).values
            box_size = max_pos - min_pos + 2.0  # Add padding

            # Orthogonal lattice
            params = torch.tensor(
                [
                    box_size[0].item(),
                    box_size[1].item(),
                    box_size[2].item(),
                    math.pi / 2,
                    math.pi / 2,
                    math.pi / 2,
                ],
                device=device,
            )
            params_list.append(params)

        return torch.stack(params_list, dim=0)

    def loss(self, output: dict) -> torch.Tensor:
        """
        Conditional Flow Matching loss.

        L = E[ ||v_θ(x_t, t) - u_t(x|x_1)||² ]
        """
        # Fractional coordinate loss (per-atom MSE)
        loss_frac = F.mse_loss(output["pred_v_frac"], output["target_v_frac"])

        # Lattice parameter loss (per-batch MSE)
        loss_lattice = F.mse_loss(output["pred_v_lattice"], output["target_v_lattice"])

        # Combined loss
        return loss_frac + loss_lattice

    @torch.no_grad()
    def sample(
        self,
        data: Batch,
        num_steps: int = 100,
        return_trajectory: bool = False,
    ) -> dict:
        """
        Generate samples by integrating the learned vector field.

        Uses simple Euler integration:
        x_{k+1} = x_k + (1/num_steps) * v_θ(x_k, t_k)

        For fractional coords, we use the exponential map (wrapping).
        """
        device = next(self.parameters()).device
        batch = self._batch_from_ptr(data.ptr)
        batch_size = int(batch.max().item()) + 1
        num_atoms = int(data.ptr[-1].item())

        # Get conditioning
        cond = self.encode_condition(data)

        # Initialize from base distribution
        frac_coords = self.frac_base.sample(num_atoms, device)
        lattice_params = self.lattice_base.sample(
            batch_size, device, scale=data.radius * 2
        )

        dt = 1.0 / num_steps
        trajectory = (
            [(frac_coords.clone(), lattice_params.clone())]
            if return_trajectory
            else None
        )

        # Integrate ODE from t=0 to t=1
        for step in range(num_steps):
            t = torch.full((batch_size,), step / num_steps, device=device)

            # Predict vector field
            v_frac, v_lattice = self.vector_field(
                data.z
                if hasattr(data, "z") and data.z.size(0) == num_atoms
                else torch.ones(num_atoms, dtype=torch.long, device=device) * 14,
                frac_coords,
                lattice_params,
                t,
                batch,
                cond,
            )

            # Euler step on torus (with wrapping)
            frac_coords = torus_exp_map(frac_coords, dt * v_frac)

            # Euler step for lattice (Euclidean)
            lattice_params = lattice_params + dt * v_lattice

            # Ensure valid lattice parameters
            lattice_params[:, :3] = torch.clamp(
                lattice_params[:, :3], min=0.5
            )  # Lengths > 0
            lattice_params[:, 3:] = torch.clamp(
                lattice_params[:, 3:], 0.3, math.pi - 0.3
            )  # Valid angles

            if return_trajectory:
                trajectory.append((frac_coords.clone(), lattice_params.clone()))

        # Convert final fractional to Cartesian
        lattice_matrix = lattice_params_to_matrix(lattice_params)
        cart_coords = torch.bmm(
            frac_coords.unsqueeze(1), lattice_matrix[batch]
        ).squeeze(1)

        result = {
            "pos": cart_coords,
            "frac_coords": frac_coords,
            "lattice_params": lattice_params,
            "lattice": lattice_matrix,
        }

        if return_trajectory:
            result["trajectory"] = trajectory

        return result


# =============================================================================
# Training utilities
# =============================================================================


def train_step(model: FlowMM, data: Batch, optimizer: torch.optim.Optimizer) -> float:
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        cell_pos=torch.randn(4, 3),  # 2 unit cells with 2 atoms each
        cell_z=torch.tensor([14, 14, 14, 14]),  # Si atoms
        cell_ptr=torch.tensor([0, 2, 4]),
        radius=torch.tensor([5.0, 6.0]),
    ).to(device)

    # Create model
    model = FlowMM(
        hidden_dim=128,
        num_layers=2,
        cond_hidden_dim=64,
        cond_num_layers=2,
    ).to(device)

    # Test forward pass
    output = model(data)
    loss = model.loss(output)
    print(f"Training loss: {loss.item():.6f}")

    # Test sampling
    result = model.sample(data, num_steps=50)
    print(f"Generated positions shape: {result['pos'].shape}")
    print(f"Lattice parameters shape: {result['lattice_params'].shape}")

    print("\nFlowMM model test passed!")
