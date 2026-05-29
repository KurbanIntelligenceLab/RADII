"""
Corrected CDVAE (Crystal Diffusion Variational Autoencoder) for RADII Benchmark

Architecture based on the ICLR 2022 paper:
"Crystal Diffusion Variational Autoencoder for Periodic Material Generation"

Key components:
1. Encoder: GNN (DimeNet++/GemNet style) that encodes crystal -> per-atom embeddings -> aggregated latent z
2. Decoder: Score network that performs denoising via Langevin dynamics
3. Separate prediction heads for: coordinates, atom types, lattice parameters

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
from torch_geometric.nn import radius_graph, MessagePassing, global_mean_pool
from torch_geometric.nn.models import SchNet
from torch_geometric.nn.pool import global_add_pool
from torch_scatter import scatter

torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])


# =============================================================================
# Building Blocks
# =============================================================================


class SinusoidsEmbedding(nn.Module):
    """Sinusoidal embedding for continuous values (distances, sigma, etc.)"""

    def __init__(
        self, n_frequencies: int = 32, n_min: float = 0.0, n_max: float = 10.0
    ):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.register_buffer("frequencies", torch.linspace(n_min, n_max, n_frequencies))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [...] -> [..., 2 * n_frequencies]
        x = x.unsqueeze(-1)  # [..., 1]
        emb = torch.cat(
            [torch.sin(x * self.frequencies), torch.cos(x * self.frequencies)], dim=-1
        )
        return emb


class GaussianSmearing(nn.Module):
    """Gaussian smearing for distance embeddings."""

    def __init__(self, start: float = 0.0, stop: float = 5.0, num_gaussians: int = 50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class InteractionBlock(MessagePassing):
    """Message passing block for GNN encoder."""

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__(aggr="add")
        self.mlp_msg = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mlp_upd = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.mlp_upd(torch.cat([x, out], dim=-1))
        return x + out  # Residual

    def message(
        self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        return self.mlp_msg(torch.cat([x_i, x_j, edge_attr], dim=-1))


class GNNEncoder(nn.Module):
    """
    GNN Encoder that produces per-atom embeddings.
    Simplified version inspired by DimeNet++/GemNet.
    """

    def __init__(
        self,
        max_atomic_number: int = 100,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_gaussians: int = 50,
        cutoff: float = 5.0,
    ):
        super().__init__()
        self.cutoff = cutoff

        # Atom embedding
        self.atom_emb = nn.Embedding(max_atomic_number, hidden_dim)

        # Distance embedding
        self.dist_emb = GaussianSmearing(0.0, cutoff, num_gaussians)
        edge_dim = num_gaussians

        # Interaction blocks
        self.interactions = nn.ModuleList(
            [InteractionBlock(hidden_dim, edge_dim) for _ in range(num_layers)]
        )

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        z: torch.Tensor,  # Atomic numbers [N]
        pos: torch.Tensor,  # Positions [N, 3]
        batch: torch.Tensor,  # Batch indices [N]
    ) -> torch.Tensor:
        # Build graph
        edge_index = radius_graph(pos, self.cutoff, batch=batch, max_num_neighbors=32)

        # Edge features (distances)
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.dist_emb(dist)

        # Initial node features
        h = self.atom_emb(z)

        # Message passing
        for interaction in self.interactions:
            h = interaction(h, edge_index, edge_attr)

        return self.out_proj(h)  # [N, hidden_dim]


class ScoreNetwork(MessagePassing):
    """
    Score network for denoising coordinates.
    Predicts the score ∇_x log p(x) for Langevin dynamics.
    """

    def __init__(
        self,
        max_atomic_number: int = 100,
        hidden_dim: int = 256,
        latent_dim: int = 256,
        num_layers: int = 4,
        num_gaussians: int = 50,
        cutoff: float = 5.0,
    ):
        super().__init__(aggr="add")
        self.cutoff = cutoff
        self.hidden_dim = hidden_dim

        # Atom embedding
        self.atom_emb = nn.Embedding(max_atomic_number, hidden_dim)

        # Sigma (noise level) embedding
        self.sigma_emb = SinusoidsEmbedding(n_frequencies=32, n_min=0.0, n_max=1.0)
        sigma_dim = 64  # 2 * 32

        # Distance embedding
        self.dist_emb = GaussianSmearing(0.0, cutoff, num_gaussians)

        # Project latent + sigma to node features
        self.latent_proj = nn.Linear(latent_dim + sigma_dim, hidden_dim)

        # Input projection (combines atom emb + latent)
        self.input_proj = nn.Linear(2 * hidden_dim, hidden_dim)

        # Message passing layers
        edge_dim = num_gaussians + 3  # distance emb + unit vector
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "msg": nn.Sequential(
                            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
                            nn.SiLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                        ),
                        "upd": nn.Sequential(
                            nn.Linear(2 * hidden_dim, hidden_dim),
                            nn.SiLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                        ),
                    }
                )
            )

        # Output: predict score (gradient direction for denoising)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

        # Output: predict atom type logits
        self.atom_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, max_atomic_number),
        )

    def forward(
        self,
        z: torch.Tensor,  # Atom types [N] (possibly noisy/predicted)
        pos: torch.Tensor,  # Noisy positions [N, 3]
        batch: torch.Tensor,  # Batch indices [N]
        latent: torch.Tensor,  # Latent from encoder [B, latent_dim]
        sigma: torch.Tensor,  # Noise level [B] or [N]
    ):
        # Build graph from noisy positions
        edge_index = radius_graph(pos, self.cutoff, batch=batch, max_num_neighbors=32)

        # Edge features
        row, col = edge_index
        edge_vec = pos[row] - pos[col]
        dist = edge_vec.norm(dim=-1, keepdim=True)
        unit_vec = edge_vec / (dist + 1e-8)
        dist_emb = self.dist_emb(dist.squeeze(-1))
        edge_attr = torch.cat([dist_emb, unit_vec], dim=-1)

        # Sigma embedding (broadcast to nodes)
        if sigma.dim() == 1 and sigma.size(0) != pos.size(0):
            sigma = sigma[batch]
        sigma_emb = self.sigma_emb(sigma)  # [N, sigma_dim]

        # Latent broadcast to nodes
        latent_node = latent[batch]  # [N, latent_dim]
        cond = self.latent_proj(
            torch.cat([latent_node, sigma_emb], dim=-1)
        )  # [N, hidden_dim]

        # Initial node features
        h = self.atom_emb(z)
        h = self.input_proj(torch.cat([h, cond], dim=-1))

        # Message passing
        for layer in self.layers:
            # Compute messages
            msg_input = torch.cat([h[row], h[col], edge_attr], dim=-1)
            msg = layer["msg"](msg_input)
            agg = scatter(msg, col, dim=0, dim_size=h.size(0), reduce="add")

            # Update
            h = h + layer["upd"](torch.cat([h, agg], dim=-1))

        # Predict outputs
        score = self.score_head(h)  # [N, 3] - direction to denoise
        atom_logits = self.atom_head(h)  # [N, max_atomic_number]

        return score, atom_logits


# =============================================================================
# Main CDVAE Model
# =============================================================================


class CDVAEUnitCell(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="radii",
    repo_url="https://github.com/KurbanIntelligenceLab/RADII",
    pipeline_tag="other",
    license="mit",
    tags=["materials-science", "crystal-structures", "generative-models", "kdd-2026"],
):
    """
    Crystal Diffusion VAE adapted for RADII benchmark.

    Conditions on unit cell + radius to generate nanoparticle.

    Architecture:
    1. Unit cell encoder (SchNet) -> condition embedding
    2. Target structure encoder (GNN) -> per-atom embeddings -> aggregated latent z
    3. Score network decoder -> denoises coordinates via Langevin dynamics
    """

    def __init__(
        self,
        # Encoder params
        max_atomic_number: int = 100,
        hidden_dim: int = 256,
        latent_dim: int = 256,
        encoder_num_layers: int = 4,
        # Decoder params
        decoder_num_layers: int = 4,
        # Common params
        num_gaussians: int = 50,
        cutoff_radius: float = 5.0,
        # Conditioning params
        r_emb_dim: int = 64,
        # Diffusion params
        num_noise_levels: int = 50,
        sigma_min: float = 0.01,
        sigma_max: float = 1.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.cutoff_radius = cutoff_radius
        self.num_noise_levels = num_noise_levels

        # Noise schedule (geometric)
        sigmas = torch.exp(
            torch.linspace(math.log(sigma_max), math.log(sigma_min), num_noise_levels)
        )
        self.register_buffer("sigmas", sigmas)

        # Unit cell encoder (for conditioning)
        # PyG SchNet returns [B, 1] (property prediction). We need [B, hidden_dim].
        # Use SchNet's internal representation + global_add_pool (same fix as ADiT).
        self._schnet = SchNet(
            hidden_channels=hidden_dim,
            num_filters=hidden_dim,
            num_interactions=3,
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

        # Condition projection (cell_emb + radius_emb -> hidden_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(hidden_dim + r_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Target structure encoder (encodes the full nanoparticle)
        self.encoder = GNNEncoder(
            max_atomic_number=max_atomic_number,
            hidden_dim=hidden_dim,
            num_layers=encoder_num_layers,
            num_gaussians=num_gaussians,
            cutoff=cutoff_radius,
        )

        # VAE bottleneck: per-atom embeddings -> aggregated -> mu/logvar
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Initialize logvar bias for stable training
        nn.init.zeros_(self.fc_logvar.bias)
        self.fc_logvar.bias.data.fill_(-4.0)

        # Combine latent with condition for decoder
        self.latent_cond_proj = nn.Linear(latent_dim + hidden_dim, latent_dim)

        # Score network decoder
        self.decoder = ScoreNetwork(
            max_atomic_number=max_atomic_number,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=decoder_num_layers,
            num_gaussians=num_gaussians,
            cutoff=cutoff_radius,
        )

        # Number of atoms predictor (from latent)
        self.num_atoms_head = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _batch_from_ptr(ptr: torch.Tensor) -> torch.Tensor:
        diffs = ptr[1:] - ptr[:-1]
        return torch.repeat_interleave(
            torch.arange(diffs.size(0), device=ptr.device), diffs
        )

    def _process_cell_ptr(self, cell_ptr: torch.Tensor) -> torch.Tensor:
        """Handle different cell_ptr formats."""
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
        return global_add_pool(h, cell_batch)  # [B, hidden_dim]

    def encode_condition(self, data: Batch) -> torch.Tensor:
        """Encode unit cell + radius into condition embedding."""
        cell_ptr = self._process_cell_ptr(data.cell_ptr)
        cell_batch = self._batch_from_ptr(cell_ptr)
        cell_emb = self._encode_cell(
            data.cell_z, data.cell_pos, cell_batch
        )  # [B, hidden_dim]
        radius = data.radius.view(-1, 1)
        r_emb = self.r_emb(radius)  # [B, r_emb_dim]
        cond = self.cond_proj(torch.cat([cell_emb, r_emb], dim=-1))  # [B, hidden_dim]
        return cond

    def encode(self, data: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode target structure to latent distribution parameters."""
        batch = self._batch_from_ptr(data.ptr)

        # Per-atom embeddings
        h = self.encoder(data.z, data.pos, batch)  # [N, hidden_dim]

        # Aggregate to graph-level
        h_graph = global_mean_pool(h, batch)  # [B, hidden_dim]

        # VAE parameters
        mu = self.fc_mu(h_graph)  # [B, latent_dim]
        logvar = self.fc_logvar(h_graph)  # [B, latent_dim]

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data: Batch):
        """
        Training forward pass.

        Args:
            data: Batch containing:
                - z, pos, ptr: nanoparticle structure
                - cell_z, cell_pos, cell_ptr: unit cell structure
                - radius: target radius

        Returns:
            Dictionary with predictions and latent parameters
        """
        batch = self._batch_from_ptr(data.ptr)
        batch_size = batch.max().item() + 1

        # Encode condition (unit cell + radius)
        cond = self.encode_condition(data)  # [B, hidden_dim]

        # Encode target structure
        mu, logvar = self.encode(data)  # [B, latent_dim]
        z_latent = self.reparameterize(mu, logvar)  # [B, latent_dim]

        # Combine latent with condition
        z_cond = self.latent_cond_proj(
            torch.cat([z_latent, cond], dim=-1)
        )  # [B, latent_dim]

        # Sample noise level for score matching
        noise_idx = torch.randint(
            0, self.num_noise_levels, (batch_size,), device=data.z.device
        )
        sigma = self.sigmas[noise_idx]  # [B]

        # Add noise to positions
        noise = torch.randn_like(data.pos)
        sigma_node = sigma[batch].unsqueeze(-1)  # [N, 1]
        noisy_pos = data.pos + sigma_node * noise

        # Predict score (denoising direction) and atom types
        score_pred, atom_logits = self.decoder(data.z, noisy_pos, batch, z_cond, sigma)

        # Target score: -noise / sigma (direction to denoise)
        score_target = -noise / sigma_node

        # Predict number of atoms
        num_atoms_pred = self.num_atoms_head(torch.cat([z_latent, cond], dim=-1))

        return {
            "score_pred": score_pred,
            "score_target": score_target,
            "atom_logits": atom_logits,
            "atom_target": data.z,
            "num_atoms_pred": num_atoms_pred.squeeze(-1),
            "num_atoms_target": scatter(
                torch.ones_like(batch), batch, reduce="sum"
            ).float(),
            "mu": mu,
            "logvar": logvar,
            "sigma": sigma,
        }

    def loss(
        self,
        output: dict,
        lambda_coord: float = 1.0,
        lambda_atom: float = 1.0,
        lambda_num: float = 0.1,
        lambda_kl: float = 1e-4,
    ) -> dict:
        """Compute training loss."""
        # Score matching loss (coordinate denoising)
        coord_loss = F.mse_loss(output["score_pred"], output["score_target"])

        # Atom type loss
        atom_loss = F.cross_entropy(output["atom_logits"], output["atom_target"])

        # Number of atoms loss
        num_loss = F.mse_loss(output["num_atoms_pred"], output["num_atoms_target"])

        # KL divergence
        mu, logvar = output["mu"], output["logvar"]
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total = (
            lambda_coord * coord_loss
            + lambda_atom * atom_loss
            + lambda_num * num_loss
            + lambda_kl * kl_loss
        )

        return {
            "total": total,
            "coord": coord_loss,
            "atom": atom_loss,
            "num_atoms": num_loss,
            "kl": kl_loss,
        }

    @torch.no_grad()
    def sample(
        self,
        data: Batch,
        num_atoms: torch.Tensor = None,
        num_langevin_steps: int = 100,
        step_size: float = 0.01,
    ) -> dict:
        """
        Generate nanoparticle via Langevin dynamics.

        Args:
            data: Batch with cell_z, cell_pos, cell_ptr, radius
            num_atoms: Number of atoms per sample [B], if None will predict
            num_langevin_steps: Steps of Langevin dynamics
            step_size: Langevin step size

        Returns:
            Generated positions and atom types
        """
        device = next(self.parameters()).device

        # Encode condition
        cond = self.encode_condition(data)  # [B, hidden_dim]
        batch_size = cond.size(0)

        # Sample latent from prior
        z_latent = torch.randn(batch_size, self.latent_dim, device=device)
        z_cond = self.latent_cond_proj(torch.cat([z_latent, cond], dim=-1))

        # Predict or use given number of atoms
        if num_atoms is None:
            num_atoms_pred = self.num_atoms_head(torch.cat([z_latent, cond], dim=-1))
            num_atoms = torch.clamp(
                num_atoms_pred.squeeze(-1).round().long(), min=1, max=500
            )

        # Create batch indices
        total_atoms = num_atoms.sum().item()
        batch = torch.repeat_interleave(
            torch.arange(batch_size, device=device), num_atoms
        )

        # Initialize: random positions and atom types
        pos = torch.randn(total_atoms, 3, device=device) * data.radius[batch].unsqueeze(
            -1
        )
        z_atoms = torch.randint(
            1, 50, (total_atoms,), device=device
        )  # Random atom types

        # Langevin dynamics: denoise from high noise to low noise
        for i in range(self.num_noise_levels):
            sigma_idx = i
            sigma = self.sigmas[sigma_idx]
            sigma_tensor = sigma.expand(batch_size)

            for _ in range(num_langevin_steps // self.num_noise_levels + 1):
                # Get score
                score, atom_logits = self.decoder(
                    z_atoms, pos, batch, z_cond, sigma_tensor
                )

                # Langevin update
                noise = torch.randn_like(pos) if i < self.num_noise_levels - 1 else 0
                pos = pos + step_size * score + math.sqrt(2 * step_size) * noise

                # Update atom types (take argmax of logits)
                z_atoms = atom_logits.argmax(dim=-1)

        return {
            "positions": pos,
            "atom_types": z_atoms,
            "batch": batch,
            "num_atoms": num_atoms,
        }

    @torch.no_grad()
    def reconstruct(
        self, data: Batch, num_langevin_steps: int = 100, step_size: float = 0.01
    ) -> dict:
        """Reconstruct from encoded latent."""
        batch = self._batch_from_ptr(data.ptr)
        batch_size = batch.max().item() + 1

        # Encode
        cond = self.encode_condition(data)
        mu, logvar = self.encode(data)
        z_latent = mu  # Use mean for reconstruction
        z_cond = self.latent_cond_proj(torch.cat([z_latent, cond], dim=-1))

        # Start from noisy version of true positions
        pos = data.pos + torch.randn_like(data.pos) * self.sigmas[0]
        z_atoms = data.z.clone()

        # Langevin dynamics
        for i in range(self.num_noise_levels):
            sigma = self.sigmas[i]
            sigma_tensor = sigma.expand(batch_size)

            for _ in range(num_langevin_steps // self.num_noise_levels + 1):
                score, atom_logits = self.decoder(
                    z_atoms, pos, batch, z_cond, sigma_tensor
                )
                noise = torch.randn_like(pos) if i < self.num_noise_levels - 1 else 0
                pos = pos + step_size * score + math.sqrt(2 * step_size) * noise
                z_atoms = atom_logits.argmax(dim=-1)

        return {
            "positions": pos,
            "atom_types": z_atoms,
            "batch": batch,
        }
