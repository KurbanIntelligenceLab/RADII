"""
ADiT (All-atom Diffusion Transformer) for RADII Benchmark - Model Only

Architecture:
1. AtomVAE: Encodes/decodes atomic structures to/from latent space
2. DiT: Diffusion Transformer operating in latent space with adaLN-Zero
3. UnitCellEncoder: SchNet-based encoder for conditioning

Reference: "All-atom Diffusion Transformers: Unified generative modelling of molecules and materials"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torch_geometric.data import Batch
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.nn.models import SchNet
from torch_geometric.nn.pool import global_add_pool

torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])


# =============================================================================
# Helpers
# =============================================================================


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: x * (1 + scale) + shift"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def timestep_embedding(
    t: torch.Tensor, dim: int, max_period: int = 10000
) -> torch.Tensor:
    """Sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# =============================================================================
# VAE Components
# =============================================================================


class AtomEncoder(nn.Module):
    """Encodes atoms (type + position) into latent space using Transformer."""

    def __init__(
        self,
        max_atomic_number: int = 100,
        atom_emb_dim: int = 64,
        hidden_dim: int = 256,
        latent_dim: int = 8,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.atom_emb = nn.Embedding(max_atomic_number, atom_emb_dim)
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        self.input_proj = nn.Linear(atom_emb_dim + hidden_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self.to_latent_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_latent_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor):
        atom_feat = self.atom_emb(z)
        pos_feat = self.pos_encoder(pos)
        x = self.input_proj(torch.cat([atom_feat, pos_feat], dim=-1))

        batch_size = batch.max().item() + 1
        mu_list, logvar_list = [], []

        for b in range(batch_size):
            mask = batch == b
            x_b = self.transformer(x[mask].unsqueeze(0)).squeeze(0)
            mu_list.append(self.to_latent_mu(x_b))
            logvar_list.append(self.to_latent_logvar(x_b))

        return torch.cat(mu_list, dim=0), torch.cat(logvar_list, dim=0)


class AtomDecoder(nn.Module):
    """Decodes latent representations back to atom types and positions."""

    def __init__(
        self,
        max_atomic_number: int = 100,
        latent_dim: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.latent_proj = nn.Linear(latent_dim, hidden_dim)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            decoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self.to_atom_type = nn.Linear(hidden_dim, max_atomic_number)
        self.to_position = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, latent: torch.Tensor, batch: torch.Tensor):
        x = self.latent_proj(latent)

        batch_size = batch.max().item() + 1
        x_out_list = []

        for b in range(batch_size):
            mask = batch == b
            x_out_list.append(self.transformer(x[mask].unsqueeze(0)).squeeze(0))

        x = torch.cat(x_out_list, dim=0)
        return self.to_atom_type(x), self.to_position(x)


class AtomVAE(nn.Module):
    """VAE for atomic structures."""

    def __init__(
        self,
        max_atomic_number: int = 100,
        atom_emb_dim: int = 64,
        hidden_dim: int = 256,
        latent_dim: int = 8,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        kl_weight: float = 1e-4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

        self.encoder = AtomEncoder(
            max_atomic_number,
            atom_emb_dim,
            hidden_dim,
            latent_dim,
            num_layers,
            num_heads,
            dropout,
        )
        self.decoder = AtomDecoder(
            max_atomic_number, latent_dim, hidden_dim, num_layers, num_heads, dropout
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def encode(self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor):
        return self.encoder(z, pos, batch)

    def decode(self, latent: torch.Tensor, batch: torch.Tensor):
        return self.decoder(latent, batch)

    def forward(self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor):
        mu, logvar = self.encode(z, pos, batch)
        latent = self.reparameterize(mu, logvar)
        atom_logits, pos_pred = self.decode(latent, batch)
        return atom_logits, pos_pred, mu, logvar, latent

    def loss(self, atom_logits, pos_pred, mu, logvar, z_target, pos_target):
        recon_atom = F.cross_entropy(atom_logits, z_target)
        recon_pos = F.mse_loss(pos_pred, pos_target)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return {
            "total": recon_atom + recon_pos + self.kl_weight * kl_loss,
            "recon_atom": recon_atom,
            "recon_pos": recon_pos,
            "kl": kl_loss,
        }


# =============================================================================
# DiT Components
# =============================================================================


class AdaLNZeroBlock(nn.Module):
    """DiT block with adaLN-Zero conditioning."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)

        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

        # adaLN modulation: 6 params (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN_modulation(c).chunk(
            6, dim=-1
        )

        x_norm = modulate(self.norm1(x), shift1, scale1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate1.unsqueeze(1) * attn_out

        x_norm = modulate(self.norm2(x), shift2, scale2)
        x = x + gate2.unsqueeze(1) * self.mlp(x_norm)

        return x


class DiT(nn.Module):
    """Diffusion Transformer for latent space generation."""

    def __init__(
        self,
        latent_dim: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_num_atoms: int = 1000,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.pos_emb = nn.Embedding(max_num_atoms, hidden_dim)

        self.time_emb = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.blocks = nn.ModuleList(
            [
                AdaLNZeroBlock(hidden_dim, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim)
        )
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        batch: torch.Tensor,
        condition: torch.Tensor = None,
    ):
        batch_size = batch.max().item() + 1

        t_emb = timestep_embedding(t, self.hidden_dim)
        c = self.time_emb(t_emb)
        if condition is not None:
            c = c + condition

        x = self.input_proj(z)

        output_list = []
        for b in range(batch_size):
            mask = batch == b
            x_b = x[mask]
            x_b = x_b + self.pos_emb(torch.arange(x_b.shape[0], device=x_b.device))
            x_b = x_b.unsqueeze(0)

            for block in self.blocks:
                x_b = block(x_b, c[b : b + 1])

            shift, scale = self.final_adaLN(c[b : b + 1]).chunk(2, dim=-1)
            x_b = modulate(self.final_norm(x_b), shift, scale)
            output_list.append(x_b.squeeze(0))

        return self.output_proj(torch.cat(output_list, dim=0))


# =============================================================================
# Unit Cell Encoder
# =============================================================================


class UnitCellEncoder(nn.Module):
    """
    SchNet-based encoder for unit cell conditioning.
    Uses SchNet's representation (embedding + interaction blocks) but outputs graph
    embeddings (batch, hidden_dim) instead of scalar property predictions.
    PyG's SchNet outputs (batch, 1) for property prediction; we need (batch, hidden_dim).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_gaussians: int = 50,
        cutoff: float = 5.0,
    ):
        super().__init__()

        self.schnet = SchNet(
            hidden_channels=hidden_dim,
            num_filters=hidden_dim,
            num_interactions=num_layers,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            readout="add",
        )
        self.condition_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor):
        # Use SchNet's representation backbone but skip the property-prediction head.
        # SchNet's forward does: embed -> interactions -> lin1 -> act -> lin2 -> readout
        # where lin2 projects to 1. We want the (N, hidden_dim) after interactions.
        h = self.schnet.embedding(z)
        edge_index, edge_weight = self.schnet.interaction_graph(pos, batch)
        edge_attr = self.schnet.distance_expansion(edge_weight)
        for interaction in self.schnet.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
        # h: (N, hidden_dim). Pool to (batch_size, hidden_dim)
        out = global_add_pool(h, batch)
        return self.condition_proj(out)


# =============================================================================
# Full ADiT Model
# =============================================================================


class ADiTUnitCell(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="radii",
    repo_url="https://github.com/KurbanIntelligenceLab/RADII",
    pipeline_tag="other",
    license="mit",
    tags=["materials-science", "crystal-structures", "generative-models", "kdd-2026"],
):
    """
    All-atom Diffusion Transformer for RADII benchmark Task 1.
    Generates nanoparticle structures conditioned on unit cells.
    """

    def __init__(
        self,
        max_atomic_number: int = 100,
        atom_emb_dim: int = 64,
        vae_hidden_dim: int = 256,
        latent_dim: int = 8,
        vae_num_layers: int = 4,
        vae_num_heads: int = 8,
        kl_weight: float = 1e-4,
        dit_hidden_dim: int = 256,
        dit_num_layers: int = 6,
        dit_num_heads: int = 8,
        mlp_ratio: float = 4.0,
        cell_hidden_dim: int = 256,
        cell_num_layers: int = 3,
        cutoff_radius: float = 5.0,
        beta_min: float = 0.0001,
        beta_max: float = 0.02,
        num_diffusion_steps: int = 1000,
        dropout: float = 0.1,
        max_num_atoms: int = 18200,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_diffusion_steps = num_diffusion_steps

        # Diffusion schedule
        betas = torch.linspace(beta_min, beta_max, num_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

        # Precompute posterior variance for sampling
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

        # VAE
        self.vae = AtomVAE(
            max_atomic_number,
            atom_emb_dim,
            vae_hidden_dim,
            latent_dim,
            vae_num_layers,
            vae_num_heads,
            dropout,
            kl_weight,
        )

        # Unit cell encoder
        self.cell_encoder = UnitCellEncoder(
            cell_hidden_dim, cell_num_layers, cutoff=cutoff_radius
        )
        self.cell_proj = nn.Linear(cell_hidden_dim, dit_hidden_dim)

        # DiT (max_num_atoms must cover largest nanoparticle; dataset has ~2.6k atoms/sample avg)
        self.dit = DiT(
            latent_dim,
            dit_hidden_dim,
            dit_num_layers,
            dit_num_heads,
            mlp_ratio,
            dropout,
            max_num_atoms=max_num_atoms,
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

    def q_sample(
        self,
        z0: torch.Tensor,
        t: torch.Tensor,
        batch: torch.Tensor,
        noise: torch.Tensor = None,
    ):
        """Forward diffusion: add noise to latent."""
        if noise is None:
            noise = torch.randn_like(z0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][batch].unsqueeze(-1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][batch].unsqueeze(
            -1
        )

        return sqrt_alpha * z0 + sqrt_one_minus_alpha * noise, noise

    def forward(self, data: Batch, t: torch.Tensor = None):
        """Forward pass for DiT training (assumes VAE is frozen)."""
        batch = self._batch_from_ptr(data.ptr)
        batch_size = batch.max().item() + 1

        cell_ptr = self._process_cell_ptr(data.cell_ptr)
        cell_batch = self._batch_from_ptr(cell_ptr)

        # Encode unit cell (ensure cell_z is 1D for SchNet: (N,) not (N,1))
        cell_z = data.cell_z.flatten().long()
        condition = self.cell_encoder(cell_z, data.cell_pos, cell_batch)
        condition = self.cell_proj(condition)

        # Encode nanoparticle to latent (no grad for DiT training)
        with torch.no_grad():
            mu, logvar = self.vae.encode(data.z, data.pos, batch)
            z0 = self.vae.reparameterize(mu, logvar)

        # Sample timesteps
        if t is None:
            t = torch.randint(
                0, self.num_diffusion_steps, (batch_size,), device=data.z.device
            )

        # Forward diffusion
        zt, noise = self.q_sample(z0, t, batch)

        # Predict noise
        noise_pred = self.dit(
            zt, t.float() / self.num_diffusion_steps, batch, condition
        )

        return {
            "loss": F.mse_loss(noise_pred, noise),
            "noise_pred": noise_pred,
            "noise": noise,
        }

    def train_vae_step(self, data: Batch):
        """Training step for VAE."""
        batch = self._batch_from_ptr(data.ptr)
        atom_logits, pos_pred, mu, logvar, _ = self.vae(data.z, data.pos, batch)
        return self.vae.loss(atom_logits, pos_pred, mu, logvar, data.z, data.pos)

    @torch.no_grad()
    def sample(self, data: Batch, num_steps: int = None, guidance_scale: float = 1.0):
        """Generate nanoparticles conditioned on unit cells using DDPM."""
        device = next(self.parameters()).device
        num_steps = num_steps or self.num_diffusion_steps

        # Process unit cell
        cell_ptr = self._process_cell_ptr(data.cell_ptr)
        cell_batch = self._batch_from_ptr(cell_ptr)
        batch_size = cell_batch.max().item() + 1

        cell_z = data.cell_z.flatten().long()
        condition = self.cell_proj(self.cell_encoder(cell_z, data.cell_pos, cell_batch))

        # Create batch for generated atoms
        num_atoms_per_sample = data.num_atoms
        total_atoms = num_atoms_per_sample.sum().item()
        batch = torch.repeat_interleave(
            torch.arange(batch_size, device=device), num_atoms_per_sample
        )

        # Start from noise
        zt = torch.randn(total_atoms, self.latent_dim, device=device)

        # DDPM reverse process
        for i in reversed(range(num_steps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            t_norm = t.float() / self.num_diffusion_steps

            # Predict noise (with optional CFG)
            noise_pred = self.dit(zt, t_norm, batch, condition)
            if guidance_scale > 1.0:
                noise_pred_uncond = self.dit(zt, t_norm, batch, None)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred - noise_pred_uncond
                )

            # DDPM update
            alpha_t = self.alphas_cumprod[i]
            alpha_t_prev = self.alphas_cumprod_prev[i]
            beta_t = self.betas[i]

            # Predict x0
            z0_pred = (zt - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            z0_pred = torch.clamp(z0_pred, -10, 10)  # Stability clamp

            # Posterior mean
            posterior_mean = (
                torch.sqrt(alpha_t_prev) * beta_t / (1 - alpha_t) * z0_pred
                + torch.sqrt(self.alphas[i]) * (1 - alpha_t_prev) / (1 - alpha_t) * zt
            )

            if i > 0:
                noise = torch.randn_like(zt)
                zt = posterior_mean + torch.sqrt(self.posterior_variance[i]) * noise
            else:
                zt = posterior_mean

        # Decode
        atom_logits, pos_pred = self.vae.decode(zt, batch)

        return {
            "positions": pos_pred,
            "atom_types": atom_logits.argmax(dim=-1),
            "latent": zt,
            "batch": batch,
        }
