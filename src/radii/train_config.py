"""Shared config for task_1: TrainConfig and ModelConfig (data, training, scheduler, model params)."""

import torch
import random
import numpy as np


# =============================================================================
# TrainConfig: data, training, eval, device, scheduler params
# =============================================================================
class TrainConfig:
    # Data
    DATA_ROOT = "radii"
    BATCH_SIZE = 2
    NUM_WORKERS = 0
    DATA_NUM_WORKERS = 4
    LOADED_FRAC = 1
    TRAIN_RATIO = 0.8

    # Training
    LR = 1e-4
    NUM_EPOCHS = 50
    SEEDS = [1, 2, 3]
    GRAD_CLIP_NORM = 1.0

    # Evaluation / sampling
    VAE_PRETRAIN_EPOCHS = 2
    EVAL_BATCH_SIZE = 1
    EVAL_NUM_STEPS = 25
    EVAL_GUIDANCE_SCALE = 1.0
    EVAL_LANGEVIN_STEPS = 25
    EVAL_STEP_SIZE = 0.01

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_MODE = "min"
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5

    ID_RADII = [11, 13, 15, 17, 19, 21]
    OOD_RADII = [6, 7, 29, 30]

    @staticmethod
    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def make_scheduler(optimizer):
        """Build ReduceLROnPlateau scheduler from TrainConfig."""
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=TrainConfig.SCHEDULER_MODE,
            factor=TrainConfig.SCHEDULER_FACTOR,
            patience=TrainConfig.SCHEDULER_PATIENCE,
        )

    @staticmethod
    def setup_torch(safe_globals=None):
        """Register safe globals for serialization (if provided) and set backend flags (cudnn benchmark, tf32)."""
        if safe_globals is not None:
            torch.serialization.add_safe_globals(safe_globals)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


# =============================================================================
# ModelConfig: per-model params (~500-550k params each)
# =============================================================================
class ModelConfig:
    @staticmethod
    def _model_to_dict(cls):
        return {
            k: v
            for k, v in vars(cls).items()
            if not k.startswith("_")
            and not callable(v)
            and type(v) not in (classmethod, staticmethod)
        }

    class ADiT:
        max_atomic_number = 100
        atom_emb_dim = 12
        vae_hidden_dim = 24
        latent_dim = 8
        vae_num_layers = 2
        vae_num_heads = 4
        kl_weight = 1e-4
        dit_hidden_dim = 24
        dit_num_layers = 2
        dit_num_heads = 4
        mlp_ratio = 4.0
        cell_hidden_dim = 24
        cell_num_layers = 2
        cutoff_radius = 5.0
        beta_min = 0.0001
        beta_max = 0.02
        num_diffusion_steps = 1000
        dropout = 0.1
        max_num_atoms = 18200

        @classmethod
        def to_dict(cls):
            return ModelConfig._model_to_dict(cls)

    class CDVAE:
        max_atomic_number = 100
        hidden_dim = 92
        latent_dim = 92
        encoder_num_layers = 2
        decoder_num_layers = 2
        num_gaussians = 50
        cutoff_radius = 5.0
        r_emb_dim = 32
        num_noise_levels = 50
        sigma_min = 0.01
        sigma_max = 1.0

        @classmethod
        def to_dict(cls):
            return ModelConfig._model_to_dict(cls)

    class DiffCSP:
        max_atomic_number = 100
        hidden_dim = 120
        num_layers = 2
        num_gaussians = 50
        cutoff_radius = 7.0
        time_dim = 64
        cond_hidden_dim = 120
        cond_num_layers = 2
        r_emb_dim = 32
        num_diffusion_steps = 1000
        beta_start = 1e-4
        beta_end = 0.02

        @classmethod
        def to_dict(cls):
            return ModelConfig._model_to_dict(cls)

    class FlowMM:
        hidden_dim = 120
        num_layers = 2
        num_gaussians = 50
        cutoff_radius = 5.0
        time_dim = 64
        max_atomic_number = 100
        cond_hidden_dim = 120
        cond_num_layers = 2
        sigma_min = 1e-4

        @classmethod
        def to_dict(cls):
            return ModelConfig._model_to_dict(cls)

    class MatterGen:
        hidden_dim = 130
        num_layers = 2
        cutoff_radius = 5.0
        time_dim = 64
        num_diffusion_steps = 100
        cond_hidden_dim = 88
        cond_num_layers = 2
        num_atom_types = 100
        num_gaussians = 50

        @classmethod
        def to_dict(cls):
            return ModelConfig._model_to_dict(cls)
