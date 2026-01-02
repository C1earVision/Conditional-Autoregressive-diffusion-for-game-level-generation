from dataclasses import dataclass, field

@dataclass
class AutoencoderConfig:
    num_tile_types: int = 13
    embedding_dim: int = 32
    latent_dim: int = 128
    patch_height: int = 14
    patch_width: int = 16

@dataclass
class DiffusionConfig:
    latent_dim: int = 128
    time_emb_dim: int = 256
    context_emb_dim: int = 128
    hidden_dims: list = field(default_factory=lambda: [256, 512, 512])
    num_res_blocks: int = 2
    cond_dropout: float = 0.05
    context_dropout: float = 0.3

@dataclass
class NormalizerConfig:
    norm: float = 11.0
    std: float = 1.0

@dataclass
class NoiseScheduleConfig:
    num_timesteps: int = 500
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule_type: str = 'linear'
    min_alpha_cumprod: float = 1e-4