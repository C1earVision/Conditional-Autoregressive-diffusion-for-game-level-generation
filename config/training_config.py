from dataclasses import dataclass


@dataclass
class AutoencoderTrainingConfig:
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    difficulty_loss_weight: float = 0.0
    print_every: int = 10
    save_path: str = './checkpoints/autoencoder.pth'

@dataclass
class DiffusionTrainingConfig:
    batch_size: int = 16
    num_epochs: int = 600
    learning_rate: float = 5e-4
    save_interval: int = 300
    save_path: str = './checkpoints/diffusion.pth'


