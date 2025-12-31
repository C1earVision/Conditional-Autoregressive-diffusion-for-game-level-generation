import torch
from torch.utils.data import DataLoader
from torch import nn

class LatentNormalizer:

    def __init__(self, target_norm: float = 11.0):
        self.target_norm = target_norm
        self.scale_factor = None
        self.original_mean_norm = None

    def fit(self, latents: torch.Tensor):
        norms = latents.norm(dim=-1)
        self.original_mean_norm = norms.mean().item()
        self.scale_factor = self.target_norm / self.original_mean_norm

        print(f"Latent Normalizer fitted:")
        print(f"  Original mean norm: {self.original_mean_norm:.2f}")
        print(f"  Target mean norm: {self.target_norm:.2f}")
        print(f"  Scale factor: {self.scale_factor:.6f}")

    def fit_from_dataloader(self, dataloader: DataLoader, model: nn.Module, device: str = 'cuda'):
        model.eval()
        all_norms = []

        print("Computing latent statistics from data...")
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    patches = batch[0]
                else:
                    patches = batch

                patches = patches.to(device)
                latents = model.encode(patches)
                norms = latents.norm(dim=-1)
                all_norms.append(norms.cpu())

        all_norms = torch.cat(all_norms)
        self.original_mean_norm = all_norms.mean().item()
        self.scale_factor = self.target_norm / self.original_mean_norm

        print(f"Latent Normalizer fitted:")
        print(f"  Original mean norm: {self.original_mean_norm:.2f}")
        print(f"  Original std: {all_norms.std().item():.2f}")
        print(f"  Target mean norm: {self.target_norm:.2f}")
        print(f"  Scale factor: {self.scale_factor:.6f}")

    def normalize(self, latents: torch.Tensor) -> torch.Tensor:
        return latents * self.scale_factor

    def denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        return latents / self.scale_factor

    def save(self, path: str):
        torch.save({
            'scale_factor': self.scale_factor,
            'target_norm': self.target_norm,
            'original_mean_norm': self.original_mean_norm
        }, path)
        print(f"Latent normalizer saved to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.scale_factor = checkpoint['scale_factor']
        self.target_norm = checkpoint['target_norm']
        self.original_mean_norm = checkpoint['original_mean_norm']
        print(f"Latent normalizer loaded from {path}")
        print(f"  Scale factor: {self.scale_factor:.6f}")
        print(f"  Original norm: {self.original_mean_norm:.2f} -> Target: {self.target_norm:.2f}")
