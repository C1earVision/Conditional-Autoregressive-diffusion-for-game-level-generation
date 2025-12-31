import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class NoiseSchedule:
    def __init__(self,
                 num_timesteps: int = 500,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02,
                 schedule_type: str = 'linear',
                 device: str = 'cpu',
                 min_alpha_cumprod: float = 1e-4):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_type = schedule_type
        self.device = device
        self.min_alpha_cumprod = min_alpha_cumprod
        self.betas = self._linear_schedule()

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod = torch.clamp(self.alphas_cumprod, min=min_alpha_cumprod, max=1.0)

        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas.clamp(min=1e-8))
        safe_denominator = torch.clamp(1.0 - self.alphas_cumprod, min=1e-8)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / safe_denominator
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-8, max=1.0)
        self._to_device(device)
        self._verify_schedule()

    def _linear_schedule(self) -> torch.Tensor:
        return torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)


    def _to_device(self, device: str):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)

    def _verify_schedule(self):
        print(f"\n{'='*60}")
        print(f"Noise Schedule Verification ({self.schedule_type}, T={self.num_timesteps})")
        print(f"{'='*60}")
        print(f"  t=0 (clean):")
        print(f"    alphas_cumprod: {self.alphas_cumprod[0]:.6f}")
        print(f"    sqrt_alphas_cumprod: {self.sqrt_alphas_cumprod[0]:.6f}")
        print(f"  t={self.num_timesteps-1} (noisy):")
        print(f"    alphas_cumprod: {self.alphas_cumprod[-1]:.6f}")
        print(f"    sqrt_alphas_cumprod: {self.sqrt_alphas_cumprod[-1]:.6f}")
        print(f"  Minimum sqrt_alphas_cumprod: {self.sqrt_alphas_cumprod.min():.6f}")
        print(f"  Maximum sqrt_alphas_cumprod: {self.sqrt_alphas_cumprod.max():.6f}")
        if self.sqrt_alphas_cumprod.min() < 0.01:
            print(f" WARNING: sqrt_alphas_cumprod minimum is very small!")
            print(f"Consider increasing min_alpha_cumprod parameter")
        else:
            print(f"  ✓ Schedule is numerically stable")
        print(f"{'='*60}\n")

    def visualize_schedule(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        timesteps = np.arange(self.num_timesteps)
        axes[0, 0].plot(timesteps, self.betas.cpu().numpy())
        axes[0, 0].set_title('Beta Schedule (Noise Variance)')
        axes[0, 0].set_xlabel('Timestep')
        axes[0, 0].set_ylabel('β_t')
        axes[0, 0].grid(True)
        axes[0, 1].plot(timesteps, self.alphas_cumprod.cpu().numpy())
        axes[0, 1].set_title('Cumulative Alpha Product')
        axes[0, 1].set_xlabel('Timestep')
        axes[0, 1].set_ylabel('ᾱ_t')
        axes[0, 1].grid(True)
        axes[0, 1].axhline(y=self.min_alpha_cumprod, color='r', linestyle='--',
                          label=f'Min clamp ({self.min_alpha_cumprod})')
        axes[0, 1].legend()
        axes[1, 0].plot(timesteps, self.sqrt_alphas_cumprod.cpu().numpy(), label='Signal')
        axes[1, 0].plot(timesteps, self.sqrt_one_minus_alphas_cumprod.cpu().numpy(), label='Noise')
        axes[1, 0].set_title('Signal vs Noise Over Time')
        axes[1, 0].set_xlabel('Timestep')
        axes[1, 0].set_ylabel('Coefficient')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        snr = self.alphas_cumprod / (1 - self.alphas_cumprod.clamp(max=0.9999))
        axes[1, 1].plot(timesteps, snr.cpu().numpy())
        axes[1, 1].set_title('Signal-to-Noise Ratio')
        axes[1, 1].set_xlabel('Timestep')
        axes[1, 1].set_ylabel('SNR')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()


class ForwardDiffusion:

    def __init__(self, noise_schedule: NoiseSchedule):
        self.schedule = noise_schedule
        self.num_timesteps = noise_schedule.num_timesteps
        self.device = noise_schedule.device

    def add_noise(self,
                  x_0: torch.Tensor,
                  t: torch.Tensor,
    ):

        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.schedule.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.schedule.sqrt_one_minus_alphas_cumprod[t]
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1)
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

        return x_t, noise

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)

    def get_noise_level(self, t: int) -> float:
        return self.schedule.sqrt_one_minus_alphas_cumprod[t].item()