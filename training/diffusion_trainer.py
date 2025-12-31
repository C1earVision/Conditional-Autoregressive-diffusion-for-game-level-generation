import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config.training_config import DiffusionTrainingConfig
diffusion_config = DiffusionTrainingConfig()
class DiffusionTrainer:
    def __init__(
        self,
        unet,
        noise_schedule,
        forward_diffusion,
        learning_rate: float = diffusion_config.learning_rate,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.unet = unet.to(device)
        self.schedule = noise_schedule
        self.forward = forward_diffusion
        self.device = device

        for name in ["sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
                     "posterior_variance", "alphas", "betas", "alphas_cumprod"]:
            val = getattr(self.schedule, name, None)
            if isinstance(val, torch.Tensor):
                setattr(self.schedule, name, val.to(device))

        self.optimizer = optim.AdamW(self.unet.parameters(), lr=learning_rate, weight_decay=1e-3)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=learning_rate * 0.01
        )
        self.criterion = nn.MSELoss()

        self.train_losses = []
        self.val_losses = []
        self.epoch_losses = []

        print(f"\n{'='*70}")
        print(f"CFG Trainer Initialized")
        print(f"{'='*70}")
        print(f"  Device: {device}")
        print(f"  Condition dropout: {unet.cond_dropout*100:.0f}%")
        print(f"{'='*70}\n")

    def train_step(self, batch_data):
        """Single training step."""
        self.unet.train()

        current_latent, prev_latents, curr_play, prev_play = batch_data
        current_latent = current_latent.to(self.device).float()
        prev_latents = prev_latents.to(self.device).float()
        curr_play = curr_play.to(self.device).float()
        prev_play = prev_play.to(self.device).float()

        batch_size = current_latent.shape[0]
        timesteps = self.forward.sample_timesteps(batch_size).to(self.device).long()

        noisy_latent, noise = self.forward.add_noise(current_latent, timesteps)

        if curr_play.dim() == 2:
            target_diff = curr_play.squeeze(1)
        else:
            target_diff = curr_play

        predicted_noise = self.unet(
            x=noisy_latent,
            timesteps=timesteps,
            previous_latents=prev_latents,
            previous_playabilities=prev_play,
            target_difficulty=target_diff
        )

        loss = self.criterion(predicted_noise, noise)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {'total': loss.item(), 'noise': loss.item()}

    def validate(self, val_loader: DataLoader) -> float:
        self.unet.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_data in val_loader:
                current_latent, prev_latents, curr_play, prev_play = batch_data
                current_latent = current_latent.to(self.device).float()
                prev_latents = prev_latents.to(self.device).float()
                curr_play = curr_play.to(self.device).float()
                prev_play = prev_play.to(self.device).float()

                batch_size = current_latent.shape[0]
                timesteps = self.forward.sample_timesteps(batch_size).to(self.device).long()
                noisy_latents, noise = self.forward.add_noise(current_latent, timesteps)

                if curr_play.dim() == 2:
                    target_diff = curr_play.squeeze(1)
                else:
                    target_diff = curr_play

                predicted_noise = self.unet(
                    x=noisy_latents,
                    timesteps=timesteps,
                    previous_latents=prev_latents,
                    previous_playabilities=prev_play,
                    target_difficulty=target_diff
                )

                loss = self.criterion(predicted_noise, noise)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        return avg_loss

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = diffusion_config.num_epochs,
        save_interval: int = diffusion_config.save_interval,
        save_path: str = diffusion_config.save_path
    ):
        print("=" * 70)
        print("TRAINING CFG DIFFUSION MODEL")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.unet.parameters()):,}")
        print(f"Epochs: {num_epochs}")

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=self.optimizer.param_groups[0]['lr'] * 0.01
        )

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            epoch_loss = 0
            self.unet.train()
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch_data in progress_bar:
                losses = self.train_step(batch_data)
                epoch_loss += losses['total']
                progress_bar.set_postfix({
                    'loss': f"{losses['total']:.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })

            avg_epoch_loss = epoch_loss / len(train_loader)
            self.epoch_losses.append(avg_epoch_loss)
            self.scheduler.step()

            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(save_path.replace('.pth', '_best.pth'))
                print(f"Epoch {epoch+1}/{num_epochs} | Train: {avg_epoch_loss:.4f} | Val: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{num_epochs} | Train: {avg_epoch_loss:.4f}")

            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(save_path.replace('.pth', f'_epoch{epoch+1}.pth'))

        self.save_checkpoint(save_path)
        print(f"\n✓ Training complete! Best val loss: {best_val_loss:.4f}")

    def save_checkpoint(self, path: str):
        torch.save({
            'unet_state_dict': self.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch_losses': self.epoch_losses,
            'val_losses': self.val_losses,
        }, path)
        print(f"✓ Checkpoint saved to {path}")

    def plot_losses(self, save_path: Optional[str] = None):
        plt.figure(figsize=(8, 5))
        plt.plot(self.epoch_losses, label='Train Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Loss plot saved to {save_path}")
            plt.close()
        else:
            plt.show()