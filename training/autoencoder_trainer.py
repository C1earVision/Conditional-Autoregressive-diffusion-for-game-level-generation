import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from config.training_config import AutoencoderTrainingConfig

device = 'cuda'
ae_config = AutoencoderTrainingConfig()


class AutoencoderTrainer:
    def __init__(self,
                 model,
                 learning_rate: float = ae_config.learning_rate,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 difficulty_loss_weight: float = ae_config.difficulty_loss_weight):

        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        self.difficulty_loss_weight = difficulty_loss_weight
        self.difficulty_criterion = nn.MSELoss()

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                patches, diffs = batch
                patches = patches.to(self.device)
                diffs = diffs.to(self.device).view(-1)
            else:
                patches = batch.to(self.device)
                diffs = None
            if self.difficulty_loss_weight > 0.0 and diffs is not None:
                reconstruction, latent, diff_pred = self.model(patches, return_difficulty=True)
            else:
                reconstruction, latent = self.model(patches)
                diff_pred = None
            recon_loss = self.criterion(reconstruction, patches)

            loss = recon_loss
            if (self.difficulty_loss_weight > 0.0) and (diffs is not None):
                diff_loss = self.difficulty_criterion(diff_pred, diffs)
                loss = loss + self.difficulty_loss_weight * diff_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        return avg_loss

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    patches, diffs = batch
                    patches = patches.to(self.device)
                    diffs = diffs.to(self.device).view(-1)
                else:
                    patches = batch.to(self.device)
                    diffs = None
                if self.difficulty_loss_weight > 0.0 and diffs is not None:
                    reconstruction, latent, diff_pred = self.model(patches, return_difficulty=True)
                else:
                    reconstruction, latent = self.model(patches)
                    diff_pred = None

                recon_loss = self.criterion(reconstruction, patches)
                loss = recon_loss
                if (self.difficulty_loss_weight > 0.0) and (diffs is not None):
                    diff_loss = self.difficulty_criterion(diff_pred, diffs)
                    loss = loss + self.difficulty_loss_weight * diff_loss

                predictions = torch.argmax(reconstruction, dim=1)
                accuracy = (predictions == patches).float().mean()

                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        avg_accuracy = total_accuracy / max(1, num_batches)
        return avg_loss, avg_accuracy

    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              num_epochs: int = ae_config.num_epochs,
              print_every: int = ae_config.print_every):

        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Difficulty loss weight: {self.difficulty_loss_weight}")
        print("-" * 70)

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            if val_loader is not None:
                val_loss, val_accuracy = self.validate(val_loader)
                self.val_losses.append(val_loss)

                if (epoch + 1) % print_every == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}] "
                          f"Train Loss: {train_loss:.4f} | "
                          f"Val Loss: {val_loss:.4f} | "
                          f"Val Accuracy: {val_accuracy:.4f}")
            else:
                if (epoch + 1) % print_every == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}")

        print("-" * 70)
        print("Training complete!")

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Autoencoder Training Progress')
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_model(self, path: str = './checkpoints/autoencoder.pth'):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'difficulty_loss_weight': self.difficulty_loss_weight
        }

        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str = './checkpoints/autoencoder.pth'):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.difficulty_loss_weight = checkpoint.get('difficulty_loss_weight', 0.0)

        print(f"Model loaded from {path}")