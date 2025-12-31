import torch
from torch.utils.data import DataLoader
import pickle
import numpy as np
from models.autoencoder import Autoencoder
from training.autoencoder_trainer import AutoencoderTrainer
from data.dataset import PatchDatasetCreator
from config.model_config import AutoencoderConfig
from config.training_config import AutoencoderTrainingConfig
ae_config = AutoencoderConfig()
ae_train_config = AutoencoderTrainingConfig()
with open('output/processed/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

patches = np.load('output/processed/patches.npy')
difficulties = [(d['final_score']) for d in metadata]

print("Level Autoencoder")
print("=" * 70)
print(f'Patches shape: {patches.shape}')
print(f'Difficulties shape: {np.array(difficulties).shape}')
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
print("-" * 70)
dataset = PatchDatasetCreator(patches, difficulties=difficulties)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Train size: {train_size}, Val size: {val_size}")
print("-" * 70)
model = Autoencoder(
  num_tile_types=ae_config.num_tile_types,
  embedding_dim=ae_config.embedding_dim,
  latent_dim=ae_config.latent_dim,
  patch_height=ae_config.patch_height,
  patch_width=ae_config.patch_width,
)


print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Latent dimension: {model.latent_dim}")
print("-" * 70)
trainer = AutoencoderTrainer(model, learning_rate=ae_train_config.learning_rate, difficulty_loss_weight=ae_train_config.difficulty_loss_weight)

print("\nStarting training...")
trainer.train(train_loader, val_loader, num_epochs=ae_train_config.num_epochs, print_every=ae_train_config.print_every)

print("\n" + "=" * 70)
print("Testing reconstruction on a sample:")
model.eval()
with torch.no_grad():
    sample = patches[0:1]
    sample_tensor = torch.from_numpy(sample).long().to(trainer.device)
    latent = model.encode(sample_tensor)
    reconstruction = model.reconstruct(sample_tensor)

    print(f"Original shape: {sample.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    accuracy = (reconstruction.cpu().numpy() == sample).mean()
    print(f"Reconstruction accuracy: {accuracy:.2%}")

trainer.save_model(ae_train_config.save_path)