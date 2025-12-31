import torch
from torch.utils.data import DataLoader
from models.autoencoder import Autoencoder
from models.latent_normalizer import LatentNormalizer
from data.dataset import PatchDatasetCreator
import pickle as pkl
import numpy as np
import yaml
from config.model_config import (
    AutoencoderConfig,
    NormalizerConfig
)
ae_config = AutoencoderConfig()
normalizer_config = NormalizerConfig()
with open('output/processed/metadata.pkl', 'rb') as f:
    metadata = pkl.load(f)

patches = np.load('output/processed/patches.npy')
difficulties = [(d['final_score']) for d in metadata]


with open('config/generation_config.yaml', 'r') as f:
    generation_config = yaml.safe_load(f)
device = 'cuda'
autoencoder_path=generation_config['models']['autoencoder_path']

autoencoder = Autoencoder(
    num_tile_types=ae_config.num_tile_types,
    embedding_dim=ae_config.embedding_dim,
    latent_dim=ae_config.latent_dim,
    patch_height=ae_config.patch_height,
    patch_width=ae_config.patch_width
)

ae_checkpoint = torch.load(autoencoder_path, map_location=device)
autoencoder.load_state_dict(ae_checkpoint['model_state_dict'])
autoencoder.to(device)
autoencoder.eval()
print("✓ Autoencoder loaded")
for param in autoencoder.parameters():
    param.requires_grad = False

print("✓ Autoencoder loaded!")

print(f"Loaded {len(patches)} patches")
print("num_scores:", len(difficulties))

dataset = PatchDatasetCreator(patches, difficulties=difficulties)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
latents = []
difficulties = []

with torch.no_grad():
    for batch in dataloader:
        patches, diffs = batch
        patches = patches.to(device)
        latent = autoencoder.encode(patches)

        latents.append(latent.cpu())
        difficulties.append(diffs.cpu())
latents = torch.cat(latents, dim=0)
difficulties = torch.cat(difficulties, 0)

print(f"Encoded latents shape: {latents.shape}")
print(f"Latent dimension: {latents.shape[1]}")
print(f"Difficulties shape: {difficulties.shape}")
print(f"latents: norm={latents.norm(dim=-1).mean():.2f}, std={latents.std():.2f}")

print("\n" + "="*70)
print("CREATING LATENT NORMALIZER")
print("="*70)

normalizer = LatentNormalizer(target_norm=normalizer_config.norm)
normalizer.fit(latents)
latents_norm = normalizer.normalize(latents)
print(f"Normalized latents: norm={latents_norm.norm(dim=-1).mean():.2f}, std={latents_norm.std():.2f}")
torch.save(latents_norm, "output/processed/latents.pt")
normalizer.save('checkpoints/latent_normalizer.pth')