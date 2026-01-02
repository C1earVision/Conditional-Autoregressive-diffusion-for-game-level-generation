import torch
import pickle as pkl
from torch.utils.data import DataLoader, random_split
from models.autoencoder import Autoencoder
from models.diffusion import DiffusionUNet
from training.diffusion_trainer import DiffusionTrainer
from data.dataset import AutoregressivePatchDatasetCreator
from models.noise_scheduler import NoiseSchedule, ForwardDiffusion
from config.model_config import (
    AutoencoderConfig,
    DiffusionConfig,
    NoiseScheduleConfig
)
from config.training_config import (
    AutoencoderTrainingConfig,
    DiffusionTrainingConfig,
)
ae_config = AutoencoderConfig()
diff_config = DiffusionConfig()
schedule_config = NoiseScheduleConfig()
device = 'cuda'
ae_train_config = AutoencoderTrainingConfig()
diff_train_config = DiffusionTrainingConfig()
autoencoder_path = ae_train_config.save_path
save_path = diff_train_config.save_path
batch_size = diff_train_config.batch_size
num_epochs = diff_train_config.num_epochs
save_interval = diff_train_config.save_interval
with open('output/processed/metadata.pkl', 'rb') as f:
    metadata = pkl.load(f)

difficulties = [(d['final_score']) for d in metadata]

latents = torch.load('output/processed/latents.pt')
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
print("\n" + "="*70)
print("STEP 4: Creating Training Dataset")
print("="*70)

print(f'scores min/max: {min(difficulties)}/{max(difficulties)}')
dataset = AutoregressivePatchDatasetCreator(latents, difficulties, num_prev=5)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
batch = next(iter(train_loader))
latents_check = batch[0]
print(f"Latent norm: {latents_check.norm(dim=-1).mean():.4f}")
print(f"Latent std: {latents_check.std():.4f}")

print(f"  Train: {train_size} samples")
print(f"  Val: {val_size} samples")
print("\n" + "="*70)
print("STEP 5: Creating Conditional U-Net")
print("="*70)
unet = DiffusionUNet(
    latent_dim=diff_config.latent_dim,
    time_emb_dim=diff_config.time_emb_dim,
    context_emb_dim=diff_config.context_emb_dim,
    hidden_dims=diff_config.hidden_dims,
    num_res_blocks=diff_config.num_res_blocks,
    cond_dropout=diff_config.cond_dropout,
    context_dropout=diff_config.context_dropout,
).to(device)

print("="*70)
print("MODEL INITIALIZATION CHECK")
print("="*70)
print(f"Output projection type: {type(unet.output_proj)}")
print("="*70)

num_params = sum(p.numel() for p in unet.parameters())
print(f"✓ Model created: {num_params:,} parameters")
schedule = NoiseSchedule(
    num_timesteps=schedule_config.num_timesteps,
    schedule_type=schedule_config.schedule_type,
    device=device,
)
forward = ForwardDiffusion(schedule)

trainer = DiffusionTrainer(
    unet=unet,
    noise_schedule=schedule,
    forward_diffusion=forward,
    learning_rate=diff_train_config.learning_rate,
    device=device
)

print("\n" + "="*70)
print("STEP 7: Training")
print("="*70)
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=num_epochs,
    save_interval=save_interval,
    save_path=save_path
)

trainer.plot_losses('output/visualizations/losses.png')

full_checkpoint = {
    'trainer_state': {
        'epoch_losses': trainer.epoch_losses,
        'val_losses': trainer.val_losses,
        'train_losses': trainer.train_losses,
    },
    'unet_state_dict': unet.state_dict(),
    'val_loader_dataset_indices': val_dataset.indices,
    'config': {
        'latent_dim': diff_config.latent_dim,
        'time_emb_dim': diff_config.time_emb_dim,
        'context_emb_dim': diff_config.context_emb_dim,
        'hidden_dims': diff_config.hidden_dims,
        'num_res_blocks': diff_config.num_res_blocks,
        'batch_size': batch_size,
    }
}

torch.save(full_checkpoint, 'checkpoints/full_training_state.pth')
print("✓ Full training state saved")


with open('checkpoints/val_loader_info.pkl', 'wb') as f:
    pkl.dump({
        'val_indices': val_dataset.indices,
        'batch_size': batch_size,
    }, f)