import torch
import argparse
from models.autoencoder import Autoencoder
from models.diffusion import DiffusionUNet
from generation.sampler import Sampler
from models.latent_normalizer import LatentNormalizer
from models.noise_scheduler import NoiseSchedule
from generation.stitcher import PatchStitcher
from data.parser import LevelParser
import yaml
from config.model_config import (
    AutoencoderConfig,
    DiffusionConfig,
    NoiseScheduleConfig,
    NormalizerConfig
)

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

arg_parser = argparse.ArgumentParser(description='Generate levels with diffusion model')
arg_parser.add_argument('--difficulty', type=float, default=None, help='Difficulty target (0.0-1.0)')
arg_parser.add_argument('--temperature', type=float, default=None, help='Sampling temperature')
arg_parser.add_argument('--guidance', type=float, default=None, help='CFG guidance scale')
arg_parser.add_argument('--patches', type=int, default=None, help='Number of patches per level')
arg_parser.add_argument('--num_levels', type=int, default=None, help='Number of levels to generate')
args = arg_parser.parse_args()

device = 'cuda'
ae_config = AutoencoderConfig()
diff_config = DiffusionConfig()
schedule_config = NoiseScheduleConfig()
normalizer_config = NormalizerConfig()
parser = LevelParser()

print("="*70)
print(f"GENERATING LEVELS")
print("="*70)

device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

with open('config/generation_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

patches_per_level = args.patches if args.patches is not None else config['generation']['patches_per_level']
num_levels = args.num_levels if args.num_levels is not None else config['generation']['num_levels']
difficulty_target = args.difficulty if args.difficulty is not None else config['generation']['difficulty_target']
temperature = args.temperature if args.temperature is not None else config['generation']['temperature']
guidance_scale = args.guidance if args.guidance is not None else config['generation']['guidance_scale']

autoencoder_path = config['models']['autoencoder_path']
diffusion_path = config['models']['diffusion_path']
normalizer_path = config['models']['normalizer_path']
output_dir = config['output']['directory']

# Load seed context from training data (first latent of first level)
import pickle as pkl
latents_all = torch.load('output/processed/latents.pt')
with open('output/processed/metadata.pkl', 'rb') as f:
    metadata = pkl.load(f)
difficulties_all = [d['final_score'] for d in metadata]
seed_latent = latents_all[0]  # First latent (normalized)
seed_difficulty = difficulties_all[0]  # First difficulty
print(f"✓ Loaded seed context: latent shape={seed_latent.shape}, difficulty={seed_difficulty:.3f}")


autoencoder = Autoencoder(
    num_tile_types=ae_config.num_tile_types,
    embedding_dim=ae_config.embedding_dim,
    latent_dim=ae_config.latent_dim,
    patch_height=ae_config.patch_height,
    patch_width=ae_config.patch_width
)

ae_checkpoint = torch.load(autoencoder_path, map_location=device)

state = ae_checkpoint['model_state_dict']

try:
    autoencoder.load_state_dict(state)
except Exception as e:
    missing, unexpected = autoencoder.load_state_dict(state, strict=False)
    print("Warning: loaded autoencoder with strict=False. Missing keys:", missing, "Unexpected:", unexpected)
autoencoder.to(device)
autoencoder.eval()
print("✓ Autoencoder loaded")

unet = DiffusionUNet(
    latent_dim=diff_config.latent_dim,
    time_emb_dim=diff_config.time_emb_dim,
    context_emb_dim=diff_config.context_emb_dim,
    hidden_dims=diff_config.hidden_dims,
    num_res_blocks=diff_config.num_res_blocks,
    cond_dropout=diff_config.cond_dropout,
).to(device)

diff_checkpoint = torch.load(diffusion_path, map_location=device)
unet_state = diff_checkpoint['unet_state_dict']


try:
    unet.load_state_dict(unet_state)
except Exception as e:
    missing, unexpected = unet.load_state_dict(unet_state, strict=False)
    print("Warning: loaded UNet with strict=False. Missing keys:", missing, "Unexpected:", unexpected)

unet.to(device)
unet.eval()
print("✓ Autoregressive Diffusion U-Net loaded")

schedule = NoiseSchedule(
    num_timesteps=schedule_config.num_timesteps,
    schedule_type=schedule_config.schedule_type,
    device=device,
)

normalizer = LatentNormalizer(target_norm=normalizer_config.norm)
normalizer.load(normalizer_path)

sampler = Sampler(
    unet,
    schedule,
    device=device
)
print("✓ Autoregressive sampler initialized")

stitcher = PatchStitcher(patch_height=14, patch_width=16, stride=4)
generated_levels = []


for level_idx in range(num_levels):
    print(f"\n{'='*70}")
    print(f"Generating Level {level_idx + 1}/{num_levels}")
    print(f"{'='*70}")

    print(f"Generating {patches_per_level} patches sequentially (autoregressive)...")
    for attr in ['scale','mean','target_norm','original_norm','scale_factor']:
        if hasattr(normalizer, attr):
            print(attr, getattr(normalizer, attr))
        else:
            print("no attr", attr)

    latents = sampler.sample_level(
        num_patches=patches_per_level,
        normalizer=normalizer,
        difficulty_target=difficulty_target,
        temperature=temperature,
        guidance_scale=guidance_scale,
        show_progress=True,
        seed_latent=seed_latent,
        seed_difficulty=seed_difficulty,
    )

    latents_denorm = normalizer.denormalize(latents)

    print("normalizer attributes:")
    for a in ["scale_factor","target_norm","original_mean_norm","original_std_norm"]:
        if hasattr(normalizer, a):
            print(f"  {a} =", getattr(normalizer, a))
        else:
            print(f"  {a} = <missing>")

    print("\nRaw sampler stats (per-dim):", "min/max:", float(latents.min()), float(latents.max()))
    print("Raw mean/std (per-dim):", float(latents.mean()), float(latents.std()))

    ld = latents_denorm.detach().cpu()
    print("\nDenorm per-dim stats: mean/std:", float(ld.mean()), float(ld.std()))
    print("Denorm min/max:", float(ld.min()), float(ld.max()))

    l2 = ld.norm(dim=-1)
    print(f"Denorm latents norm = {l2}")
    print("\nL2 norms (denorm) stats: mean/std/min/max:", float(l2.mean()), float(l2.std()), float(l2.min()), float(l2.max()))
    if not torch.is_tensor(latents_denorm):
        raise RuntimeError("Sampler did not return a torch.Tensor. Got: %s" % type(latents_denorm))

    print(f"✓ Generated {latents_denorm.shape[0]} latents (shape: {latents_denorm.shape})")

    with torch.no_grad():
        latents_denorm = latents_denorm.to(device)
        decoded_logits = autoencoder.decoder(latents_denorm)

        if decoded_logits.dim() == 4:
            patches = torch.argmax(decoded_logits, dim=1)
        elif decoded_logits.dim() == 3:
            patches = decoded_logits.long()
        elif decoded_logits.dim() == 5:
            decoded_logits = decoded_logits.squeeze(1)
            patches = torch.argmax(decoded_logits, dim=1)
        else:
            patches = torch.argmax(decoded_logits, dim=-1)

    patches_np = patches.cpu().numpy()

    if patches_np.shape[1] != stitcher.patch_height:
        print(f"Warning: patch height mismatch: decoded H={patches_np.shape[1]}, stitcher.patch_height={stitcher.patch_height}")

    level = stitcher.stitch_patches_to_level(patches_np)

    print(f"\n--- Generated Level {level_idx + 1} ---")
    level_str = parser.decode_level(level)
    print(level_str)


    generated_levels.append(level_str)

print(f"\n{'='*70}")
print(f"✓ GENERATION COMPLETE! Generated {len(generated_levels)} levels")
print(f"{'='*70}")
print(f"Saving levels to output/generated_levels/")

from mario_gpt import MarioDataset, MarioLM
from mario_gpt.utils import view_level, convert_level_to_png, join_list_of_list, characterize

mario_lm = MarioLM()
for i in range(len(generated_levels)):
    _map = generated_levels[i].split("\n")
    _map_png = convert_level_to_png(_map,  mario_lm.tokenizer)[0]
    _map_png.save(f'output/generated_levels/level_{i+1}.png')
    print(f'✓ Saved level {i+1} to output/generated_levels/level_{i+1}.png')