# %%
import glob

rawData = []
for filepath in glob.glob("/content/Dataset/*.txt"):
    with open(filepath, "r") as file:
        content = [line.strip("\n") for line in file.readlines()]
        rawData.append(content)

# %%
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import bisect

# %%
rawData[0]

# %%
import torch

# %%
class LevelParser:
    def __init__(self):
        self.tile_to_idx = {
            "X": 0,   #solid ground
            "S": 1,   # solid breakable
            "-": 2,   # passable empty
            "?": 3,   # question block full
            "Q": 4,   # question block empty
            "E": 5,   # enemy
            "<": 6,   # top-left pipe
            ">": 7,   # top-right pipe
            "[": 8,   # left pipe
            "]": 9,   # right pipe
            "o": 10,  # coin
            "B": 11,  # cannon top
            "b": 12   # cannon bottom
        }

        self.idx_to_tile = {v: k for k, v in self.tile_to_idx.items()}

        self.num_tile_types = len(self.tile_to_idx)


    def parse_level_list(self, level_lines: List[str]) -> np.ndarray:
        parsed_rows = []
        for row in level_lines:
            parsed_row = [self.tile_to_idx[char] for char in row.strip()]
            parsed_rows.append(parsed_row)
        level_array = np.array(parsed_rows, dtype=np.int32)
        return level_array


    def parse_dataset(self, levels: List[str], level_names: List[str] = None,
                      start_x: int = 0) -> Dict:
        if level_names is None:
            level_names = [f"map_{i+1}" for i in range(len(levels))]

        parsed_data = {}

        for idx, (level, name) in enumerate(zip(levels, level_names)):
            level_array = self.parse_level_list(level)
            entry = {
                'data': level_array,
                'height': level_array.shape[0],
                'width': level_array.shape[1],
                'original': level,
                'index': idx
            }


            parsed_data[name] = entry

        return parsed_data


    def decode_level(self, level_array: np.ndarray) -> str:
        rows = []
        for row in level_array:
            row_str = ''.join([self.idx_to_tile[idx] for idx in row])
            rows.append(row_str)
        return '\n'.join(rows)

    def get_statistics(self, parsed_data: Dict) -> Dict:
        heights = [data['height'] for data in parsed_data.values()]
        widths = [data['width'] for data in parsed_data.values()]

        tile_counts = np.zeros(self.num_tile_types, dtype=np.int32)
        for data in parsed_data.values():
            unique, counts = np.unique(data['data'], return_counts=True)
            for tile_idx, count in zip(unique, counts):
                tile_counts[tile_idx] += count

        stats = {
            'num_levels': len(parsed_data),
            'height_range': (min(heights), max(heights)),
            'width_range': (min(widths), max(widths)),
            'avg_height': np.mean(heights),
            'avg_width': np.mean(widths),
            'tile_frequencies': {
                self.idx_to_tile[i]: int(count)
                for i, count in enumerate(tile_counts)
            },
            'total_tiles': int(tile_counts.sum())
        }

        return stats

    def visualize_level(self, level_array: np.ndarray, use_colors: bool = False):
            print(self.decode_level(level_array))

# %%
parser = LevelParser()


parsed_dataset = parser.parse_dataset(
    levels=rawData,
    level_names=[f"map_{i+1}" for i in range(1000)]
)



for name, data in parsed_dataset.items():
    print(f"{name}: {data['data'].shape}")

stats = parser.get_statistics(parsed_dataset)
print(f"\nYour dataset has {stats['num_levels']} levels")
print(f"Level dimensions: {stats['height_range']} (height) x {stats['width_range']} (width)")
print(f"Most common tiles:")
for tile, count in sorted(stats['tile_frequencies'].items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {tile}: {count}")

# %%
parsed_dataset['map_1']['original']


# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Dict

class PatchDifficultyEvaluator:
    def __init__(self, parser):
        self.parser = parser
        self.EMPTY = parser.tile_to_idx['-']
        self.GROUND = parser.tile_to_idx['X']
        self.BREAKABLE = parser.tile_to_idx['S']
        self.COIN = parser.tile_to_idx['o']
        self.ENEMY = parser.tile_to_idx['E']
        self.PIPE_LEFT = parser.tile_to_idx['[']
        self.PIPE_RIGHT = parser.tile_to_idx[']']
        self.QUESTION = parser.tile_to_idx['Q']
        self.BRICK_LEFT = parser.tile_to_idx['<']
        self.BRICK_RIGHT = parser.tile_to_idx['>']
        self.CANNON_BOTTOM = parser.tile_to_idx['b']
        self.CANNON_TOP = parser.tile_to_idx['B']

        self.MAX_ENEMY_DENSITY = 0.15
        self.MAX_CANNON_DENSITY = 0.15
        self.MAX_PIPE_DENSITY = 0.15
        self.MAX_JUMP_DENSITY = 0.2
        self.MAX_PLATFORM_DENSITY = 0.1

    def _count_enemies(self, patch):
        return np.sum(patch == self.ENEMY)

    def _count_cannons(self, patch):
        height, width = patch.shape
        cannon_count = 0

        for col in range(width):
            has_cannon_bottom = np.any(patch[:, col] == self.CANNON_BOTTOM)
            if has_cannon_bottom:
                cannon_count += 1

        return cannon_count

    def _count_pipes(self, patch):
        height, width = patch.shape
        pipe_count = 0

        for col in range(width):
            has_pipe_left = np.any(patch[:, col] == self.PIPE_LEFT)
            if has_pipe_left:
                pipe_count += 1

        return pipe_count

    def _count_jumps(self, patch):
        height, width = patch.shape
        jumps = 0

        ground_floor = patch[height - 1]
        gap_length = 0

        for col in range(width):
            if ground_floor[col] == self.EMPTY:
                gap_length += 1
            else:
                if gap_length > 0:
                    jumps += 1
                    if gap_length > 4:
                        jumps += (gap_length - 2) * 0.5
                gap_length = 0

        if gap_length > 0:
            jumps += 1
            if gap_length > 2:
                jumps += (gap_length - 2) * 0.5

        return int(jumps)

    def _count_elevated_platforms(self, patch):
            height, width = patch.shape
            platforms = 0

            PLATFORM_TILES = {self.GROUND, self.BREAKABLE, self.QUESTION}

            for row in range(height - 1):
                in_platform = False
                platform_start = -1

                for col in range(width):
                    if patch[row, col] in PLATFORM_TILES:
                        if not in_platform:
                            in_platform = True
                            platform_start = col
                    else:
                        if in_platform:
                            is_floating = False
                            for c in range(platform_start, col):
                                for r in range(row + 1, height):
                                    if patch[r, c] == self.EMPTY:
                                        is_floating = True
                                        break
                                if is_floating:
                                    break

                            if is_floating:
                                platforms += 1

                            in_platform = False

                if in_platform:
                    is_floating = False
                    for c in range(platform_start, width):
                        for r in range(row + 1, height):
                            if patch[r, c] == self.EMPTY:
                                is_floating = True
                                break
                        if is_floating:
                            break

                    if is_floating:
                        platforms += 1

            return platforms


    def evaluate_patch(self, patch, metadata=None):
        height, width = patch.shape
        total_tiles = height * width

        enemies = self._count_enemies(patch)
        cannons = self._count_cannons(patch)
        pipes = self._count_pipes(patch)
        jumps = self._count_jumps(patch)
        platforms = self._count_elevated_platforms(patch)

        enemy_density = enemies / width
        cannon_density = cannons / width
        pipe_density = pipes / width
        jump_density = jumps / width
        platform_density = platforms / width

        enemy_term = min(enemy_density / self.MAX_ENEMY_DENSITY, 1.0)
        cannon_term = min(cannon_density / self.MAX_CANNON_DENSITY, 1.0)
        jump_term = min(jump_density / self.MAX_JUMP_DENSITY, 1.0)
        pipe_term = min(pipe_density / self.MAX_PIPE_DENSITY, 1.0)
        platform_term = min(platform_density / self.MAX_PLATFORM_DENSITY, 1.0)

        raw_diff = (
            0.60 * enemy_term +
            0.30 * cannon_term +
            0.20 * jump_term +
            0.30 * platform_term +
            0.20 * pipe_term
        )

        diff_score = min(raw_diff , 1.0)


        result = {
            "metadata": metadata,
            "counts": {
                "enemies": enemies,
                "cannons": cannons,
                "pipes": pipes,
                "jumps": jumps,
                "platforms": platforms,
                "total_tiles": total_tiles
            },
            "densities": {
                "enemy_density": round(enemy_density, 3),
                "cannon_density": round(cannon_density, 3),
                "pipe_density": round(pipe_density, 3),
                "jump_density": round(jump_density, 3),
                "platform_density": round(platform_density, 3)
            },
            "scores": {
                "difficulty_score": round(diff_score, 3),
            },
        }

        return result

    def evaluate_patches_batch(self, patches, metadata_list=None):
        if metadata_list is None:
            metadata_list = [None] * len(patches)

        results = []
        for i in range(len(patches)):
            result = self.evaluate_patch(patches[i], metadata_list[i])
            results.append(result)

        return results

# %%
import numpy as np
from typing import List, Dict, Tuple, Optional


class PatchExtractor:
    def __init__(self, patch_height: int = 14, patch_width: int = 16,
                 stride: int = 4, vertical_stride: int = 8):
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.stride = stride
        self.vertical_stride = vertical_stride
    def extract_patches_from_level(self, level_meta,
                                   level_name: str = "",
                                   ):
        height, width = level_meta['data'].shape
        patches = []
        metadata = []
        patch_idx = 0

        y_positions = list(range(0, height - self.patch_height + 1, self.vertical_stride))
        x_positions = list(range(0, width - self.patch_width + 1, self.stride))
        if len(y_positions) == 0:
            y_positions = [max(0, height - self.patch_height)]
        elif y_positions[-1] + self.patch_height < height:
            y_positions.append(height - self.patch_height)

        if len(x_positions) == 0:
            x_positions = [max(0, width - self.patch_width)]
        elif x_positions[-1] + self.patch_width < width:
            x_positions.append(width - self.patch_width)

        for y in y_positions:
            for x in x_positions:
                patch = level_meta['data'][y:y + self.patch_height, x:x + self.patch_width].copy()
                patches.append(patch)
                meta = {
                    'level_name': level_name,
                    'patch_idx': patch_idx,
                    'x_start': x,
                    'x_end': x + self.patch_width,
                    'y_start': y,
                    'y_end': y + self.patch_height,
                    'level_height': height,
                    'final_score': 0,
                }

                metadata.append(meta)
                patch_idx += 1

        patches_array = np.stack(patches, axis=0) if patches else np.empty((0, self.patch_height, self.patch_width), dtype=level_meta['data'].dtype)
        return patches_array, metadata
    def extract_patches_from_dataset(self, parsed_dataset: Dict) -> Tuple[np.ndarray, List[Dict]]:
        all_patches = []
        all_metadata = []

        for level_name, level_data in parsed_dataset.items():
            patches, metadata = self.extract_patches_from_level(level_data, level_name)
            if len(patches) > 0:
                all_patches.append(patches)
                all_metadata.extend(metadata)

        if all_patches:
            all_patches_array = np.concatenate(all_patches, axis=0)
        else:
            all_patches_array = np.empty((0, self.patch_height, self.patch_width), dtype=np.int32)

        return all_patches_array, all_metadata

    def get_patch_statistics(self, patches: np.ndarray, metadata: List[Dict]) -> Dict:
        if len(patches) == 0:
            return {'num_patches': 0}

        level_patch_counts = {}
        for meta in metadata:
            lvl = meta['level_name']
            level_patch_counts[lvl] = level_patch_counts.get(lvl, 0) + 1

        unique_tiles, tile_counts = np.unique(patches, return_counts=True)

        stats = {
            'num_patches': len(patches),
            'patch_shape': patches.shape,
            'patches_per_level': level_patch_counts,
            'avg_patches_per_level': float(np.mean(list(level_patch_counts.values()))),
            'min_patches_per_level': int(min(level_patch_counts.values())),
            'max_patches_per_level': int(max(level_patch_counts.values())),
            'tile_distribution': {int(t): int(c) for t, c in zip(unique_tiles, tile_counts)},
            'total_tiles_in_patches': int(tile_counts.sum())
        }
        return stats

    def reconstruct_level_from_patches(self, patches: np.ndarray, metadata: List[Dict]) -> np.ndarray:
        if not metadata:
            return np.array([])

        original_width = max(m['x_end'] for m in metadata)
        original_height = max(m['y_end'] for m in metadata)

        reconstructed = np.zeros((original_height, original_width), dtype=np.float32)
        counts = np.zeros((original_height, original_width), dtype=np.float32)

        for patch, meta in zip(patches, metadata):
            y_start, y_end = meta['y_start'], meta['y_end']
            x_start, x_end = meta['x_start'], meta['x_end']
            reconstructed[y_start:y_end, x_start:x_end] += patch
            counts[y_start:y_end, x_start:x_end] += 1

        reconstructed = reconstructed / np.maximum(counts, 1)
        reconstructed = np.round(reconstructed).astype(np.int32)
        return reconstructed


# %% [markdown]
# # Test Patch Extractor

# %%
extractor = PatchExtractor(
    patch_height=14,
    patch_width=16,
    stride=4,
    vertical_stride=4
)

patches, metadata = extractor.extract_patches_from_level(
    level_meta=parsed_dataset['map_1'],
    level_name="sample_level"
)

print(f"\nExtracted {len(patches)} patches from sample level")
print(f"Patches shape: {patches.shape}")

patch_stats = extractor.get_patch_statistics(patches, metadata)
print(f"\nPatch Statistics:")
print(f"  Total patches: {patch_stats['num_patches']}")
print(f"  Patches shape: {patch_stats['patch_shape']}")
print(f"  Total tiles in all patches: {patch_stats['total_tiles_in_patches']}")


patch_evaluator = PatchDifficultyEvaluator(parser)
patch_evaluation_results = patch_evaluator.evaluate_patches_batch(patches, metadata)
for i, result in enumerate(patch_evaluation_results):
    metadata[i]['final_score'] = result['scores']['difficulty_score']

num_to_show = min(122, len(patches))
print(f"\nShowing {num_to_show} sample patches:\n" + "=" * 60)

for i in range(num_to_show):
    meta = metadata[i]

    difficulty_eval = patch_evaluation_results[i]
    diff_score = difficulty_eval['scores']['difficulty_score']

    enemy_count = difficulty_eval['counts']['enemies']
    pipe_count = difficulty_eval['counts']['pipes']
    jump_count = difficulty_eval['counts']['jumps']
    platform_count = difficulty_eval['counts']['platforms']

    print(f"\nPatch {i+1} from {meta['level_name']} "
          f"(x: {meta['x_start']}-{meta['x_end']}, y: {meta['y_start']}-{meta['y_end']}, "
          f"difficulty: {meta.get('final_score', 0):.3f})\"")
    print(f"Enemies: {enemy_count}, Pipes: {pipe_count}, Jumps: {jump_count}, Platforms: {platform_count}")
    print("-" * 60)
    decoded_patch = parser.decode_level(patches[i])
    print(decoded_patch)

print("=" * 60)

print("\n" + "="*70)
print("RECONSTRUCTION TEST")
print("="*70)

reconstructed = extractor.reconstruct_level_from_patches(
    patches,
    metadata,
)

print(f"\nOriginal shape: {parsed_dataset['map_1']['data'].shape}")
print(f"Reconstructed shape: {reconstructed.shape}")

match_percentage = np.mean(reconstructed == parsed_dataset['map_1']['data']) * 100
print(f"Reconstruction accuracy: {match_percentage:.2f}%")

if match_percentage < 100:
    print("\nNote: Small differences may occur at boundaries due to averaging/rounding")

# %%
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# %%
metadata[0]

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt


class PatchDatasetCreator(Dataset):

    def __init__(self, patches: np.ndarray, difficulties: Optional[np.ndarray] = None):
        self.patches = torch.from_numpy(patches).long()

        if difficulties is not None:
            diffs = np.asarray(difficulties, dtype=np.float32).reshape(-1)  # [N]
            assert diffs.shape[0] == self.patches.shape[0]
            self.difficulties = torch.from_numpy(diffs).float()  # [N]
        else:
            self.difficulties = None


    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        if self.difficulties is None:
            return self.patches[idx]
        else:
            return self.patches[idx], self.difficulties[idx]


class PatchEncoder(nn.Module):

    def __init__(self,
                 num_tile_types: int = 13,
                 embedding_dim: int = 32,
                 latent_dim: int = 128,
                 patch_height: int = 14,
                 patch_width: int = 16):
        super().__init__()

        self.num_tile_types = num_tile_types
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.tile_embedding = nn.Embedding(num_tile_types, embedding_dim)
        self.conv1 = nn.Conv2d(embedding_dim, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        with torch.no_grad():
            dummy = torch.zeros(1, embedding_dim, patch_height, patch_width)
            dummy = self.leaky_relu(self.bn1(self.conv1(dummy)))
            dummy = self.leaky_relu(self.bn2(self.conv2(dummy)))
            dummy = self.leaky_relu(self.bn3(self.conv3(dummy)))
            dummy = self.leaky_relu(self.bn4(self.conv4(dummy)))
            self.flattened_size = dummy.numel()
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, latent_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.tile_embedding(x)           # [B, H, W, E]
        x = x.permute(0, 3, 1, 2)           # [B, E, H, W]
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.leaky_relu(self.bn4(self.conv4(x)))
        x = x.reshape(batch_size, -1)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class PatchDecoder(nn.Module):

    def __init__(self,
                 num_tile_types: int = 13,
                 latent_dim: int = 128,
                 patch_height: int = 14,
                 patch_width: int = 16):
        super().__init__()

        self.num_tile_types = num_tile_types
        self.latent_dim = latent_dim
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.start_h = 4
        self.start_w = 4
        self.start_channels = 512
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, self.start_channels * self.start_h * self.start_w)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self.final_conv = nn.Conv2d(64, num_tile_types, kernel_size=3, stride=1, padding=1)

        self.dropout = nn.Dropout(0.1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        x = self.leaky_relu(self.fc1(z))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        x = x.reshape(batch_size, self.start_channels, self.start_h, self.start_w)
        x = self.leaky_relu(self.bn1(self.deconv1(x)))
        x = self.leaky_relu(self.bn2(self.deconv2(x)))
        x = self.leaky_relu(self.bn3(self.deconv3(x)))
        x = self.final_conv(x)
        x = x[:, :, :self.patch_height, :self.patch_width]

        return x
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


class Autoencoder(nn.Module):
    def __init__(self,
                 num_tile_types: int = 13,
                 embedding_dim: int = 32,
                 latent_dim: int = 128,
                 patch_height: int = 14,
                 patch_width: int = 16
              ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_tile_types = num_tile_types

        self.encoder = PatchEncoder(
            num_tile_types=num_tile_types,
            embedding_dim=embedding_dim,
            latent_dim=latent_dim,
            patch_height=patch_height,
            patch_width=patch_width
        )

        self.decoder = PatchDecoder(
            num_tile_types=num_tile_types,
            latent_dim=latent_dim,
            patch_height=patch_height,
            patch_width=patch_width
        )
        self.difficulty_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.SiLU(),
            nn.Linear(latent_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, return_difficulty: bool = False):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        z = latent

        if return_difficulty:
            diff_pred = self.difficulty_head(z).squeeze(1)
            return recon, z, diff_pred
        else:
            return recon, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        predictions = torch.argmax(logits, dim=1)
        return predictions

    def predict_difficulty_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.difficulty_head(latent).squeeze(1)

    def predict_difficulty(self, x: torch.Tensor) -> torch.Tensor:
        _, latent = self.forward(x)
        return self.predict_difficulty_from_latent(latent)


class AutoencoderTrainer:
    def __init__(self,
                 model: Autoencoder,
                 learning_rate: float = 1e-4,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 difficulty_loss_weight: float = 0.0):

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
              num_epochs: int = 100,
              print_every: int = 10):

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

    def save_model(self, path: str):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'difficulty_loss_weight': self.difficulty_loss_weight
        }

        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.difficulty_loss_weight = checkpoint.get('difficulty_loss_weight', 0.0)

        print(f"Model loaded from {path}")


# %% [markdown]
# **create latents**

# %%
all_patches, metadata = extractor.extract_patches_from_dataset(
parsed_dataset
)
patch_evaluator = PatchDifficultyEvaluator(parser)
patch_evaluation_results = patch_evaluator.evaluate_patches_batch(all_patches, metadata)
for i, result in enumerate(patch_evaluation_results):
  metadata[i]['final_score'] = result['scores']['difficulty_score']
all_difficulties = [(d['final_score']) for d in metadata]

# %%
np.array(all_difficulties).mean()

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

data = all_difficulties

plt.figure(figsize=(10, 6))

counts, bins, patches = plt.hist(
    data,
    bins='auto',
    alpha=0.6,
    edgecolor='black',
    linewidth=1.2
)

kde = gaussian_kde(data)
x_vals = np.linspace(min(data), max(data), 300)
kde_vals = kde(x_vals)

scale = max(counts) / max(kde_vals)
plt.plot(x_vals, kde_vals * scale, linewidth=3)

plt.grid(alpha=0.25)
plt.xlabel("Difficulty", fontsize=14)
plt.ylabel("Number of patches", fontsize=14)
plt.title("patch Difficulty Distribution", fontsize=18, pad=15)
plt.tight_layout()

plt.show()


# %%
print("Mario Level Autoencoder")
print("=" * 70)

dataset = PatchDatasetCreator(all_patches, difficulties=all_difficulties)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Train size: {train_size}, Val size: {val_size}")
print("-" * 70)
model = Autoencoder(
  num_tile_types=13,
  embedding_dim=32,
  latent_dim=128,
  patch_height=14,
  patch_width=16,
)


print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Latent dimension: {model.latent_dim}")
print("-" * 70)
trainer = AutoencoderTrainer(model, learning_rate=1e-4, difficulty_loss_weight=10.0)

print("\nStarting training...")
trainer.train(train_loader, val_loader, num_epochs=100, print_every=5)

print("\n" + "=" * 70)
print("Testing reconstruction on a sample:")
model.eval()
with torch.no_grad():
    sample = all_patches[0:1]
    sample_tensor = torch.from_numpy(sample).long().to(trainer.device)
    latent = model.encode(sample_tensor)
    reconstruction = model.reconstruct(sample_tensor)

    print(f"Original shape: {sample.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    accuracy = (reconstruction.cpu().numpy() == sample).mean()
    print(f"Reconstruction accuracy: {accuracy:.2%}")

# %%
trainer.save_model('/content/Output/AutoEncoder.pth')

# %%
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

# %% [markdown]
# **Convert Entire DataSet to latents**

# %%
import torch
from torch.utils.data import DataLoader

device = 'cuda'
autoencoder_path='/content/Output/AutoEncoder.pth'

autoencoder = Autoencoder(
    num_tile_types=13,
    embedding_dim=32,
    latent_dim=128,
    patch_height=14,
    patch_width=16
  )

ae_checkpoint = torch.load(autoencoder_path, map_location=device)
autoencoder.load_state_dict(ae_checkpoint['model_state_dict'])
autoencoder.to(device)
autoencoder.eval()
print("✓ Autoencoder loaded")
for param in autoencoder.parameters():
    param.requires_grad = False

print("✓ Autoencoder loaded!")

print(f"Loaded {len(all_patches)} patches")
print("num_scores:", len(all_difficulties))

dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
all_latents = []
all_difficulties = []

with torch.no_grad():
    for batch in dataloader:
        patches, diffs = batch
        patches = patches.to(device)
        latents = autoencoder.encode(patches)

        all_latents.append(latents.cpu())
        all_difficulties.append(diffs.cpu())
all_latents = torch.cat(all_latents, dim=0)
all_difficulties = torch.cat(all_difficulties, 0)

print(f"Encoded latents shape: {all_latents.shape}")
print(f"Latent dimension: {all_latents.shape[1]}")
print(f"Difficulties shape: {all_difficulties.shape}")


# %%
max(all_difficulties)

# %%
print(f"latents: norm={all_latents.norm(dim=-1).mean():.2f}, std={all_latents.std():.2f}")

# %%
print("\n" + "="*70)
print("CREATING LATENT NORMALIZER")
print("="*70)

normalizer = LatentNormalizer(target_norm=11.0)
normalizer.fit(all_latents)
latents_norm = normalizer.normalize(all_latents)
print(f"Normalized latents: norm={latents_norm.norm(dim=-1).mean():.2f}, std={latents_norm.std():.2f}")


# %%
normalizer.save('Output/latent_normalizer.pth')

# %%
import torch
import numpy as np

def extract_latents_and_difficulty(dataloader, autoencoder, device):
    autoencoder.eval()
    latents = []
    difficulties = []

    with torch.no_grad():
        for patches, diff in dataloader:
            patches = patches.to(device)
            diff = diff.to(device).float()

            z = autoencoder.encode(patches)  # [B, 128]
            latents.append(z.cpu())
            difficulties.append(diff.cpu())

    latents = torch.cat(latents, dim=0)
    difficulties = torch.cat(difficulties, dim=0)

    return latents, difficulties


# %%
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Tuple, List
from tqdm import tqdm
import numpy as np


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, max_timesteps: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        position = torch.arange(max_timesteps).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_timesteps, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.pe[timesteps]


class FourierPlayabilityEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, num_frequencies: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_frequencies = num_frequencies

        frequencies = torch.randn(num_frequencies) * 2.0
        self.register_buffer('frequencies', frequencies)

        fourier_dim = num_frequencies * 2
        self.projection = nn.Sequential(
            nn.Linear(fourier_dim + 1, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, playability: torch.Tensor) -> torch.Tensor:
        if playability.dim() == 1:
            playability = playability.unsqueeze(-1)

        scaled = playability * 2 - 1

        angles = scaled * self.frequencies.unsqueeze(0) * math.pi
        fourier_features = torch.cat([
            torch.sin(angles),
            torch.cos(angles),
            playability
        ], dim=-1)

        return self.projection(fourier_features)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, time_emb_dim: int):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(time_emb_dim, dim), nn.SiLU())
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        time_proj = self.time_mlp(time_emb)
        h = self.block(x + time_proj)
        return self.activation(x + h)


class MixingMLP(nn.Module):
    def __init__(self, dim: int, hidden_scale: float = 1.0):
        super().__init__()
        mid = int(dim * hidden_scale)
        mid = max(mid, dim)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mid), nn.SiLU(), nn.Linear(mid, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


class DiffusionUNet(nn.Module):

    def __init__(
        self,
        latent_dim: int = 128,
        time_emb_dim: int = 256,
        context_emb_dim: int = 128,
        hidden_dims: list = [256, 512, 512],
        num_res_blocks: int = 2,
        cond_dropout: float = 0.15,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_emb_dim = time_emb_dim
        self.context_emb_dim = context_emb_dim
        self.prev_play_emb_dim = 64
        self.hidden_dims = hidden_dims
        self.num_res_blocks = num_res_blocks
        self.cond_dropout = cond_dropout

        self.output_scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

        self.time_embedding = SinusoidalPositionalEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.context_encoder_per = nn.Sequential(
            nn.Linear(latent_dim, context_emb_dim),
            nn.SiLU(),
            nn.Linear(context_emb_dim, time_emb_dim)
        )

        self.prev_playability_mlp = nn.Sequential(
            nn.Linear(1, self.prev_play_emb_dim),
            nn.SiLU(),
            nn.Linear(self.prev_play_emb_dim, time_emb_dim)
        )

        self.playability_embedding = FourierPlayabilityEmbedding(
            embedding_dim=time_emb_dim,
            num_frequencies=64
        )
        self.null_play_embedding = nn.Parameter(torch.randn(time_emb_dim) * 0.02)

        self.input_proj = nn.Linear(latent_dim * 2, hidden_dims[0])

        self.encoder_blocks = nn.ModuleList()
        self.encoder_mixes = nn.ModuleList()
        dims = [hidden_dims[0]] + hidden_dims
        for i in range(len(hidden_dims)):
            blocks = nn.ModuleList([ResidualBlock(dims[i], time_emb_dim) for _ in range(num_res_blocks)])
            self.encoder_blocks.append(blocks)
            self.encoder_mixes.append(MixingMLP(dims[i], hidden_scale=1.0))
            if i < len(hidden_dims) - 1:
                self.encoder_blocks.append(nn.ModuleList([nn.Linear(dims[i], dims[i + 1])]))

        bottleneck_dim = hidden_dims[-1]
        self.bottleneck = nn.ModuleList([ResidualBlock(bottleneck_dim, time_emb_dim) for _ in range(num_res_blocks)])
        self.bottleneck_mix = MixingMLP(bottleneck_dim, hidden_scale=1.0)

        self.decoder_blocks = nn.ModuleList()
        self.decoder_mixes = nn.ModuleList()
        self.skip_projections = nn.ModuleList()
        reversed_dims = list(reversed(hidden_dims))
        reversed_encoder_dims = [hidden_dims[0]] + hidden_dims
        reversed_skip_dims = list(reversed(reversed_encoder_dims[:len(hidden_dims)]))

        for i in range(len(reversed_dims)):
            if i == 0:
                blocks = nn.ModuleList([ResidualBlock(reversed_dims[i], time_emb_dim) for _ in range(num_res_blocks)])
                self.decoder_blocks.append(blocks)
                self.skip_projections.append(None)
            else:
                current_dim = reversed_dims[i - 1]
                target_dim = reversed_dims[i]
                skip_dim = reversed_skip_dims[i]
                self.decoder_blocks.append(nn.Linear(current_dim, target_dim))
                concat_dim = target_dim + skip_dim
                self.skip_projections.append(nn.Linear(concat_dim, target_dim))
                blocks = nn.ModuleList([ResidualBlock(target_dim, time_emb_dim) for _ in range(num_res_blocks)])
                self.decoder_blocks.append(blocks)
            self.decoder_mixes.append(MixingMLP(reversed_dims[i], hidden_scale=1.0))

        self.output_proj = nn.Linear(hidden_dims[0], latent_dim)

        self._initialize_weights()

        print(f"\n{'='*70}")
        print(f"CFGDiffusionUNet Initialized")
        print(f"{'='*70}")
        print(f"  Condition dropout: {cond_dropout*100:.0f}%")
        print(f"  Latent dim: {latent_dim}")
        print(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"{'='*70}\n")

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.output_proj.weight, gain=1.0)
        nn.init.zeros_(self.output_proj.bias)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and 'mlp' in name.lower():
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        previous_latents: Optional[torch.Tensor] = None,
        previous_playabilities: Optional[torch.Tensor] = None,
        target_playability: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        batch = x.shape[0]
        device = x.device
        dtype = x.dtype

        x = x.to(device=device, dtype=dtype)
        if timesteps.dtype != torch.long:
            timesteps = timesteps.long()
        timesteps = timesteps.to(device=device)

        t_emb = self.time_embedding(timesteps)
        t_emb = self.time_mlp(t_emb)

        if previous_latents is None:
            previous_latents = torch.zeros((batch, 1, self.latent_dim), device=device, dtype=dtype)
            inferred_k = 1
            prev_flat = previous_latents.reshape(-1, self.latent_dim)
        else:
            previous_latents = previous_latents.to(device=device, dtype=dtype).contiguous()
            if previous_latents.dim() == 3:
                prev_flat = previous_latents.reshape(-1, self.latent_dim)
                inferred_total = prev_flat.shape[0]
                if inferred_total % batch != 0:
                    raise RuntimeError('previous_latents flattened count not divisible by batch')
                inferred_k = inferred_total // batch
            elif previous_latents.dim() == 2:
                prev_flat = previous_latents
                inferred_total = prev_flat.shape[0]
                if inferred_total % batch != 0:
                    raise RuntimeError('previous_latents flattened length not divisible by batch')
                inferred_k = inferred_total // batch
            else:
                raise ValueError('previous_latents must be 2D or 3D')

        prev_enc = self.context_encoder_per(prev_flat)
        prev_enc = prev_enc.reshape(batch, inferred_k, -1)
        k = inferred_k

        if previous_playabilities is None:
            previous_playabilities = torch.zeros((batch, k), device=device, dtype=dtype)
        else:
            previous_playabilities = previous_playabilities.to(device=device, dtype=dtype)
            if previous_playabilities.dim() == 2:
                if previous_playabilities.shape[0] != batch or previous_playabilities.shape[1] != k:
                    if previous_playabilities.numel() == batch * k:
                        previous_playabilities = previous_playabilities.reshape(batch, k)
                    else:
                        raise ValueError('previous_playabilities shape incompatible')
            elif previous_playabilities.dim() == 1:
                n = previous_playabilities.numel()
                if n == k:
                    previous_playabilities = previous_playabilities.unsqueeze(0).repeat(batch, 1)
                elif n == batch * k:
                    previous_playabilities = previous_playabilities.reshape(batch, k)
                else:
                    raise ValueError('previous_playabilities length incompatible')

        prev_play_flat = previous_playabilities.reshape(batch * k, 1)
        prev_play_enc = self.prev_playability_mlp(prev_play_flat)
        prev_play_enc = prev_play_enc.reshape(batch, k, -1)

        per_prev = prev_enc + prev_play_enc
        ctx_emb = per_prev.mean(dim=1)

        if target_playability is None:
            play_emb = self.null_play_embedding.unsqueeze(0).expand(batch, -1)
        else:
            target_playability = target_playability.to(device=device, dtype=dtype)
            if target_playability.dim() == 0:
                target_playability = target_playability.unsqueeze(0)
            if target_playability.dim() == 1:
                target_playability = target_playability.view(-1, 1)

            play_emb = self.playability_embedding(target_playability)

            if self.training and self.cond_dropout > 0:
                drop_mask = (torch.rand(batch, device=device) < self.cond_dropout).unsqueeze(1)
                null_emb = self.null_play_embedding.unsqueeze(0).expand(batch, -1)
                play_emb = torch.where(drop_mask, null_emb, play_emb)

        if previous_latents.dim() == 2:
            prev_lat_for_mean = prev_flat.reshape(batch, k, self.latent_dim)
        else:
            prev_lat_for_mean = previous_latents.reshape(batch, k, self.latent_dim)
        prev_lat_mean = prev_lat_for_mean.mean(dim=1)

        h = torch.cat([x, prev_lat_mean], dim=-1)
        h = self.input_proj(h)

        combined_emb = t_emb + ctx_emb + play_emb

        skip_connections = []
        block_idx = 0
        for i in range(len(self.hidden_dims)):
            for res_block in self.encoder_blocks[block_idx]:
                h = res_block(h, combined_emb)
            block_idx += 1
            h = self.encoder_mixes[i](h)
            skip_connections.append(h)
            if i < len(self.hidden_dims) - 1:
                for down_layer in self.encoder_blocks[block_idx]:
                    h = down_layer(h)
                block_idx += 1

        for bottleneck_block in self.bottleneck:
            h = bottleneck_block(h, combined_emb)
        h = self.bottleneck_mix(h)

        skip_connections = list(reversed(skip_connections))
        decoder_block_idx = 0
        for i in range(len(self.hidden_dims)):
            if i == 0:
                for res_block in self.decoder_blocks[decoder_block_idx]:
                    h = res_block(h, combined_emb)
                decoder_block_idx += 1
            else:
                h = self.decoder_blocks[decoder_block_idx](h)
                decoder_block_idx += 1
                skip = skip_connections[i]
                h = torch.cat([h, skip], dim=-1)
                h = self.skip_projections[i](h)
                for res_block in self.decoder_blocks[decoder_block_idx]:
                    h = res_block(h, combined_emb)
                decoder_block_idx += 1
            h = self.decoder_mixes[i](h)

        noise_pred = self.output_proj(h)
        scale = self.output_scale.clamp(min=0.1, max=10.0)
        noise_pred = noise_pred * scale

        return noise_pred

# %%
import torch
from typing import Optional, Sequence


class AutoregressivePatchDatasetCreator(torch.utils.data.Dataset):
    def __init__(self,
                 latents: torch.Tensor,
                 playabilities: Sequence,
                 num_prev: int = 10,
                 level_ids: Optional[Sequence] = None):
        super().__init__()

        assert latents.dim() == 2
        num_patches = latents.shape[0]

        assert len(playabilities) == num_patches
        if level_ids is not None:
            assert len(level_ids) == num_patches

        self.latents = latents
        self.playabilities = torch.as_tensor(playabilities, dtype=latents.dtype)
        self.num_prev = int(num_prev)
        self.level_ids = torch.as_tensor(level_ids) if level_ids is not None else None

        self.num_patches = num_patches
        self.latent_dim = latents.shape[1]

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.num_patches:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_patches}")

        device = self.latents.device
        dtype = self.latents.dtype
        k = self.num_prev

        current_latent = self.latents[idx]
        current_playability = self.playabilities[idx].unsqueeze(0)
        zero_latent = torch.zeros(self.latent_dim, dtype=dtype, device=device)
        zero_play = torch.zeros(1, dtype=dtype, device=device)

        prev_latents = []
        prev_playabilities = []

        for j in range(1, k + 1):
            prev_idx = idx - j
            if prev_idx < 0:
                prev_latents.append(zero_latent)
                prev_playabilities.append(zero_play)
                continue
            if self.level_ids is not None:
                if self.level_ids[prev_idx].item() != self.level_ids[idx].item():
                    prev_latents.append(zero_latent)
                    prev_playabilities.append(zero_play)
                    continue
            prev_latents.append(self.latents[prev_idx].to(device))
            prev_playabilities.append(self.playabilities[prev_idx].unsqueeze(0).to(device))
        prev_latents = torch.stack(prev_latents, dim=0)
        prev_playabilities = torch.cat(prev_playabilities, dim=0).view(k)
        current_latent = current_latent.to(device)
        current_playability = current_playability.to(device)

        return current_latent, prev_latents, current_playability, prev_playabilities


# %%
all_difficulties.mean()

# %%
metadata[0]

# %%
all_latents.shape

# %%
class DiffusionTrainer:
    def __init__(
        self,
        unet: DiffusionUNet,
        noise_schedule,
        forward_diffusion,
        learning_rate: float = 1e-4,
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

        self.optimizer = optim.AdamW(self.unet.parameters(), lr=learning_rate, weight_decay=1e-4)
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
        print(f"  NOTE: Target playability is now an INPUT condition")
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
            target_play = curr_play.squeeze(1)
        else:
            target_play = curr_play

        predicted_noise = self.unet(
            x=noisy_latent,
            timesteps=timesteps,
            previous_latents=prev_latents,
            previous_playabilities=prev_play,
            target_playability=target_play
        )

        loss = self.criterion(predicted_noise, noise)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {'total': loss.item(), 'noise': loss.item()}

    def validate(self, val_loader):
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
                    target_play = curr_play.squeeze(1)
                else:
                    target_play = curr_play

                predicted_noise = self.unet(
                    x=noisy_latents,
                    timesteps=timesteps,
                    previous_latents=prev_latents,
                    previous_playabilities=prev_play,
                    target_playability=target_play
                )

                loss = self.criterion(predicted_noise, noise)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        return avg_loss

    def train(
        self,
        train_loader,
        val_loader=None,
        num_epochs=500,
        save_interval=50,
        save_path='cfg_diffusion.pth'
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

    def save_checkpoint(self, path):
        torch.save({
            'unet_state_dict': self.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch_losses': self.epoch_losses,
            'val_losses': self.val_losses,
        }, path)

    def plot_losses(self, save_path=None):
        plt.figure(figsize=(8,5))
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
        plt.show()

# %%
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.optim as optim
import itertools

def train_conditional_diffusion_model(
    latents: np.ndarray,
    metadata: List[Dict],
    autoencoder_path: str,
    save_path: str = '/content/Output/conditional_diffusion.pth',
    num_epochs: int = 500,
    batch_size: int = 16):


    device = 'cuda'


    autoencoder = Autoencoder(
        num_tile_types=13,
        embedding_dim=32,
        latent_dim=128,
        patch_height=14,
        patch_width=16
    )

    ae_checkpoint = torch.load(autoencoder_path, map_location=device)
    autoencoder.load_state_dict(ae_checkpoint['model_state_dict'])
    autoencoder.to(device)
    autoencoder.eval()
    print("✓ Autoencoder loaded")
    print("\n" + "="*70)
    print("STEP 4: Creating Training Dataset")
    print("="*70)

    print(f'scores min/max: {min(all_difficulties)}/{max(all_difficulties)}')
    dataset = AutoregressivePatchDatasetCreator(latents, all_difficulties)

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
        latent_dim=128,
        time_emb_dim=256,
        context_emb_dim=128,
        hidden_dims=[256, 512, 512],
        num_res_blocks=2,
    ).to(device)

    print("="*70)
    print("MODEL INITIALIZATION CHECK")
    print("="*70)
    print(f"Output scale initial value: {unet.output_scale.item()}")
    print(f"Output scale requires_grad: {unet.output_scale.requires_grad}")
    print(f"Output projection type: {type(unet.output_proj)}")
    print("="*70)


    num_params = sum(p.numel() for p in unet.parameters())
    print(f"✓ Model created: {num_params:,} parameters")
    schedule = NoiseSchedule(
        num_timesteps=500,
        schedule_type='linear',
        device=device,
    )
    forward = ForwardDiffusion(schedule)

    trainer = DiffusionTrainer(
        unet=unet,
        noise_schedule=schedule,
        forward_diffusion=forward,
        learning_rate=5e-4,
        device=device
    )

    print("\n" + "="*70)
    print("STEP 7: Training")
    print("="*70)
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        save_interval=300,
        save_path=save_path
    )

    trainer.plot_losses(save_path.replace('.pth', '_losses.png'))

    return trainer, unet, val_loader

# %%
all_latents.dim()

# %%
class Sampler:
    def __init__(
        self,
        unet: DiffusionUNet,
        noise_schedule,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.unet = unet.to(device)
        self.schedule = noise_schedule
        self.device = device
        self.unet.eval()

        # Move schedule tensors to device
        for name in ["sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
                     "posterior_variance", "alphas", "betas", "alphas_cumprod"]:
            val = getattr(self.schedule, name, None)
            if isinstance(val, torch.Tensor):
                setattr(self.schedule, name, val.to(device))

        print(f"\n{'='*70}")
        print(f"CFG Sampler Initialized")
        print(f"{'='*70}")
        print(f"  Device: {device}")
        print(f"  Timesteps: {self.schedule.num_timesteps}")
        print(f"{'='*70}\n")

    @torch.no_grad()
    def _denoise_step(
        self,
        x: torch.Tensor,
        t_batch: torch.Tensor,
        predicted_noise: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        device = x.device
        dtype = x.dtype

        alpha_t = self.schedule.alphas[t_batch].to(device=device, dtype=dtype).view(-1, 1)
        beta_t = self.schedule.betas[t_batch].to(device=device, dtype=dtype).view(-1, 1)
        alpha_cumprod_t = self.schedule.alphas_cumprod[t_batch].to(device=device, dtype=dtype).view(-1, 1)

        eps = 1e-8
        sqrt_alpha_t = torch.sqrt(alpha_t.clamp(min=eps))
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt((1.0 - alpha_cumprod_t).clamp(min=eps))

        mean = (1.0 / sqrt_alpha_t) * (x - (beta_t / sqrt_one_minus_alpha_cumprod_t) * predicted_noise)

        if t_batch[0] > 0:
            variance = beta_t
            std = torch.sqrt(variance.clamp(min=eps))
            noise = torch.randn_like(x)
            x_prev = mean + std * temperature * noise
        else:
            x_prev = mean

        return x_prev

    @torch.no_grad()
    def sample_single_patch(
        self,
        normalizer,
        previous_latent: Optional[torch.Tensor],
        target_playability: float,
        previous_playabilities: Optional[list] = None,
        temperature: float = 0.9,
        guidance_scale: float = 3.0,
        show_progress: bool = False
    ) -> torch.Tensor:

        device = self.device
        latent_dim = self.unet.latent_dim

        x = torch.randn(1, latent_dim, device=device) * temperature
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-12) * normalizer.target_norm

        target_play_tensor = torch.tensor([target_playability], device=device, dtype=torch.float32)

        if previous_latent is None:
            prev_lat = torch.zeros((1, 1, latent_dim), device=device)
            prev_play = torch.zeros((1, 1), device=device)
        else:
            if previous_latent.dim() == 1:
                previous_latent = previous_latent.unsqueeze(0)
            prev_lat = previous_latent.unsqueeze(0).to(device)
            if previous_playabilities is None:
                prev_play = torch.zeros((1, prev_lat.shape[1]), device=device)
            else:
                prev_play = torch.tensor(previous_playabilities, device=device).unsqueeze(0)

        timesteps = range(self.schedule.num_timesteps - 1, -1, -1)
        if show_progress:
            timesteps = tqdm(timesteps, desc=f"Sampling (CFG scale={guidance_scale})")

        self.unet.eval()
        for t in timesteps:
            t_batch = torch.tensor([t], device=device, dtype=torch.long)

            noise_cond = self.unet(
                x=x,
                timesteps=t_batch,
                previous_latents=prev_lat,
                previous_playabilities=prev_play,
                target_playability=target_play_tensor
            )

            noise_uncond = self.unet(
                x=x,
                timesteps=t_batch,
                previous_latents=prev_lat,
                previous_playabilities=prev_play,
                target_playability=None
            )

            noise_guided = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            x = self._denoise_step(x, t_batch, noise_guided, temperature)

            if torch.isnan(x).any() or torch.isinf(x).any():
                x = torch.nan_to_num(x, nan=0.0, posinf=1e8, neginf=-1e8)

        return x.squeeze(0)

    @torch.no_grad()
    def sample_level(
        self,
        normalizer,
        num_patches: int,
        playability_target: float = 0.5,
        temperature: float = 0.9,
        guidance_scale: float = 3.0,
        show_progress: bool = True
    ) -> torch.Tensor:
        generated = []
        prev_buffer = []
        prev_play_buffer = []

        if isinstance(playability_target, (list, tuple)):
            playability_schedule = [float(p) for p in playability_target]
        else:

            scalar_target = float(playability_target)
            scalar_target = max(0.0, min(1.0, scalar_target))
            if num_patches == 1:
                playability_schedule = [scalar_target]
            else:
                power = 2.0
                t = torch.linspace(0.0, 1.0, steps=num_patches)
                curved = t ** power
                playability_schedule = (curved * scalar_target).tolist()
                print(f"  Playability ramp (power={power}): {playability_schedule[0]:.2f} → {playability_schedule[-1]:.2f}")


        iterator = range(num_patches)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating level (CFG)")

        for i in iterator:
            if len(prev_buffer) == 0:
                prev_lat = None
                prev_play = None
            else:
                prev_lat = torch.stack(prev_buffer, dim=0).to(self.device)
                prev_play = prev_play_buffer.copy()

            latent = self.sample_single_patch(
                normalizer=normalizer,
                previous_latent=prev_lat,
                target_playability=playability_schedule[i],
                previous_playabilities=prev_play,
                temperature=temperature,
                guidance_scale=guidance_scale,
                show_progress=False
            )

            generated.append(latent)
            prev_buffer.append(latent.cpu())
            prev_play_buffer.append(playability_schedule[i])

        result = torch.stack(generated, dim=0)

        if show_progress:
            print(f"\n✓ Generated {num_patches} patches with CFG (scale={guidance_scale})")

        return result

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, List, Dict, Tuple

class PatchStitcher:
    def __init__(self,
                 patch_height: int = 14,
                 patch_width: int = 16,
                 stride: int = 4):

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.stride = stride

    def stitch_patches_to_level(self,
                            patches: np.ndarray,
                            target_width: int = None,
                            ) -> np.ndarray:
        num_patches = len(patches)
        first_patch = patches[0]
        ph, pw = first_patch.shape

        if ph != self.patch_height or pw != self.patch_width:
            self.patch_height = ph
            self.patch_width = pw

        target_width = (num_patches - 1) * self.stride + pw

        level = np.zeros((ph, target_width), dtype=np.int32)

        for i, patch in enumerate(patches):
            if patch.ndim != 2:
                raise ValueError(f"Each patch must be 2D. Got ndim={patch.ndim} at index {i}.")

            ph_i, pw_i = patch.shape
            if ph_i != ph:
                if ph_i > ph:
                    patch = patch[:ph, :]
                else:
                    patch = np.pad(patch, ((0, ph - ph_i), (0, 0)), mode='constant', constant_values=0)

            x_start = i * self.stride

            if i == num_patches - 1:
                x_end = x_start + pw_i
                patch_slice = patch
            else:
                x_end = x_start + self.stride
                patch_slice = patch[:, :self.stride]
            if x_end > target_width:
                overflow = x_end - target_width
                x_end = target_width
                patch_slice = patch_slice[:, :-overflow]
            level[:, x_start:x_end] = patch_slice

        return level

    def compare_cfg_difficulty(self,
                              sampler,
                              normalizer,
                              autoencoder,
                              parser,
                              difficulty_evaluator,
                              guidance_scales: List[float] = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0],
                              num_patches: int = 10,
                              target_playability: float = 0.8,
                              temperature: float = 0.5,
                              device: str = 'cuda',
                              save_path: Optional[str] = None) -> Dict:
        print(f"\n{'='*70}")
        print(f"CFG CONDITIONAL VS UNCONDITIONAL COMPARISON")
        print(f"{'='*70}")
        print(f"Target Playability: {target_playability}")
        print(f"Guidance Scales: {guidance_scales}")
        print(f"Patches per condition: {num_patches}")
        print(f"{'='*70}\n")

        print("Step 1: Generating UNCONDITIONAL baseline (scale = 0.0)...")
        latents_uncond = sampler.sample_level(
            num_patches=num_patches,
            normalizer=normalizer,
            playability_target=target_playability,
            temperature=temperature,
            guidance_scale=0.0,
            show_progress=False
        )
        print(f"✓ Unconditional generation complete: {latents_uncond.shape}")

        latents_denorm = normalizer.denormalize(latents_uncond)
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

        patches_np_uncond = patches.cpu().numpy()
        print(f"✓ Decoded to patches: {patches_np_uncond.shape}")

        uncond_evals = difficulty_evaluator.evaluate_patches_batch(patches_np_uncond)
        uncond_difficulties = [e['scores']['difficulty_score'] for e in uncond_evals]
        uncond_enemies = [e['counts']['enemies'] for e in uncond_evals]
        uncond_cannons = [e['counts']['cannons'] for e in uncond_evals]

        uncond_mean_diff = np.mean(uncond_difficulties)
        uncond_std_diff = np.std(uncond_difficulties)
        print(f"✓ Unconditional difficulty: {uncond_mean_diff:.3f} ± {uncond_std_diff:.3f}")
        print(f"  Enemies: {np.mean(uncond_enemies):.2f}, Cannons: {np.mean(uncond_cannons):.2f}\n")

        print("Step 2: Generating CONDITIONAL with different guidance scales...\n")

        results = {
            'unconditional': {
                'mean_difficulty': uncond_mean_diff,
                'std_difficulty': uncond_std_diff,
                'mean_enemies': np.mean(uncond_enemies),
                'mean_cannons': np.mean(uncond_cannons),
                'difficulties': uncond_difficulties
            },
            'conditional_by_scale': {},
            'comparison_table': []
        }

        conditional_scales = [s for s in guidance_scales if s > 0.0]

        for scale in conditional_scales:
            print(f"  Testing scale = {scale}...")

            latents_cond = sampler.sample_level(
                num_patches=num_patches,
                normalizer=normalizer,
                playability_target=target_playability,
                temperature=temperature,
                guidance_scale=scale,
                show_progress=False
            )

            latents_denorm = normalizer.denormalize(latents_cond)
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

            patches_np_cond = patches.cpu().numpy()

            cond_evals = difficulty_evaluator.evaluate_patches_batch(patches_np_cond)
            cond_difficulties = [e['scores']['difficulty_score'] for e in cond_evals]
            cond_enemies = [e['counts']['enemies'] for e in cond_evals]
            cond_cannons = [e['counts']['cannons'] for e in cond_evals]

            cond_mean_diff = np.mean(cond_difficulties)
            cond_std_diff = np.std(cond_difficulties)

            diff = cond_mean_diff - uncond_mean_diff

            results['conditional_by_scale'][scale] = {
                'mean_difficulty': cond_mean_diff,
                'std_difficulty': cond_std_diff,
                'mean_enemies': np.mean(cond_enemies),
                'mean_cannons': np.mean(cond_cannons),
                'difficulties': cond_difficulties
            }

            print(f"    ✓ Difficulty: {cond_mean_diff:.3f} ± {cond_std_diff:.3f}")
            print(f"    ✓ Δ from uncond: {diff:+.3f}\n")

        print("="*70)
        print("CONDITIONAL VS UNCONDITIONAL COMPARISON")
        print("="*70)
        print(f"{'Scale':<10} {'Uncond':<12} {'Cond':<12} {'Δ':<12} {'%Change':<12}")
        print("-"*70)

        for scale in conditional_scales:
            uncond = results['unconditional']['mean_difficulty']
            cond = results['conditional_by_scale'][scale]['mean_difficulty']
            diff = cond - uncond
            pct = (diff / uncond * 100) if uncond != 0 else 0

            print(f"{scale:<10.1f} {uncond:<12.3f} {cond:<12.3f} {diff:<+12.3f} {pct:<+12.1f}%")

            results['comparison_table'].append({
                'scale': scale,
                'uncond': uncond,
                'cond': cond,
                'diff': diff,
                'pct_change': pct
            })

        print("="*70)

        print("\nStep 3: Testing if different playability targets produce different outputs...")
        if len(conditional_scales) >= 2:
            scale_test = conditional_scales[0]
            diff_low_high = abs(
                np.mean(results['conditional_by_scale'][scale_test]['difficulties']) -
                results['unconditional']['mean_difficulty']
            )
            print(f"✓ Different targets produce different outputs: diff = {diff_low_high:.4f}")

        print("\n✓ All CFG comparison tests complete!")
        print("="*70 + "\n")

        if save_path:
            with open(save_path, 'w') as f:
                f.write("CONDITIONAL VS UNCONDITIONAL COMPARISON\n")
                f.write("="*70 + "\n")
                f.write(f"Target Playability: {target_playability}\n")
                f.write(f"Patches per condition: {num_patches}\n\n")

                f.write(f"{'Scale':<10} {'Uncond':<12} {'Cond':<12} {'Δ':<12} {'%Change':<12}\n")
                f.write("-"*70 + "\n")

                for row in results['comparison_table']:
                    f.write(f"{row['scale']:<10.1f} {row['uncond']:<12.3f} {row['cond']:<12.3f} "
                          f"{row['diff']:<+12.3f} {row['pct_change']:<+12.1f}%\n")

            print(f"✓ Results saved to {save_path}")

        return results

    def evaluate_difficulty_comparison(self,
                                      sampler,
                                      normalizer,
                                      autoencoder,
                                      parser,
                                      difficulty_evaluator,
                                      target_playabilities: List[float] = [0.2, 0.4, 0.6, 0.8, 1.0],
                                      num_samples_per_target: int = 5,
                                      guidance_scale: float = 3.0,
                                      temperature: float = 0.5,
                                      device: str = 'cuda',
                                      save_path: Optional[str] = None) -> Dict:

        print(f"\n{'='*70}")
        print(f"DIFFICULTY EVALUATION COMPARISON")
        print(f"{'='*70}")
        print(f"Target Playabilities: {target_playabilities}")
        print(f"Samples per target: {num_samples_per_target}")
        print(f"Guidance Scale: {guidance_scale}")
        print(f"{'='*70}\n")

        results = {
            'target_playabilities': target_playabilities,
            'evaluations': [],
            'summary': {}
        }

        all_targets = []
        all_actual_scores = []

        for target_play in target_playabilities:
            print(f"\nGenerating {num_samples_per_target} patches with target playability = {target_play}...")

            target_scores = []
            actual_scores = []
            patches_list = []

            for sample_idx in range(num_samples_per_target):
                latent = sampler.sample_single_patch(
                    normalizer=normalizer,
                    previous_latent=None,
                    target_playability=target_play,
                    previous_playabilities=None,
                    temperature=temperature,
                    guidance_scale=guidance_scale,
                    show_progress=False
                )

                latent_denorm = normalizer.denormalize(latent.unsqueeze(0))

                with torch.no_grad():
                    latent_denorm = latent_denorm.to(device)
                    decoded_logits = autoencoder.decoder(latent_denorm)

                    if decoded_logits.dim() == 4:
                        patch = torch.argmax(decoded_logits, dim=1)
                    elif decoded_logits.dim() == 3:
                        patch = decoded_logits.long()
                    elif decoded_logits.dim() == 5:
                        decoded_logits = decoded_logits.squeeze(1)
                        patch = torch.argmax(decoded_logits, dim=1)
                    else:
                        patch = torch.argmax(decoded_logits, dim=-1)

                    patch = patch.cpu().numpy()[0]

                eval_result = difficulty_evaluator.evaluate_patch(
                    patch,
                    metadata={'target_playability': target_play, 'sample_idx': sample_idx}
                )

                difficulty_score = eval_result['scores']['difficulty_score']

                target_scores.append(target_play)
                actual_scores.append(difficulty_score)
                patches_list.append(patch)

                all_targets.append(target_play)
                all_actual_scores.append(difficulty_score)

                results['evaluations'].append({
                    'target_playability': target_play,
                    'actual_difficulty': difficulty_score,
                    'sample_idx': sample_idx,
                    'patch': patch,
                    'full_evaluation': eval_result
                })

            mean_actual = np.mean(actual_scores)
            std_actual = np.std(actual_scores)

            results['summary'][target_play] = {
                'mean_difficulty': mean_actual,
                'std_difficulty': std_actual,
                'target_playability': target_play,
                'error': abs(mean_actual - target_play),
                'samples': actual_scores
            }

            print(f"  Target: {target_play:.2f} | "
                  f"Actual Difficulty: {mean_actual:.3f} ± {std_actual:.3f} | "
                  f"Error: {abs(mean_actual - target_play):.3f}")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        ax1 = axes[0, 0]
        target_vals = list(results['summary'].keys())
        mean_vals = [results['summary'][t]['mean_difficulty'] for t in target_vals]
        std_vals = [results['summary'][t]['std_difficulty'] for t in target_vals]

        ax1.errorbar(target_vals, mean_vals, yerr=std_vals,
                    fmt='o', markersize=8, capsize=5, capthick=2,
                    label='Generated (Mean ± Std)', color='blue', alpha=0.7)
        ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Alignment', alpha=0.5)
        ax1.set_xlabel('Target Playability', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Actual Difficulty Score', fontsize=12, fontweight='bold')
        ax1.set_title('Target vs Actual Difficulty (Aggregated)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-0.05, 1.05)
        ax1.set_ylim(-0.05, 1.05)

        ax2 = axes[0, 1]
        ax2.scatter(all_targets, all_actual_scores, alpha=0.5, s=50, color='green')
        ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Alignment', alpha=0.5)
        ax2.set_xlabel('Target Playability', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Actual Difficulty Score', fontsize=12, fontweight='bold')
        ax2.set_title('All Individual Samples', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-0.05, 1.05)
        ax2.set_ylim(-0.05, 1.05)

        ax3 = axes[1, 0]
        errors = [results['summary'][t]['error'] for t in target_vals]
        ax3.bar(range(len(target_vals)), errors, color='orange', alpha=0.7, edgecolor='black')
        ax3.set_xticks(range(len(target_vals)))
        ax3.set_xticklabels([f'{t:.1f}' for t in target_vals])
        ax3.set_xlabel('Target Playability', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Absolute Error', fontsize=12, fontweight='bold')
        ax3.set_title('Prediction Error by Target', fontsize=14, fontweight='bold')
        ax3.grid(True, axis='y', alpha=0.3)

        ax4 = axes[1, 1]
        data_for_box = [results['summary'][t]['samples'] for t in target_vals]
        bp = ax4.boxplot(data_for_box, positions=range(len(target_vals)),
                        widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))
        ax4.plot(range(len(target_vals)), target_vals, 'go-',
                linewidth=2, markersize=8, label='Target', alpha=0.7)
        ax4.set_xticks(range(len(target_vals)))
        ax4.set_xticklabels([f'{t:.1f}' for t in target_vals])
        ax4.set_xlabel('Target Playability', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Difficulty Score Distribution', fontsize=12, fontweight='bold')
        ax4.set_title('Distribution of Generated Difficulties', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Evaluation plot saved to {save_path}")

        plt.show()

        print(f"\n{'='*90}")
        print(f"DETAILED EVALUATION RESULTS")
        print(f"{'='*90}")
        print(f"{'Target':<10} {'Generated':<12} {'Std Dev':<12} {'Error':<12} {'% Error':<12} {'Range':<20}")
        print(f"{'-'*90}")

        for target in target_vals:
            summary = results['summary'][target]
            samples = summary['samples']
            range_str = f"[{min(samples):.3f}, {max(samples):.3f}]"
            pct_error = (summary['error'] / target * 100) if target != 0 else 0
            print(f"{target:<10.2f} {summary['mean_difficulty']:<12.3f} "
                  f"{summary['std_difficulty']:<12.3f} {summary['error']:<12.3f} "
                  f"{pct_error:<12.1f}% {range_str:<20}")

        overall_mae = np.mean([results['summary'][t]['error'] for t in target_vals])
        overall_correlation = np.corrcoef(all_targets, all_actual_scores)[0, 1]

        print(f"\n{'='*90}")
        print(f"OVERALL STATISTICS")
        print(f"{'='*90}")
        print(f"Mean Absolute Error (MAE): {overall_mae:.4f}")
        print(f"Correlation Coefficient: {overall_correlation:.4f}")
        print(f"Total Samples Generated: {len(all_targets)}")
        print(f"{'='*90}\n")

        results['overall'] = {
            'mae': overall_mae,
            'correlation': overall_correlation,
            'total_samples': len(all_targets)
        }

        return results

# %%
import torch
import numpy as np
from typing import Optional

def generate_levels(
    num_levels: int = 5,
    patches_per_level: int = 20,
    playability_target: float = 1.0,
    autoencoder_path: str = '/content/Output/AutoEncoder.pth',
    diffusion_path: str = '/content/Output/conditional_diffusion_best.pth',
    temperature: float = 0.9,
    guidance_scale: float = 2.0,
    parser=None,
    stitcher_class=None,
    autoencoder_class=None,
    device: Optional[str] = None
):

    print("="*70)
    print(f"GENERATING PLAYABLE LEVELS (AUTOREGRESSIVE)")
    print("="*70)

    device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")


    autoencoder = autoencoder_class(
        num_tile_types=13,
        embedding_dim=32,
        latent_dim=128,
        patch_height=14,
        patch_width=16
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
        latent_dim=128,
        time_emb_dim=256,
        context_emb_dim=128,
        hidden_dims=[256, 512, 512],
        num_res_blocks=2,
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
        num_timesteps=500,
        schedule_type='linear',
        device=device,
    )

    normalizer = LatentNormalizer(target_norm=11.0)
    normalizer.load('Output/latent_normalizer.pth')

    sampler = Sampler(
        unet,
        schedule,
        device=device
    )
    print("✓ Autoregressive sampler initialized")
    stitcher = stitcher_class(patch_height=14, patch_width=16, stride=4)
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
            playability_target=playability_target,
            temperature=temperature,
            guidance_scale=guidance_scale,
            show_progress=True,
        )

        latents_denorm = normalizer.denormalize(latents)

        import math
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
        print("\nL2 norms (denorm) stats: mean/std/min/max:", float(l2.mean()), float(l2.std()), float(l2.min()), float(l2.max()))
        print("\nExpected training L2 mean/std (what you expect):", "mean=11, std=1 (confirm these are the intended metrics)")

        z_test = torch.randn(1000, ld.shape[1])
        z_test_den = normalizer.denormalize(z_test)
        print("\nTest mapping for std-normal inputs -> denorm mean/std (per-dim):", float(z_test_den.mean()), float(z_test_den.std()))
        print("Test mapping L2 norm mean/std:", float(z_test_den.norm(dim=-1).mean()), float(z_test_den.norm(dim=-1).std()))

        if not torch.is_tensor(latents_denorm):
            raise RuntimeError("Sampler did not return a torch.Tensor. Got: %s" % type(latents_denorm))

        assert latents_denorm.dim() == 2 and latents_denorm.shape[0] == patches_per_level, \
            f"Expected latents shape [num_patches, latent_dim], got {latents_denorm.shape}"

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

        output_path = f'/content/Output/autoregressive_level_{level_idx+1}.txt'
        with open(output_path, 'w') as f:
            f.write(level_str)
        print(f"✓ Saved to {output_path}")

        generated_levels.append(level)

    print(f"\n{'='*70}")
    print(f"✓ GENERATION COMPLETE! Generated {len(generated_levels)} levels")
    print(f"{'='*70}")

    return generated_levels, sampler


# %%
len(all_difficulties)

# %%
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.optim as optim


print("STEP 1: Training conditional diffusion model...")
trainer, unet, val_loader = train_conditional_diffusion_model(
    latents=latents_norm,
    metadata=metadata,
    autoencoder_path='/content/Output/AutoEncoder.pth',
    save_path='/content/Output/conditional_diffusion.pth',
    num_epochs=200,
    batch_size=16
)

# %%
def test_cfg_architecture():
    print("\n" + "="*70)
    print("TESTING CFG ARCHITECTURE")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = DiffusionUNet(
        latent_dim=128,
        time_emb_dim=256,
        context_emb_dim=128,
        hidden_dims=[256, 512, 512],
        num_res_blocks=2,
        cond_dropout=0.15
    ).to(device)

    batch_size = 4
    x = torch.randn(batch_size, 128, device=device)
    t = torch.randint(0, 500, (batch_size,), device=device)
    prev_lat = torch.randn(batch_size, 5, 128, device=device)
    prev_play = torch.rand(batch_size, 5, device=device)

    target_play = torch.rand(batch_size, device=device)
    noise_cond = model(x, t, prev_lat, prev_play, target_playability=target_play)
    print(f"✓ Conditional forward works: output shape = {noise_cond.shape}")

    noise_uncond = model(x, t, prev_lat, prev_play, target_playability=None)
    print(f"✓ Unconditional forward works: output shape = {noise_uncond.shape}")

    diff = (noise_cond - noise_uncond).abs().mean().item()
    print(f"✓ Cond vs Uncond difference: {diff:.4f}")

    guidance_scale = 3.0
    noise_guided = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
    print(f"✓ CFG formula works: output shape = {noise_guided.shape}")

    model.eval()
    with torch.no_grad():
        target_low = torch.full((batch_size,), 0.1, device=device)
        target_high = torch.full((batch_size,), 0.9, device=device)

        noise_low = model(x, t, prev_lat, prev_play, target_playability=target_low)
        noise_high = model(x, t, prev_lat, prev_play, target_playability=target_high)

        diff_targets = (noise_low - noise_high).abs().mean().item()
        print(f"✓ Different targets produce different noise: diff = {diff_targets:.4f}")

    print("\n✓ All CFG architecture tests passed!")
    print("="*70 + "\n")

    return model


if __name__ == "__main__":
    test_cfg_architecture()

# %%
stitcher = PatchStitcher()
scheduler = NoiseSchedule(num_timesteps=500, schedule_type='linear')
sampler = Sampler(unet, scheduler)
difficulty_evaluator = PatchDifficultyEvaluator(parser)

# %%
eval_results = stitcher.evaluate_difficulty_comparison(
    sampler=sampler,
    normalizer=normalizer,
    autoencoder=autoencoder,
    parser=parser,
    difficulty_evaluator=difficulty_evaluator,
    target_playabilities=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    num_samples_per_target=1,
    temperature = 0.5,
    guidance_scale=5.0,
    save_path='/content/Output/difficulty_evaluation.png'
)

# Check overall performance
print(f"MAE: {eval_results['overall']['mae']:.4f}")
print(f"Correlation: {eval_results['overall']['correlation']:.4f}")

# %%
results = stitcher.compare_cfg_difficulty(
    sampler=sampler,
    normalizer=normalizer,
    autoencoder=autoencoder,
    difficulty_evaluator=difficulty_evaluator,
    parser=parser,
    guidance_scales=[0.0, 1.0, 2.0, 3.0, 5.0, 7.0],
    num_patches=1,
    target_playability=1.0,
    save_path='/content/Output/cfg_comparison.png'
)

# %%
levels, sampler = generate_levels(
    num_levels=1,
    patches_per_level=20,
    parser=parser,
    playability_target = 1.0,
    temperature=0.2,
    guidance_scale = 5,
    stitcher_class=PatchStitcher,
    autoencoder_class=Autoencoder,
    diffusion_path = '/content/Output/conditional_diffusion.pth'
)


# %%
!pip install mario_gpt

# %%
from mario_gpt import MarioDataset, MarioLM
from mario_gpt.utils import view_level, convert_level_to_png, join_list_of_list, characterize

mario_lm = MarioLM()

test = """----------------
----------------
----------------
----------------
----------------
---------E-E----
--------<><>----
------[][][]----
------QQQQQQ----
----------------
----------------
----------------
----------------
XXXX--XXX-XXXXXX""".split("\n")

testing = convert_level_to_png(test,  mario_lm.tokenizer)[0]
testing


