import numpy as np
from typing import List, Dict, Tuple


class PatchExtractor:
    def __init__(self, patch_height: int = 14, patch_width: int = 16,
                 stride: int = 2, vertical_stride: int = 8):
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
