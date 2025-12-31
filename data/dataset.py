import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Sequence


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


class AutoregressivePatchDatasetCreator(Dataset):
    def __init__(self,
                 latents: torch.Tensor,
                 difficulties: Sequence,
                 num_prev: int = 10,
                 level_ids: Optional[Sequence] = None):
        super().__init__()

        assert latents.dim() == 2
        num_patches = latents.shape[0]

        assert len(difficulties) == num_patches
        if level_ids is not None:
            assert len(level_ids) == num_patches

        self.latents = latents
        self.difficulties = torch.as_tensor(difficulties, dtype=latents.dtype)
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
        current_difficulty = self.difficulties[idx].unsqueeze(0)
        zero_latent = torch.zeros(self.latent_dim, dtype=dtype, device=device)
        zero_diff = torch.zeros(1, dtype=dtype, device=device)

        prev_latents = []
        prev_difficulties = []

        for j in range(1, k + 1):
            prev_idx = idx - j
            if prev_idx < 0:
                prev_latents.append(zero_latent)
                prev_difficulties.append(zero_diff)
                continue
            if self.level_ids is not None:
                if self.level_ids[prev_idx].item() != self.level_ids[idx].item():
                    prev_latents.append(zero_latent)
                    prev_difficulties.append(zero_diff)
                    continue
            prev_latents.append(self.latents[prev_idx].to(device))
            prev_difficulties.append(self.difficulties[prev_idx].unsqueeze(0).to(device))
        prev_latents = torch.stack(prev_latents, dim=0)
        prev_difficulties = torch.cat(prev_difficulties, dim=0).view(k)
        current_latent = current_latent.to(device)
        current_difficulty = current_difficulty.to(device)

        return current_latent, prev_latents, current_difficulty, prev_difficulties
