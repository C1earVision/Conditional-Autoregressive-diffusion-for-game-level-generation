import math
import torch
import torch.nn as nn


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


class FourierDifficultyEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, num_frequencies: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_frequencies = num_frequencies

        frequencies = torch.exp(torch.linspace(-2.0, 4.0, num_frequencies))
        self.register_buffer('frequencies', frequencies)

        fourier_dim = num_frequencies * 2
        self.projection = nn.Sequential(
            nn.Linear(fourier_dim + 1, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, difficulty: torch.Tensor) -> torch.Tensor:
        if difficulty.dim() == 1:
            difficulty = difficulty.unsqueeze(-1)

        scaled = difficulty * 2 - 1

        angles = scaled * self.frequencies.unsqueeze(0) * math.pi
        fourier_features = torch.cat([
            torch.sin(angles),
            torch.cos(angles),
            difficulty
        ], dim=-1)

        return self.projection(fourier_features)