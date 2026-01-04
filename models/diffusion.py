import torch
import torch.nn as nn
from typing import Optional
from .embeddings import FourierDifficultyEmbedding, SinusoidalPositionalEmbedding

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, time_emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(time_emb_dim, dim), nn.SiLU())
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        time_proj = self.time_mlp(time_emb)
        h = self.block(x + time_proj)
        return self.activation(x + self.dropout(h))


class MixingMLP(nn.Module):
    def __init__(self, dim: int, hidden_scale: float = 1.0, dropout: float = 0.0):
        super().__init__()
        mid = int(dim * hidden_scale)
        mid = max(mid, dim)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mid),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mlp(self.norm(x)))


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
        self.prev_diff_emb_dim = 64
        self.hidden_dims = hidden_dims
        self.num_res_blocks = num_res_blocks
        self.cond_dropout = cond_dropout

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

        self.prev_difficulty_mlp = nn.Sequential(
            nn.Linear(1, self.prev_diff_emb_dim),
            nn.SiLU(),
            nn.Linear(self.prev_diff_emb_dim, time_emb_dim)
        )

        self.difficulty_embedding = FourierDifficultyEmbedding(
            embedding_dim=time_emb_dim,
            num_frequencies=128
        )
        self.null_diff_embedding = nn.Parameter(torch.randn(time_emb_dim) * 0.02)
        self.null_context_embedding = nn.Parameter(torch.randn(time_emb_dim) * 0.02)
        
        # Learnable scale for difficulty embedding to strengthen its influence
        self.diff_scale = nn.Parameter(torch.ones(1) * 2.0)
        
        self.output_scale = nn.Parameter(torch.ones(1))

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
            current_dim = reversed_dims[i]
            skip_dim = reversed_skip_dims[i]
            concat_dim = current_dim + skip_dim
            self.skip_projections.append(nn.Linear(concat_dim, current_dim))
            
            blocks = nn.ModuleList([ResidualBlock(current_dim, time_emb_dim) for _ in range(num_res_blocks)])
            self.decoder_blocks.append(blocks)
            self.decoder_mixes.append(MixingMLP(current_dim, hidden_scale=1.0))
            
            if i < len(reversed_dims) - 1:
                next_dim = reversed_dims[i + 1]
                self.decoder_blocks.append(nn.Linear(current_dim, next_dim))

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
        previous_difficulties: Optional[torch.Tensor] = None,
        target_difficulty: Optional[torch.Tensor] = None,
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

        if self.training and self.cond_dropout > 0:
            cond_drop_mask = (torch.rand(batch, device=device) < self.cond_dropout)
        else:
            cond_drop_mask = None

        if previous_difficulties is None:
            previous_difficulties = torch.zeros((batch, k), device=device, dtype=dtype)
        else:
            previous_difficulties = previous_difficulties.to(device=device, dtype=dtype)
            if previous_difficulties.dim() == 2:
                if previous_difficulties.shape[0] != batch or previous_difficulties.shape[1] != k:
                    if previous_difficulties.numel() == batch * k:
                        previous_difficulties = previous_difficulties.reshape(batch, k)
                    else:
                        raise ValueError('previous_difficulties shape incompatible')
            elif previous_difficulties.dim() == 1:
                n = previous_difficulties.numel()
                if n == k:
                    previous_difficulties = previous_difficulties.unsqueeze(0).repeat(batch, 1)
                elif n == batch * k:
                    previous_difficulties = previous_difficulties.reshape(batch, k)
                else:
                    raise ValueError('previous_difficulties length incompatible')

        prev_diff_flat = previous_difficulties.reshape(batch * k, 1)
        prev_diff_enc = self.prev_difficulty_mlp(prev_diff_flat)
        prev_diff_enc = prev_diff_enc.reshape(batch, k, -1)

        per_prev = prev_enc + prev_diff_enc
        ctx_emb = per_prev.mean(dim=1)
        
        if cond_drop_mask is not None:
            mask_2d = cond_drop_mask.view(-1, 1)
            null_ctx = self.null_context_embedding.unsqueeze(0).expand(batch, -1)
            ctx_emb = torch.where(mask_2d.expand_as(ctx_emb), null_ctx, ctx_emb)

        if target_difficulty is None:
            diff_emb = self.null_diff_embedding.unsqueeze(0).expand(batch, -1)
        else:
            target_difficulty = target_difficulty.to(device=device, dtype=dtype)
            if target_difficulty.dim() == 0:
                target_difficulty = target_difficulty.unsqueeze(0)
            if target_difficulty.dim() == 1:
                target_difficulty = target_difficulty.view(-1, 1)

            diff_emb = self.difficulty_embedding(target_difficulty)

            if cond_drop_mask is not None:
                mask_2d = cond_drop_mask.view(-1, 1)
                null_emb = self.null_diff_embedding.unsqueeze(0).expand(batch, -1)
                diff_emb = torch.where(mask_2d, null_emb, diff_emb)

        if previous_latents.dim() == 2:
            prev_lat_for_mean = prev_flat.reshape(batch, k, self.latent_dim)
        else:
            prev_lat_for_mean = previous_latents.reshape(batch, k, self.latent_dim)
        prev_lat_mean = prev_lat_for_mean.mean(dim=1)

        if cond_drop_mask is not None:
            mask_2d = cond_drop_mask.view(-1, 1)
            prev_lat_mean = torch.where(mask_2d.expand_as(prev_lat_mean), torch.zeros_like(prev_lat_mean), prev_lat_mean)

        h = torch.cat([x, prev_lat_mean], dim=-1)
        h = self.input_proj(h)

        combined_emb = t_emb + ctx_emb + (diff_emb * self.diff_scale)


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
            skip = skip_connections[i]
            h = torch.cat([h, skip], dim=-1)
            h = self.skip_projections[i](h)
            
            for res_block in self.decoder_blocks[decoder_block_idx]:
                h = res_block(h, combined_emb)
            decoder_block_idx += 1
            
            h = self.decoder_mixes[i](h)
            
            if i < len(self.hidden_dims) - 1:
                h = self.decoder_blocks[decoder_block_idx](h)
                decoder_block_idx += 1

        noise_pred = self.output_proj(h)
        scale = self.output_scale.clamp(min=0.1, max=10.0)
        noise_pred = noise_pred * scale

        return noise_pred