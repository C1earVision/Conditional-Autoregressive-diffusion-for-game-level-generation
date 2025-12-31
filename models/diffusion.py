import torch
import torch.nn as nn
from typing import Optional
from .embeddings import FourierDifficultyEmbedding, SinusoidalPositionalEmbedding


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, time_emb_dim: int, dropout: float = 0.1):
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
    def __init__(self, dim: int, hidden_scale: float = 1.0, dropout: float = 0.1):
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

        self.prev_difficulty_mlp = nn.Sequential(
            nn.Linear(1, self.prev_diff_emb_dim),
            nn.SiLU(),
            nn.Linear(self.prev_diff_emb_dim, time_emb_dim)
        )

        self.difficulty_embedding = FourierDifficultyEmbedding(
            embedding_dim=time_emb_dim,
            num_frequencies=64
        )
        self.null_diff_embedding = nn.Parameter(torch.randn(time_emb_dim) * 0.02)

        self.input_proj = nn.Linear(latent_dim * 2, hidden_dims[0])

        # Build encoder with clear structure
        self.encoder_levels = nn.ModuleList()
        self.encoder_downsamples = nn.ModuleList()
        
        for i in range(len(hidden_dims)):
            # Residual blocks at this level
            current_dim = hidden_dims[i] if i > 0 else hidden_dims[0]
            blocks = nn.ModuleList([
                ResidualBlock(current_dim, time_emb_dim) 
                for _ in range(num_res_blocks)
            ])
            mix = MixingMLP(current_dim, hidden_scale=1.0)
            
            self.encoder_levels.append(nn.ModuleDict({
                'blocks': blocks,
                'mix': mix
            }))
            
            # Downsampling to next level (if not last)
            if i < len(hidden_dims) - 1:
                self.encoder_downsamples.append(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1])
                )

        # Bottleneck
        bottleneck_dim = hidden_dims[-1]
        self.bottleneck_blocks = nn.ModuleList([
            ResidualBlock(bottleneck_dim, time_emb_dim) 
            for _ in range(num_res_blocks)
        ])
        self.bottleneck_mix = MixingMLP(bottleneck_dim, hidden_scale=1.0)

        # Build decoder - mirror of encoder
        self.decoder_levels = nn.ModuleList()
        self.decoder_upsamples = nn.ModuleList()
        self.skip_projections = nn.ModuleList()
        
        # Decoder processes in reverse: [512, 512, 256] for hidden_dims=[256, 512, 512]
        for i in range(len(hidden_dims)):
            decoder_idx = len(hidden_dims) - 1 - i
            current_dim = hidden_dims[decoder_idx]
            
            # Skip connection projection (concatenate then project)
            # Skip comes from encoder at the same level
            skip_dim = current_dim
            concat_dim = current_dim + skip_dim
            skip_proj = nn.Linear(concat_dim, current_dim)
            self.skip_projections.append(skip_proj)
            
            # Residual blocks at this level
            blocks = nn.ModuleList([
                ResidualBlock(current_dim, time_emb_dim) 
                for _ in range(num_res_blocks)
            ])
            mix = MixingMLP(current_dim, hidden_scale=1.0)
            
            self.decoder_levels.append(nn.ModuleDict({
                'blocks': blocks,
                'mix': mix
            }))
            
            # Upsampling to next level (if not last)
            if i < len(hidden_dims) - 1:
                next_decoder_idx = decoder_idx - 1
                next_dim = hidden_dims[next_decoder_idx]
                self.decoder_upsamples.append(
                    nn.Linear(current_dim, next_dim)
                )

        self.output_proj = nn.Linear(hidden_dims[0], latent_dim)

        self._initialize_weights()

        print(f"\n{'='*70}")
        print(f"CFGDiffusionUNet Initialized")
        print(f"{'='*70}")
        print(f"  Condition dropout: {cond_dropout*100:.0f}%")
        print(f"  Latent dim: {latent_dim}")
        print(f"  Hidden dims: {hidden_dims}")
        print(f"  Encoder levels: {len(self.encoder_levels)}")
        print(f"  Decoder levels: {len(self.decoder_levels)}")
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

        # Time embedding
        t_emb = self.time_embedding(timesteps)
        t_emb = self.time_mlp(t_emb)

        # Process previous latents
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

        # Process previous difficulties
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

        # Process target difficulty
        if target_difficulty is None:
            diff_emb = self.null_diff_embedding.unsqueeze(0).expand(batch, -1)
        else:
            target_difficulty = target_difficulty.to(device=device, dtype=dtype)
            if target_difficulty.dim() == 0:
                target_difficulty = target_difficulty.unsqueeze(0)
            if target_difficulty.dim() == 1:
                target_difficulty = target_difficulty.view(-1, 1)

            diff_emb = self.difficulty_embedding(target_difficulty)

            if self.training and self.cond_dropout > 0:
                drop_mask = (torch.rand(batch, device=device) < self.cond_dropout).unsqueeze(1)
                null_emb = self.null_diff_embedding.unsqueeze(0).expand(batch, -1)
                diff_emb = torch.where(drop_mask, null_emb, diff_emb)

        # Prepare input
        if previous_latents.dim() == 2:
            prev_lat_for_mean = prev_flat.reshape(batch, k, self.latent_dim)
        else:
            prev_lat_for_mean = previous_latents.reshape(batch, k, self.latent_dim)
        prev_lat_mean = prev_lat_for_mean.mean(dim=1)

        h = torch.cat([x, prev_lat_mean], dim=-1)
        h = self.input_proj(h)

        # Combined embedding
        combined_emb = t_emb + ctx_emb + diff_emb

        # Encoder forward pass
        skip_connections = []
        for i, level in enumerate(self.encoder_levels):
            # Process residual blocks
            for res_block in level['blocks']:
                h = res_block(h, combined_emb)
            h = level['mix'](h)
            
            # Save skip connection BEFORE downsampling
            skip_connections.append(h)
            
            # Downsample if not last level
            if i < len(self.encoder_downsamples):
                h = self.encoder_downsamples[i](h)

        # Bottleneck
        for bottleneck_block in self.bottleneck_blocks:
            h = bottleneck_block(h, combined_emb)
        h = self.bottleneck_mix(h)

        # Decoder forward pass
        # Skip connections are in encoder order, we need them in decoder order (reversed)
        skip_connections = list(reversed(skip_connections))
        
        for i, level in enumerate(self.decoder_levels):
            # Concatenate with skip connection
            skip = skip_connections[i]
            h = torch.cat([h, skip], dim=-1)
            h = self.skip_projections[i](h)
            
            # Process residual blocks
            for res_block in level['blocks']:
                h = res_block(h, combined_emb)
            h = level['mix'](h)
            
            # Upsample if not last level
            if i < len(self.decoder_upsamples):
                h = self.decoder_upsamples[i](h)

        # Output projection
        noise_pred = self.output_proj(h)
        scale = self.output_scale.clamp(min=0.1, max=10.0)
        noise_pred = noise_pred * scale

        return noise_pred