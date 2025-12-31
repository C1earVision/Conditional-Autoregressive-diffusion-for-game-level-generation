import torch
import torch.nn as nn


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