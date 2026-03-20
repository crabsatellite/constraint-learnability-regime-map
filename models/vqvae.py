"""
3D VQ-VAE for Minecraft builds.

Compresses 32x32x32 voxel grids into 8x8x8 discrete latent codes.
Each code represents a 4x4x4 region of the build.

Architecture:
  Encoder: 32³ -> 16³ -> 8³ (strided conv3d)
  Quantizer: 8³ continuous -> 8³ discrete (codebook lookup)
  Decoder: 8³ -> 16³ -> 32³ (transposed conv3d)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock3D(nn.Module):
    """Residual block with GroupNorm for 3D data."""

    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class VectorQuantizer(nn.Module):
    """
    Vector Quantization with EMA codebook updates and dead code revival.
    """

    def __init__(self, num_codes=1024, code_dim=256, beta=0.25,
                 ema_decay=0.99, revival_threshold=2):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta
        self.ema_decay = ema_decay
        self.revival_threshold = revival_threshold

        # Codebook
        self.embedding = nn.Embedding(num_codes, code_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_codes, 1.0 / num_codes)

        # EMA tracking
        self.register_buffer('ema_cluster_size', torch.zeros(num_codes))
        self.register_buffer('ema_embed_sum', self.embedding.weight.data.clone())
        self.register_buffer('usage_count', torch.zeros(num_codes, dtype=torch.long))

    def forward(self, z):
        """
        Args:
            z: (B, C, D, H, W) continuous latent
        Returns:
            z_q: (B, C, D, H, W) quantized latent
            indices: (B, D, H, W) codebook indices
            loss: commitment + codebook loss
            perplexity: codebook usage metric
        """
        B, C, D, H, W = z.shape

        # (B, C, D, H, W) -> (B*D*H*W, C)
        z_flat = z.permute(0, 2, 3, 4, 1).reshape(-1, C)

        # Find nearest codebook entry
        # dist = ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z.e
        dist = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * z_flat @ self.embedding.weight.t()
        )
        indices = dist.argmin(dim=1)
        z_q_flat = self.embedding(indices)

        # EMA codebook update (only during training)
        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(indices, self.num_codes).float()
                cluster_size = one_hot.sum(0)
                embed_sum = one_hot.t() @ z_flat

                self.ema_cluster_size.mul_(self.ema_decay).add_(
                    cluster_size, alpha=1 - self.ema_decay
                )
                self.ema_embed_sum.mul_(self.ema_decay).add_(
                    embed_sum, alpha=1 - self.ema_decay
                )

                # Laplace smoothing
                n = self.ema_cluster_size.sum()
                cluster_size_smooth = (
                    (self.ema_cluster_size + 1e-5)
                    / (n + self.num_codes * 1e-5)
                    * n
                )
                self.embedding.weight.data.copy_(
                    self.ema_embed_sum / cluster_size_smooth.unsqueeze(1)
                )

                # Track usage for dead code revival
                self.usage_count += (cluster_size > 0).long()

        # Losses
        # Commitment loss: encourage encoder output to stay close to codes
        commitment_loss = F.mse_loss(z_flat, z_q_flat.detach())
        # Codebook loss is handled by EMA, but we still use straight-through
        loss = self.beta * commitment_loss

        # Straight-through estimator
        z_q_flat = z_flat + (z_q_flat - z_flat).detach()

        # Reshape back
        z_q = z_q_flat.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)
        indices = indices.reshape(B, D, H, W)

        # Perplexity (codebook utilization metric)
        with torch.no_grad():
            avg_probs = F.one_hot(indices.flatten(), self.num_codes).float().mean(0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q, indices, loss, perplexity

    def revive_dead_codes(self, z_flat):
        """Replace unused codes with random encoder outputs."""
        with torch.no_grad():
            dead = self.usage_count < self.revival_threshold
            n_dead = dead.sum().item()
            if n_dead > 0 and len(z_flat) > 0:
                rand_idx = torch.randint(0, len(z_flat), (n_dead,), device=z_flat.device)
                self.embedding.weight.data[dead] = z_flat[rand_idx]
                self.ema_embed_sum[dead] = z_flat[rand_idx]
                self.ema_cluster_size[dead] = 1.0
                self.usage_count[dead] = 0
            # Reset usage tracking periodically
            self.usage_count.zero_()
        return n_dead

    def decode_indices(self, indices):
        """Convert codebook indices back to continuous latent vectors."""
        return self.embedding(indices).permute(0, 4, 1, 2, 3)


class VQVAE3D(nn.Module):
    """
    3D VQ-VAE: 32³ voxel grid -> latent³ discrete codes -> 32³ reconstruction.

    Input: (B, 32, 32, 32) int64 token grid (0 = air, 1+ = block types)
    Output: (B, vocab_size, 32, 32, 32) logits per voxel

    n_downsample=2: 32->16->8 (8³ latent, 512 tokens) — recommended
    n_downsample=3: 32->16->8->4 (4³ latent, 64 tokens) — legacy
    """

    def __init__(self, vocab_size=513, embed_dim=32, hidden_dim=128,
                 code_dim=256, num_codes=2048, latent_size=4, n_downsample=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Infer n_downsample from latent_size if not provided (backwards compat)
        if n_downsample is None:
            import math
            n_downsample = int(math.log2(32 // latent_size))
        self.n_downsample = n_downsample
        self.latent_size = 32 // (2 ** n_downsample)

        # Token embedding (block type -> vector)
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # Build encoder and decoder dynamically
        if n_downsample == 2:
            # Encoder: 32³ -> 16³ -> 8³
            self.encoder = nn.Sequential(
                nn.Conv3d(embed_dim, hidden_dim, 4, stride=2, padding=1),  # 32->16
                nn.GroupNorm(8, hidden_dim),
                nn.SiLU(),
                ResBlock3D(hidden_dim),
                nn.Conv3d(hidden_dim, code_dim, 4, stride=2, padding=1),  # 16->8
                nn.GroupNorm(min(8, code_dim), code_dim),
                nn.SiLU(),
                ResBlock3D(code_dim),
                ResBlock3D(code_dim),
                ResBlock3D(code_dim),
            )
            # Decoder: 8³ -> 16³ -> 32³
            self.decoder = nn.Sequential(
                ResBlock3D(code_dim),
                ResBlock3D(code_dim),
                ResBlock3D(code_dim),
                nn.ConvTranspose3d(code_dim, hidden_dim, 4, stride=2, padding=1),  # 8->16
                nn.GroupNorm(8, hidden_dim),
                nn.SiLU(),
                ResBlock3D(hidden_dim),
                nn.ConvTranspose3d(hidden_dim, hidden_dim // 2, 4, stride=2, padding=1),  # 16->32
            )
            head_ch = hidden_dim // 2
        else:
            # Encoder: 32³ -> 16³ -> 8³ -> 4³ (legacy)
            self.encoder = nn.Sequential(
                nn.Conv3d(embed_dim, hidden_dim // 2, 4, stride=2, padding=1),  # 32->16
                nn.GroupNorm(8, hidden_dim // 2),
                nn.SiLU(),
                nn.Conv3d(hidden_dim // 2, hidden_dim, 4, stride=2, padding=1),  # 16->8
                nn.GroupNorm(8, hidden_dim),
                nn.SiLU(),
                ResBlock3D(hidden_dim),
                nn.Conv3d(hidden_dim, code_dim, 4, stride=2, padding=1),  # 8->4
                nn.GroupNorm(min(8, code_dim), code_dim),
                nn.SiLU(),
                ResBlock3D(code_dim),
                ResBlock3D(code_dim),
            )
            # Decoder: 4³ -> 8³ -> 16³ -> 32³
            self.decoder = nn.Sequential(
                ResBlock3D(code_dim),
                ResBlock3D(code_dim),
                nn.ConvTranspose3d(code_dim, hidden_dim, 4, stride=2, padding=1),  # 4->8
                nn.GroupNorm(8, hidden_dim),
                nn.SiLU(),
                ResBlock3D(hidden_dim),
                nn.ConvTranspose3d(hidden_dim, hidden_dim // 2, 4, stride=2, padding=1),  # 8->16
                nn.GroupNorm(8, hidden_dim // 2),
                nn.SiLU(),
                nn.ConvTranspose3d(hidden_dim // 2, hidden_dim // 2, 4, stride=2, padding=1),  # 16->32
            )
            head_ch = hidden_dim // 2

        # Vector Quantizer
        self.quantizer = VectorQuantizer(
            num_codes=num_codes,
            code_dim=code_dim,
            beta=0.25,
            ema_decay=0.99,
        )

        # Output head
        self.head = nn.Sequential(
            nn.GroupNorm(min(8, head_ch), head_ch),
            nn.SiLU(),
            nn.Conv3d(head_ch, vocab_size, 1),
        )

    def encode(self, x):
        """Encode voxel grid to continuous latent."""
        # x: (B, 32, 32, 32) int64
        emb = self.token_embed(x)         # (B, 32, 32, 32, embed_dim)
        emb = emb.permute(0, 4, 1, 2, 3)  # (B, embed_dim, 32, 32, 32)
        z = self.encoder(emb)              # (B, code_dim, 8, 8, 8)
        return z

    def decode(self, z_q):
        """Decode quantized latent to voxel logits."""
        h = self.decoder(z_q)   # (B, hidden_dim, 32, 32, 32)
        logits = self.head(h)   # (B, vocab_size, 32, 32, 32)
        return logits

    def forward(self, x):
        """
        Full forward pass: encode -> quantize -> decode.

        Args:
            x: (B, 32, 32, 32) int64 voxel grid
        Returns:
            logits: (B, vocab_size, 32, 32, 32)
            vq_loss: scalar
            perplexity: scalar
            indices: (B, 8, 8, 8) codebook indices
        """
        z = self.encode(x)
        z_q, indices, vq_loss, perplexity = self.quantizer(z)
        logits = self.decode(z_q)
        return logits, vq_loss, perplexity, indices

    def encode_to_indices(self, x):
        """Encode voxel grid directly to discrete codebook indices."""
        with torch.no_grad():
            z = self.encode(x)
            _, indices, _, _ = self.quantizer(z)
        return indices

    def decode_from_indices(self, indices):
        """Decode codebook indices back to voxel logits."""
        with torch.no_grad():
            z_q = self.quantizer.decode_indices(indices)
            logits = self.decode(z_q)
        return logits
