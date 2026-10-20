"""
JED-M2 Pure PyTorch Visual Patch Encoder (Phase 6D).
Converts raw screenshot images [B, 3, H, W] into spatial visual patch tokens [B, N_p, hidden_dim].

STRICT BUILD RULE: Pure PyTorch without torchvision, Hugging Face, or compiled C++/CUDA extensions.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from jed_m2.config import JEDM2Config
from jed_m2.mla import RMSNorm


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """
    Generates 2D sinusoidal positional embeddings for a square grid [grid_size x grid_size].

    Args:
        embed_dim: Feature dimension D (must be even).
        grid_size: Spatial grid side length (e.g., 14 for 224x224 with 16x16 patches).

    Returns:
        Positional embedding tensor of shape [1, grid_size * grid_size, embed_dim].
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")  # [2, grid_size, grid_size]
    grid = torch.stack(grid, dim=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    
    # 1D sin/cos for height and width
    dim_per_coord = embed_dim // 2
    omega = torch.arange(dim_per_coord // 2, dtype=torch.float32)
    omega = 1.0 / (10000.0 ** (omega / (dim_per_coord // 2)))

    out_h = grid[0].reshape(-1, 1) @ omega.reshape(1, -1)  # [grid_size*grid_size, dim//4]
    out_w = grid[1].reshape(-1, 1) @ omega.reshape(1, -1)  # [grid_size*grid_size, dim//4]

    pos_embed_h = torch.cat([torch.sin(out_h), torch.cos(out_h)], dim=1)
    pos_embed_w = torch.cat([torch.sin(out_w), torch.cos(out_w)], dim=1)

    pos_embed = torch.cat([pos_embed_h, pos_embed_w], dim=1)  # [N_p, embed_dim]
    return pos_embed.unsqueeze(0)  # [1, N_p, embed_dim]


class VisionPatchEncoder(nn.Module):
    """
    Lightweight, Pure PyTorch Visual Patch Encoder for Large Action Models.
    Projects RGB screenshot images [B, 3, H, W] to visual tokens [B, N_patches, hidden_dim].
    
    Architecture:
    1. 2D Convolutional Patch Projection (kernel_size=patch_size, stride=patch_size)
    2. Spatial Flattening: [B, hidden_dim, H/P, W/P] -> [B, N_p, hidden_dim]
    3. 2D Spatial Positional Embeddings (with dynamic 2D interpolation for variable resolutions)
    4. Root-Mean-Square Normalization (RMSNorm)
    """

    def __init__(self, config: Optional[JEDM2Config] = None):
        super().__init__()
        self.config = config if config is not None else JEDM2Config()
        self.hidden_dim = self.config.hidden_dim
        self.image_size = getattr(self.config, "image_size", 224)
        self.patch_size = getattr(self.config, "patch_size", 16)
        self.num_channels = getattr(self.config, "num_channels", 3)
        self.grid_size = self.image_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size

        # 1. Non-overlapping patch projection: [B, 3, H, W] -> [B, hidden_dim, H/P, W/P]
        self.patch_proj = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        # 2. Learnable 2D spatial positional embedding initialized with 2D sincos grid
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, self.hidden_dim)
        )
        self._init_pos_embed()

        # 3. Layer normalization
        self.norm = RMSNorm(self.hidden_dim, eps=self.config.layer_norm_eps)

    def _init_pos_embed(self) -> None:
        """Initializes positional embeddings with 2D sinusoidal representation."""
        pos_embed = get_2d_sincos_pos_embed(self.hidden_dim, self.grid_size)
        self.pos_embed.data.copy_(pos_embed)

    def _interpolate_pos_embed(self, h_patches: int, w_patches: int) -> torch.Tensor:
        """
        Dynamically interpolates 2D positional embeddings when input resolution differs from default.

        Args:
            h_patches: Number of vertical patches (H // patch_size).
            w_patches: Number of horizontal patches (W // patch_size).

        Returns:
            Interpolated position embedding tensor [1, h_patches * w_patches, hidden_dim].
        """
        if h_patches == self.grid_size and w_patches == self.grid_size:
            return self.pos_embed

        # [1, N_orig, D] -> [1, orig_grid, orig_grid, D] -> [1, D, orig_grid, orig_grid]
        pos_embed = self.pos_embed.reshape(1, self.grid_size, self.grid_size, self.hidden_dim)
        pos_embed = pos_embed.permute(0, 3, 1, 2)

        # Bicubic 2D interpolation for continuous spatial alignment
        pos_embed = F.interpolate(
            pos_embed,
            size=(h_patches, w_patches),
            mode="bicubic",
            align_corners=False,
        )

        # [1, D, h_p, w_p] -> [1, h_p, w_p, D] -> [1, h_p * w_p, D]
        pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
        return pos_embed

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract visual patch tokens from screenshot images.

        Args:
            images: Screen image tensor of shape [batch_size, 3, H, W].

        Returns:
            Visual patch tokens of shape [batch_size, N_patches, hidden_dim].
        """
        if images.ndim != 4:
            raise ValueError(f"Expected 4D image tensor [B, C, H, W], got shape {images.shape}")

        b, c, h, w = images.shape
        if c != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} image channels, got {c}")

        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(
                f"Image dimensions ({h}, {w}) must be divisible by patch size {self.patch_size}"
            )

        h_patches = h // self.patch_size
        w_patches = w // self.patch_size

        # 1. Patch projection: [B, 3, H, W] -> [B, hidden_dim, h_patches, w_patches]
        x = self.patch_proj(images)

        # 2. Flatten spatial dimensions: [B, hidden_dim, N_p] -> [B, N_p, hidden_dim]
        x = x.flatten(2).transpose(1, 2)

        # 3. Add 2D spatial positional embeddings: [B, N_p, hidden_dim]
        pos_embed = self._interpolate_pos_embed(h_patches, w_patches).to(
            dtype=x.dtype, device=x.device
        )
        x = x + pos_embed

        # 4. RMSNorm
        x = self.norm(x)
        return x
