"""
UNet baseline wrapper for Gray-Scott dataset.

Wraps a standard UNet2d as a single-step predictor with teacher forcing.
For each of the T input frames, UNet predicts the next frame independently.
Output is stacked to match our model's [B, T, H, W, 18] interface.

Active channels: CH_A=5 (species A), CH_B=6 (species B).
UNet operates on 2 channels only, results are placed back into 18-ch tensor.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unet import UNet2d


# Gray-Scott channel indices in 18-channel layout
CH_A = 5  # scalar[2] = concentration_u (activator)
CH_B = 6  # scalar[3] = concentration_v (inhibitor)


class UNetGrayScott(nn.Module):
    """UNet baseline for Gray-Scott with teacher-forcing multi-step output.

    Forward signature matches our model:
        input:  [B, T, H, W, 18]
        output: [B, T, H, W, 18]
    """

    def __init__(self, init_features: int = 32):
        super().__init__()
        self.unet = UNet2d(in_channels=2, out_channels=2, init_features=init_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, H, W, 18] — T input frames (teacher forcing)

        Returns:
            [B, T, H, W, 18] — T predicted next-frames
        """
        B, T, H, W, C = x.shape

        # Extract active channels: [B, T, H, W, 2]
        ab = torch.stack([x[..., CH_A], x[..., CH_B]], dim=-1)

        # Reshape for batch UNet: [B*T, 2, H, W]
        ab_flat = ab.reshape(B * T, H, W, 2).permute(0, 3, 1, 2)

        # Single-step prediction (teacher forcing: each frame predicted independently)
        pred_flat = self.unet(ab_flat)  # [B*T, 2, H, W]

        # Reshape back: [B, T, H, W, 2]
        pred = pred_flat.permute(0, 2, 3, 1).reshape(B, T, H, W, 2)

        # Place back into 18-channel tensor
        output = torch.zeros_like(x)
        output[..., CH_A] = pred[..., 0]
        output[..., CH_B] = pred[..., 1]

        return output
