"""
V3 Improved 2D Decoder for PDE Foundation Model.

Improvements over decoder_v2:
    A: Cross-patch mixing at 8×8 resolution (CrossPatchMixer)
    B: Upsample + Conv2d replaces ConvTranspose2d (no checkerboard)
    C: Encoder → Decoder skip connections at 4×4 and 8×8
    D: Increased capacity (configurable hidden_channels, extra ResBlocks)

Identity initialization:
    - Skip merge Conv1x1: [I, 0] → ignores skip at init
    - CrossPatchMixer: zero-init last conv → identity at init
    - Encoder + Transformer weights fully preserved from pretrained checkpoint
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from typing import Dict, Optional

from pretrain.attention_v2 import RMSNorm


class ResBlock2D(nn.Module):
    """Residual block with GroupNorm (same as encoder_v2)."""

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='replicate'),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='replicate'),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class UpsampleConv(nn.Module):
    """Bilinear upsample + Conv2d (replaces ConvTranspose2d to avoid checkerboard)."""

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode='replicate')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class SkipMerge(nn.Module):
    """
    Merge decoder features with encoder skip features via concat + 1×1 Conv.

    Identity-init: weight = [I, 0], bias = 0
    → At initialization, output = decoder features (skip ignored).
    """

    def __init__(self, decoder_ch: int, skip_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(decoder_ch + skip_ch, decoder_ch, 1)
        self._identity_init(decoder_ch, skip_ch)

    def _identity_init(self, decoder_ch: int, skip_ch: int) -> None:
        # weight shape: [decoder_ch, decoder_ch + skip_ch, 1, 1]
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        # Set identity: output[i] = decoder_input[i], skip ignored
        with torch.no_grad():
            self.conv.weight[:decoder_ch, :decoder_ch, 0, 0] = torch.eye(decoder_ch)

    def forward(self, decoder_feat: torch.Tensor, skip_feat: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([decoder_feat, skip_feat], dim=1))


class CrossPatchMixer(nn.Module):
    """
    Cross-patch convolution at intermediate resolution (e.g., 8×8 per patch).

    Applied after assembling patches into global spatial grid.
    Zero-init last layer → identity at initialization (residual connection).

    Args:
        channels: Feature channels
        kernel_size: Conv kernel size (default 3)
        num_layers: Number of conv layers (default 2)
    """

    def __init__(self, channels: int, kernel_size: int = 3, num_layers: int = 2):
        super().__init__()
        pad = kernel_size // 2
        layers: list[nn.Module] = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(channels, channels, kernel_size,
                                    padding=pad, padding_mode='replicate'))
            if i < num_layers - 1:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)
        self._zero_init_last()

    def _zero_init_last(self) -> None:
        """Zero-init last conv → identity at start."""
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B*T, C, H_global, W_global] at sub-patch resolution."""
        return x + self.net(x)


class CNNDecoderV2(nn.Module):
    """
    Improved CNN Decoder: 1×1 → 16×16 with:
        B: Upsample + Conv (no ConvTranspose2d)
        C: Skip connections at 4×4 and 8×8
        D: Extra ResBlocks, configurable hidden channels

    Architecture:
        Stage 1: 1×1 → 2×2 (UpsampleConv + ResBlock × 2)
        Stage 2: 2×2 → 4×4 (UpsampleConv + ResBlock + SkipMerge + ResBlock)
        Stage 3: 4×4 → 8×8 (UpsampleConv + ResBlock + SkipMerge + ResBlock)
        Stage 4: 8×8 → 16×16 (UpsampleConv, output channels)

    Stages 1-3 return 8×8 features for cross-patch mixing.
    Stage 4 is called separately via final_upsample().
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 256,
        out_channels: int = 18,
        enc_8x8_channels: int = 128,
        enc_4x4_channels: int = 256,
        use_skip: bool = True,
    ):
        super().__init__()
        self.use_skip = use_skip
        hid = hidden_channels

        # Stage 1: 1×1 → 2×2
        self.stage1 = nn.Sequential(
            UpsampleConv(in_channels, hid),
            nn.GELU(),
            ResBlock2D(hid),
            ResBlock2D(hid),
        )

        # Stage 2: 2×2 → 4×4
        self.stage2_up = nn.Sequential(
            UpsampleConv(hid, hid),
            nn.GELU(),
            ResBlock2D(hid),
        )
        if use_skip:
            self.skip_merge_4x4 = SkipMerge(hid, enc_4x4_channels)
        self.stage2_post = ResBlock2D(hid)

        # Stage 3: 4×4 → 8×8
        self.stage3_up = nn.Sequential(
            UpsampleConv(hid, hid),
            nn.GELU(),
            ResBlock2D(hid),
        )
        if use_skip:
            self.skip_merge_8x8 = SkipMerge(hid, enc_8x8_channels)
        self.stage3_post = ResBlock2D(hid)

        # Stage 4: 8×8 → 16×16 (called separately)
        self.stage4 = UpsampleConv(hid, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        skip_4x4: Optional[torch.Tensor] = None,
        skip_8x8: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run stages 1-3, return 8×8 features for cross-patch mixing.

        Args:
            x: [N, in_channels, 1, 1]
            skip_4x4: [N, enc_4x4_ch, 4, 4] encoder intermediate (optional)
            skip_8x8: [N, enc_8x8_ch, 8, 8] encoder intermediate (optional)

        Returns:
            feat_8x8: [N, hidden_channels, 8, 8]
        """
        # Stage 1: 1×1 → 2×2
        x = self.stage1(x)

        # Stage 2: 2×2 → 4×4
        x = self.stage2_up(x)
        if self.use_skip and skip_4x4 is not None:
            x = self.skip_merge_4x4(x, skip_4x4)
        x = self.stage2_post(x)

        # Stage 3: 4×4 → 8×8
        x = self.stage3_up(x)
        if self.use_skip and skip_8x8 is not None:
            x = self.skip_merge_8x8(x, skip_8x8)
        x = self.stage3_post(x)

        return x

    def final_upsample(self, x: torch.Tensor) -> torch.Tensor:
        """
        Stage 4: 8×8 → 16×16.

        Args:
            x: [N, hidden_channels, 8, 8]
        Returns:
            out: [N, out_channels, 16, 16]
        """
        return self.stage4(x)


class PatchifyDecoderV2(nn.Module):
    """
    Improved 2D Patchify Decoder with A+B+C+D.

    Converts token sequence back to spatial output with:
        A: Cross-patch mixing at 8×8 sub-patch resolution
        B: Upsample+Conv (no checkerboard)
        C: Encoder skip connections at 4×4 and 8×8
        D: Increased capacity
        E: Post-assembly 5×5 Conv smoothing at full resolution (cross-patch boundary)

    Args:
        out_channels: Output channels (e.g., 18)
        hidden_dim: Transformer hidden dimension (768)
        patch_size: Patch size (16)
        stem_channels: Channels after projection (256)
        decoder_hidden: Hidden channels in CNN decoder (256)
        enc_8x8_channels: Encoder 8×8 intermediate channels (128)
        enc_4x4_channels: Encoder 4×4 intermediate channels (256)
        use_skip: Enable skip connections (C)
        cross_patch_cfg: Cross-patch mixer config dict
        gradient_checkpointing: Checkpoint decoder activations to save memory
    """

    def __init__(
        self,
        out_channels: int = 18,
        hidden_dim: int = 768,
        patch_size: int = 16,
        stem_channels: int = 256,
        decoder_hidden: int = 256,
        enc_8x8_channels: int = 128,
        enc_4x4_channels: int = 256,
        use_skip: bool = True,
        cross_patch_cfg: Optional[Dict] = None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.stem_channels = stem_channels
        self.gradient_checkpointing = gradient_checkpointing

        # Project from transformer hidden dim (same as v2, weights loadable)
        self.norm = RMSNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, stem_channels)

        # CNN Decoder V2 with skip connections
        self.cnn_decoder = CNNDecoderV2(
            in_channels=stem_channels,
            hidden_channels=decoder_hidden,
            out_channels=out_channels,
            enc_8x8_channels=enc_8x8_channels,
            enc_4x4_channels=enc_4x4_channels,
            use_skip=use_skip,
        )

        # Cross-patch mixer (A)
        cp_cfg = cross_patch_cfg or {}
        if cp_cfg.get('enabled', True):
            self.cross_patch_mixer = CrossPatchMixer(
                channels=decoder_hidden,
                kernel_size=cp_cfg.get('kernel_size', 3),
                num_layers=cp_cfg.get('num_layers', 2),
            )

        # Post-assembly smoothing (E): 5×5 Conv at full resolution
        # Identity-init: weight = delta function at center → pass-through at init
        self.post_smooth = nn.Conv2d(
            out_channels, out_channels, kernel_size=5,
            padding=2, padding_mode='replicate',
        )
        self._identity_init_post_smooth()

    def _identity_init_post_smooth(self) -> None:
        """Identity-init post_smooth: delta kernel at center → output = input."""
        nn.init.zeros_(self.post_smooth.weight)
        nn.init.zeros_(self.post_smooth.bias)
        center = self.post_smooth.kernel_size[0] // 2
        with torch.no_grad():
            for c in range(self.out_channels):
                self.post_smooth.weight[c, c, center, center] = 1.0

    def reinit_identity(self) -> None:
        """
        Re-apply identity initialization for skip merges, cross-patch mixer,
        and post-assembly smoothing.

        Must be called after model._init_weights() which overwrites Conv2d inits.
        """
        cnn = self.cnn_decoder
        if hasattr(cnn, 'skip_merge_4x4'):
            cnn.skip_merge_4x4._identity_init(
                cnn.skip_merge_4x4.conv.out_channels,
                cnn.skip_merge_4x4.conv.in_channels - cnn.skip_merge_4x4.conv.out_channels,
            )
        if hasattr(cnn, 'skip_merge_8x8'):
            cnn.skip_merge_8x8._identity_init(
                cnn.skip_merge_8x8.conv.out_channels,
                cnn.skip_merge_8x8.conv.in_channels - cnn.skip_merge_8x8.conv.out_channels,
            )
        if hasattr(self, 'cross_patch_mixer'):
            self.cross_patch_mixer._zero_init_last()
        if hasattr(self, 'post_smooth'):
            self._identity_init_post_smooth()

    def forward(self, x: torch.Tensor, shape_info: Dict) -> torch.Tensor:
        """
        Args:
            x: [B, T*n_h*n_w, D] token sequence
            shape_info: dict with T, n_h, n_w, H, W, encoder_intermediates (optional)

        Returns:
            output: [B, T, H, W, C]
        """
        B = x.shape[0]
        T = shape_info['T']
        n_h = shape_info['n_h']
        n_w = shape_info['n_w']
        H = shape_info.get('H', n_h * self.patch_size)
        W = shape_info.get('W', n_w * self.patch_size)
        P = self.patch_size

        # Step 1: Norm and project
        x = self.norm(x)
        x = self.proj(x)  # [B, T*n_h*n_w, stem_channels]

        # Step 2: Reshape to patches
        x = x.reshape(B * T * n_h * n_w, self.stem_channels, 1, 1)

        # Get encoder skip features
        intermediates = shape_info.get('encoder_intermediates', {})
        skip_4x4 = intermediates.get('feat_4x4')  # [B*T*n_h*n_w, 256, 4, 4]
        skip_8x8 = intermediates.get('feat_8x8')  # [B*T*n_h*n_w, 128, 8, 8]

        use_ckpt = self.gradient_checkpointing

        # Step 3: CNN decoder stages 1-3 (with skip connections)
        if use_ckpt:
            feat_8x8 = grad_checkpoint(
                self.cnn_decoder, x, skip_4x4, skip_8x8, use_reentrant=False,
            )
        else:
            feat_8x8 = self.cnn_decoder(x, skip_4x4=skip_4x4, skip_8x8=skip_8x8)
        # feat_8x8: [B*T*n_h*n_w, hid_ch, 8, 8]

        # Step 4: Cross-patch mixing at 8×8 (A)
        if hasattr(self, 'cross_patch_mixer'):
            hid_ch = feat_8x8.shape[1]
            # Reassemble patches at 8×8 resolution
            feat = feat_8x8.reshape(B * T, n_h, n_w, hid_ch, 8, 8)
            # [B*T, n_h, n_w, C, 8, 8] → [B*T, C, n_h, 8, n_w, 8]
            feat = feat.permute(0, 3, 1, 4, 2, 5)
            # → [B*T, C, n_h*8, n_w*8]
            feat = feat.reshape(B * T, hid_ch, n_h * 8, n_w * 8)

            # Apply cross-patch mixing (checkpointed — this is the memory bottleneck)
            if use_ckpt:
                feat = grad_checkpoint(
                    self.cross_patch_mixer, feat, use_reentrant=False,
                )
            else:
                feat = self.cross_patch_mixer(feat)

            # Split back to per-patch
            # [B*T, C, n_h*8, n_w*8] → [B*T, C, n_h, 8, n_w, 8]
            feat = feat.reshape(B * T, hid_ch, n_h, 8, n_w, 8)
            # → [B*T, n_h, n_w, C, 8, 8] → [B*T*n_h*n_w, C, 8, 8]
            feat = feat.permute(0, 2, 4, 1, 3, 5)
            feat_8x8 = feat.reshape(B * T * n_h * n_w, hid_ch, 8, 8)

        # Step 5: Final upsample 8×8 → 16×16
        if use_ckpt:
            x = grad_checkpoint(
                self.cnn_decoder.final_upsample, feat_8x8, use_reentrant=False,
            )
        else:
            x = self.cnn_decoder.final_upsample(feat_8x8)
        # x: [B*T*n_h*n_w, out_channels, 16, 16]

        # Step 6: Reassemble patches
        x = x.reshape(B, T, n_h, n_w, self.out_channels, P, P)
        x = x.permute(0, 1, 2, 5, 3, 6, 4)  # [B, T, n_h, P, n_w, P, C]
        x = x.reshape(B, T, H, W, self.out_channels)

        # Step 7: Post-assembly smoothing at full resolution (E)
        # 5×5 Conv on global grid to smooth patch boundary discontinuities
        x = x.reshape(B * T, H, W, self.out_channels).permute(0, 3, 1, 2)
        x = self.post_smooth(x)  # [B*T, C, H, W]
        x = x.permute(0, 2, 3, 1).reshape(B, T, H, W, self.out_channels)

        return x


if __name__ == "__main__":
    """Test decoder_v3."""
    print("=" * 60)
    print("Testing PatchifyDecoderV2 (A+B+C+D)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config
    out_ch = 18
    hidden_dim = 768
    patch_size = 16
    stem_ch = 256
    hid_ch = 256
    enc_8x8_ch = 128
    enc_4x4_ch = 256

    decoder = PatchifyDecoderV2(
        out_channels=out_ch,
        hidden_dim=hidden_dim,
        patch_size=patch_size,
        stem_channels=stem_ch,
        decoder_hidden=hid_ch,
        enc_8x8_channels=enc_8x8_ch,
        enc_4x4_channels=enc_4x4_ch,
        use_skip=True,
        cross_patch_cfg={'enabled': True, 'kernel_size': 3, 'num_layers': 2},
    ).to(device)

    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"\nDecoder V2 parameters: {n_params:,}")

    # Test cases: (B, T, n_h, n_w, H, W)
    test_cases = [
        (2, 8, 8, 8, 128, 128),
        (2, 8, 16, 16, 256, 256),
        (1, 4, 16, 32, 256, 512),
    ]

    for B, T, n_h, n_w, H, W in test_cases:
        N = B * T * n_h * n_w
        seq_len = T * n_h * n_w

        tokens = torch.randn(B, seq_len, hidden_dim, device=device)
        skip_8x8 = torch.randn(N, enc_8x8_ch, 8, 8, device=device)
        skip_4x4 = torch.randn(N, enc_4x4_ch, 4, 4, device=device)

        shape_info = {
            'T': T, 'n_h': n_h, 'n_w': n_w, 'H': H, 'W': W,
            'encoder_intermediates': {
                'feat_8x8': skip_8x8,
                'feat_4x4': skip_4x4,
            },
        }

        output = decoder(tokens, shape_info)
        assert output.shape == (B, T, H, W, out_ch), \
            f"Expected {(B, T, H, W, out_ch)}, got {output.shape}"
        print(f"  [{B}, {seq_len}, 768] → [{B}, {T}, {H}, {W}, {out_ch}] ✓")

    # Test without skip connections
    print("\nTesting without skip connections:")
    decoder_no_skip = PatchifyDecoderV2(
        out_channels=out_ch, hidden_dim=hidden_dim, patch_size=patch_size,
        stem_channels=stem_ch, decoder_hidden=hid_ch,
        use_skip=False,
        cross_patch_cfg={'enabled': True},
    ).to(device)

    B, T, n_h, n_w, H, W = 2, 8, 8, 8, 128, 128
    tokens = torch.randn(B, T * n_h * n_w, hidden_dim, device=device)
    shape_info = {'T': T, 'n_h': n_h, 'n_w': n_w, 'H': H, 'W': W}
    output = decoder_no_skip(tokens, shape_info)
    assert output.shape == (B, T, H, W, out_ch)
    print(f"  No-skip: [{B}, {T * n_h * n_w}, 768] → {output.shape} ✓")

    # Test without cross-patch mixing
    print("\nTesting without cross-patch mixing:")
    decoder_no_cp = PatchifyDecoderV2(
        out_channels=out_ch, hidden_dim=hidden_dim, patch_size=patch_size,
        stem_channels=stem_ch, decoder_hidden=hid_ch,
        use_skip=False,
        cross_patch_cfg={'enabled': False},
    ).to(device)
    output = decoder_no_cp(tokens, shape_info)
    assert output.shape == (B, T, H, W, out_ch)
    print(f"  No-CP: [{B}, {T * n_h * n_w}, 768] → {output.shape} ✓")

    # Test identity init of SkipMerge (on CPU for exact check)
    print("\nTesting SkipMerge identity initialization (CPU):")
    merge_cpu = SkipMerge(decoder_ch=256, skip_ch=128)
    dec_feat = torch.randn(4, 256, 8, 8)
    skip_feat = torch.randn(4, 128, 8, 8)
    out = merge_cpu(dec_feat, skip_feat)
    diff = (out - dec_feat).abs().max().item()
    print(f"  SkipMerge(dec, skip) - dec = {diff:.2e} (should be 0)")
    assert diff < 1e-6, f"SkipMerge not identity at init! diff={diff}"

    # Test CrossPatchMixer identity init (on CPU for exact check)
    print("\nTesting CrossPatchMixer identity initialization (CPU):")
    mixer_cpu = CrossPatchMixer(channels=256)
    inp = torch.randn(4, 256, 64, 64)
    out = mixer_cpu(inp)
    diff = (out - inp).abs().max().item()
    print(f"  CrossPatchMixer(x) - x = {diff:.2e} (should be 0)")
    assert diff < 1e-6, f"CrossPatchMixer not identity at init! diff={diff}"

    # Test post_smooth identity init (on CPU for exact check)
    print("\nTesting post_smooth identity initialization (CPU):")
    decoder_cpu = PatchifyDecoderV2(
        out_channels=out_ch, hidden_dim=hidden_dim, patch_size=patch_size,
        stem_channels=stem_ch, decoder_hidden=hid_ch,
        use_skip=False, cross_patch_cfg={'enabled': False},
    )
    inp = torch.randn(4, out_ch, 128, 128)
    out = decoder_cpu.post_smooth(inp)
    diff = (out - inp).abs().max().item()
    print(f"  post_smooth(x) - x = {diff:.2e} (should be 0)")
    assert diff < 1e-6, f"post_smooth not identity at init! diff={diff}"

    # Test full decoder identity: reinit_identity should make new components transparent
    print("\nTesting reinit_identity preserves output:")
    decoder_cpu.reinit_identity()
    inp2 = torch.randn(4, out_ch, 64, 64)
    out2 = decoder_cpu.post_smooth(inp2)
    diff2 = (out2 - inp2).abs().max().item()
    print(f"  After reinit_identity: post_smooth diff = {diff2:.2e} (should be 0)")
    assert diff2 < 1e-6, f"reinit_identity broke post_smooth! diff={diff2}"

    print("\n" + "=" * 60)
    print("All decoder_v3 tests passed!")
    print("=" * 60)
