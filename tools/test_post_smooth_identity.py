"""
Test: verify post_smooth identity init on V2 decoder.

1. V2 (no post_smooth): load checkpoint → output_v2
2. V2 + post_smooth_kernel=5: load checkpoint → only post_smooth keys missing → output_ps
3. |output_v2 - output_ps| should be exactly 0 (identity init)
"""

import torch
import yaml
import sys
sys.path.insert(0, '.')

from pretrain.model_v3 import PDEModelV3


def load_checkpoint(model: PDEModelV3, ckpt_path: str) -> tuple[list, list]:
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))

    cleaned = {}
    for k, v in state_dict.items():
        k = k.removeprefix('module.').removeprefix('_orig_mod.')
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    return missing, unexpected


def test_model(config: dict, label: str, ckpt_path: str, device: torch.device):
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"{'='*60}")

    model = PDEModelV3(config)
    missing, unexpected = load_checkpoint(model, ckpt_path)

    print(f"Total missing: {len(missing)}, unexpected: {len(unexpected)}")
    for k in missing:
        print(f"  MISSING: {k}")
    for k in unexpected:
        print(f"  UNEXPECTED: {k}")

    model = model.to(device).eval()

    # Synthetic input: [B=1, T=8, H=128, W=128, C=18]
    torch.manual_seed(42)
    x = torch.randn(1, 8, 128, 128, 18, device=device)

    with torch.no_grad():
        out = model(x)

    print(f"\nInput:  shape={x.shape}, mean={x.mean():.6f}, std={x.std():.6f}")
    print(f"Output: shape={out.shape}, mean={out.mean():.6f}, std={out.std():.6f}")

    # Check post_smooth identity init
    if hasattr(model.decoder, 'post_smooth'):
        w = model.decoder.post_smooth.weight
        b = model.decoder.post_smooth.bias
        print(f"\npost_smooth weight: shape={w.shape}")
        print(f"  weight sum={w.sum():.6f} (expect {w.shape[0]:.0f})")
        print(f"  bias sum={b.sum():.6f} (expect 0)")
        # Check delta kernel
        center = w.shape[2] // 2
        diag_sum = sum(w[c, c, center, center].item() for c in range(w.shape[0]))
        print(f"  diagonal center sum={diag_sum:.6f} (expect {w.shape[0]:.0f})")
        off_diag = w.sum().item() - diag_sum
        print(f"  off-diagonal sum={off_diag:.6f} (expect 0)")

    return out


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt_path = '/home/msai/song0304/code/PDE/checkpoints_v4_2donly/best_tf.pt'

    # --- V2 baseline (no post_smooth) ---
    with open('configs/pretrain_v4_2donly.yaml', 'r') as f:
        config_v2 = yaml.safe_load(f)

    out_v2 = test_model(config_v2, "V2 Decoder (baseline)", ckpt_path, device)

    # --- V2 + post_smooth ---
    with open('configs/pretrain_v4_2donly_postsmooth.yaml', 'r') as f:
        config_ps = yaml.safe_load(f)

    out_ps = test_model(config_ps, "V2 + post_smooth 5×5", ckpt_path, device)

    # --- Compare ---
    print(f"\n{'='*60}")
    print("Comparison: V2 vs V2+post_smooth")
    print(f"{'='*60}")
    diff = (out_v2 - out_ps).abs()
    print(f"|V2 - V2+PS|: mean={diff.mean():.8f}, max={diff.max():.8f}")
    print(f"Exact match (atol=1e-6)? {torch.allclose(out_v2, out_ps, atol=1e-6)}")
    print(f"Close match (atol=1e-4)? {torch.allclose(out_v2, out_ps, atol=1e-4)}")

    if diff.max() < 1e-5:
        print("\n✓ PASS: Identity init verified — post_smooth is transparent")
    else:
        print("\n✗ FAIL: post_smooth is NOT identity — check init!")


if __name__ == '__main__':
    main()
