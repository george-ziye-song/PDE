"""
Test: compare v2 and v3 decoder output after loading the same checkpoint.
Verify whether identity init works and diagnose weight loading gaps.
"""

import torch
import yaml
import sys
sys.path.insert(0, '.')

from pretrain.model_v3 import PDEModelV3


def load_checkpoint(model: PDEModelV3, ckpt_path: str) -> tuple[list, list]:
    """Load checkpoint and return (missing, unexpected) keys."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))

    cleaned = {}
    for k, v in state_dict.items():
        k = k.removeprefix('module.').removeprefix('_orig_mod.')
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    return missing, unexpected


def test_model(config: dict, label: str, ckpt_path: str, device: torch.device):
    """Create model, load weights, run one forward pass."""
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"{'='*60}")

    model = PDEModelV3(config)
    missing, unexpected = load_checkpoint(model, ckpt_path)

    # Show decoder-related missing/unexpected keys
    dec_missing = [k for k in missing if k.startswith('decoder')]
    dec_unexpected = [k for k in unexpected if k.startswith('decoder')]

    print(f"\nTotal missing: {len(missing)}, unexpected: {len(unexpected)}")
    print(f"Decoder missing: {len(dec_missing)}")
    for k in dec_missing:
        print(f"  MISSING: {k}")
    print(f"Decoder unexpected: {len(dec_unexpected)}")
    for k in dec_unexpected:
        print(f"  UNEXPECTED: {k}")

    model = model.to(device).eval()

    # Synthetic input: [B=1, T=8, H=128, W=128, C=18]
    torch.manual_seed(42)
    x = torch.randn(1, 8, 128, 128, 18, device=device)

    with torch.no_grad():
        out = model(x)

    print(f"\nInput:  shape={x.shape}, mean={x.mean():.4f}, std={x.std():.4f}")
    print(f"Output: shape={out.shape}, mean={out.mean():.4f}, std={out.std():.4f}")
    print(f"Output abs range: min={out.abs().min():.4f}, max={out.abs().max():.4f}")

    # Check if output is close to input (autoregressive: output should predict next step)
    # For a well-initialized model, output should be in a reasonable range
    rmse = ((out - x).pow(2).mean()).sqrt()
    print(f"RMSE(output - input): {rmse:.4f}")

    return out


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt_path = '/home/msai/song0304/code/PDE/checkpoints_v4_2donly/best_tf.pt'

    # --- V2 config (original decoder) ---
    with open('configs/pretrain_v4_2donly.yaml', 'r') as f:
        config_v2 = yaml.safe_load(f)
    # Make sure it uses v2 decoder (no version key or version != v3)
    if 'decoder' in config_v2.get('model', {}):
        config_v2['model']['decoder'].pop('version', None)

    out_v2 = test_model(config_v2, "V2 Decoder", ckpt_path, device)

    # --- V3 config ---
    with open('configs/pretrain_v4_2donly_dec_v3.yaml', 'r') as f:
        config_v3 = yaml.safe_load(f)

    out_v3 = test_model(config_v3, "V3 Decoder", ckpt_path, device)

    # --- Compare ---
    print(f"\n{'='*60}")
    print("Comparison")
    print(f"{'='*60}")
    diff = (out_v2 - out_v3).abs()
    print(f"V2 output: mean={out_v2.mean():.4f}, std={out_v2.std():.4f}")
    print(f"V3 output: mean={out_v3.mean():.4f}, std={out_v3.std():.4f}")
    print(f"|V2 - V3|: mean={diff.mean():.4f}, max={diff.max():.4f}")
    print(f"Are they close (atol=0.01)? {torch.allclose(out_v2, out_v3, atol=0.01)}")


if __name__ == '__main__':
    main()
