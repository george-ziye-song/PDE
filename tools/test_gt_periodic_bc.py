"""
Test periodic boundary condition loss on GT data.

For a doubly-periodic domain [0,Lx) × [0,Ly) with N×M grid points,
the periodic BC loss measures smoothness at the wrap-around boundary
via linear extrapolation (value continuity only):

  |2*f[-1] - f[-2] - f[0]|²

Truncation: |f''(bnd)|² * dx⁴ → O(dx⁴) on smooth GT data.
Signal:     ≈ Δ² for boundary seam of magnitude Δ.

GT baselines:
  Shear Flow:  val_x ~ 1e-8,  val_y ~ 1e-10
  Gray-Scott:  val_x ~ 1.3e-3, val_y ~ 1.2e-3

Note: derivative matching (one-sided stencil comparison) was tested but
rejected — GT baseline too large due to |f'''|² truncation in sharp fields.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/test_gt_periodic_bc.py \
        --dataset shear_flow --device cuda

    CUDA_VISIBLE_DEVICES=0 python tools/test_gt_periodic_bc.py \
        --dataset gray_scott --device cuda
"""

import argparse
import h5py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple


# ── Periodic BC Loss ─────────────────────────────────────────────────

class PeriodicBCLoss(nn.Module):
    """
    Periodic boundary condition loss for doubly-periodic 2D domains.

    PINN-style enforcement adapted for discrete grids:
      Value continuity via linear extrapolation from interior:
        |2*f[-1] - f[-2] - f[0]|²
      Measures smoothness at the wrap-around boundary.

    Truncation: |f''(bnd)|² * dx⁴ → O(dx⁴) on smooth GT data.
    Signal:     ≈ Δ² for boundary seam of magnitude Δ.

    GT baselines:
      Shear Flow:  val_x ~ 1e-8,  val_y ~ 1e-10
      Gray-Scott:  val_x ~ 1.3e-3, val_y ~ 1.2e-3

    Note: derivative matching (one-sided stencil comparison) was tested but
    rejected — GT baseline too large due to |f'''|² truncation in sharp fields.
    """

    def __init__(self, dx: float, dy: float):
        super().__init__()
        self.dx = dx
        self.dy = dy

    def _value_loss_x(self, f: torch.Tensor) -> torch.Tensor:
        """Value continuity at x-boundary (linear extrapolation, dim=-2).

        Extrapolate: f̂[0] = 2*f[-1] - f[-2], should equal f[0].
        Equivalent to 2nd-order boundary smoothness: |f[0] - 2f[-1] + f[-2]|²
        """
        extrap = 2 * f[..., -1, :] - f[..., -2, :]
        return torch.mean((extrap - f[..., 0, :]) ** 2)

    def _value_loss_y(self, f: torch.Tensor) -> torch.Tensor:
        """Value continuity at y-boundary (linear extrapolation, dim=-1)."""
        extrap = 2 * f[..., :, -1] - f[..., :, -2]
        return torch.mean((extrap - f[..., :, 0]) ** 2)

    def forward(
        self, fields: Dict[str, torch.Tensor], reduction: str = 'mean',
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute periodic BC loss for multiple fields.

        Args:
            fields: {name: tensor[B, T, H, W]} or {name: tensor[T, H, W]}
            reduction: 'mean' (scalar), 'none' (per-field dict)

        Returns:
            total_loss, losses_dict
        """
        losses = {}
        device = next(iter(fields.values())).device
        total = torch.tensor(0.0, device=device)

        for name, f in fields.items():
            vx = self._value_loss_x(f)
            vy = self._value_loss_y(f)

            field_total = vx + vy
            losses[f'{name}_val_x'] = vx
            losses[f'{name}_val_y'] = vy
            losses[f'{name}_total'] = field_total
            total = total + field_total

        losses['total'] = total
        return total, losses


# ── Data loading ─────────────────────────────────────────────────────

def load_shear_flow_sample(filepath: str, sid: int) -> Dict[str, torch.Tensor]:
    """Load u, v, p, s from shear flow H5."""
    with h5py.File(filepath, 'r') as f:
        vec = np.array(f['vector'][sid], dtype=np.float32)
        scl = np.array(f['scalar'][sid], dtype=np.float32)
        si = f['scalar_indices'][:]

    u = torch.from_numpy(vec[..., 0])
    v = torch.from_numpy(vec[..., 1])

    # tracer → scalar_idx=11, pressure → scalar_idx=12
    s_data, p_data = None, None
    for i, idx in enumerate(si):
        if idx == 11:
            s_data = torch.from_numpy(scl[..., i])
        elif idx == 12:
            p_data = torch.from_numpy(scl[..., i])

    return {'u': u, 'v': v, 'p': p_data, 's': s_data}


def load_gray_scott_sample(filepath: str, sid: int) -> Dict[str, torch.Tensor]:
    """Load A, B from Gray-Scott H5."""
    with h5py.File(filepath, 'r') as f:
        scl = np.array(f['scalar'][sid], dtype=np.float32)
        si = f['scalar_indices'][:]

    A_data, B_data = None, None
    for i, idx in enumerate(si):
        if idx == 2:
            A_data = torch.from_numpy(scl[..., i])
        elif idx == 3:
            B_data = torch.from_numpy(scl[..., i])

    return {'A': A_data, 'B': B_data}


# ── Main ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Test periodic BC loss on GT data")
    p.add_argument('--dataset', required=True, choices=['shear_flow', 'gray_scott'])
    p.add_argument('--device', type=str, default='cuda')
    # Shear flow defaults
    p.add_argument('--sf_data', default='./data/finetune/shear_flow_clean.h5')
    p.add_argument('--sf_nx', type=int, default=256)
    p.add_argument('--sf_ny', type=int, default=512)
    p.add_argument('--sf_Lx', type=float, default=1.0)
    p.add_argument('--sf_Ly', type=float, default=2.0)
    # Gray-Scott defaults
    p.add_argument('--gs_data', default='/scratch-share/SONG0304/finetune/gray_scott_train.h5')
    p.add_argument('--gs_nx', type=int, default=128)
    p.add_argument('--gs_ny', type=int, default=128)
    p.add_argument('--gs_Lx', type=float, default=2.0157)
    p.add_argument('--gs_Ly', type=float, default=2.0157)
    # Common
    p.add_argument('--max_samples', type=int, default=10,
                   help='Max samples to test (for speed)')
    p.add_argument('--chunk_size', type=int, default=50)
    return p.parse_args()


def test_dataset(
    dataset_name: str,
    data_path: str,
    bc_loss_fn: PeriodicBCLoss,
    load_fn,
    device: torch.device,
    max_samples: int,
    chunk_size: int,
):
    """Run periodic BC loss test on all samples."""

    with h5py.File(data_path, 'r') as f:
        if 'vector' in f:
            n_samples = f['vector'].shape[0]
            n_timesteps = f['vector'].shape[1]
        else:
            n_samples = f['scalar'].shape[0]
            n_timesteps = f['scalar'].shape[1]

    n_test = min(n_samples, max_samples)

    print(f"\n{'=' * 75}")
    print(f"  Periodic BC Loss on GT — {dataset_name}")
    print(f"{'=' * 75}")
    print(f"  Data:      {data_path}")
    print(f"  Samples:   {n_test}/{n_samples}")
    print(f"  Timesteps: {n_timesteps}")
    print(f"  dx={bc_loss_fn.dx:.6e}, dy={bc_loss_fn.dy:.6e}")
    print(f"  Expected:  O(dx⁴) ≈ {bc_loss_fn.dx**4:.2e} (value continuity)")
    print(f"{'=' * 75}\n")

    all_losses = {}  # key → list of per-sample values

    for sid in range(n_test):
        print(f"  Sample {sid:3d} ...", end='', flush=True)

        fields_cpu = load_fn(data_path, sid)  # each [T, H, W]

        # Process in temporal chunks
        T = next(iter(fields_cpu.values())).shape[0]
        chunk_losses = {}

        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            fields_gpu = {
                k: v[start:end].unsqueeze(0).to(device)  # [1, chunk_T, H, W]
                for k, v in fields_cpu.items()
            }

            with torch.no_grad():
                _, losses = bc_loss_fn(fields_gpu)

            for k, v in losses.items():
                if k not in chunk_losses:
                    chunk_losses[k] = []
                chunk_losses[k].append(v.item())

            del fields_gpu
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        # Average across chunks
        sample_losses = {k: np.mean(v) for k, v in chunk_losses.items()}

        for k, v in sample_losses.items():
            if k not in all_losses:
                all_losses[k] = []
            all_losses[k].append(v)

        print(f"  total={sample_losses['total']:.4e}")

    # ── Summary ──
    print(f"\n{'=' * 75}")
    print(f"  SUMMARY (mean ± std across {n_test} samples)")
    print(f"{'=' * 75}")

    # Group by field
    field_names = set()
    for k in all_losses.keys():
        if '_' in k and k != 'total':
            parts = k.rsplit('_', 2)
            if len(parts) >= 3:
                field_names.add(k.rsplit('_', 2)[0])

    for fname in sorted(field_names):
        print(f"\n  Field: {fname}")
        for suffix in ['val_x', 'val_y', 'total']:
            key = f'{fname}_{suffix}'
            if key in all_losses:
                arr = np.array(all_losses[key])
                print(f"    {suffix:>8s}: mean={arr.mean():.4e}  std={arr.std():.4e}  "
                      f"min={arr.min():.4e}  max={arr.max():.4e}")

    total_arr = np.array(all_losses['total'])
    print(f"\n  TOTAL:    mean={total_arr.mean():.4e}  std={total_arr.std():.4e}  "
          f"min={total_arr.min():.4e}  max={total_arr.max():.4e}")

    # ── Comparison with expected order ──
    dx4 = bc_loss_fn.dx ** 4
    print(f"\n  dx⁴ = {dx4:.4e} (value truncation order)")
    print(f"  total / dx⁴ = {total_arr.mean() / dx4:.2f}")
    print(f"{'=' * 75}")


def main():
    args = parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    device = torch.device(args.device)

    if args.dataset == 'shear_flow':
        dx = args.sf_Lx / args.sf_nx
        dy = args.sf_Ly / args.sf_ny
        bc_loss_fn = PeriodicBCLoss(dx, dy).to(device)
        test_dataset(
            'Shear Flow', args.sf_data, bc_loss_fn,
            load_shear_flow_sample, device,
            args.max_samples, args.chunk_size,
        )
    elif args.dataset == 'gray_scott':
        dx = args.gs_Lx / args.gs_nx
        dy = args.gs_Ly / args.gs_ny
        bc_loss_fn = PeriodicBCLoss(dx, dy).to(device)
        test_dataset(
            'Gray-Scott', args.gs_data, bc_loss_fn,
            load_gray_scott_sample, device,
            args.max_samples, args.chunk_size,
        )


if __name__ == '__main__':
    main()
