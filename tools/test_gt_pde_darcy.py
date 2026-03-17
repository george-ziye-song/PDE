"""
Test DarcyFlowPDELoss on GT data from PDEBench.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/test_gt_pde_darcy.py \
        --data data/finetune/2D_DarcyFlow_beta1.0_Train.hdf5 \
        --device cuda --n_samples 50
"""

import argparse
import h5py
import torch
import numpy as np

from finetune.pde_loss_verified import DarcyFlowPDELoss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--n_samples', type=int, default=50)
    p.add_argument('--dtype', type=str, default='float64',
                   choices=['float32', 'float64'])
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == 'float64' else torch.float32

    # Create PDE loss (interior only, no ghost cells)
    pde_fn = DarcyFlowPDELoss(
        nx=128, ny=128, Lx=1.0, Ly=1.0, forcing=1.0,
    ).to(device=device, dtype=dtype)

    with h5py.File(args.data, 'r') as fh:
        n_total = fh['nu'].shape[0]
        beta = fh.attrs.get('beta', '?')

    n_test = min(args.n_samples, n_total)

    print(f"{'=' * 90}")
    print(f"  DarcyFlowPDELoss GT Test: -div(a*grad(u)) = 1")
    print(f"  Data: {args.data} (beta={beta}, {n_total} samples)")
    print(f"  dtype={args.dtype}, device={args.device}")
    print(f"{'=' * 90}")

    header = f"{'Samp':>4} | {'Interior MSE':>12} | {'a range':>18} | {'u range':>18}"
    print(header)
    print("-" * len(header))

    all_mse = []

    with h5py.File(args.data, 'r') as fh:
        for sid in range(n_test):
            np_dtype = np.float64 if dtype == torch.float64 else np.float32
            a_np = np.array(fh['nu'][sid], dtype=np_dtype)
            u_np = np.array(fh['tensor'][sid, 0], dtype=np_dtype)

            a_t = torch.from_numpy(a_np).to(device=device, dtype=dtype)
            u_t = torch.from_numpy(u_np).to(device=device, dtype=dtype)

            with torch.no_grad():
                loss_val, _ = pde_fn(a_t, u_t, reduction='mean')

            mse = float(loss_val)
            all_mse.append(mse)

            print(f"{sid:4d} | {mse:12.4e} | "
                  f"[{a_np.min():.3f}, {a_np.max():.3f}] | "
                  f"[{u_np.min():.4f}, {u_np.max():.4f}]")

    arr = np.array(all_mse)

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY ({n_test} samples, dtype={args.dtype})")
    print(f"{'=' * 70}")
    print(f"  Interior [1:-1,1:-1]:  mean={np.mean(arr):.4e}  median={np.median(arr):.4e}  "
          f"min={np.min(arr):.4e}  max={np.max(arr):.4e}")

    # Quality tiers
    thresh_good = 1e-5
    thresh_ok = 1e-3
    n_good = np.sum(arr < thresh_good)
    n_ok = np.sum((arr >= thresh_good) & (arr < thresh_ok))
    n_bad = np.sum(arr >= thresh_ok)
    print(f"\n  Quality tiers:")
    print(f"    GOOD (< 1e-5):  {n_good}/{n_test} ({100*n_good/n_test:.0f}%)")
    print(f"    OK (1e-5~1e-3): {n_ok}/{n_test} ({100*n_ok/n_test:.0f}%)")
    print(f"    BAD (> 1e-3):   {n_bad}/{n_test} ({100*n_bad/n_test:.0f}%)")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
