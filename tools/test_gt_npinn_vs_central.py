"""
Compare GT PDE residual: Central Diff (original) vs n-PINN Conservative Upwind

Tests both ShearFlowPDELoss (central) and ShearFlowPDELossNPINN (upwind) on
ground truth data to measure the numerical floor of each scheme.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/test_gt_npinn_vs_central.py \
        --data ./data/finetune/shear_flow_clean.h5 --device cuda
"""

import argparse
import h5py
import torch
import numpy as np
from typing import Dict

from finetune.pde_loss_verified import ShearFlowPDELoss, ShearFlowPDELossNPINN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--Lx', type=float, default=1.0)
    p.add_argument('--Ly', type=float, default=2.0)
    p.add_argument('--dt', type=float, default=0.1)
    p.add_argument('--nu', type=float, default=1e-4)
    p.add_argument('--D', type=float, default=1e-3)
    p.add_argument('--dtype', type=str, default='float64',
                   choices=['float32', 'float64'])
    return p.parse_args()


def eval_pde_chunked(
    pde_fn, u: torch.Tensor, v: torch.Tensor,
    p: torch.Tensor, s: torch.Tensor,
    chunk: int = 30,
) -> Dict[str, float]:
    """Evaluate PDE loss in sliding chunks to avoid OOM."""
    T = u.shape[0]
    stride = chunk - 2
    totals = []
    components: Dict[str, list] = {
        'continuity': [], 'u_momentum': [], 'v_momentum': [], 'tracer': []
    }

    start = 0
    with torch.no_grad():
        while start < T - 2:
            end = min(start + chunk, T)
            if end - start < 3:
                break
            u_c = u[start:end].unsqueeze(0)
            v_c = v[start:end].unsqueeze(0)
            p_c = p[start:end].unsqueeze(0)
            s_c = s[start:end].unsqueeze(0)

            total, losses = pde_fn(u_c, v_c, p_c, s_c, reduction='mean')
            totals.append(float(total))
            for k in components:
                components[k].append(float(losses[k]))

            start += stride

    return {
        'total': np.mean(totals),
        **{k: np.mean(v) for k, v in components.items()},
    }


def main():
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == 'float64' else torch.float32

    nx, ny = 256, 512

    # Create both PDE loss functions
    pde_central = ShearFlowPDELoss(
        nx=nx, ny=ny, Lx=args.Lx, Ly=args.Ly,
        dt=args.dt, nu=args.nu, D=args.D,
    ).to(device=device, dtype=dtype)

    pde_npinn = ShearFlowPDELossNPINN(
        nx=nx, ny=ny, Lx=args.Lx, Ly=args.Ly,
        dt=args.dt, nu=args.nu, D=args.D,
        use_div_correction=True,
    ).to(device=device, dtype=dtype)

    # Also test without div correction
    pde_npinn_nodiv = ShearFlowPDELossNPINN(
        nx=nx, ny=ny, Lx=args.Lx, Ly=args.Ly,
        dt=args.dt, nu=args.nu, D=args.D,
        use_div_correction=False,
    ).to(device=device, dtype=dtype)

    with h5py.File(args.data, 'r') as f:
        n_samples = f['vector'].shape[0]
        scalar_indices = f['scalar_indices'][:]

    tracer_ci = int(np.where(scalar_indices == 11)[0][0])
    press_ci = int(np.where(scalar_indices == 12)[0][0])

    print(f"{'=' * 120}")
    print(f"  GT PDE Residual Comparison: Central Diff vs n-PINN Conservative Upwind")
    print(f"{'=' * 120}")
    print(f"  Data: {args.data}  ({n_samples} samples)")
    print(f"  dtype={args.dtype}, div_correction=True/False")
    print(f"{'=' * 120}")

    header = (
        f"{'Samp':>4} | {'Central':>12} | {'nPINN+div':>12} | {'nPINN-div':>12} | "
        f"{'C/nP+d ratio':>12} | {'Central cont':>12} | {'nPINN cont':>12} | "
        f"{'Central umom':>12} | {'nPINN umom':>12}"
    )
    print(header)
    print("-" * len(header))

    all_central = []
    all_npinn = []
    all_npinn_nodiv = []

    for sid in range(n_samples):
        with h5py.File(args.data, 'r') as f:
            vec = np.array(f['vector'][sid], dtype=np.float64 if dtype == torch.float64 else np.float32)
            scl = np.array(f['scalar'][sid], dtype=np.float64 if dtype == torch.float64 else np.float32)

        u = torch.from_numpy(vec[..., 0]).to(device=device, dtype=dtype)
        v = torch.from_numpy(vec[..., 1]).to(device=device, dtype=dtype)
        p = torch.from_numpy(scl[..., press_ci]).to(device=device, dtype=dtype)
        s = torch.from_numpy(scl[..., tracer_ci]).to(device=device, dtype=dtype)

        res_central = eval_pde_chunked(pde_central, u, v, p, s)
        res_npinn = eval_pde_chunked(pde_npinn, u, v, p, s)
        res_npinn_nodiv = eval_pde_chunked(pde_npinn_nodiv, u, v, p, s)

        ratio = res_central['total'] / max(res_npinn['total'], 1e-30)

        all_central.append(res_central['total'])
        all_npinn.append(res_npinn['total'])
        all_npinn_nodiv.append(res_npinn_nodiv['total'])

        print(
            f"{sid:4d} | {res_central['total']:12.4e} | {res_npinn['total']:12.4e} | "
            f"{res_npinn_nodiv['total']:12.4e} | {ratio:12.2f}x | "
            f"{res_central['continuity']:12.4e} | {res_npinn['continuity']:12.4e} | "
            f"{res_central['u_momentum']:12.4e} | {res_npinn['u_momentum']:12.4e}"
        )

        del u, v, p, s
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Summary
    arr_c = np.array(all_central)
    arr_n = np.array(all_npinn)
    arr_nd = np.array(all_npinn_nodiv)

    print(f"\n{'=' * 80}")
    print(f"  SUMMARY ({n_samples} samples, dtype={args.dtype})")
    print(f"{'=' * 80}")
    print(f"  Central diff:     mean={np.mean(arr_c):.4e}  median={np.median(arr_c):.4e}  "
          f"min={np.min(arr_c):.4e}  max={np.max(arr_c):.4e}")
    print(f"  nPINN +div_corr:  mean={np.mean(arr_n):.4e}  median={np.median(arr_n):.4e}  "
          f"min={np.min(arr_n):.4e}  max={np.max(arr_n):.4e}")
    print(f"  nPINN -div_corr:  mean={np.mean(arr_nd):.4e}  median={np.median(arr_nd):.4e}  "
          f"min={np.min(arr_nd):.4e}  max={np.max(arr_nd):.4e}")
    print(f"\n  Central / nPINN+div ratio: mean={np.mean(arr_c/arr_n):.2f}x")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
