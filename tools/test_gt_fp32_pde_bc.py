"""
Test GT PDE loss (fp32) and BC loss (old supervised vs new PeriodicBC) per sample.

This measures the floor that training will see, since model outputs are fp32.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/test_gt_fp32_pde_bc.py \
        --data ./data/finetune/shear_flow_clean.h5 --device cuda
"""

import argparse
import h5py
import torch
import numpy as np
from typing import Dict, Tuple

from finetune.pde_loss_verified import ShearFlowPDELoss, PeriodicBCLoss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--Lx', type=float, default=1.0)
    p.add_argument('--Ly', type=float, default=2.0)
    p.add_argument('--dt', type=float, default=0.1)
    p.add_argument('--nu', type=float, default=1e-4)
    p.add_argument('--D', type=float, default=1e-3)
    p.add_argument('--t_input', type=int, default=8)
    return p.parse_args()


def compute_old_bc_loss(output: torch.Tensor, target: torch.Tensor) -> float:
    """Old supervised boundary RMSE (mean over 4 edges)."""
    # output, target: [1, T, H, W, C]
    bc_top = torch.mean((output[:, :, 0, :, :] - target[:, :, 0, :, :]) ** 2)
    bc_bot = torch.mean((output[:, :, -1, :, :] - target[:, :, -1, :, :]) ** 2)
    bc_left = torch.mean((output[:, :, :, 0, :] - target[:, :, :, 0, :]) ** 2)
    bc_right = torch.mean((output[:, :, :, -1, :] - target[:, :, :, -1, :]) ** 2)
    return float((bc_top + bc_bot + bc_left + bc_right) / 4)


def main():
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    device = torch.device(args.device)

    nx, ny = 256, 512
    dx, dy = args.Lx / nx, args.Ly / ny

    pde_fn = ShearFlowPDELoss(
        nx=nx, ny=ny, Lx=args.Lx, Ly=args.Ly,
        dt=args.dt, nu=args.nu, D=args.D,
    ).float().to(device)

    bc_fn = PeriodicBCLoss(dx, dy).float().to(device)

    with h5py.File(args.data, 'r') as f:
        n_samples = f['vector'].shape[0]
        n_timesteps = f['vector'].shape[1]
        scalar_indices = f['scalar_indices'][:]

    tracer_ci = int(np.where(scalar_indices == 11)[0][0])
    press_ci = int(np.where(scalar_indices == 12)[0][0])

    t_input = args.t_input

    print(f"{'=' * 95}")
    print(f"  GT Floor Analysis (fp32) — PDE loss + BC loss")
    print(f"{'=' * 95}")
    print(f"  Data: {args.data}  ({n_samples} samples, {n_timesteps} timesteps)")
    print(f"  t_input={t_input}, dtype=float32")
    print(f"{'=' * 95}")
    print(f"{'Samp':>4} | {'PDE total':>12} | {'PDE cont':>10} | {'PDE u-mom':>10} | "
          f"{'PDE v-mom':>10} | {'PDE tracer':>10} | {'Old BC':>10} | {'New BC':>10} | {'New/Old':>8}")
    print(f"{'-' * 4}-+-{'-' * 12}-+-{'-' * 10}-+-{'-' * 10}-+-"
          f"{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 8}")

    all_pde = []
    all_old_bc = []
    all_new_bc = []

    for sid in range(n_samples):
        with h5py.File(args.data, 'r') as f:
            vec = np.array(f['vector'][sid], dtype=np.float32)  # [T, H, W, 3]
            scl = np.array(f['scalar'][sid], dtype=np.float32)  # [T, H, W, 2]

        u = torch.from_numpy(vec[..., 0]).to(device)   # [T, H, W]
        v = torch.from_numpy(vec[..., 1]).to(device)
        p = torch.from_numpy(scl[..., press_ci]).to(device)
        s = torch.from_numpy(scl[..., tracer_ci]).to(device)

        # ── PDE loss (sliding window, mean over all windows) ──
        pde_totals = []
        pde_components = {'continuity': [], 'u_momentum': [], 'v_momentum': [], 'tracer': []}

        # Use chunks of t_input+1 timesteps (same as training: input t_input, predict next)
        # But PDE needs 3 consecutive timesteps for central diff, so use t_input frames
        # Actually PDE is computed on model output [B, t_input, H, W].
        # For GT, simulate: take consecutive t_input-frame windows, compute PDE on them
        chunk = min(30, n_timesteps)  # process in chunks
        stride = chunk - 2

        with torch.no_grad():
            start = 0
            while start < n_timesteps - 2:
                end = min(start + chunk, n_timesteps)
                if end - start < 3:
                    break

                u_c = u[start:end].unsqueeze(0)
                v_c = v[start:end].unsqueeze(0)
                p_c = p[start:end].unsqueeze(0)
                s_c = s[start:end].unsqueeze(0)

                total, losses = pde_fn(u_c, v_c, p_c, s_c, reduction='mean')

                pde_totals.append(float(total))
                for k in pde_components:
                    pde_components[k].append(float(losses[k]))

                start += stride

        mean_pde = np.mean(pde_totals)
        mean_comp = {k: np.mean(v) for k, v in pde_components.items()}

        # ── BC loss: old (supervised) vs new (PeriodicBC) ──
        # Simulate training scenario: take t_input consecutive frames as "output"
        # Old BC: compare output boundary with target boundary (shift by 1)
        # New BC: self-consistency on output boundaries

        old_bc_vals = []
        new_bc_vals = []

        with torch.no_grad():
            # Sample several windows
            for t_start in range(0, n_timesteps - t_input - 1, t_input):
                # Construct 18-channel tensor like training
                t_end = t_start + t_input
                # "output" = GT[t_start+1 : t_end+1]  (what model would predict)
                out_vec = vec[t_start + 1:t_end + 1]  # [t_input, H, W, 3]
                out_scl = scl[t_start + 1:t_end + 1]  # [t_input, H, W, 2]
                tgt_vec = vec[t_start + 1:t_end + 1]  # same as output for GT
                tgt_scl = scl[t_start + 1:t_end + 1]

                # Build 18-channel: [1, t_input, H, W, 18]
                out_18 = np.zeros((1, t_input, nx, ny, 18), dtype=np.float32)
                out_18[..., 0] = out_vec[..., 0]
                out_18[..., 1] = out_vec[..., 1]
                out_18[..., 2] = out_vec[..., 2]
                out_18[..., 14] = out_scl[..., tracer_ci]
                out_18[..., 15] = out_scl[..., press_ci]

                tgt_18 = out_18.copy()  # GT = GT for this test

                out_t = torch.from_numpy(out_18).to(device)
                tgt_t = torch.from_numpy(tgt_18).to(device)

                # Old BC: supervised boundary RMSE
                old_val = compute_old_bc_loss(out_t, tgt_t)
                old_bc_vals.append(old_val)

                # New BC: PeriodicBCLoss on output
                fields = {
                    'vx': out_t[..., 0],
                    'vy': out_t[..., 1],
                    'tracer': out_t[..., 14],
                    'pressure': out_t[..., 15],
                }
                new_val, _ = bc_fn(fields)
                new_bc_vals.append(float(new_val))

        mean_old_bc = np.mean(old_bc_vals)
        mean_new_bc = np.mean(new_bc_vals)
        ratio = mean_new_bc / max(mean_old_bc, 1e-30)

        all_pde.append(mean_pde)
        all_old_bc.append(mean_old_bc)
        all_new_bc.append(mean_new_bc)

        print(f"{sid:4d} | {mean_pde:12.4e} | {mean_comp['continuity']:10.4e} | "
              f"{mean_comp['u_momentum']:10.4e} | {mean_comp['v_momentum']:10.4e} | "
              f"{mean_comp['tracer']:10.4e} | {mean_old_bc:10.4e} | {mean_new_bc:10.4e} | "
              f"{ratio:8.2f}x")

        del u, v, p, s
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY (across {n_samples} samples)")
    print(f"{'=' * 60}")
    print(f"  PDE total:    mean={np.mean(all_pde):.4e}  "
          f"min={np.min(all_pde):.4e}  max={np.max(all_pde):.4e}")
    print(f"  Old BC (sup): mean={np.mean(all_old_bc):.4e}  "
          f"min={np.min(all_old_bc):.4e}  max={np.max(all_old_bc):.4e}")
    print(f"  New BC (pbc): mean={np.mean(all_new_bc):.4e}  "
          f"min={np.min(all_new_bc):.4e}  max={np.max(all_new_bc):.4e}")
    print(f"\n  Old BC = 0 for GT (output == target), measuring only numerical noise")
    print(f"  New BC = self-consistency at periodic boundaries (nonzero even for perfect data)")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
