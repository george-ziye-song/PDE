"""
Comprehensive per-sample PDE residual analysis for Gray-Scott.

Loads ALL samples directly from H5 (bypasses train/val split),
computes per-timestep PDE loss, and reports:
  1. Per-sample overall PDE loss
  2. Per-sample temporal breakdown (which time ranges satisfy ~1e-5)
  3. Summary table for dataset curation

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/test_gt_pde_all_samples_gs.py \
        --data /scratch-share/SONG0304/finetune/gray_scott_train.h5 \
        --threshold 1e-5 --device cuda
"""

import argparse
import h5py
import torch
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

from finetune.pde_loss_verified import GrayScottPDELoss


def parse_args():
    p = argparse.ArgumentParser(description="Per-sample PDE residual analysis (Gray-Scott)")
    p.add_argument('--data', type=str, required=True, help='Path to gray_scott H5 file')
    p.add_argument('--threshold', type=float, default=1e-5,
                   help='PDE loss threshold for constraint')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--chunk_size', type=int, default=50,
                   help='Temporal chunk size for GPU processing')
    # Physics params (matching config)
    p.add_argument('--nx', type=int, default=128)
    p.add_argument('--ny', type=int, default=128)
    p.add_argument('--dx', type=float, default=2.0157 / 128)
    p.add_argument('--dy', type=float, default=2.0157 / 128)
    p.add_argument('--dt', type=float, default=10.0)
    p.add_argument('--F', type=float, default=0.098)
    p.add_argument('--k', type=float, default=0.057)
    p.add_argument('--D_A', type=float, default=1.81e-5)
    p.add_argument('--D_B', type=float, default=1.39e-5)
    return p.parse_args()


def load_sample_from_h5(
    filepath: str, sample_idx: int, scalar_indices: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load A, B for one sample directly from H5.

    Gray-Scott: A at total ch=5 (scalar_idx=2), B at total ch=6 (scalar_idx=3)

    Returns:
        A: [T, H, W], B: [T, H, W]
    """
    with h5py.File(filepath, 'r') as f:
        # scalar: [N, T, H, W, C_s]
        scl = np.array(f['scalar'][sample_idx], dtype=np.float32)

        # A → total channel 5 → scalar index 2 (5-3=2)
        # B → total channel 6 → scalar index 3 (6-3=3)
        a_target = 2
        b_target = 3

        A_data = None
        B_data = None
        for i, idx in enumerate(scalar_indices):
            if idx == a_target:
                A_data = torch.from_numpy(scl[..., i])
            elif idx == b_target:
                B_data = torch.from_numpy(scl[..., i])

        if A_data is None or B_data is None:
            # Fallback: check if vector contains the data or try direct mapping
            # Some datasets might store differently
            raise RuntimeError(
                f"Could not find A (scalar_idx={a_target}) or "
                f"B (scalar_idx={b_target}) in scalar_indices={scalar_indices.tolist()}\n"
                f"scalar shape: {scl.shape}"
            )

    return A_data, B_data


def compute_per_timestep_pde(
    pde_fn: GrayScottPDELoss,
    A: torch.Tensor,
    B: torch.Tensor,
    device: torch.device,
    chunk_size: int = 50,
) -> Dict[str, np.ndarray]:
    """Compute PDE residual MSE per timestep using temporal chunking.

    Returns dict of arrays, each shape [T-2], for each PDE component.
    """
    T = A.shape[0]
    keys = ['A_equation', 'B_equation', 'total']
    all_per_t = {k: [] for k in keys}

    stride = chunk_size - 2
    start = 0

    while start < T - 2:
        end = min(start + chunk_size, T)
        if end - start < 3:
            break

        A_c = A[start:end].unsqueeze(0).to(device)
        B_c = B[start:end].unsqueeze(0).to(device)

        with torch.no_grad():
            _, losses = pde_fn(A_c, B_c, reduction='none')

        for key in keys:
            residual_sq = losses[key]
            per_t = residual_sq.mean(dim=(-1, -2)).squeeze(0).cpu().numpy()

            if start == 0:
                all_per_t[key].append(per_t)
            else:
                all_per_t[key].append(per_t[1:])

        del A_c, B_c, losses
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        start += stride

    result = {}
    for key in keys:
        result[key] = np.concatenate(all_per_t[key])

    return result


def find_valid_ranges(
    per_t_loss: np.ndarray, threshold: float
) -> List[Tuple[int, int]]:
    """Find contiguous ranges where per-timestep loss < threshold."""
    mask = per_t_loss < threshold
    ranges = []
    in_range = False
    start = 0

    for i, valid in enumerate(mask):
        real_t = i + 1
        if valid and not in_range:
            start = real_t
            in_range = True
        elif not valid and in_range:
            ranges.append((start, real_t - 1))
            in_range = False

    if in_range:
        ranges.append((start, len(mask)))

    return ranges


def main():
    args = parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    device = torch.device(args.device)
    threshold = args.threshold

    pde_fn = GrayScottPDELoss(
        nx=args.nx, ny=args.ny, dx=args.dx, dy=args.dy,
        dt=args.dt, F=args.F, k=args.k, D_A=args.D_A, D_B=args.D_B,
    ).to(device)

    # Read metadata
    with h5py.File(args.data, 'r') as f:
        keys = list(f.keys())
        print(f"  H5 keys: {keys}")

        if 'vector' in f:
            n_samples = f['vector'].shape[0]
            n_timesteps = f['vector'].shape[1]
            spatial_shape = f['vector'].shape[2:4]
        elif 'scalar' in f:
            n_samples = f['scalar'].shape[0]
            n_timesteps = f['scalar'].shape[1]
            spatial_shape = f['scalar'].shape[2:4]
        else:
            # Old format
            sample_keys = sorted([k for k in keys if k.isdigit()], key=int)
            n_samples = len(sample_keys)
            first = f[sample_keys[0]]
            if 'data' in first:
                n_timesteps = first['data'].shape[0]
                spatial_shape = first['data'].shape[1:3]
            else:
                raise RuntimeError(f"Unknown H5 format, keys={keys}")

        scalar_indices = f['scalar_indices'][:] if 'scalar_indices' in f else np.array([])

    print(f"{'=' * 80}")
    print(f"  Comprehensive Per-Sample PDE Residual Analysis — Gray-Scott")
    print(f"{'=' * 80}")
    print(f"  Data:       {args.data}")
    print(f"  Samples:    {n_samples}")
    print(f"  Timesteps:  {n_timesteps}")
    print(f"  Spatial:    {spatial_shape}")
    print(f"  Threshold:  {threshold:.0e}")
    print(f"  Device:     {device}")
    print(f"  scalar_idx: {scalar_indices.tolist()}")
    print(f"  Physics:    F={args.F}, k={args.k}, D_A={args.D_A:.2e}, D_B={args.D_B:.2e}, dt={args.dt}")
    print(f"{'=' * 80}\n")

    all_results = []
    all_per_t = []

    for sid in range(n_samples):
        print(f"  Processing sample {sid:2d}/{n_samples} ...", end='', flush=True)

        A, B = load_sample_from_h5(args.data, sid, scalar_indices)
        per_t = compute_per_timestep_pde(pde_fn, A, B, device, args.chunk_size)

        total_arr = per_t['total']
        a_arr = per_t['A_equation']
        b_arr = per_t['B_equation']

        overall_mean = float(total_arr.mean())
        valid_ranges = find_valid_ranges(total_arr, threshold)
        n_valid_t = int((total_arr < threshold).sum())
        n_total_t = len(total_arr)

        result = {
            'sample_idx': sid,
            'overall_mean': overall_mean,
            'overall_max': float(total_arr.max()),
            'overall_min': float(total_arr.min()),
            'A_mean': float(a_arr.mean()),
            'B_mean': float(b_arr.mean()),
            'n_valid_t': n_valid_t,
            'n_total_t': n_total_t,
            'valid_fraction': n_valid_t / n_total_t,
            'valid_ranges': valid_ranges,
        }
        all_results.append(result)
        all_per_t.append(total_arr)

        status = "PASS" if overall_mean < threshold else "FAIL"
        print(f"  mean={overall_mean:.4e}  valid_t={n_valid_t}/{n_total_t}  [{status}]")

    # ── Summary Table ──
    print(f"\n{'=' * 80}")
    print(f"  SUMMARY TABLE (threshold={threshold:.0e})")
    print(f"{'=' * 80}")
    print(f"{'Sample':>6} | {'Mean PDE':>12} | {'Max PDE':>12} | {'Valid t':>10} | {'%':>6} | {'Status':>6} | "
          f"{'A_eq':>12} | {'B_eq':>12}")
    print(f"{'-' * 6}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 10}-+-{'-' * 6}-+-{'-' * 6}-+-"
          f"{'-' * 12}-+-{'-' * 12}")

    for r in all_results:
        status = "PASS" if r['overall_mean'] < threshold else "FAIL"
        print(f"{r['sample_idx']:6d} | {r['overall_mean']:12.4e} | {r['overall_max']:12.4e} | "
              f"{r['n_valid_t']:4d}/{r['n_total_t']:<4d} | {r['valid_fraction']*100:5.1f}% | {status:>6} | "
              f"{r['A_mean']:12.4e} | {r['B_mean']:12.4e}")

    # ── Per-Sample Valid Ranges ──
    print(f"\n{'=' * 80}")
    print(f"  VALID TEMPORAL RANGES (PDE < {threshold:.0e})")
    print(f"{'=' * 80}")

    for r in all_results:
        sid = r['sample_idx']
        ranges = r['valid_ranges']
        if len(ranges) == 0:
            print(f"  Sample {sid:2d}: NO valid range")
        else:
            range_strs = [f"t={s}-{e}" for s, e in ranges]
            longest = max(ranges, key=lambda x: x[1] - x[0])
            print(f"  Sample {sid:2d}: {', '.join(range_strs)}  "
                  f"(longest: t={longest[0]}-{longest[1]}, len={longest[1]-longest[0]+1})")

    # ── Temporal Breakdown ──
    print(f"\n{'=' * 80}")
    print(f"  TEMPORAL BREAKDOWN (early=0-33%, mid=33-66%, late=66-100%)")
    print(f"{'=' * 80}")

    n_residual_t = n_timesteps - 2
    third = n_residual_t // 3
    early_end = third
    mid_end = 2 * third

    print(f"  Periods: early=t[1-{early_end}], mid=t[{early_end+1}-{mid_end}], late=t[{mid_end+1}-{n_residual_t}]")
    print(f"{'Sample':>6} | {'Early Mean':>12} | {'Mid Mean':>12} | {'Late Mean':>12} | "
          f"{'Early OK':>10} | {'Mid OK':>10} | {'Late OK':>10}")
    print(f"{'-' * 6}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}-+-"
          f"{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}")

    for i, r in enumerate(all_results):
        arr = all_per_t[i]
        early = arr[:early_end]
        mid = arr[early_end:mid_end]
        late = arr[mid_end:]

        early_ok = int((early < threshold).sum())
        mid_ok = int((mid < threshold).sum())
        late_ok = int((late < threshold).sum())

        print(f"{r['sample_idx']:6d} | {early.mean():12.4e} | {mid.mean():12.4e} | {late.mean():12.4e} | "
              f"{early_ok:4d}/{len(early):<4d} | {mid_ok:4d}/{len(mid):<4d} | {late_ok:4d}/{len(late):<4d}")

    # ── Recommendation ──
    print(f"\n{'=' * 80}")
    print(f"  RECOMMENDATION")
    print(f"{'=' * 80}")

    pass_samples = [r for r in all_results if r['overall_mean'] < threshold]
    fail_samples = [r for r in all_results if r['overall_mean'] >= threshold]
    print(f"  Overall PASS: {len(pass_samples)}/{n_samples} samples")
    print(f"  Overall FAIL: {len(fail_samples)}/{n_samples} samples")
    if fail_samples:
        fail_ids = [r['sample_idx'] for r in fail_samples]
        print(f"  Failed sample IDs: {fail_ids}")

    # Check late period
    late_pass = 0
    for i in range(len(all_results)):
        arr = all_per_t[i]
        late = arr[mid_end:]
        if late.mean() < threshold:
            late_pass += 1
    print(f"\n  Late period PASS: {late_pass}/{n_samples}")

    # Cross-sample statistics
    stacked = np.stack(all_per_t, axis=0)
    per_t_mean = stacked.mean(axis=0)
    per_t_max = stacked.max(axis=0)

    all_pass_mask = stacked.max(axis=0) < threshold
    first_all_pass = np.argmax(all_pass_mask) if all_pass_mask.any() else -1
    if first_all_pass >= 0:
        real_t = first_all_pass + 1
        print(f"  First timestep where ALL samples < {threshold:.0e}: t={real_t}")
    else:
        print(f"  No single timestep where ALL samples pass simultaneously")
        n_pass_per_t = (stacked < threshold).sum(axis=0)
        best_t_idx = np.argmax(n_pass_per_t)
        print(f"  Best timestep: t={best_t_idx+1}, {n_pass_per_t[best_t_idx]}/{n_samples} samples pass")

    print(f"\n  Cross-sample mean PDE by temporal region:")
    print(f"    Early (t=1-{early_end}):   mean={per_t_mean[:early_end].mean():.4e}  "
          f"max_across_samples={per_t_max[:early_end].max():.4e}")
    print(f"    Mid   (t={early_end+1}-{mid_end}): mean={per_t_mean[early_end:mid_end].mean():.4e}  "
          f"max_across_samples={per_t_max[early_end:mid_end].max():.4e}")
    print(f"    Late  (t={mid_end+1}-{n_residual_t}): mean={per_t_mean[mid_end:].mean():.4e}  "
          f"max_across_samples={per_t_max[mid_end:].max():.4e}")

    # Check for duplicate samples
    print(f"\n  Checking for duplicate samples...")
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if np.allclose(all_per_t[i], all_per_t[j], rtol=1e-4):
                print(f"    WARNING: Sample {i} and {j} appear to be duplicates!")

    print(f"\n{'=' * 80}")
    print(f"  Analysis complete.")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
