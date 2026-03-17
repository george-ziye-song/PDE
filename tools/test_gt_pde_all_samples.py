"""
Comprehensive per-sample PDE residual analysis for Shear Flow.

Loads ALL samples directly from H5 (bypasses train/val split),
computes per-timestep PDE loss, and reports:
  1. Per-sample overall PDE loss
  2. Per-sample temporal breakdown (which time ranges satisfy ~1e-5)
  3. Summary table for dataset curation

Usage:
    python tools/test_gt_pde_all_samples.py \
        --data ./data/finetune/shear_flow_train.h5 \
        --threshold 1e-5

    # Use GPU for faster computation
    CUDA_VISIBLE_DEVICES=0 python tools/test_gt_pde_all_samples.py \
        --data ./data/finetune/shear_flow_train.h5 --device cuda
"""

import argparse
import h5py
import torch
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

from finetune.pde_loss_verified import ShearFlowPDELoss


def parse_args():
    p = argparse.ArgumentParser(description="Per-sample PDE residual analysis")
    p.add_argument('--data', type=str, required=True, help='Path to shear_flow H5 file')
    p.add_argument('--threshold', type=float, default=1e-5,
                   help='PDE loss threshold for constraint')
    p.add_argument('--device', type=str, default='cuda',
                   help='Device: cuda or cpu')
    # Physics params (matching config)
    p.add_argument('--nx', type=int, default=256)
    p.add_argument('--ny', type=int, default=512)
    p.add_argument('--Lx', type=float, default=1.0)
    p.add_argument('--Ly', type=float, default=2.0)
    p.add_argument('--dt', type=float, default=0.1)
    p.add_argument('--nu', type=float, default=1e-4)
    p.add_argument('--D', type=float, default=1e-3)
    return p.parse_args()


def load_sample_from_h5(
    filepath: str, sample_idx: int, scalar_indices: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load u, v, p, s for one sample directly from H5.

    Returns:
        u: [T, H, W], v: [T, H, W], p: [T, H, W], s: [T, H, W]
    """
    with h5py.File(filepath, 'r') as f:
        # vector: [N, T, H, W, 3] → Vx=0, Vy=1
        # Use float64 to avoid NaN in PDE computation
        vec = np.array(f['vector'][sample_idx], dtype=np.float64)  # [T, H, W, 3]
        u = torch.from_numpy(vec[..., 0])  # [T, H, W]
        v = torch.from_numpy(vec[..., 1])  # [T, H, W]

        # scalar: [N, T, H, W, C_s], mapped via scalar_indices
        scl = np.array(f['scalar'][sample_idx], dtype=np.float64)  # [T, H, W, C_s]

        # Find tracer (total_ch=14 → scalar_idx=11) and pressure (total_ch=15 → scalar_idx=12)
        tracer_target = 11  # 14 - 3
        press_target = 12   # 15 - 3

        s_data = None
        p_data = None
        for i, idx in enumerate(scalar_indices):
            if idx == tracer_target:
                s_data = torch.from_numpy(scl[..., i])
            elif idx == press_target:
                p_data = torch.from_numpy(scl[..., i])

        if s_data is None or p_data is None:
            raise RuntimeError(
                f"Could not find tracer (idx={tracer_target}) or "
                f"pressure (idx={press_target}) in scalar_indices={scalar_indices}"
            )

    return u, v, p_data, s_data


def compute_per_timestep_pde(
    pde_fn: ShearFlowPDELoss,
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    s: torch.Tensor,
    device: torch.device,
    chunk_size: int = 30,
) -> Dict[str, np.ndarray]:
    """Compute PDE residual MSE per timestep using temporal chunking.

    Central time derivative uses t-1, t, t+1, so residual is defined at t=1..T-2.
    Processes in overlapping chunks to fit in GPU memory.

    Returns dict of arrays, each shape [T-2], for each PDE component.
    """
    T = u.shape[0]
    keys = ['continuity', 'u_momentum', 'v_momentum', 'tracer', 'total']
    all_per_t = {k: [] for k in keys}

    # Process in chunks with overlap=2 (need t-1 and t+1 for central diff)
    stride = chunk_size - 2  # effective new timesteps per chunk
    start = 0

    while start < T - 2:  # need at least 3 timesteps
        end = min(start + chunk_size, T)
        if end - start < 3:
            break

        # Move chunk to device: [1, chunk_T, H, W]
        u_c = u[start:end].unsqueeze(0).to(device)
        v_c = v[start:end].unsqueeze(0).to(device)
        p_c = p[start:end].unsqueeze(0).to(device)
        s_c = s[start:end].unsqueeze(0).to(device)

        with torch.no_grad():
            _, losses = pde_fn(u_c, v_c, p_c, s_c, reduction='none')

        # losses[key] is [1, chunk_T-2, H, W]
        for key in keys:
            residual_sq = losses[key]
            per_t = residual_sq.mean(dim=(-1, -2)).squeeze(0).cpu().numpy()

            if start == 0:
                # First chunk: keep all
                all_per_t[key].append(per_t)
            else:
                # Subsequent chunks: skip first timestep (already covered by previous)
                all_per_t[key].append(per_t[1:])

        # Free GPU memory
        del u_c, v_c, p_c, s_c, losses
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
    """Find contiguous ranges where per-timestep loss < threshold.

    Args:
        per_t_loss: [T-2] array, index 0 corresponds to real timestep 1
        threshold: constraint threshold

    Returns:
        List of (start_t, end_t) real timestep ranges (inclusive)
    """
    mask = per_t_loss < threshold
    ranges = []
    in_range = False
    start = 0

    for i, valid in enumerate(mask):
        real_t = i + 1  # offset by 1 because central diff drops first/last
        if valid and not in_range:
            start = real_t
            in_range = True
        elif not valid and in_range:
            ranges.append((start, real_t - 1))
            in_range = False

    if in_range:
        ranges.append((start, len(mask)))  # last real_t = len(mask)

    return ranges


def main():
    args = parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    device = torch.device(args.device)
    threshold = args.threshold

    pde_fn = ShearFlowPDELoss(
        nx=args.nx, ny=args.ny, Lx=args.Lx, Ly=args.Ly,
        dt=args.dt, nu=args.nu, D=args.D,
    ).double().to(device)

    # Read metadata
    with h5py.File(args.data, 'r') as f:
        n_samples = f['vector'].shape[0]
        n_timesteps = f['vector'].shape[1]
        spatial_shape = f['vector'].shape[2:4]
        scalar_indices = f['scalar_indices'][:] if 'scalar_indices' in f else np.array([])

    print(f"{'=' * 80}")
    print(f"  Comprehensive Per-Sample PDE Residual Analysis — Shear Flow")
    print(f"{'=' * 80}")
    print(f"  Data:       {args.data}")
    print(f"  Samples:    {n_samples}")
    print(f"  Timesteps:  {n_timesteps}")
    print(f"  Spatial:    {spatial_shape}")
    print(f"  Threshold:  {threshold:.0e}")
    print(f"  Device:     {device}")
    print(f"  scalar_idx: {scalar_indices.tolist()}")
    print(f"{'=' * 80}\n")

    # Storage for all results
    all_results = []  # list of dicts per sample
    all_per_t = []    # per-timestep total loss arrays

    for sid in range(n_samples):
        print(f"  Processing sample {sid:2d}/{n_samples} ...", end='', flush=True)

        u, v, p, s = load_sample_from_h5(args.data, sid, scalar_indices)
        per_t = compute_per_timestep_pde(pde_fn, u, v, p, s, device)

        total_arr = per_t['total']
        cont_arr = per_t['continuity']
        umom_arr = per_t['u_momentum']
        vmom_arr = per_t['v_momentum']
        tracer_arr = per_t['tracer']

        overall_mean = float(total_arr.mean())
        valid_ranges = find_valid_ranges(total_arr, threshold)
        n_valid_t = int((total_arr < threshold).sum())
        n_total_t = len(total_arr)

        result = {
            'sample_idx': sid,
            'overall_mean': overall_mean,
            'overall_max': float(total_arr.max()),
            'overall_min': float(total_arr.min()),
            'cont_mean': float(cont_arr.mean()),
            'umom_mean': float(umom_arr.mean()),
            'vmom_mean': float(vmom_arr.mean()),
            'tracer_mean': float(tracer_arr.mean()),
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
          f"{'Cont':>10} | {'U-Mom':>10} | {'V-Mom':>10} | {'Tracer':>10}")
    print(f"{'-' * 6}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 10}-+-{'-' * 6}-+-{'-' * 6}-+-"
          f"{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}")

    for r in all_results:
        status = "PASS" if r['overall_mean'] < threshold else "FAIL"
        print(f"{r['sample_idx']:6d} | {r['overall_mean']:12.4e} | {r['overall_max']:12.4e} | "
              f"{r['n_valid_t']:4d}/{r['n_total_t']:<4d} | {r['valid_fraction']*100:5.1f}% | {status:>6} | "
              f"{r['cont_mean']:10.4e} | {r['umom_mean']:10.4e} | {r['vmom_mean']:10.4e} | {r['tracer_mean']:10.4e}")

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
            # Find the longest contiguous valid range
            longest = max(ranges, key=lambda x: x[1] - x[0])
            print(f"  Sample {sid:2d}: {', '.join(range_strs)}  "
                  f"(longest: t={longest[0]}-{longest[1]}, len={longest[1]-longest[0]+1})")

    # ── Temporal Breakdown (early / mid / late) ──
    print(f"\n{'=' * 80}")
    print(f"  TEMPORAL BREAKDOWN (early=0-65, mid=66-132, late=133-199)")
    print(f"{'=' * 80}")

    # Residual timesteps are 1..T-2 (=1..198 for T=200)
    # Map to thirds: early=1-66, mid=67-132, late=133-198
    third = (n_timesteps - 2) // 3  # ~66
    early_end = third
    mid_end = 2 * third

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

    # Summarize pass/fail
    pass_samples = [r for r in all_results if r['overall_mean'] < threshold]
    fail_samples = [r for r in all_results if r['overall_mean'] >= threshold]
    print(f"  Overall PASS: {len(pass_samples)}/{n_samples} samples")
    print(f"  Overall FAIL: {len(fail_samples)}/{n_samples} samples")
    if fail_samples:
        fail_ids = [r['sample_idx'] for r in fail_samples]
        print(f"  Failed sample IDs: {fail_ids}")

    # Check if late period is universally better
    late_pass = 0
    for i, r in enumerate(all_results):
        arr = all_per_t[i]
        late = arr[mid_end:]
        if late.mean() < threshold:
            late_pass += 1
    print(f"\n  Late period (t≥{mid_end+1}) PASS: {late_pass}/{n_samples}")

    # Find the global best starting timestep
    print(f"\n  Per-timestep statistics across all samples:")
    stacked = np.stack(all_per_t, axis=0)  # [n_samples, T-2]
    per_t_mean = stacked.mean(axis=0)
    per_t_max = stacked.max(axis=0)

    # Find first timestep where ALL samples pass
    all_pass_mask = stacked.max(axis=0) < threshold
    first_all_pass = np.argmax(all_pass_mask) if all_pass_mask.any() else -1
    if first_all_pass >= 0:
        real_t = first_all_pass + 1
        print(f"  First timestep where ALL samples < {threshold:.0e}: t={real_t}")
    else:
        print(f"  No single timestep where ALL samples pass simultaneously")
        # Find timestep where most samples pass
        n_pass_per_t = (stacked < threshold).sum(axis=0)
        best_t_idx = np.argmax(n_pass_per_t)
        print(f"  Best timestep: t={best_t_idx+1}, {n_pass_per_t[best_t_idx]}/{n_samples} samples pass")

    # Percentile analysis over timesteps
    print(f"\n  Cross-sample mean PDE by temporal region:")
    print(f"    Early (t=1-{early_end}):   mean={per_t_mean[:early_end].mean():.4e}  "
          f"max_across_samples={per_t_max[:early_end].max():.4e}")
    print(f"    Mid   (t={early_end+1}-{mid_end}): mean={per_t_mean[early_end:mid_end].mean():.4e}  "
          f"max_across_samples={per_t_max[early_end:mid_end].max():.4e}")
    print(f"    Late  (t={mid_end+1}-{n_timesteps-2}): mean={per_t_mean[mid_end:].mean():.4e}  "
          f"max_across_samples={per_t_max[mid_end:].max():.4e}")

    print(f"\n{'=' * 80}")
    print(f"  Analysis complete. Review above to decide dataset trimming strategy.")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
