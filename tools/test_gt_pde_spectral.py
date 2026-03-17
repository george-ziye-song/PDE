"""
Spectral PDE residual analysis for Shear Flow.

Uses FFT-based spatial derivatives (exact for doubly-periodic domains)
and 4th-order central difference for time derivative.
This eliminates spatial discretization error and reduces temporal error to O(dt^4).

Equations:
    div(u) = 0                       (continuity)
    du/dt + u·∇u = -∇p + ν∇²u       (momentum)
    ds/dt + u·∇s = D∇²s             (tracer)

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/test_gt_pde_spectral.py \
        --data ./data/finetune/shear_flow_train.h5 --device cuda
"""

import argparse
import h5py
import torch
import numpy as np
from typing import Dict


def parse_args():
    p = argparse.ArgumentParser(description="Spectral PDE residual analysis")
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--Lx', type=float, default=1.0)
    p.add_argument('--Ly', type=float, default=2.0)
    p.add_argument('--dt', type=float, default=0.1)
    p.add_argument('--nu', type=float, default=1e-4)
    p.add_argument('--D', type=float, default=1e-3)
    return p.parse_args()


class SpectralOps:
    """FFT-based derivative operators for doubly-periodic domain [0,Lx)×[0,Ly)."""

    def __init__(self, nx: int, ny: int, Lx: float, Ly: float, device: torch.device):
        kx = 2 * torch.pi * torch.fft.fftfreq(nx, d=Lx / nx)
        ky = 2 * torch.pi * torch.fft.fftfreq(ny, d=Ly / ny)
        self.kx = kx.to(device=device, dtype=torch.float64)[:, None]   # [nx, 1]
        self.ky = ky.to(device=device, dtype=torch.float64)[None, :]   # [1, ny]
        self.k2 = self.kx ** 2 + self.ky ** 2                          # [nx, ny]

    def dx(self, f: torch.Tensor) -> torch.Tensor:
        """df/dx via FFT. f: [..., nx, ny]"""
        return torch.fft.ifft2(1j * self.kx * torch.fft.fft2(f)).real

    def dy(self, f: torch.Tensor) -> torch.Tensor:
        """df/dy via FFT. f: [..., nx, ny]"""
        return torch.fft.ifft2(1j * self.ky * torch.fft.fft2(f)).real

    def laplacian(self, f: torch.Tensor) -> torch.Tensor:
        """∇²f via FFT. f: [..., nx, ny]"""
        return torch.fft.ifft2(-self.k2 * torch.fft.fft2(f)).real

    def divergence(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """∇·(u,v) = du/dx + dv/dy"""
        uv_hat = torch.fft.fft2(torch.stack([u, v], dim=0))
        div_hat = 1j * self.kx * uv_hat[0] + 1j * self.ky * uv_hat[1]
        return torch.fft.ifft2(div_hat).real


def compute_spectral_pde(
    ops: SpectralOps,
    u: torch.Tensor, v: torch.Tensor,
    p: torch.Tensor, s: torch.Tensor,
    dt: float, nu: float, D: float,
) -> Dict[str, np.ndarray]:
    """Compute PDE residuals using spectral spatial + 4th-order temporal.

    Args:
        u, v, p, s: [T, H, W] on device, float64
    Returns:
        dict of per-timestep MSE arrays, each shape [T-4]
    """
    # 4th-order central diff: (-f[t+2]+8f[t+1]-8f[t-1]+f[t-2])/(12dt)
    du_dt = (-u[4:] + 8 * u[3:-1] - 8 * u[1:-3] + u[:-4]) / (12 * dt)
    dv_dt = (-v[4:] + 8 * v[3:-1] - 8 * v[1:-3] + v[:-4]) / (12 * dt)
    ds_dt = (-s[4:] + 8 * s[3:-1] - 8 * s[1:-3] + s[:-4]) / (12 * dt)

    # Fields at mid-timesteps (t=2..T-3)
    u_m, v_m, p_m, s_m = u[2:-2], v[2:-2], p[2:-2], s[2:-2]

    # 1. Continuity: ∇·u = 0
    R_cont = ops.divergence(u_m, v_m)

    # 2. x-momentum: du/dt + u·∇u = -∇p + ν∇²u
    R_u = du_dt + u_m * ops.dx(u_m) + v_m * ops.dy(u_m) + ops.dx(p_m) - nu * ops.laplacian(u_m)

    # 3. y-momentum: dv/dt + u·∇v = -∇p + ν∇²v
    R_v = dv_dt + u_m * ops.dx(v_m) + v_m * ops.dy(v_m) + ops.dy(p_m) - nu * ops.laplacian(v_m)

    # 4. Tracer: ds/dt + u·∇s = D∇²s
    R_s = ds_dt + u_m * ops.dx(s_m) + v_m * ops.dy(s_m) - D * ops.laplacian(s_m)

    per_t = lambda r: (r ** 2).mean(dim=(-1, -2)).cpu().numpy()

    cont = per_t(R_cont)
    u_mom = per_t(R_u)
    v_mom = per_t(R_v)
    tracer = per_t(R_s)

    return {
        'continuity': cont,
        'u_momentum': u_mom,
        'v_momentum': v_mom,
        'tracer': tracer,
        'total': cont + u_mom + v_mom + tracer,
    }


def main():
    args = parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    device = torch.device(args.device)

    with h5py.File(args.data, 'r') as f:
        n_samples = f['vector'].shape[0]
        n_timesteps = f['vector'].shape[1]
        nx, ny = f['vector'].shape[2:4]
        scalar_indices = f['scalar_indices'][:]

    tracer_ci = int(np.where(scalar_indices == 11)[0][0])
    press_ci = int(np.where(scalar_indices == 12)[0][0])

    print(f"{'=' * 90}")
    print(f"  Spectral PDE Residual Analysis — Shear Flow")
    print(f"{'=' * 90}")
    print(f"  Data:     {args.data}")
    print(f"  Samples:  {n_samples},  Timesteps: {n_timesteps},  Grid: {nx}×{ny}")
    print(f"  Lx={args.Lx}, Ly={args.Ly}, dt={args.dt}, ν={args.nu}, D={args.D}")
    print(f"  Spatial:  Spectral (FFT, exact for periodic)")
    print(f"  Temporal: 4th-order central diff, truncation O(dt⁴={args.dt**4:.0e})")
    print(f"  Device:   {device}")
    print(f"{'=' * 90}\n")

    ops = SpectralOps(nx, ny, args.Lx, args.Ly, device)

    all_results = []
    all_per_t = []

    for sid in range(n_samples):
        print(f"  Sample {sid:2d}/{n_samples} ...", end='', flush=True)

        with h5py.File(args.data, 'r') as f:
            vec = np.array(f['vector'][sid], dtype=np.float64)
            scl = np.array(f['scalar'][sid], dtype=np.float64)

        u = torch.from_numpy(vec[..., 0]).to(device)
        v = torch.from_numpy(vec[..., 1]).to(device)
        p = torch.from_numpy(scl[..., press_ci]).to(device)
        s = torch.from_numpy(scl[..., tracer_ci]).to(device)

        with torch.no_grad():
            res = compute_spectral_pde(ops, u, v, p, s, args.dt, args.nu, args.D)

        total_arr = res['total']
        mean_pde = float(total_arr.mean())

        result = {
            'sid': sid,
            'mean': mean_pde,
            'max': float(total_arr.max()),
            'cont': float(res['continuity'].mean()),
            'u_mom': float(res['u_momentum'].mean()),
            'v_mom': float(res['v_momentum'].mean()),
            'tracer': float(res['tracer'].mean()),
        }
        all_results.append(result)
        all_per_t.append(total_arr)

        print(f"  mean={mean_pde:.4e}  max={result['max']:.4e}")

        del u, v, p, s
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ── Summary Table ──
    print(f"\n{'=' * 100}")
    print(f"  SUMMARY  (Spectral spatial + 4th-order temporal)")
    print(f"{'=' * 100}")
    print(f"{'Samp':>4} | {'Mean Total':>12} | {'Max Total':>12} | "
          f"{'Continuity':>12} | {'U-Mom':>12} | {'V-Mom':>12} | {'Tracer':>12}")
    print(f"{'-' * 4}-+-{'-' * 12}-+-{'-' * 12}-+-"
          f"{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}")

    for r in all_results:
        print(f"{r['sid']:4d} | {r['mean']:12.4e} | {r['max']:12.4e} | "
              f"{r['cont']:12.4e} | {r['u_mom']:12.4e} | {r['v_mom']:12.4e} | {r['tracer']:12.4e}")

    # ── Temporal Breakdown ──
    T_valid = all_per_t[0].shape[0]
    third = T_valid // 3

    print(f"\n{'=' * 80}")
    print(f"  TEMPORAL BREAKDOWN  (T_valid={T_valid}, early/mid/late ~{third} each)")
    print(f"{'=' * 80}")
    print(f"{'Samp':>4} | {'Early':>12} | {'Mid':>12} | {'Late':>12}")
    print(f"{'-' * 4}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}")

    for i, r in enumerate(all_results):
        arr = all_per_t[i]
        print(f"{r['sid']:4d} | {arr[:third].mean():12.4e} | "
              f"{arr[third:2*third].mean():12.4e} | {arr[2*third:].mean():12.4e}")

    # ── Duplicate Detection ──
    print(f"\n  Checking for duplicate samples ...")
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            rel = abs(all_results[i]['mean'] - all_results[j]['mean']) / max(all_results[i]['mean'], 1e-30)
            if rel < 1e-4:
                print(f"    ⚠ Sample {i} and {j} appear IDENTICAL (rel diff={rel:.2e})")

    # ── Recommendation ──
    print(f"\n{'=' * 80}")
    print(f"  RECOMMENDATION  (IQR-based outlier detection)")
    print(f"{'=' * 80}")

    means = np.array([r['mean'] for r in all_results])
    log_means = np.log10(means)

    median = np.median(log_means)
    q25 = np.percentile(log_means, 25)
    q75 = np.percentile(log_means, 75)
    iqr = q75 - q25
    outlier_log = q75 + 1.5 * iqr
    outlier_threshold = 10 ** outlier_log

    print(f"  Log10 stats:  median={median:.2f}, Q25={q25:.2f}, Q75={q75:.2f}, IQR={iqr:.2f}")
    print(f"  Outlier threshold: 10^({q75:.2f}+1.5×{iqr:.2f}) = {outlier_threshold:.4e}")

    bad = [r['sid'] for r in all_results if r['mean'] > outlier_threshold]
    good = [r['sid'] for r in all_results if r['mean'] <= outlier_threshold]

    print(f"\n  REMOVE ({len(bad)}): {bad}")
    print(f"  KEEP   ({len(good)}): {good}")

    # Sorted ranking
    print(f"\n  Ranked by mean PDE (ascending):")
    ranked = sorted(all_results, key=lambda r: r['mean'])
    for rank, r in enumerate(ranked, 1):
        marker = " *** REMOVE" if r['sid'] in bad else ""
        print(f"    {rank:2d}. Sample {r['sid']:2d}  mean={r['mean']:.4e}{marker}")

    print(f"\n{'=' * 80}")


if __name__ == '__main__':
    main()
