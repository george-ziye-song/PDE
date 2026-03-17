"""
GT PDE residual verification for Gray-Scott dataset.

Computes per-equation RMS residuals to determine eq_scales
for normalized PDE loss (same pattern as NS-PwC).

Gray-Scott equations:
    dA/dt = D_A * ∇²A - A*B² + F*(1-A)
    dB/dt = D_B * ∇²B + A*B² - (F+k)*B

Numerical method:
    - Spatial: 4th-order central difference Laplacian (periodic BC)
    - Temporal: 2nd-order central difference

Usage:
    python tools/test_gt_pde_gray_scott.py \
        --data /scratch-share/SONG0304/finetune/gray_scott_test.h5
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


def fd_laplacian_4th(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """4th-order central FD Laplacian with periodic BC."""
    d2f_dx2 = (
        -np.roll(f, -2, axis=-2) + 16 * np.roll(f, -1, axis=-2)
        - 30 * f
        + 16 * np.roll(f, 1, axis=-2) - np.roll(f, 2, axis=-2)
    ) / (12 * dx ** 2)
    d2f_dy2 = (
        -np.roll(f, -2, axis=-1) + 16 * np.roll(f, -1, axis=-1)
        - 30 * f
        + 16 * np.roll(f, 1, axis=-1) - np.roll(f, 2, axis=-1)
    ) / (12 * dy ** 2)
    return d2f_dx2 + d2f_dy2


def compute_residuals(
    A: np.ndarray,
    B: np.ndarray,
    dx: float,
    dy: float,
    dt: float,
    F: float,
    k: float,
    D_A: float,
    D_B: float,
) -> dict[str, np.ndarray]:
    """Compute Gray-Scott PDE residuals.

    Args:
        A, B: shape (T, H, W)
    Returns:
        dict with 'A_equation', 'B_equation' residual arrays
    """
    # 2nd-order central time difference
    dA_dt = (A[2:] - A[:-2]) / (2 * dt)
    dB_dt = (B[2:] - B[:-2]) / (2 * dt)

    # Spatial terms at mid-frame
    A_n = A[1:-1]
    B_n = B[1:-1]

    lap_A = fd_laplacian_4th(A_n, dx, dy)
    lap_B = fd_laplacian_4th(B_n, dx, dy)

    AB2 = A_n * B_n ** 2

    R_A = dA_dt - D_A * lap_A + AB2 - F * (1 - A_n)
    R_B = dB_dt - D_B * lap_B - AB2 + (F + k) * B_n

    return {
        'A_equation': R_A,
        'B_equation': R_B,
    }


def main():
    parser = argparse.ArgumentParser(description="GT PDE residual for Gray-Scott")
    parser.add_argument("--data", type=str, required=True, help="Path to H5 file")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--dx", type=float, default=2.0157 / 128)
    parser.add_argument("--dy", type=float, default=2.0157 / 128)
    parser.add_argument("--dt", type=float, default=10.0)
    parser.add_argument("--F", type=float, default=0.098)
    parser.add_argument("--k", type=float, default=0.057)
    parser.add_argument("--D_A", type=float, default=1.81e-5)
    parser.add_argument("--D_B", type=float, default=1.39e-5)
    args = parser.parse_args()

    print(f"Loading: {args.data}")
    with h5py.File(args.data, 'r') as f:
        # scalar: [N, T, H, W, C_s]
        scalar = f['scalar']
        n_samples_total = scalar.shape[0]
        n_time = scalar.shape[1]
        print(f"  Samples={n_samples_total}, T={n_time}, "
              f"Grid={scalar.shape[2]}×{scalar.shape[3]}, Scalar_ch={scalar.shape[4]}")

        n_proc = min(n_samples_total, args.max_samples)

        eq_names = ['A_equation', 'B_equation']
        all_rms = {k: [] for k in eq_names}

        for s_idx in range(n_proc):
            scl = np.array(scalar[s_idx], dtype=np.float64)  # [T, H, W, C_s]
            # Gray-Scott: A=scalar[2], B=scalar[3]
            A = scl[:, :, :, 2]  # [T, H, W]
            B = scl[:, :, :, 3]  # [T, H, W]

            res = compute_residuals(
                A, B, args.dx, args.dy, args.dt,
                args.F, args.k, args.D_A, args.D_B,
            )

            for eq in eq_names:
                rms = np.sqrt(np.mean(res[eq] ** 2))
                all_rms[eq].append(rms)

            if s_idx < 3 or s_idx == n_proc - 1:
                print(f"  Sample {s_idx}: "
                      f"A_eq={all_rms['A_equation'][-1]:.4e}, "
                      f"B_eq={all_rms['B_equation'][-1]:.4e}")

    # Summary
    print(f"\n{'='*70}")
    print(f"Summary over {n_proc} samples")
    print(f"{'='*70}")

    print(f"\n{'Equation':<15} | {'Mean RMS':>12} | {'Std RMS':>12} | {'Min RMS':>12} | {'Max RMS':>12}")
    print("-" * 70)
    for eq in eq_names:
        arr = np.array(all_rms[eq])
        print(f"{eq:<15} | {arr.mean():>12.4e} | {arr.std():>12.4e} | "
              f"{arr.min():>12.4e} | {arr.max():>12.4e}")

    # eq_scales for config
    print(f"\n--- eq_scales for config (Mean RMS values) ---")
    print("physics:")
    print("  eq_scales:")
    for eq in eq_names:
        arr = np.array(all_rms[eq])
        print(f"    {eq}: {arr.mean():.4e}")


if __name__ == "__main__":
    main()
