"""
Preprocess PDEBench 2D Darcy Flow data to our standard H5 format.

Input: 2D_DarcyFlow_betaX.X_Train.hdf5
  - nu: (N, 128, 128) — coefficient field a(x)
  - tensor: (N, 1, 128, 128) — solution field u(x)
  - x-coordinate, y-coordinate: (128,)

Output: darcy_flow_betaX.X.h5
  - vector: (N, T, H, W, 3) — zeros (no velocity)
  - scalar: (N, T, H, W, 2) — [a(x), u(x)]
  - scalar_indices: [8, 12]  (8=geometry/mask, 12=pressure)
  - x-coordinate, y-coordinate: (128,)
  - attrs: beta, forcing (=beta), equation, bc, nx, ny

Darcy Flow: -div(a(x)*grad(u(x))) = f, f=beta
  - Steady state (T=1), no velocity, vector_dim=0
  - Dirichlet BC: u=0 on boundary
  - Domain: [0,1]^2, cell-centered 128x128

Usage:
    python tools/preprocess_darcy_flow.py \
        --input data/finetune/2D_DarcyFlow_beta1.0_Train.hdf5 \
        --output data/finetune/darcy_flow_beta1.0.h5
"""

import argparse
import h5py
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, required=True)
    p.add_argument('--output', type=str, required=True)
    p.add_argument('--max_samples', type=int, default=None,
                   help='Limit number of samples (default: all)')
    return p.parse_args()


def main():
    args = parse_args()

    with h5py.File(args.input, 'r') as fin:
        n_total = fin['nu'].shape[0]
        n_samples = min(args.max_samples, n_total) if args.max_samples else n_total
        beta = float(fin.attrs.get('beta', 0.0))
        nx = fin['nu'].shape[1]  # 128
        ny = fin['nu'].shape[2]  # 128
        T = 1  # steady state
        C_vec = 3
        C_scl = 2  # [a(x), u(x)]

        print(f"Input: {args.input}")
        print(f"  beta={beta}, N={n_total}, T={T}, H={nx}, W={ny}")
        print(f"  C_vec={C_vec} (zeros), C_scl={C_scl} (a(x), u(x))")
        print(f"  Using {n_samples} samples")

        x_coord = np.array(fin['x-coordinate'][:])
        y_coord = np.array(fin['y-coordinate'][:])

        with h5py.File(args.output, 'w') as fout:
            # Create datasets
            vec_ds = fout.create_dataset(
                'vector', shape=(n_samples, T, nx, ny, C_vec),
                dtype='float32', chunks=(1, T, nx, ny, C_vec),
            )
            scl_ds = fout.create_dataset(
                'scalar', shape=(n_samples, T, nx, ny, C_scl),
                dtype='float32', chunks=(1, T, nx, ny, C_scl),
            )

            # Write data in chunks
            chunk_size = 100
            for start in range(0, n_samples, chunk_size):
                end = min(start + chunk_size, n_samples)
                batch_a = np.array(fin['nu'][start:end], dtype=np.float32)   # (B, 128, 128)
                batch_u = np.array(fin['tensor'][start:end, 0], dtype=np.float32)  # (B, 128, 128)

                # vector: zeros
                vec_ds[start:end] = 0.0

                # scalar: [a(x), u(x)]
                scl_ds[start:end, 0, :, :, 0] = batch_a
                scl_ds[start:end, 0, :, :, 1] = batch_u

                print(f"  Written samples {start}-{end-1}")

            # Metadata
            # scalar_indices: 8=geometry(a(x) coeff), 12=pressure(u(x) solution)
            fout.create_dataset('scalar_indices', data=np.array([8, 12], dtype=np.int64))
            fout.create_dataset('x-coordinate', data=x_coord)
            fout.create_dataset('y-coordinate', data=y_coord)
            fout.attrs['beta'] = beta
            fout.attrs['equation'] = 'darcy_flow'
            fout.attrs['forcing'] = beta  # f = beta
            fout.attrs['bc'] = 'dirichlet_zero'
            fout.attrs['nx'] = nx
            fout.attrs['ny'] = ny

    print(f"\nOutput: {args.output}")
    print(f"  vector: ({n_samples}, {T}, {nx}, {ny}, {C_vec}) — zeros")
    print(f"  scalar: ({n_samples}, {T}, {nx}, {ny}, {C_scl}) — [a(x), u(x)]")
    print(f"  scalar_indices: [8, 12] (8=geometry, 12=pressure)")
    print(f"  forcing: f = {beta} (= beta)")


if __name__ == '__main__':
    main()
