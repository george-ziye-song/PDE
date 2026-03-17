"""
Preprocess camlab-ethz Wave-Gauss (acoustic wave equation) NetCDF data to our HDF5 format.

Input:
  solution_0.nc, solution_1.nc, solution_2.nc — displacement u(x,y,t)
    variable "solution": shape (N_chunk, 16, 128, 128)
  c_0.nc — spatially varying wave speed (time-independent)
    variable "c": shape (N_total, 128, 128)

Output: HDF5 with vector (N, T, H, W, 3) and scalar (N, T, H, W, 2)

Channel mapping:
  vector[..., :] = 0 (no velocity fields, vector_dim=0)
  scalar[..., 0] = u (displacement)
  scalar[..., 1] = c (wave speed, replicated across T)
  scalar_indices = [0, 1]

Usage:
    python tools/preprocess_wave_gauss.py \
        --input_dir /scratch-share/SONG0304/finetune/Wave-Gauss \
        --output /scratch-share/SONG0304/finetune/wave_gauss.hdf5
"""

import argparse
import h5py
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Preprocess Wave-Gauss NetCDF → HDF5")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing solution_*.nc and c_0.nc")
    parser.add_argument("--output", type=str, required=True,
                        help="Output HDF5 path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    # Discover solution files
    sol_files = sorted(input_dir.glob("solution_*.nc"))
    if not sol_files:
        raise FileNotFoundError(f"No solution_*.nc files in {input_dir}")

    c_path = input_dir / "c_0.nc"
    if not c_path.exists():
        raise FileNotFoundError(f"Wave speed file not found: {c_path}")

    print(f"Processing {len(sol_files)} solution files from {input_dir}")

    # Get dimensions from first file (NC4 is HDF5, read with h5py)
    with h5py.File(str(sol_files[0]), "r") as f:
        sol_shape = f["solution"].shape  # (N_chunk, 15, 128, 128)
        first_n, n_time, nx, ny = sol_shape
        print(f"  Per-file: samples={first_n}, T={n_time}, Grid={nx}x{ny}")

    # Count total samples across solution files
    sample_counts: list[int] = []
    for sol_path in sol_files:
        with h5py.File(str(sol_path), "r") as f:
            sample_counts.append(f["solution"].shape[0])
    total_samples = sum(sample_counts)
    print(f"  Total samples: {total_samples}")

    # Verify wave speed file has matching sample count
    with h5py.File(str(c_path), "r") as f:
        c_total = f["c"].shape[0]
        print(f"  Wave speed samples: {c_total}")
        if c_total != total_samples:
            print(f"  WARNING: c samples ({c_total}) != solution samples ({total_samples})")

    # Create output HDF5
    with h5py.File(args.output, "w") as out:
        # vector: (N, T, H, W, 3) — all zeros (no velocity)
        vec_ds = out.create_dataset(
            "vector", shape=(total_samples, n_time, nx, ny, 3),
            dtype=np.float32, chunks=(1, n_time, nx, ny, 3),
            fillvalue=0.0,
        )
        # scalar: (N, T, H, W, 2) — u (displacement), c (wave speed)
        sca_ds = out.create_dataset(
            "scalar", shape=(total_samples, n_time, nx, ny, 2),
            dtype=np.float32, chunks=(1, n_time, nx, ny, 2),
        )
        # scalar_indices: displacement at 0, wave speed at 1
        out.create_dataset("scalar_indices", data=np.array([0, 1], dtype=np.int64))

        # Load wave speed field once (fits in memory: N_total x 128 x 128 float32)
        with h5py.File(str(c_path), "r") as f_c:
            c_all = f_c["c"][:].astype(np.float32)  # (N_total, 128, 128)
        print(f"  c range: [{c_all.min():.3f}, {c_all.max():.3f}]")

        offset = 0
        for sol_path in sol_files:
            print(f"  Loading {sol_path.name}...")
            with h5py.File(str(sol_path), "r") as f:
                n_s = f["solution"].shape[0]
                chunk_size = 100
                for start in range(0, n_s, chunk_size):
                    end = min(start + chunk_size, n_s)
                    chunk_n = end - start

                    # solution: (chunk, T, H, W)
                    u = f["solution"][start:end].astype(np.float32)

                    # scalar channel 0: displacement u → (chunk, T, H, W, 1)
                    # scalar channel 1: wave speed c → replicate across T
                    sca_data = np.zeros((chunk_n, n_time, nx, ny, 2), dtype=np.float32)
                    sca_data[..., 0] = u
                    # c is time-independent: (chunk, H, W) → broadcast to (chunk, T, H, W)
                    c_chunk = c_all[offset + start:offset + end]  # (chunk, H, W)
                    sca_data[..., 1] = c_chunk[:, np.newaxis, :, :]

                    # vector is all zeros — fillvalue handles it, skip writing
                    sca_ds[offset + start:offset + end] = sca_data

                    if start == 0:
                        print(f"    u range: [{u.min():.3f}, {u.max():.3f}]")
                        print(f"    c range (chunk): [{c_chunk.min():.3f}, {c_chunk.max():.3f}]")

                offset += n_s

    print(f"\nSaved: {args.output}")
    print(f"  vector: ({total_samples}, {n_time}, {nx}, {ny}, 3)  [all zeros]")
    print(f"  scalar: ({total_samples}, {n_time}, {nx}, {ny}, 2)  [u, c]")
    print(f"  scalar_indices: [0, 1]")


if __name__ == "__main__":
    main()
