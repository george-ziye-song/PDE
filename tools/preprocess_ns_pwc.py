"""
Preprocess POSEIDON NS-PwC NetCDF data to our HDF5 format.

Input: velocity_*.nc files (sample, time, channel, x, y) with channels [u_x, u_y, tracer]
Output: HDF5 with vector (N, T, H, W, 3) and scalar (N, T, H, W, 1)

NetCDF4 files are HDF5 under the hood — read directly with h5py to avoid
netCDF4 library version issues.

Channel mapping:
  vector[..., 0] = u_x, vector[..., 1] = u_y, vector[..., 2] = 0 (pad)
  scalar[..., 0] = tracer → scalar_indices=[11] → channel 14 in 18-ch layout (passive_tracer)

Usage:
    python tools/preprocess_ns_pwc.py \
        --input_dir /scratch-share/SONG0304/finetune/NS-PwC \
        --output /scratch-share/SONG0304/finetune/ns_pwc.hdf5
"""

import argparse
import h5py
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Preprocess NS-PwC NetCDF → HDF5")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing velocity_*.nc files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output HDF5 path")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Max number of .nc files to process")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    nc_files = sorted(input_dir.glob("velocity_*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No velocity_*.nc files in {input_dir}")

    if args.max_files:
        nc_files = nc_files[:args.max_files]
    print(f"Processing {len(nc_files)} files from {input_dir}")

    # Get dimensions from first file (NC4 is HDF5, read with h5py)
    with h5py.File(str(nc_files[0]), "r") as f:
        vel_shape = f["velocity"].shape  # (sample, time, channel, x, y)
        first_n, n_time, n_ch, nx, ny = vel_shape
        print(f"  Per-file: samples={first_n}, T={n_time}, Ch={n_ch}, Grid={nx}x{ny}")

    # Count total samples
    sample_counts = []
    for nc_path in nc_files:
        with h5py.File(str(nc_path), "r") as f:
            sample_counts.append(f["velocity"].shape[0])
    total_samples = sum(sample_counts)
    print(f"  Total samples: {total_samples}")

    # Create output HDF5
    with h5py.File(args.output, "w") as out:
        # vector: (N, T, H, W, 3) — u_x, u_y, 0
        vec_ds = out.create_dataset(
            "vector", shape=(total_samples, n_time, nx, ny, 3),
            dtype=np.float32, chunks=(1, n_time, nx, ny, 3),
        )
        # scalar: (N, T, H, W, 1) — tracer
        sca_ds = out.create_dataset(
            "scalar", shape=(total_samples, n_time, nx, ny, 1),
            dtype=np.float32, chunks=(1, n_time, nx, ny, 1),
        )
        # scalar_indices: tracer at scalar position 11 → channel 14 (passive_tracer)
        out.create_dataset("scalar_indices", data=np.array([11], dtype=np.int64))

        offset = 0
        for nc_path in nc_files:
            print(f"  Loading {nc_path.name}...")
            with h5py.File(str(nc_path), "r") as f:
                n_s = f["velocity"].shape[0]
                # velocity: (sample, time, channel, x, y)
                # Process in chunks to avoid memory issues
                chunk_size = 100
                for start in range(0, n_s, chunk_size):
                    end = min(start + chunk_size, n_s)
                    vel = f["velocity"][start:end].astype(np.float32)  # (chunk, T, C, X, Y)

                    # Separate channels
                    ux = vel[:, :, 0, :, :]       # (chunk, T, X, Y)
                    uy = vel[:, :, 1, :, :]
                    tracer = vel[:, :, 2, :, :]

                    chunk_n = end - start
                    # vector: (chunk, T, X, Y, 3)
                    vec_data = np.zeros((chunk_n, n_time, nx, ny, 3), dtype=np.float32)
                    vec_data[..., 0] = ux
                    vec_data[..., 1] = uy
                    # vec_data[..., 2] = 0 (Vz pad)

                    # scalar: (chunk, T, X, Y, 1)
                    sca_data = tracer[..., np.newaxis]

                    vec_ds[offset + start:offset + end] = vec_data
                    sca_ds[offset + start:offset + end] = sca_data

                    if start == 0:
                        print(f"    ux range: [{ux.min():.3f}, {ux.max():.3f}]")
                        print(f"    tracer range: [{tracer.min():.3f}, {tracer.max():.3f}]")

                offset += n_s

    print(f"\nSaved: {args.output}")
    print(f"  vector: ({total_samples}, {n_time}, {nx}, {ny}, 3)")
    print(f"  scalar: ({total_samples}, {n_time}, {nx}, {ny}, 1)")
    print(f"  scalar_indices: [0]")


if __name__ == "__main__":
    main()
