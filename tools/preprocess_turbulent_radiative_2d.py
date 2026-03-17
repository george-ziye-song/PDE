"""
Preprocess Turbulent Radiative Layer 2D HDF5 data to FinetuneDataset format.

Input: turbulent_radiative_layer_tcool_*.hdf5 files
  Each file contains:
    t0_fields/density  [8, 101, 128, 384]
    t0_fields/pressure [8, 101, 128, 384]
    t1_fields/velocity [8, 101, 128, 384, 2]
    attrs['tcool'] or scalars/tcool

Output: HDF5 with
  vector: (N_total, T, H, W, 3)  — [vx, vy, 0]
  scalar: (N_total, T, H, W, 2)  — [density, pressure]
  scalar_indices: [4, 12]         — density@ch7 (CFD density), pressure@ch15 (CFD pressure)
  nu: (N_total,)                  — tcool per sample

Usage:
    python tools/preprocess_turbulent_radiative_2d.py \
        --input_dir /scratch-share/SONG0304/finetune/turbulent_radiative_2d/data/train \
        --output /scratch-share/SONG0304/finetune/turbulent_radiative_2d.hdf5
"""

import argparse
import glob
import os

import h5py
import numpy as np
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Turbulent Radiative 2D -> HDF5")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing turbulent_radiative_layer_tcool_*.hdf5 files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output HDF5 path")
    parser.add_argument("--chunk_size", type=int, default=4,
                        help="Samples to load at once per file (default: 4)")
    args = parser.parse_args()

    src_files = sorted(glob.glob(
        os.path.join(args.input_dir, "turbulent_radiative_layer_tcool_*.hdf5")
    ))
    if not src_files:
        raise FileNotFoundError(f"No turbulent_radiative_layer_tcool_*.hdf5 in {args.input_dir}")

    print(f"Found {len(src_files)} source files in {args.input_dir}")

    # First pass: count total samples and verify shapes
    total_n = 0
    file_info: list[tuple[str, int, int, int, int, float]] = []
    for fp in src_files:
        with h5py.File(fp, "r") as f:
            n = f["t0_fields/density"].shape[0]
            T = f["t0_fields/density"].shape[1]
            Nx = f["t0_fields/density"].shape[2]
            Ny = f["t0_fields/density"].shape[3]
            # tcool can be in attrs or scalars group
            if "tcool" in f.attrs:
                tcool = float(f.attrs["tcool"])
            else:
                tcool = float(f["scalars/tcool"][()])
            file_info.append((fp, n, T, Nx, Ny, tcool))
            total_n += n
            print(f"  {Path(fp).name}: N={n}, T={T}, grid={Nx}x{Ny}, tcool={tcool:.4f}")

    T = file_info[0][2]
    Nx = file_info[0][3]
    Ny = file_info[0][4]
    print(f"\nTotal samples: {total_n}, T={T}, grid={Nx}x{Ny}")

    # Create output HDF5
    with h5py.File(args.output, "w") as dst:
        vec_ds = dst.create_dataset(
            "vector", shape=(total_n, T, Nx, Ny, 3), dtype="float32",
            chunks=(1, 1, Nx, Ny, 3),
        )
        sca_ds = dst.create_dataset(
            "scalar", shape=(total_n, T, Nx, Ny, 2), dtype="float32",
            chunks=(1, 1, Nx, Ny, 2),
        )
        dst.create_dataset("scalar_indices", data=np.array([4, 12], dtype=np.int64))
        nu_ds = dst.create_dataset("nu", shape=(total_n,), dtype=np.float64)

        offset = 0
        for fp, n, _, _, _, tcool in file_info:
            print(f"\nConverting {Path(fp).name} (samples {offset}..{offset + n - 1}, tcool={tcool:.4f})")

            with h5py.File(fp, "r") as src:
                for start in range(0, n, args.chunk_size):
                    end = min(start + args.chunk_size, n)
                    chunk_n = end - start

                    vel = src["t1_fields/velocity"][start:end].astype(np.float32)  # [chunk, T, Nx, Ny, 2]
                    vec = np.zeros((chunk_n, T, Nx, Ny, 3), dtype=np.float32)
                    vec[..., :2] = vel
                    vec_ds[offset + start:offset + end] = vec

                    dens = src["t0_fields/density"][start:end].astype(np.float32)  # [chunk, T, Nx, Ny]
                    pres = src["t0_fields/pressure"][start:end].astype(np.float32)
                    sca = np.stack([dens, pres], axis=-1)  # [chunk, T, Nx, Ny, 2]
                    sca_ds[offset + start:offset + end] = sca

                    nu_ds[offset + start:offset + end] = tcool

                    # Print value ranges for first chunk
                    if start == 0:
                        print(f"  vx  range: [{vel[..., 0].min():.6f}, {vel[..., 0].max():.6f}]")
                        print(f"  vy  range: [{vel[..., 1].min():.6f}, {vel[..., 1].max():.6f}]")
                        print(f"  density  range: [{dens.min():.6f}, {dens.max():.6f}]")
                        print(f"  pressure range: [{pres.min():.6f}, {pres.max():.6f}]")

                    print(f"  Wrote samples {offset + start}..{offset + end - 1}")

            offset += n

    # Verification
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    with h5py.File(args.output, "r") as f:
        for k in f.keys():
            ds = f[k]
            if hasattr(ds, "shape"):
                print(f"  {k}: shape={ds.shape}, dtype={ds.dtype}")
        nu = f["nu"][:]
        print(f"  nu: min={nu.min():.6e}, max={nu.max():.6e}, unique={len(np.unique(nu))}")
        print(f"  scalar_indices: {f['scalar_indices'][:]}")

    size_mb = os.path.getsize(args.output) / (1024 ** 2)
    print(f"\nOutput file: {args.output}")
    print(f"File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
