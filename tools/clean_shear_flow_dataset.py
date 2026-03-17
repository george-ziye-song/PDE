"""
Clean Shear Flow dataset: remove bad samples only (keep ALL timesteps).

Based on spectral PDE residual analysis (FFT spatial + 4th-order temporal):
  - Samples 5, 13, 17, 21, 28: mean PDE > 1e-3, late period > 1e-4 (data quality issue)
  - Sample 6: exact duplicate of sample 0

Usage:
    python tools/clean_shear_flow_dataset.py \
        --input ./data/finetune/shear_flow_train.h5 \
        --output ./data/finetune/shear_flow_clean.h5 \
        --remove 5 6 13 17 21 28

    # Preview only (no write)
    python tools/clean_shear_flow_dataset.py \
        --input ./data/finetune/shear_flow_train.h5 --dry_run \
        --remove 5 6 13 17 21 28
"""

import argparse
import h5py
import numpy as np
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Clean Shear Flow H5 dataset")
    p.add_argument('--input', type=str, required=True, help='Input H5 file')
    p.add_argument('--output', type=str, default=None,
                   help='Output H5 file (default: input with _clean suffix)')
    p.add_argument('--remove', type=int, nargs='+',
                   default=[5, 6, 13, 17, 21, 28],
                   help='Sample indices to remove')
    p.add_argument('--dry_run', action='store_true',
                   help='Print summary without writing')
    return p.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    if args.output is None:
        output_path = input_path.parent / (input_path.stem + '_clean.h5')
    else:
        output_path = Path(args.output)

    remove_set = set(args.remove)

    with h5py.File(input_path, 'r') as f:
        n_samples = f['vector'].shape[0]
        n_timesteps = f['vector'].shape[1]
        spatial_shape = f['vector'].shape[2:]  # (H, W, 3)

        keep_indices = [i for i in range(n_samples) if i not in remove_set]

        print(f"{'=' * 60}")
        print(f"  Shear Flow Dataset Cleaning (samples only)")
        print(f"{'=' * 60}")
        print(f"  Input:       {input_path}")
        print(f"  Output:      {output_path}")
        print(f"  Original:    {n_samples} samples, {n_timesteps} timesteps")
        print(f"  Remove:      {sorted(remove_set)} ({len(remove_set)} samples)")
        print(f"  Keep:        {len(keep_indices)} samples, ALL {n_timesteps} timesteps")
        print(f"{'=' * 60}")

        if args.dry_run:
            print(f"\n  [DRY RUN] No file written.")
            print(f"\n  Sample mapping (old → new):")
            for new_idx, old_idx in enumerate(keep_indices):
                print(f"    {old_idx:2d} → {new_idx:2d}")
            return

        print(f"\n  Writing cleaned dataset...")

        with h5py.File(output_path, 'w') as out:
            # vector: [N_keep, T, H, W, 3]
            vec_src = f['vector']
            new_n = len(keep_indices)
            vec_shape = (new_n,) + vec_src.shape[1:]
            vec_dst = out.create_dataset('vector', shape=vec_shape, dtype=vec_src.dtype)

            for new_idx, old_idx in enumerate(keep_indices):
                vec_dst[new_idx] = vec_src[old_idx]
                if (new_idx + 1) % 5 == 0:
                    print(f"    vector: {new_idx + 1}/{new_n}")

            print(f"    vector: {vec_shape}")

            # scalar
            if 'scalar' in f and f['scalar'].shape[-1] > 0:
                scl_src = f['scalar']
                scl_shape = (new_n,) + scl_src.shape[1:]
                scl_dst = out.create_dataset('scalar', shape=scl_shape, dtype=scl_src.dtype)

                for new_idx, old_idx in enumerate(keep_indices):
                    scl_dst[new_idx] = scl_src[old_idx]

                print(f"    scalar: {scl_shape}")

            # scalar_indices
            if 'scalar_indices' in f:
                out.create_dataset('scalar_indices', data=f['scalar_indices'][:])
                print(f"    scalar_indices: {f['scalar_indices'][:].tolist()}")

            # Copy other per-sample or global datasets
            skip_keys = {'vector', 'scalar', 'scalar_indices'}
            for key in f.keys():
                if key not in skip_keys:
                    data = f[key][:]
                    if data.shape[0] == n_samples:
                        out.create_dataset(key, data=data[keep_indices])
                        print(f"    {key}: {data[keep_indices].shape} (filtered)")
                    else:
                        out.create_dataset(key, data=data)
                        print(f"    {key}: {data.shape} (copied)")

    print(f"\n  Done! Saved to {output_path}")
    print(f"  New dataset: {len(keep_indices)} samples × {n_timesteps} timesteps")


if __name__ == '__main__':
    main()
