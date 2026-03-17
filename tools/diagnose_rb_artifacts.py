"""
Diagnose Rayleigh-Benard prediction artifacts:
1. Patch boundary discontinuity (16-pixel grid pattern)
2. Corner/edge value analysis
3. Spatial error distribution

Saves results to NPZ for secondary analysis.

Usage:
    python tools/diagnose_rb_artifacts.py \
        --config configs/finetune_rayleigh_benard_v3.yaml \
        --checkpoint checkpoints_rayleigh_benard_lora_v3/best_lora.pt
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

from finetune.dataset_finetune import FinetuneDataset, finetune_collate_fn
from finetune.model_lora_v3 import PDELoRAModelV3, load_lora_checkpoint

CH_VX = 0
CH_VY = 1
CH_BUOY = 3
CH_PRESS = 15


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--output', type=str, default='./diagnose_rb_artifacts.npz')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t_input = config['dataset'].get('t_input', 8)

    # Load model
    model = PDELoRAModelV3(
        config=config,
        pretrained_path=config['model'].get('pretrained_path'),
        freeze_encoder=False, freeze_decoder=False,
    )
    load_lora_checkpoint(model, args.checkpoint)
    model = model.float().to(device).eval()

    # Load validation data
    dataset = FinetuneDataset(
        data_path=config['dataset']['path'],
        temporal_length=t_input + 1,
        split='val',
        train_ratio=config['dataset'].get('train_ratio', 0.9),
        seed=config['dataset'].get('seed', 42),
        clips_per_sample=None,
        vector_dim=config['dataset'].get('vector_dim', 2),
        val_time_interval=config['dataset'].get('val_time_interval', 20),
    )

    n_samples = min(args.num_samples, len(dataset))
    patch_size = config['model'].get('patch_size', 16)

    all_errors = []
    all_pred_buoy = []
    all_gt_buoy = []
    all_pred_press = []
    all_gt_press = []
    patch_boundary_curvatures = []
    interior_curvatures = []

    print(f"Running inference on {n_samples} clips...")

    with torch.no_grad():
        for idx in range(n_samples):
            sample = dataset[idx]
            batch = finetune_collate_fn([sample])
            data = batch['data'].to(device=device, dtype=torch.float32)

            input_data = data[:, :t_input]
            target_data = data[:, 1:t_input + 1]

            output_norm, mean, std = model(input_data, return_normalized=True)
            output = output_norm * std + mean

            # Last timestep
            pred = output[0, -1].cpu().numpy()   # [H, W, 18]
            gt = target_data[0, -1].cpu().numpy()

            pred_buoy = pred[:, :, CH_BUOY]
            gt_buoy = gt[:, :, CH_BUOY]
            pred_press = pred[:, :, CH_PRESS]
            gt_press = gt[:, :, CH_PRESS]
            err_buoy = pred_buoy - gt_buoy

            all_pred_buoy.append(pred_buoy)
            all_gt_buoy.append(gt_buoy)
            all_pred_press.append(pred_press)
            all_gt_press.append(gt_press)
            all_errors.append(err_buoy)

            # Patch boundary analysis (horizontal direction, along W axis)
            P = patch_size
            for row in range(pred_buoy.shape[0]):
                line = pred_buoy[row, :]
                for b in range(P, pred_buoy.shape[1] - P, P):
                    # Curvature at boundary: |f[b-1] - 2*f[b] + f[b+1]|
                    curv_boundary = abs(line[b-1] - 2*line[b] + line[b+1])
                    patch_boundary_curvatures.append(curv_boundary)
                    # Curvature at interior (midpoint of patch)
                    mid = b - P // 2
                    curv_interior = abs(line[mid-1] - 2*line[mid] + line[mid+1])
                    interior_curvatures.append(curv_interior)

            if idx < 3:
                print(f"  Clip {idx}: buoy err MAE={np.mean(np.abs(err_buoy)):.6f}, "
                      f"pred range=[{pred_buoy.min():.4f}, {pred_buoy.max():.4f}]")

    all_errors = np.array(all_errors)           # [N, H, W]
    all_pred_buoy = np.array(all_pred_buoy)
    all_gt_buoy = np.array(all_gt_buoy)
    all_pred_press = np.array(all_pred_press)
    all_gt_press = np.array(all_gt_press)

    # === Analysis 1: Patch boundary curvature vs interior ===
    pb_curv = np.array(patch_boundary_curvatures)
    int_curv = np.array(interior_curvatures)
    print(f"\n{'='*60}")
    print("Patch Boundary Curvature Analysis (horizontal)")
    print(f"{'='*60}")
    print(f"  Boundary curvature: mean={pb_curv.mean():.6e}, median={np.median(pb_curv):.6e}")
    print(f"  Interior curvature: mean={int_curv.mean():.6e}, median={np.median(int_curv):.6e}")
    print(f"  Ratio (boundary/interior): {pb_curv.mean() / (int_curv.mean() + 1e-10):.2f}x")

    # === Analysis 2: Error by spatial region ===
    H, W = all_errors.shape[1], all_errors.shape[2]
    P = patch_size

    # Error at patch boundaries vs interior
    boundary_mask = np.zeros((H, W), dtype=bool)
    for i in range(P, H, P):
        boundary_mask[max(0,i-1):i+1, :] = True  # horizontal boundaries
    for j in range(P, W, P):
        boundary_mask[:, max(0,j-1):j+1] = True  # vertical boundaries

    err_at_boundary = all_errors[:, boundary_mask].flatten()
    err_at_interior = all_errors[:, ~boundary_mask].flatten()
    print(f"\n{'='*60}")
    print("Error at Patch Boundaries vs Interior")
    print(f"{'='*60}")
    print(f"  Boundary MAE: {np.mean(np.abs(err_at_boundary)):.6e}")
    print(f"  Interior MAE: {np.mean(np.abs(err_at_interior)):.6e}")
    print(f"  Ratio: {np.mean(np.abs(err_at_boundary)) / (np.mean(np.abs(err_at_interior)) + 1e-10):.2f}x")

    # === Analysis 3: Corner/edge values ===
    print(f"\n{'='*60}")
    print("Corner/Edge Value Analysis (pred buoyancy, last timestep)")
    print(f"{'='*60}")
    for si in range(min(3, len(all_pred_buoy))):
        pb = all_pred_buoy[si]
        gb = all_gt_buoy[si]
        print(f"  Sample {si}:")
        print(f"    Top-left     pred={pb[0,0]:.4f}, gt={gb[0,0]:.4f}, err={pb[0,0]-gb[0,0]:.4f}")
        print(f"    Top-right    pred={pb[0,-1]:.4f}, gt={gb[0,-1]:.4f}, err={pb[0,-1]-gb[0,-1]:.4f}")
        print(f"    Bottom-left  pred={pb[-1,0]:.4f}, gt={gb[-1,0]:.4f}, err={pb[-1,0]-gb[-1,0]:.4f}")
        print(f"    Bottom-right pred={pb[-1,-1]:.4f}, gt={gb[-1,-1]:.4f}, err={pb[-1,-1]-gb[-1,-1]:.4f}")
        # Edge means
        print(f"    Top row mean:    pred={pb[0,:].mean():.4f}, gt={gb[0,:].mean():.4f}")
        print(f"    Bottom row mean: pred={pb[-1,:].mean():.4f}, gt={gb[-1,:].mean():.4f}")
        print(f"    Left col mean:   pred={pb[:,0].mean():.4f}, gt={gb[:,0].mean():.4f}")
        print(f"    Right col mean:  pred={pb[:,-1].mean():.4f}, gt={gb[:,-1].mean():.4f}")

    # === Analysis 4: Row-by-row error (y-direction pattern) ===
    mean_abs_err_by_row = np.mean(np.abs(all_errors), axis=(0, 2))  # [H]
    print(f"\n{'='*60}")
    print("Error by Row (y-direction)")
    print(f"{'='*60}")
    for r in [0, 1, 2, P-1, P, P+1, H//4, H//2, 3*H//4, H-P-1, H-P, H-1]:
        if r < H:
            print(f"  Row {r:3d}: MAE={mean_abs_err_by_row[r]:.6e}")

    # === Analysis 5: GT data boundary values ===
    print(f"\n{'='*60}")
    print("GT Boundary Values (Dirichlet check)")
    print(f"{'='*60}")
    for si in range(min(3, len(all_gt_buoy))):
        gb = all_gt_buoy[si]
        print(f"  Sample {si}: top_row_mean={gb[0,:].mean():.4f}, bottom_row_mean={gb[-1,:].mean():.4f}")

    # === Analysis 6: Per-timestep error (are later timesteps worse?) ===
    print(f"\n{'='*60}")
    print("Per-timestep Error (sample 0)")
    print(f"{'='*60}")
    sample = dataset[0]
    batch = finetune_collate_fn([sample])
    data = batch['data'].to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output_norm, mean, std = model(data[:, :t_input], return_normalized=True)
        output = output_norm * std + mean
    target = data[:, 1:t_input+1]
    for t in range(t_input):
        pred_t = output[0, t, :, :, CH_BUOY].cpu().numpy()
        gt_t = target[0, t, :, :, CH_BUOY].cpu().numpy()
        mae = np.mean(np.abs(pred_t - gt_t))
        rmse = np.sqrt(np.mean((pred_t - gt_t)**2))
        print(f"  t={t}: MAE={mae:.6e}, RMSE={rmse:.6e}")

    # Save
    np.savez(args.output,
             errors=all_errors,
             pred_buoy=all_pred_buoy, gt_buoy=all_gt_buoy,
             pred_press=all_pred_press, gt_press=all_gt_press,
             boundary_curvatures=pb_curv, interior_curvatures=int_curv,
             err_by_row=mean_abs_err_by_row)
    print(f"\nSaved: {args.output}")


if __name__ == '__main__':
    main()
