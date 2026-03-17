"""
Generate demo videos comparing GT vs autoregressive prediction.

Side-by-side comparison rendered with matplotlib, saved via torchvision.io.write_video.

Usage:
    # Gray-Scott (Species A, viridis)
    python tools/generate_demo_video.py \
        --dataset gray_scott \
        --config configs/finetune_gray_scott_v3.yaml \
        --checkpoint checkpoints_gray_scott_lora_v3/best_lora.pt \
        --output demo_gray_scott.mp4

    # Shear Flow (Tracer, RdBu_r)
    python tools/generate_demo_video.py \
        --dataset shear_flow \
        --config configs/finetune_shear_flow_v3.yaml \
        --checkpoint checkpoints_shear_flow_lora_v3/best_lora.pt \
        --output demo_shear_flow.mp4

    # Show a different field
    python tools/generate_demo_video.py \
        --dataset shear_flow --field vx ...
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

from torchvision.io import write_video

from finetune.dataset_finetune import FinetuneDataset, finetune_collate_fn
from finetune.model_lora_v3 import PDELoRAModelV3, load_lora_checkpoint


# ── field configs (matching official colormaps) ──────────────────────

FIELD_MAP: Dict[str, Dict] = {
    'gray_scott': {
        'A':        {'ch': 5,  'cmap': 'viridis', 'label': 'Species A'},
        'B':        {'ch': 6,  'cmap': 'viridis', 'label': 'Species B'},
    },
    'shear_flow': {
        'tracer':   {'ch': 14, 'cmap': 'RdBu_r',  'label': 'Tracer'},
        'vx':       {'ch': 0,  'cmap': 'RdBu_r',  'label': 'Velocity X'},
        'vy':       {'ch': 1,  'cmap': 'RdBu_r',  'label': 'Velocity Y'},
        'pressure': {'ch': 15, 'cmap': 'RdBu_r',  'label': 'Pressure'},
    },
}

DATASET_DEFAULTS = {
    'gray_scott':  {'field': 'A',      'vector_dim': 0},
    'shear_flow':  {'field': 'tracer', 'vector_dim': 2},
}


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate PDE demo video")
    p.add_argument('--dataset', required=True,
                   choices=['gray_scott', 'shear_flow'])
    p.add_argument('--config', required=True, help='YAML config path')
    p.add_argument('--checkpoint', required=True, help='LoRA checkpoint')
    p.add_argument('--output', default=None, help='Output .mp4 (default: demo_{dataset}.mp4)')
    p.add_argument('--field', default=None,
                   help='Field to visualize (gray_scott: A/B, shear_flow: tracer/vx/vy/pressure)')
    p.add_argument('--num_rollout', type=int, default=None,
                   help='Autoregressive rollout steps (default: all available)')
    p.add_argument('--fps', type=int, default=8)
    p.add_argument('--clip_idx', type=int, default=0,
                   help='Which val clip to use')
    p.add_argument('--t_input', type=int, default=None,
                   help='Override t_input from config')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


# ── helpers ──────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(config: dict, ckpt_path: str, device: torch.device):
    """Load LoRA V3 model."""
    model = PDELoRAModelV3(
        config=config,
        pretrained_path=config['model']['pretrained_path'],
        freeze_encoder=config.get('model', {}).get('freeze_encoder', False),
        freeze_decoder=config.get('model', {}).get('freeze_decoder', False),
    )
    ckpt = load_lora_checkpoint(model, ckpt_path)
    if 'metrics' in ckpt:
        print(f"  Checkpoint metrics: {ckpt['metrics']}")
    return model.float().to(device).eval()


@torch.no_grad()
def sliding_window_predict(
    model, gt_data: torch.Tensor, t_input: int, num_steps: int,
) -> torch.Tensor:
    """Sliding-window prediction: each step uses GT as input.

    For each step s, feeds gt[s : s+t_input] and takes the last predicted
    frame as the prediction for timestep s+t_input.  Predictions are NEVER
    fed back into the input — always conditioned on GT.

    Args:
        model: LoRA model (on device)
        gt_data: [1, t_input + num_steps, H, W, 18] full GT on device
        t_input: number of conditioning frames
        num_steps: number of prediction steps

    Returns:
        [1, num_steps, H, W, 18] predicted frames (on CPU)
    """
    preds = []
    for s in range(num_steps):
        window = gt_data[:, s:s + t_input]          # [1, t_input, H, W, 18]
        out_norm, mean, std = model(window, return_normalized=True)
        out = out_norm * std + mean
        preds.append(out[:, -1:].cpu())              # prediction for t = s + t_input
        if (s + 1) % 10 == 0:
            print(f"  step {s + 1}/{num_steps}")
    return torch.cat(preds, dim=1)


def render_frames(
    gt: np.ndarray,
    pred: np.ndarray,
    cmap_name: str,
    field_label: str,
    t_offset: int,
    suptitle: str,
) -> np.ndarray:
    """Render side-by-side GT vs Pred frames.

    Uses gridspec with dedicated colorbar axis to ensure perfect alignment.

    Args:
        gt:   [T, H, W] ground truth field
        pred: [T, H, W] predicted field
        cmap_name: matplotlib colormap
        field_label: display label for the field
        t_offset: starting timestep index (for display)
        suptitle: figure super-title

    Returns:
        [T, frame_H, frame_W, 3] uint8 video array
    """
    num_t, data_h, data_w = gt.shape
    vmin, vmax = float(gt.min()), float(gt.max())

    # figure size adapted to data aspect ratio
    data_aspect = data_h / data_w  # H/W
    panel_w = 5.0
    panel_h = panel_w * data_aspect
    fig_w = panel_w * 2 + 1.0       # 2 panels + colorbar + gaps
    fig_h = panel_h + 1.2           # panel + title/margin

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=120)
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1, 1, 0.04],
        left=0.03, right=0.92, bottom=0.05, top=0.88,
        wspace=0.08,
    )

    ax_gt = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[0, 1])
    ax_cb = fig.add_subplot(gs[0, 2])

    im_gt = ax_gt.imshow(gt[0], cmap=cmap_name, vmin=vmin, vmax=vmax, aspect='equal')
    im_pred = ax_pred.imshow(pred[0], cmap=cmap_name, vmin=vmin, vmax=vmax, aspect='equal')

    fig.colorbar(im_gt, cax=ax_cb)

    title_gt = ax_gt.set_title('', fontsize=11)
    title_pred = ax_pred.set_title('', fontsize=11)
    ax_gt.set_xticks([]); ax_gt.set_yticks([])
    ax_pred.set_xticks([]); ax_pred.set_yticks([])

    fig.suptitle(suptitle, fontsize=13, y=0.96)

    # first draw → measure canvas
    fig.canvas.draw()
    canvas_w, canvas_h = fig.canvas.get_width_height()
    frame_w = canvas_w - (canvas_w % 2)  # H264 needs even dims
    frame_h = canvas_h - (canvas_h % 2)

    frames = np.empty((num_t, frame_h, frame_w, 3), dtype=np.uint8)

    for i in range(num_t):
        im_gt.set_data(gt[i])
        im_pred.set_data(pred[i])
        rmse = float(np.sqrt(np.mean((pred[i] - gt[i]) ** 2)))
        title_gt.set_text(f'GT ({field_label})  t={i + t_offset}')
        title_pred.set_text(f'Pred ({field_label})  t={i + t_offset}  RMSE={rmse:.4f}')
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(canvas_h, canvas_w, 4)
        frames[i] = buf[:frame_h, :frame_w, :3]

    plt.close(fig)
    return frames


# ── main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = load_config(args.config)
    ds_name = args.dataset

    field_name = args.field or DATASET_DEFAULTS[ds_name]['field']
    if field_name not in FIELD_MAP[ds_name]:
        valid = ', '.join(FIELD_MAP[ds_name].keys())
        raise ValueError(f"Unknown field '{field_name}' for {ds_name}. Choose from: {valid}")

    fc = FIELD_MAP[ds_name][field_name]
    vector_dim = DATASET_DEFAULTS[ds_name]['vector_dim']
    t_input = args.t_input or cfg.get('dataset', {}).get('t_input', 8)
    out_path = args.output or f'demo_{ds_name}.mp4'

    # ── determine num_rollout ──
    data_path = cfg['dataset']['path']
    if args.num_rollout is None:
        import h5py
        with h5py.File(data_path, 'r') as f:
            if 'vector' in f:
                # New format: vector is [N, T, H, W, 3]
                total_t = f['vector'].shape[1]
            else:
                # Old format: per-sample groups
                key0 = list(f.keys())[0]
                grp = f[key0]
                if 'vector' in grp and 'data' in grp['vector']:
                    total_t = grp['vector']['data'].shape[0]
                else:
                    total_t = grp['data'].shape[0]
        num_rollout = total_t - t_input
        print(f"  Auto num_rollout: T={total_t}, t_input={t_input} → rollout={num_rollout}")
    else:
        num_rollout = args.num_rollout

    print(f"{'=' * 55}")
    print(f"  Dataset:    {ds_name}")
    print(f"  Field:      {fc['label']}  (ch={fc['ch']}, cmap={fc['cmap']})")
    print(f"  t_input:    {t_input}")
    print(f"  Rollout:    {num_rollout} steps")
    print(f"  FPS:        {args.fps}")
    print(f"  Output:     {out_path}")
    print(f"{'=' * 55}")

    # ── load val dataset with enough temporal length ──
    temporal_length = t_input + num_rollout
    val_ds = FinetuneDataset(
        data_path=cfg['dataset']['path'],
        temporal_length=temporal_length,
        split='val',
        train_ratio=cfg['dataset'].get('train_ratio', 0.9),
        seed=cfg['dataset'].get('seed', 42),
        clips_per_sample=None,
        vector_dim=vector_dim,
        val_time_interval=cfg['dataset'].get('val_time_interval', 8),
    )

    if len(val_ds) == 0:
        raise RuntimeError(
            f"No clips found with temporal_length={temporal_length}. "
            f"Try reducing --num_rollout."
        )

    clip_idx = min(args.clip_idx, len(val_ds) - 1)
    print(f"  Val clips available: {len(val_ds)}, using idx={clip_idx}")

    sample = val_ds[clip_idx]
    batch = finetune_collate_fn([sample])
    data = batch['data'].float()  # [1, temporal_length, H, W, 18]
    print(f"  Data shape: {list(data.shape)}")

    # ── model ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nLoading model → {device}")
    model = load_model(cfg, args.checkpoint, device)

    # ── sliding-window prediction (always conditioned on GT) ──
    print(f"\nSliding-window prediction ({num_rollout} steps)...")
    pred = sliding_window_predict(model, data.to(device), t_input, num_rollout)

    # extract field
    ch = fc['ch']
    gt_seq = data[0, t_input:t_input + num_rollout, :, :, ch].numpy()   # [T, H, W]
    pred_seq = pred[0, :, :, :, ch].numpy()                              # [T, H, W]
    print(f"  GT   range: [{gt_seq.min():.4f}, {gt_seq.max():.4f}]")
    print(f"  Pred range: [{pred_seq.min():.4f}, {pred_seq.max():.4f}]")

    # ── render ──
    print("\nRendering frames...")
    display_name = ds_name.replace('_', ' ').title()
    suptitle = f"{display_name} — Sliding-Window Prediction ({fc['label']})"
    frames = render_frames(gt_seq, pred_seq, fc['cmap'], fc['label'], t_input, suptitle)
    print(f"  Frame size: {frames.shape[1]}×{frames.shape[2]}")

    # ── write video ──
    print(f"\nWriting → {out_path}")
    video_tensor = torch.from_numpy(frames)  # [T, H, W, 3] uint8
    write_video(out_path, video_tensor, fps=args.fps)

    overall_rmse = float(np.sqrt(np.mean((pred_seq - gt_seq) ** 2)))
    duration = frames.shape[0] / args.fps
    print(f"\nDone!  {out_path}")
    print(f"  Frames:   {frames.shape[0]}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  RMSE:     {overall_rmse:.6f}")


if __name__ == '__main__':
    main()
