"""
UNet Baseline for Gray-Scott Reaction-Diffusion (train from scratch).

Same loss as LoRA version: BC loss + PDE loss + stochastic RMSE anchor.
Purpose: verify whether PDE loss being small early is a dataset property
or a model property.

Channel mapping (18-channel):
- Channel 5 = scalar[2] = concentration_u (A, activator)
- Channel 6 = scalar[3] = concentration_v (B, inhibitor)

Usage:
    torchrun --nproc_per_node=4 finetune/train_gray_scott_unet.py --config configs/finetune_gray_scott_unet.yaml
"""

import os
import sys
import warnings


def _is_main_process():
    return os.environ.get('LOCAL_RANK', '0') == '0'

IS_MAIN_PROCESS = _is_main_process()

if not IS_MAIN_PROCESS:
    warnings.filterwarnings('ignore')

import argparse
import yaml
import torch
from pathlib import Path
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
)
from rich.table import Table
import logging

torch.set_float32_matmul_precision('high')

if IS_MAIN_PROCESS:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
else:
    logging.disable(logging.CRITICAL)

from finetune.dataset_finetune import (
    FinetuneDataset, FinetuneSampler, finetune_collate_fn,
    create_finetune_dataloaders, TOTAL_CHANNELS
)
from finetune.pde_loss_verified import GrayScottPDELoss
from finetune.unet_gray_scott import UNetGrayScott, CH_A, CH_B

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="UNet Baseline for Gray-Scott")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_lr_scheduler(optimizer, config: dict, total_steps: int):
    from torch.optim.lr_scheduler import LambdaLR
    import math

    warmup_steps = config['training'].get('warmup_steps', 100)
    min_lr = config['training'].get('min_lr', 1e-6)
    base_lr = config['training']['learning_rate']
    min_lr_ratio = min_lr / base_lr

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step / warmup_steps, 1e-2)  # avoid zero lr at step 0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def compute_boundary_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    channel_mask: torch.Tensor,
) -> torch.Tensor:
    """Boundary RMSE on 4 edges (supervised anchor for periodic domain)."""
    if channel_mask.dim() == 1:
        valid_ch = torch.where(channel_mask > 0)[0]
    else:
        valid_ch = torch.where(channel_mask[0] > 0)[0]

    left_pred = pred[:, :, :, 0, :][:, :, :, valid_ch]
    left_target = target[:, :, :, 0, :][:, :, :, valid_ch]
    right_pred = pred[:, :, :, -1, :][:, :, :, valid_ch]
    right_target = target[:, :, :, -1, :][:, :, :, valid_ch]
    bottom_pred = pred[:, :, 0, :, :][:, :, :, valid_ch]
    bottom_target = target[:, :, 0, :, :][:, :, :, valid_ch]
    top_pred = pred[:, :, -1, :, :][:, :, :, valid_ch]
    top_target = target[:, :, -1, :, :][:, :, :, valid_ch]

    bc_pred = torch.cat([
        left_pred.reshape(-1), right_pred.reshape(-1),
        bottom_pred.reshape(-1), top_pred.reshape(-1),
    ])
    bc_target = torch.cat([
        left_target.reshape(-1), right_target.reshape(-1),
        bottom_target.reshape(-1), top_target.reshape(-1),
    ])

    mse = torch.mean((bc_pred - bc_target) ** 2)
    return torch.sqrt(mse + 1e-8)


def compute_pde_loss(
    output: torch.Tensor,
    input_data: torch.Tensor,
    pde_loss_fn: GrayScottPDELoss,
) -> tuple:
    """Gray-Scott PDE residual loss. Prepends t0 for time derivative."""
    with torch.autocast(device_type='cuda', enabled=False):
        t0_A = input_data[:, 0:1, :, :, CH_A].float()
        t0_B = input_data[:, 0:1, :, :, CH_B].float()

        out_A = output[:, :, :, :, CH_A].float()
        out_B = output[:, :, :, :, CH_B].float()

        A = torch.cat([t0_A, out_A], dim=1)
        B = torch.cat([t0_B, out_B], dim=1)

        total_loss, losses = pde_loss_fn(A, B)

    return total_loss, losses


@torch.no_grad()
def validate(
    model, val_loader, accelerator,
    pde_loss_fn: GrayScottPDELoss,
    t_input: int = 8,
):
    """Validate and return (bc_loss, pde_loss, rmse)."""
    accelerator.wait_for_everyone()
    model.eval()

    total_bc = torch.zeros(1, device=accelerator.device)
    total_pde = torch.zeros(1, device=accelerator.device)
    total_rmse = torch.zeros(1, device=accelerator.device)
    num_batches = torch.zeros(1, device=accelerator.device)

    for batch in val_loader:
        data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
        channel_mask = batch['channel_mask'].to(device=accelerator.device)

        input_data = data[:, :t_input]
        target_data = data[:, 1:t_input + 1]

        output = model(input_data)

        bc_loss = compute_boundary_loss(output, target_data, channel_mask)
        pde_loss, _ = compute_pde_loss(output, input_data, pde_loss_fn)

        valid_ch = (torch.where(channel_mask[0] > 0)[0] if channel_mask.dim() > 1
                    else torch.where(channel_mask > 0)[0])
        mse = torch.mean((output[..., valid_ch] - target_data[..., valid_ch]) ** 2)
        rmse = torch.sqrt(mse + 1e-8)

        total_bc += bc_loss.detach()
        total_pde += pde_loss.detach()
        total_rmse += rmse.detach()
        num_batches += 1

    accelerator.wait_for_everyone()
    total_bc = accelerator.reduce(total_bc, reduction='sum')
    total_pde = accelerator.reduce(total_pde, reduction='sum')
    total_rmse = accelerator.reduce(total_rmse, reduction='sum')
    num_batches = accelerator.reduce(num_batches, reduction='sum')

    accelerator.wait_for_everyone()
    model.train()

    n = num_batches.item()
    return (
        (total_bc / num_batches).item() if n > 0 else 0,
        (total_pde / num_batches).item() if n > 0 else 0,
        (total_rmse / num_batches).item() if n > 0 else 0,
    )


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config['dataset']['seed'])

    max_epochs = config['training'].get('max_epochs', 30)
    warmup_steps = config['training'].get('warmup_steps', 100)
    log_interval = config['logging']['log_interval']
    lambda_bc = config['training'].get('lambda_bc', 1000.0)
    lambda_pde = config['training'].get('lambda_pde', 1.0)
    lambda_rmse = config['training'].get('lambda_rmse', 1.0)
    rmse_prob = config['training'].get('rmse_prob', 0.0)
    grad_clip = config['training'].get('grad_clip', 1.0)
    eval_interval = config['training'].get('eval_interval', 200)
    early_stopping_patience = config['training'].get('early_stopping_patience', 30)
    t_input = config['dataset'].get('t_input', 8)

    rmse_every_n = max(1, int(1.0 / rmse_prob)) if rmse_prob > 0 else 0

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision=config['training'].get('mixed_precision', 'no'),
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        log_with="wandb",
        kwargs_handlers=[ddp_kwargs]
    )

    if accelerator.is_main_process:
        logger.info(f"{'='*60}")
        logger.info("Gray-Scott UNet Baseline (from scratch)")
        logger.info(f"{'='*60}")
        logger.info(f"Lambda BC: {lambda_bc}, PDE: {lambda_pde}")
        logger.info(f"Stochastic RMSE: prob={rmse_prob} (every {rmse_every_n} steps)")
        logger.info(f"{'='*60}")

    temporal_length = t_input + 1

    train_loader, val_loader, train_sampler, val_sampler = create_finetune_dataloaders(
        data_path=config['dataset']['path'],
        batch_size=config['dataloader']['batch_size'],
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory'],
        seed=config['dataset']['seed'],
        temporal_length=temporal_length,
        train_ratio=config['dataset'].get('train_ratio', 0.9),
        clips_per_sample=config['dataset'].get('clips_per_sample', None),
        vector_dim=config['dataset'].get('vector_dim', 0),
        val_time_interval=config['dataset'].get('val_time_interval', 2),
    )

    steps_per_epoch = len(train_sampler)
    total_steps = max_epochs * steps_per_epoch

    if accelerator.is_main_process:
        logger.info(f"Steps per epoch: {steps_per_epoch}, Total: {total_steps}")

    # Build UNet from scratch
    init_features = config.get('model', {}).get('init_features', 32)
    model = UNetGrayScott(init_features=init_features).float()

    num_params = sum(p.numel() for p in model.parameters())
    if accelerator.is_main_process:
        logger.info(f"UNet parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=tuple(config['training'].get('betas', [0.9, 0.999]))
    )

    scheduler = get_lr_scheduler(optimizer, config, total_steps)
    model, optimizer = accelerator.prepare(model, optimizer)

    # PDE loss
    physics = config.get('physics', {})
    pde_loss_fn = GrayScottPDELoss(
        nx=physics.get('nx', 128),
        ny=physics.get('ny', 128),
        dx=physics.get('dx', 0.015748031),
        dy=physics.get('dy', 0.015748031),
        dt=physics.get('dt', 10.0),
        F=physics.get('F', 0.098),
        k=physics.get('k', 0.057),
        D_A=physics.get('D_A', 1.81e-5),
        D_B=physics.get('D_B', 1.39e-5),
    )

    global_step = 0
    best_val_loss = float('inf')
    patience_counter = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        accelerator.unwrap_model(model).load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        global_step = ckpt.get('global_step', 0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        patience_counter = ckpt.get('patience_counter', 0)
        if accelerator.is_main_process:
            logger.info(f"Resumed from step {global_step}")

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config['logging']['project'],
            config=config,
            init_kwargs={"wandb": {
                "entity": config['logging'].get('entity'),
                "name": "gray_scott-unet-baseline",
                "tags": ["gray_scott", "unet", "baseline", "from_scratch"],
            }}
        )

    save_dir = Path(config['logging']['save_dir'])
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)

    console = Console()
    early_stop = False
    start_epoch = global_step // steps_per_epoch if steps_per_epoch > 0 else 0

    model.train()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=not accelerator.is_main_process,
    ) as progress:
        train_task = progress.add_task("UNet Training", total=total_steps, completed=global_step)

        for epoch in range(start_epoch, max_epochs):
            if early_stop:
                break

            train_sampler.set_epoch(epoch)
            epoch_bc = 0.0
            epoch_pde = 0.0
            epoch_steps = 0

            for batch in train_loader:
                if early_stop:
                    break

                data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
                channel_mask = batch['channel_mask'].to(device=accelerator.device)

                input_data = data[:, :t_input]
                target_data = data[:, 1:t_input + 1]
                in_warmup = global_step < warmup_steps

                with accelerator.accumulate(model):
                    output = model(input_data)

                    bc_loss = compute_boundary_loss(output, target_data, channel_mask)
                    pde_loss, losses = compute_pde_loss(output, input_data, pde_loss_fn)

                    loss = lambda_bc * bc_loss + lambda_pde * pde_loss

                    # Stochastic RMSE anchor
                    use_rmse_this_step = rmse_every_n > 0 and (global_step % rmse_every_n == 0)
                    if use_rmse_this_step:
                        valid_ch = (torch.where(channel_mask[0] > 0)[0] if channel_mask.dim() > 1
                                    else torch.where(channel_mask > 0)[0])
                        rmse_loss = torch.sqrt(
                            torch.mean((output[..., valid_ch] - target_data[..., valid_ch]) ** 2) + 1e-8
                        )
                        loss = loss + lambda_rmse * rmse_loss

                    accelerator.backward(loss)
                    if grad_clip > 0:
                        accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                scheduler.step()
                global_step += 1
                epoch_steps += 1

                bc_reduced = accelerator.reduce(bc_loss.detach(), reduction='mean')
                pde_reduced = accelerator.reduce(pde_loss.detach(), reduction='mean')
                epoch_bc += bc_reduced.item()
                epoch_pde += pde_reduced.item()

                phase_str = "[warmup]" if in_warmup else f"[E{epoch+1}]"
                progress.update(
                    train_task, advance=1,
                    description=(
                        f"{phase_str} BC={bc_reduced.item():.4f} "
                        f"PDE={pde_reduced.item():.4f}"
                    )
                )

                if global_step % log_interval == 0:
                    accelerator.log({
                        'train/bc_loss': bc_reduced.item(),
                        'train/pde_loss': pde_reduced.item(),
                        'train/total_loss': loss.item(),
                        'train/lr': scheduler.get_last_lr()[0],
                        'train/epoch': epoch + 1,
                    }, step=global_step)

                if global_step % eval_interval == 0:
                    val_bc, val_pde, val_rmse = validate(
                        model, val_loader, accelerator, pde_loss_fn, t_input,
                    )

                    accelerator.log({
                        'val/bc_loss': val_bc,
                        'val/pde_loss': val_pde,
                        'val/rmse': val_rmse,
                    }, step=global_step)

                    if accelerator.is_main_process:
                        console.print(
                            f"\n[green]Step {global_step}/{total_steps} {phase_str}:[/green] "
                            f"val_bc={val_bc:.6f}, val_pde={val_pde:.6f}, val_rmse={val_rmse:.6f}"
                        )

                    if not in_warmup:
                        save_metric = val_rmse
                        if save_metric < best_val_loss:
                            best_val_loss = save_metric
                            patience_counter = 0
                            if accelerator.is_main_process:
                                torch.save({
                                    'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'scheduler_state_dict': scheduler.state_dict(),
                                    'global_step': global_step,
                                    'best_val_loss': best_val_loss,
                                    'patience_counter': patience_counter,
                                    'config': config,
                                }, str(save_dir / 'best_unet.pt'))
                                console.print(
                                    f"[yellow]Saved best model[/yellow] (rmse={val_rmse:.6f})"
                                )
                        else:
                            patience_counter += 1
                            if accelerator.is_main_process:
                                console.print(
                                    f"[dim]Patience: {patience_counter}/{early_stopping_patience}[/dim]"
                                )
                            if patience_counter >= early_stopping_patience:
                                if accelerator.is_main_process:
                                    console.print("[red]Early stopping![/red]")
                                early_stop = True
                                break
                    else:
                        if accelerator.is_main_process:
                            console.print("[dim](warmup - no saving)[/dim]")

                    model.train()

            if epoch_steps > 0 and accelerator.is_main_process:
                avg_bc = epoch_bc / epoch_steps
                avg_pde = epoch_pde / epoch_steps
                console.print(
                    f"\n[blue]Epoch {epoch+1}/{max_epochs}:[/blue] "
                    f"avg_bc={avg_bc:.6f}, avg_pde={avg_pde:.6f}"
                )

    accelerator.end_training()

    if accelerator.is_main_process:
        table = Table(title="Training Complete", show_header=False, border_style="green")
        table.add_row("Total Epochs", str(epoch + 1))
        table.add_row("Total Steps", str(global_step))
        table.add_row("Best Val RMSE", f"{best_val_loss:.6f}")
        table.add_row("Early Stopped", "Yes" if early_stop else "No")
        table.add_row("Checkpoint", str(save_dir / "best_unet.pt"))
        console.print(table)


if __name__ == "__main__":
    main()
