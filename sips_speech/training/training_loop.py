# Copyright (c) 2026, Mitsubishi Electric Research Laboratories. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Main training loop."""

import time
from os.path import exists, join

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sips_speech.inference.pipeline import get_inference_iterable
from sips_speech.training.checkpoint import Checkpoint
from sips_speech.training.data import InfiniteSampler
from sips_speech.utils.config import EasyDict, call_func, construct_class
from sips_speech.utils.distributed import (
    check_ddp_consistency,
    ddp_sync,
    get_rank,
    get_world_size,
    print0,
    switch_backend,
)
from sips_speech.utils.misc import make_pbar, set_random_seed


def training_loop(
    dataset_kwargs=dict(class_name="sips_speech.training.data.Dataset", path=None),
    audio_encoder_kwargs=dict(class_name="sips_speech.models.encoder.SpectrogramEncoder"),
    data_loader_kwargs=dict(
        class_name="torch.utils.data.DataLoader", pin_memory=True, num_workers=2, prefetch_factor=2
    ),
    model_kwargs=dict(class_name="sips_speech.models.model.SIPS", freq_resolution=256, spec_channels=2),
    loss_kwargs=dict(class_name="sips_speech.training.loss.DenoiserLoss"),
    optimizer_kwargs=dict(class_name="torch.optim.Adam", betas=(0.9, 0.99)),
    lr_kwargs=dict(func_name="sips_speech.training.schedulers.learning_rate_schedule"),
    ema_kwargs=dict(class_name="sips_speech.third_party.edm2.PowerFunctionEMA"),
    run_dir=".",
    seed=0,
    batch_size=32,
    batch_gpu=None,
    total_nwav=1024 << 20,
    slice_nwav=None,
    snapshot_nwav=1024 << 10,
    checkpoint_nwav=2048 << 10,
    loss_scaling=1,
    force_finite=True,
    cudnn_benchmark=True,
    device=None,
    noval=False,
    valid_dir=None,
    valid_pred_dir=None,
):
    """Run the main model training loop.

    Args:
        dataset_kwargs: Keyword arguments used to construct the training dataset.
        audio_encoder_kwargs: Keyword arguments used to construct the audio encoder.
        data_loader_kwargs: Keyword arguments used to construct the PyTorch data loader.
        model_kwargs: Keyword arguments used to construct the model.
        loss_kwargs: Keyword arguments used to construct the training loss.
        optimizer_kwargs: Keyword arguments used to construct the optimizer.
        lr_kwargs: Keyword arguments for the learning-rate scheduling function.
        ema_kwargs: Keyword arguments used to construct the EMA object. Set to None to disable EMA.
        run_dir: Output directory for logs, checkpoints, snapshots, and validation outputs.
        seed: Global random seed.
        batch_size: Total batch size for one training iteration across all ranks.
        batch_gpu: Per-GPU batch size limit. If None, uses the largest valid per-rank batch size.
        total_nwav: Total number of training audio samples to process before stopping.
        slice_nwav: Maximum number of training audio samples for this invocation. If None, no limit.
        snapshot_nwav: Interval (in processed audio samples) for saving model snapshots. If None, disabled.
        checkpoint_nwav: Interval (in processed audio samples) for saving full training checkpoints. If None, disabled.
        loss_scaling: Loss scaling factor used to reduce FP16 underflow/overflow issues.
        force_finite: Replace NaN and Inf gradients with finite values before optimizer step.
        cudnn_benchmark: Enable torch.backends.cudnn.benchmark.
        device: Device specifier for training. If None, auto-select CUDA when available, else CPU.
        noval: Disable validation when True.
        valid_dir: Directory containing noisy validation WAV files.
        valid_pred_dir: Directory containing precomputed predictor WAV files for validation.
    """
    # Initialize.
    set_random_seed(seed, get_rank())
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(device)

    writer = SummaryWriter(join(run_dir, "logs")) if get_rank() == 0 else None

    batch_gpu = batch_gpu or None
    snapshot_nwav = snapshot_nwav or None
    checkpoint_nwav = checkpoint_nwav or None

    # Validate batch size.
    batch_gpu_total = batch_size // get_world_size()
    if batch_gpu_total == 0:
        raise ValueError(f"Batch size ({batch_size}) must be at least the world size ({get_world_size()}).")
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * get_world_size()
    assert total_nwav % batch_size == 0
    assert slice_nwav is None or slice_nwav % batch_size == 0
    assert snapshot_nwav is None or (snapshot_nwav % batch_size == 0)  # and snapshot_nwav % 1024 == 0)
    assert checkpoint_nwav is None or (checkpoint_nwav % batch_size == 0)  # and checkpoint_nwav % 1024 == 0)

    # Setup dataset, encoder, and network.
    dataset_obj = construct_class(**dataset_kwargs)
    audio_encoder = construct_class(**audio_encoder_kwargs)

    model = construct_class(**model_kwargs)
    model.train().requires_grad_(True).to(device)

    # Check if validation dirs exist.
    do_validation = (not noval) and valid_dir is not None
    if do_validation and not exists(valid_dir):
        raise FileNotFoundError(f"Validation path does not exist: {valid_dir}")
    if do_validation and valid_pred_dir is None:
        raise ValueError("Validation requires valid_pred_dir with precomputed predictor WAV files.")
    if do_validation and not exists(valid_pred_dir):
        raise FileNotFoundError(f"Validation predictor path does not exist: {valid_pred_dir}")

    # Setup training state.
    state = EasyDict(cur_nwav=0, total_elapsed_time=0)
    if dist.is_initialized():
        ddp_kwargs = {}
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [device]
        ddp = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
    else:
        ddp = model
    loss_fn = construct_class(**loss_kwargs, model=ddp)
    optimizer = construct_class(params=model.parameters(), **optimizer_kwargs)
    ema = construct_class(net=model, **ema_kwargs) if ema_kwargs is not None else None

    # Load previous checkpoint.
    checkpoint = Checkpoint(state=state, model=model, optimizer=optimizer, ema=ema)
    checkpoint_path = checkpoint.load_latest(
        join(run_dir, "checkpoints"),
        pattern=r"checkpoint-(\d+).pt",
    )
    if checkpoint_path is None:
        checkpoint.load_latest(run_dir)

    # Decide when to stop.
    stop_at_nwav = total_nwav
    if slice_nwav is not None:
        granularity = (
            checkpoint_nwav
            if checkpoint_nwav is not None
            else snapshot_nwav if snapshot_nwav is not None else batch_size
        )
        slice_end_nwav = (state.cur_nwav + slice_nwav) // granularity * granularity  # round down
        stop_at_nwav = min(stop_at_nwav, slice_end_nwav)
    assert stop_at_nwav > state.cur_nwav

    # Main training loop.
    dataset_sampler = InfiniteSampler(
        dataset=dataset_obj, rank=get_rank(), num_replicas=get_world_size(), seed=seed, start_idx=state.cur_nwav
    )
    dataset_iterator = iter(
        construct_class(
            dataset=dataset_obj,
            sampler=dataset_sampler,
            batch_size=batch_gpu,
            collate_fn=dataset_obj.collate_fn,
            **data_loader_kwargs,
        )
    )
    cumulative_training_time = 0
    start_nwav = state.cur_nwav

    if (
        snapshot_nwav is not None
        and state.cur_nwav % snapshot_nwav == 0
        and (state.cur_nwav != start_nwav or start_nwav == 0)
    ):
        pbar = None
    else:
        print0("Running training...")
        pbar = make_pbar(state, snapshot_nwav)

    while True:
        done = state.cur_nwav >= stop_at_nwav

        # Save network snapshot.
        if (
            snapshot_nwav is not None
            and state.cur_nwav % snapshot_nwav == 0
            and (state.cur_nwav != start_nwav or start_nwav == 0)
        ):
            if get_rank() == 0:
                ema_list = (
                    ema.get()
                    if ema is not None
                    else optimizer.get_ema(model) if hasattr(optimizer, "get_ema") else model
                )
                ema_list = ema_list if isinstance(ema_list, list) else [(ema_list, "")]
                for ema_net, ema_suffix in ema_list:
                    fname = f"snapshot-{state.cur_nwav//1000:07d}{ema_suffix}.pt"
                    path = join(run_dir, "snapshots", fname)

                    state_dict = {
                        "ema": {k: v.detach().cpu() for k, v in ema_net.state_dict().items()},
                        "cur_nwav": state.cur_nwav,
                        "ema_suffix": ema_suffix,
                    }

                    torch.save(state_dict, path)
                    del state_dict  # conserve memory
            # reset progress bar for next interval
            if pbar is not None:
                pbar.close()
            if do_validation:
                model.eval()
                with switch_backend("valid"), torch.no_grad():
                    # Generate audio samples
                    audio_iter = get_inference_iterable(
                        model=model,
                        in_dir=valid_dir,
                        out_dir=join(run_dir, "valid", f"{state.cur_nwav}"),
                        proc_dir=valid_pred_dir,
                    )
                    # Loop over batches.
                    print0("Validation...")
                    for _ in tqdm(audio_iter, total=len(audio_iter), disable=(get_rank() != 0)):
                        pass
                model.train()

            if not done:
                print0("Training...")
                pbar = make_pbar(state, snapshot_nwav)

        # Save state checkpoint.
        if (
            checkpoint_nwav is not None
            and (done or state.cur_nwav % checkpoint_nwav == 0)
            and state.cur_nwav != start_nwav
        ):
            checkpoint.save(join(run_dir, "checkpoints", f"checkpoint-{state.cur_nwav//1000:07d}.pt"))
            check_ddp_consistency(model)

        # Stop if done.
        if done:
            print0("Training complete.")
            break

        # Evaluate loss and accumulate gradients.
        batch_start_time = time.time()
        set_random_seed(seed, get_rank(), state.cur_nwav)
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                item = next(dataset_iterator)
                clean_audio = item["clean_audio"]

                # Encode audio.
                clean_audio = audio_encoder.encode(clean_audio.to(device))

                loss, loss_dict = loss_fn(model=ddp, speech=clean_audio)
                final_loss = loss.sum().mul(loss_scaling / batch_gpu_total)
                final_loss.backward()

                if writer is not None:
                    writer.add_scalar("Loss/train_loss", final_loss.detach(), state.cur_nwav)

                # Log all loss components.
                if writer is not None:
                    for key, value in loss_dict.items():
                        writer.add_scalar(f"Loss/{key}", (value.sum() / batch_gpu_total).detach(), state.cur_nwav)

        # Run optimizer and update weights.
        lr = call_func(cur_nwav=state.cur_nwav, batch_size=batch_size, **lr_kwargs)
        if writer is not None:
            writer.add_scalar("Loss/LR", lr, state.cur_nwav)

        for g in optimizer.param_groups:
            g["lr"] = lr
        if force_finite:
            for param in model.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
        optimizer.step()

        # Update EMA and training state.
        state.cur_nwav += batch_size
        if ema is not None:
            ema.update(cur_nimg=state.cur_nwav, batch_size=batch_size)
        batch_elapsed_time = time.time() - batch_start_time
        cumulative_training_time += batch_elapsed_time
        state.total_elapsed_time += batch_elapsed_time
        pbar.update(batch_size)
        postfix = {"cur_nwav": f"{state.cur_nwav:,}"}
        if snapshot_nwav is not None:
            postfix["next_snapshot"] = f"{((state.cur_nwav // snapshot_nwav) + 1) * snapshot_nwav:,}"
        pbar.set_postfix(**postfix)
    if pbar is not None:
        pbar.close()
    if writer is not None:
        writer.close()
