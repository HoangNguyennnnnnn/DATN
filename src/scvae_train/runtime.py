import hashlib
import json
import math
import os
import random

import torch
from torch.optim.lr_scheduler import LambdaLR


def align_recon_target(recon_x: torch.Tensor, out_indices: torch.Tensor, target_x: torch.Tensor, target_indices: torch.Tensor):
    """Căn chỉnh tái tạo/mục tiêu thưa thớt (sparse recon/target) thông qua việc khớp chỉ số không gian băm (hashed spatial index matching) chính xác."""
    from src.models.sc_vae import _hash_indices, _SPARSE_HASH_BASE
    
    out_keys = _hash_indices(out_indices, _SPARSE_HASH_BASE)
    tgt_keys = _hash_indices(target_indices, _SPARSE_HASH_BASE)
    
    sorted_tgt_keys, tgt_order = torch.sort(tgt_keys)
    
    pos = torch.searchsorted(sorted_tgt_keys, out_keys)
    safe_pos = torch.clamp(pos, 0, max(sorted_tgt_keys.shape[0] - 1, 0))
    
    valid = (pos < sorted_tgt_keys.shape[0]) & (sorted_tgt_keys[safe_pos] == out_keys)
    
    # Lọc recon_x chỉ lấy các khóa hợp lệ (theo lý thuyết là tất cả chúng)
    recon_aligned = recon_x[valid]
    
    # Trích xuất CHÍNH XÁC các mục tiêu tương ứng bằng cách sử dụng các chỉ số đã được khớp
    matching_tgt_idx = tgt_order[safe_pos[valid]]
    target_aligned = target_x[matching_tgt_idx]
    
    was_mismatch = (recon_x.shape[0] != target_x.shape[0]) or not valid.all()
    
    return recon_aligned, target_aligned, was_mismatch


def is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def is_sparse_runtime_error(exc: BaseException) -> bool:
    """Phát hiện các lỗi nhân thưa/thời gian chạy (sparse kernel/runtime failures) có thể phục hồi được do các lô thưa thớt bị lỗi/trống (bad/empty sparse batches)."""
    msg = str(exc).lower()
    sparse_signatures = (
        "all_profile_res.empty",
        "can't find suitable algorithm for 0",
        "convtunersimple_tune_and_cache",
        "empty sparse batch",
    )
    return any(sig in msg for sig in sparse_signatures)


def get_lr_scheduler(optimizer, cfg, steps_per_epoch: int = 0):
    """Làm ấm tuyến tính (Linear warmup) + suy giảm cosin (cosine decay) mà không bị cảnh báo không dùng nữa (deprecation warning) từ SequentialLR.

    ``steps_per_epoch`` nên là số bước tối ưu hóa trong một epoch
    (tức là len(dataloader) // gradient_accumulation_steps). Khi được cung cấp,
    chu kỳ cosin sẽ kéo dài toàn bộ quá trình huấn luyện; nếu không, chúng ta sẽ dự phòng (fall back) bằng
    ước tính thận trọng (conservative estimate) là ``num_epochs * 2000``.
    """
    warmup_steps = max(int(cfg.lr_warmup_steps), 1)
    if steps_per_epoch > 0:
        total_steps = int(cfg.num_epochs) * int(steps_per_epoch)
    else:
        total_steps = int(cfg.num_epochs) * 2000  # Dự phòng thận trọng (conservative fallback)
    total_steps = max(total_steps, warmup_steps + 1)
    min_lr_scale = 1e-7 / max(float(cfg.learning_rate), 1e-12)
    min_lr_scale = float(min(max(min_lr_scale, 0.0), 1.0))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return 0.01 + 0.99 * (step / warmup_steps)

        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def get_resume_scheduler(
    optimizer,
    cfg,
    steps_per_epoch: int,
    resume_step: int,
    mode: str = "cosine_restart",
    extend_epochs: int = 100,
    target_min_lr: float = 1e-7,
):
    """Build a fresh LR scheduler tailored for resuming training past the original cosine end.

    The original ``get_lr_scheduler`` schedules a single cosine over the full
    ``num_epochs * steps_per_epoch`` budget. When a checkpoint is reached late in
    that budget (e.g. epoch 397 out of 500) and we want to keep training, the
    remaining budget is too small to give meaningful gradient steps. This builder
    returns a *new* scheduler that picks up from the current step with one of three
    well-defined post-cosine policies. The optimizer state is left untouched so
    Adam moments remain calibrated to the loss landscape.

    Args:
        optimizer: The optimizer to attach the scheduler to.
        cfg: SCVAEConfig (uses ``learning_rate``, ``min_lr``, ``lr_warmup_steps``).
        steps_per_epoch: optimisation steps per epoch (= len(loader)//grad_accum).
        resume_step: Optimiser step counter from the checkpoint (``scheduler._step_count``
            from the saved state, falling back to global_step // grad_accum).
        mode: One of:
          - ``"continue"``: Re-use the standard cosine over ``num_epochs`` (legacy).
          - ``"constant_min_lr"``: Hold LR at ``target_min_lr`` for ``extend_epochs``
            epochs — safest for late-stage refinement of an already-converged model.
          - ``"cosine_restart"`` (default): Half-cosine warm restart from the
            current LR scale down to ``target_min_lr`` over ``extend_epochs``.
            Mimics SGDR-style annealing without resetting optimizer momentum.
        extend_epochs: Length (in epochs) of the post-resume schedule.
        target_min_lr: Floor LR for the extended schedule.
    """
    warmup_steps = max(int(cfg.lr_warmup_steps), 1)
    base_lr = float(cfg.learning_rate)
    target_min_scale = float(min(max(target_min_lr / max(base_lr, 1e-12), 0.0), 1.0))
    extend_steps = max(int(extend_epochs) * max(int(steps_per_epoch), 1), 1)
    resume_step = int(max(resume_step, 0))

    if mode == "continue":
        return get_lr_scheduler(optimizer, cfg, steps_per_epoch)

    if mode == "constant_min_lr":
        def _const(step: int) -> float:
            # Lambda is on `base_lr`, so returning `target_min_scale` yields target_min_lr.
            return target_min_scale
        return LambdaLR(optimizer, lr_lambda=_const)

    if mode == "cosine_restart":
        # Read the current LR scale from the ORIGINAL schedule so the new cosine
        # starts smoothly from where the training was interrupted, not at base_lr.
        if steps_per_epoch > 0:
            total_steps_orig = max(int(cfg.num_epochs) * int(steps_per_epoch), warmup_steps + 1)
        else:
            total_steps_orig = max(int(cfg.num_epochs) * 2000, warmup_steps + 1)
        orig_min_scale = float(min(max(1e-7 / max(base_lr, 1e-12), 0.0), 1.0))
        if resume_step <= warmup_steps:
            start_scale = max(0.01, 0.01 + 0.99 * (resume_step / warmup_steps))
        else:
            p = (resume_step - warmup_steps) / max(total_steps_orig - warmup_steps, 1)
            p = min(max(p, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * p))
            start_scale = orig_min_scale + (1.0 - orig_min_scale) * cosine
        # Make sure we always have headroom to decay (avoid degenerate flat schedule).
        start_scale = max(start_scale, target_min_scale + 1e-9)

        def _restart(step: int) -> float:
            local = max(step - resume_step, 0)
            p = min(local / max(extend_steps, 1), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * p))
            return target_min_scale + (start_scale - target_min_scale) * cosine

        sch = LambdaLR(optimizer, lr_lambda=_restart)
        # Fast-forward internal counter so logging matches the optimizer state.
        for _ in range(resume_step):
            sch.step()
        return sch

    raise ValueError(f"Unknown resume scheduler mode: {mode!r}")


def _normalize_signature_value(value):
    if isinstance(value, dict):
        return {str(k): _normalize_signature_value(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_signature_value(v) for v in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def make_signature_record(details: dict):
    normalized = _normalize_signature_value(details or {})
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha1(payload.encode("ascii")).hexdigest()
    return {
        "digest": digest,
        "details": normalized,
    }


def _format_signature_record(record) -> str:
    if not isinstance(record, dict):
        return "missing"
    digest = str(record.get("digest", "missing"))
    return digest[:12]


def _validate_resume_signatures(
    ckpt: dict,
    expected_data_signature=None,
    expected_resume_contract=None,
    load_optimizer: bool = True,
    allow_unsafe_resume: bool = False,
):
    issues = []
    warnings = []

    def _compare(label: str, expected_record):
        if expected_record is None:
            return

        actual_record = ckpt.get(label, None)
        if actual_record is None:
            msg = (
                f"Checkpoint is missing {label} metadata "
                f"(expected={_format_signature_record(expected_record)})."
            )
            if load_optimizer and not allow_unsafe_resume:
                issues.append(
                    msg + " Full-state resume is blocked. Retry with "
                    "--resume-model-only or --allow-unsafe-resume."
                )
            else:
                warnings.append(msg)
            return

        actual_digest = str(actual_record.get("digest", "missing"))
        expected_digest = str(expected_record.get("digest", "missing"))
        if actual_digest != expected_digest:
            msg = (
                f"{label} mismatch: ckpt={actual_digest[:12]} "
                f"expected={expected_digest[:12]}"
            )
            if load_optimizer and not allow_unsafe_resume:
                issues.append(
                    msg + ". Full-state resume is blocked. Retry with "
                    "--resume-model-only or --allow-unsafe-resume."
                )
            else:
                warnings.append(msg)

    _compare("data_signature", expected_data_signature)
    _compare("resume_contract", expected_resume_contract)
    return issues, warnings


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    loss,
    path,
    global_step=None,
    batch_size=None,
    metadata=None,
):
    """Lưu checkpoint đầy đủ để resume."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        'epoch': epoch,
        'global_step': global_step,
        'batch_size': batch_size,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'loss': loss,
    }
    if isinstance(metadata, dict):
        payload.update(metadata)
    tmp_path = f"{path}.tmp.{os.getpid()}.{random.randint(0, 10**9)}"
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    print(f"  💾 Checkpoint saved: {path}")


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    load_optimizer=True,
    strict_model_load=True,
    expected_data_signature=None,
    expected_resume_contract=None,
    allow_unsafe_resume=False,
):
    """Load checkpoint và resume training."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    issues, warnings = _validate_resume_signatures(
        ckpt,
        expected_data_signature=expected_data_signature,
        expected_resume_contract=expected_resume_contract,
        load_optimizer=bool(load_optimizer),
        allow_unsafe_resume=bool(allow_unsafe_resume),
    )
    for warning in warnings:
        print(f"  ⚠️ Resume metadata: {warning}")
    if issues:
        raise RuntimeError(" ".join(issues))
    if strict_model_load:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        # Smart partial load for mismatched shapes (e.g., 10ch -> 11ch upgrade)
        model_state = model.state_dict()
        ckpt_state = ckpt['model_state_dict']
        load_state = {}
        
        for k, v in ckpt_state.items():
            if k in model_state:
                if v.shape == model_state[k].shape:
                    load_state[k] = v
                else:
                    # Attempt partial copy for mismatched layers (usually the last projection)
                    print(f"  ⚠️ Shape mismatch for {k}: ckpt={list(v.shape)}, model={list(model_state[k].shape)}. Attempting partial copy.")
                    new_v = model_state[k].clone()
                    # Calculate common slices
                    slices = [slice(0, min(v.shape[i], model_state[k].shape[i])) for i in range(v.ndim)]
                    new_v[slices] = v[slices]
                    load_state[k] = new_v
        
        incompatible = model.load_state_dict(load_state, strict=False)
        missing = len(getattr(incompatible, 'missing_keys', []))
        unexpected = len(ckpt_state.keys() - load_state.keys())
        print(f"  ⚠️ Partial model load: successfully matched={len(load_state)}, missing={missing}, unexpected={unexpected}")
    if load_optimizer and optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if load_optimizer and scheduler and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if load_optimizer and scaler and ckpt.get('scaler_state_dict'):
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    mode = "model+optimizer" if load_optimizer else "model-only"
    print(
        f"  ✅ Resumed ({mode}) from epoch {ckpt.get('epoch', 0)} "
        f"(loss={ckpt.get('loss', float('nan')):.4f})"
    )
    return ckpt
