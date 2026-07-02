"""训练循环、加权 CE、epoch 日志、断点续训与资源限制。"""
from __future__ import annotations

import ctypes
import gc
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .detect import yolo_crop_if_available
from .models import AblationMode, FERWaveletModel
from .wavelet import batch_dwt_torch


def _ts() -> str:
    """返回短时间戳 HH:MM:SS。"""
    return time.strftime("%H:%M:%S", time.localtime())


def _git_commit() -> str:
    try:
        root = Path(__file__).resolve().parents[1]
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _pil_batch_to_tensor(
    pil_list: list, face_size: int, device: torch.device,
) -> torch.Tensor:
    """将 PIL 列表 → (B,3,face_size,face_size) 张量，GPU 双线性缩放。"""
    processed = []
    for pil in pil_list:
        arr = np.array(pil.convert("RGB"), dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
        _, h, w = t.shape
        side = min(h, w)
        h_start, w_start = (h - side) // 2, (w - side) // 2
        crop = t[:, h_start:h_start + side, w_start:w_start + side]
        if side != face_size:
            crop = F.interpolate(
                crop.unsqueeze(0), size=(face_size, face_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        processed.append(crop)
    batch = torch.stack(processed, dim=0).to(device)
    return batch


# ---------------------------------------------------------------------------
# GPU 资源管理
# ---------------------------------------------------------------------------
def setup_gpu(memory_fraction: float = 0.5) -> torch.device:
    """初始化 GPU：清理缓存，设置 cuDNN 稳定性。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        total = torch.cuda.get_device_properties(0).total_memory
        limit = total * memory_fraction
        print(f"[GPU] {torch.cuda.get_device_name(0)}, "
              f"VRAM {total/1024**3:.1f}GB", flush=True)
    return device


def cool_down(seconds: int = 3):
    """epoch 间冷却：清理缓存，让 GPU 降温。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if seconds > 0:
        time.sleep(seconds)


# ---------------------------------------------------------------------------
# Windows 系统资源监控
# ---------------------------------------------------------------------------
def _get_pagefile_free_gb() -> float:
    """查询 Windows 页面文件可用量（GB）。"""
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ('dwLength', ctypes.c_ulong),
                ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong),
                ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong),
                ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong),
                ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
            ]
        ms = MEMORYSTATUSEX()
        ms.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return ms.ullAvailPageFile / 1024**3
    except Exception:
        return 999.0  # 无法检测时返回大值，跳过清理


def _check_resources():
    """检查系统资源，在页面文件不足时自动清理或优雅退出。"""
    free = _get_pagefile_free_gb()
    if free > 4:
        return free
    if free > 2:
        print(f"  [资源] 页面文件仅剩 {free:.1f}GB，执行清理...", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print(f"  [资源] 页面文件仅剩 {free:.1f}GB，临界不足，即将保存退出...", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise SystemExit(0)
    return free


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    mode: AblationMode,
    face_size: int,
    yolo_weights: Optional[str],
    device: torch.device,
    num_classes: int = 7,
) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    tp = torch.zeros(num_classes, device=device)
    fp = torch.zeros(num_classes, device=device)
    fn = torch.zeros(num_classes, device=device)
    for batch_data, y in loader:
        y = y.to(device)
        bs = y.size(0)
        if isinstance(batch_data, torch.Tensor):
            rgb_batch = batch_data.to(device)
        elif yolo_weights:
            rgb_list = []
            for pil in batch_data:
                crop = yolo_crop_if_available(pil.convert("RGB"), yolo_weights, face_size)
                rgb_list.append(pil_to_tensor01(crop))
            rgb_batch = torch.stack(rgb_list, dim=0).to(device)
        else:
            rgb_batch = _pil_batch_to_tensor(batch_data, face_size, device)
        if mode == "rgb":
            logits = model(rgb_batch, None, None)
        else:
            gray = rgb_batch.mean(dim=1, keepdim=True)
            low, high = batch_dwt_torch(gray)
            logits = model(None, low, high)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += bs
        for c in range(num_classes):
            tp[c] += ((pred == c) & (y == c)).sum()
            fp[c] += ((pred == c) & (y != c)).sum()
            fn[c] += ((pred != c) & (y == c)).sum()
    acc = correct / max(total, 1)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    macro_f1 = f1.mean().item()
    cool_down(1)
    return acc, macro_f1


# ---------------------------------------------------------------------------
# 训练（含断点续训）
# ---------------------------------------------------------------------------
CHECKPOINT_FILE = "checkpoint.pt"


def _save_checkpoint(
    out_dir: Path,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
    rows: list[dict],
    batch: int = 0,
):
    """保存完整训练状态用于断点续训。"""
    ckpt = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
        "rows": rows,
        "batch": batch,
    }
    torch.save(ckpt, out_dir / CHECKPOINT_FILE)


def _load_checkpoint(
    out_dir: Path,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float, list[dict], int] | None:
    """加载 checkpoint 返回 (start_epoch, best_val, rows, resume_batch)；无则返回 None。"""
    ckpt_path = out_dir / CHECKPOINT_FILE
    if not ckpt_path.is_file():
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt["epoch"]  # 已完成的 epoch 数
    best_val = ckpt["best_val"]
    rows = ckpt["rows"]
    resume_batch = ckpt.get("batch", 0)  # 中间断点批次（0=epoch 开头）
    print(f"  [{_ts()}] [恢复] 从 epoch {start_epoch} batch {resume_batch} 继续", flush=True)
    return start_epoch, best_val, rows, resume_batch


def _finalize_run(
    out_dir: Path,
    run_meta_path: Path,
    rows: list[dict],
    best_val: float,
    *,
    early_stopped: bool = False,
    stopped_epoch: int = 0,
) -> None:
    """写入最终指标并标记训练完成。"""
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    run_meta["best_val_macro_f1"] = best_val
    run_meta["finished_unix"] = int(time.time())
    if early_stopped:
        run_meta["early_stopped"] = True
        run_meta["stopped_epoch"] = stopped_epoch
        run_meta["patience"] = 5
    run_meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    (out_dir / "metrics.csv").write_text(
        "epoch,train_loss,val_acc,val_macro_f1\n"
        + "\n".join(
            f"{r['epoch']},{r['train_loss']:.6f},{r['val_acc']:.6f},{r['val_macro_f1']:.6f}"
            for r in rows
        ),
        encoding="utf-8",
    )
    (out_dir / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / CHECKPOINT_FILE).unlink(missing_ok=True)


def train_one_run(
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    mode: AblationMode,
    num_classes: int,
    epochs: int,
    lr: float,
    seed: int,
    face_size: int,
    yolo_weights: Optional[str],
    pretrained: bool,
    out_dir: Path,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
    single_epoch: bool = False,
    patience: int = 5,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    # 限制资源占用
    torch.set_num_threads(max(1, os.cpu_count() or 4))
    if device.type == "cuda":
        try:
            torch.cuda.set_per_process_memory_fraction(0.50)
        except Exception:
            pass  # 某些 Windows WDDM 驱动不支持此设置
    model = FERWaveletModel(
        mode=mode, num_classes=num_classes, pretrained=pretrained
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=2, min_lr=1e-6,
    )
    w = class_weights.to(device) if class_weights is not None else None
    ce = nn.CrossEntropyLoss(weight=w)

    # 尝试断点续训
    resumed = _load_checkpoint(out_dir, model, opt, device)
    if resumed is not None:
        start_epoch, best_val, rows, resume_batch = resumed
    else:
        start_epoch, best_val, rows, resume_batch = 0, -1.0, [], 0

    # 写 run_meta（首次）
    run_meta_path = out_dir / "run_meta.json"
    if not run_meta_path.is_file():
        run_meta = {
            "git_commit": _git_commit(),
            "seed": seed,
            "mode": mode,
            "epochs": epochs,
            "lr": lr,
            "face_size": face_size,
            "pretrained": pretrained,
            "started_unix": int(time.time()),
            "patience": 5,
            "lr_scheduler": "ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6)",
        }
        run_meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    # 早停检查：val_f1 连续 patience 个 epoch 未改善则停止
    if rows:
        best_row = max(rows, key=lambda r: r["val_macro_f1"])
        best_f1_epoch = best_row["epoch"]
        epochs_since_best = rows[-1]["epoch"] - best_f1_epoch
    else:
        best_f1_epoch = 0
        epochs_since_best = 0

    best_path = out_dir / "best.pt"
    total_batches = len(train_loader)
    log_interval = max(1, total_batches // 50)  # 约每 2% 打印一次
    stall_limit = 180  # 秒，无进度时输出心跳

    # 早停决策：已训练至少 patience×2 个 epoch 且连续 patience 轮未改善
    if rows and epochs_since_best >= patience and len(rows) >= patience * 2:
        print(f"  [{_ts()}] [早停] val_f1 连续 {patience} 个 epoch 未改善"
              f"（最佳 epoch {best_f1_epoch}, val_f1={best_row['val_macro_f1']:.4f}），停止训练",
              flush=True)
        _finalize_run(out_dir, run_meta_path, rows, best_val,
                      early_stopped=True, stopped_epoch=rows[-1]["epoch"])
        return {"best_val_macro_f1": best_val, "rows": rows, "run_dir": str(out_dir)}

    for ep in range(start_epoch + 1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n_batches = 0
        ep_start = time.time()
        last_log_time = time.time()
        # 创建固定顺序的批次迭代器（便于断点续训）
        train_iter = enumerate(train_loader)
        # 如果存在 batch 断点，跳过多余批次
        skip_batches = resume_batch if (resumed and resume_batch > 0) else 0
        # 仅当剩余 batch 少于跳过量时才从头开始（避免频繁重置）
        remaining = total_batches - skip_batches
        if skip_batches > remaining and skip_batches > log_interval:
            print(f"  [{_ts()}] [跳过] 跳过量 ({skip_batches}) 超过剩余量 ({remaining})，"
                  f"从头开始 epoch {ep}", flush=True)
            skip_batches = 0
            resume_batch = 0
        if skip_batches > 0:
            # 消耗前 skip_batches 个批次
            for _ in range(skip_batches):
                try:
                    next(train_iter)
                except StopIteration:
                    break
        for batch_idx, (batch_data, y) in train_iter:
            y = y.to(device)
            if isinstance(batch_data, torch.Tensor):
                rgb_batch = batch_data.to(device)
            elif yolo_weights:
                # YOLO 人脸检测路径
                rgb_list = []
                for pil in batch_data:
                    crop = yolo_crop_if_available(
                        pil.convert("RGB"), yolo_weights, face_size
                    )
                    rgb_list.append(pil_to_tensor01(crop))
                rgb_batch = torch.stack(rgb_list, dim=0).to(device)
            else:
                # GPU 双线性缩放路径（避免每 epoch PIL BICUBIC）
                rgb_batch = _pil_batch_to_tensor(batch_data, face_size, device)

            opt.zero_grad()
            if mode == "rgb":
                logits = model(rgb_batch, None, None)
            else:
                gray = rgb_batch.mean(dim=1, keepdim=True)
                low, high = batch_dwt_torch(gray)
                logits = model(None, low, high)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            n_batches += 1

            if batch_idx <= 1 or (batch_idx % log_interval == 0 and batch_idx > 0):
                avg_loss = loss_sum / n_batches
                elapsed = time.time() - ep_start
                eta = elapsed / (batch_idx + 1) * (total_batches - batch_idx - 1)
                page_free = _get_pagefile_free_gb()
                print(f"  [{_ts()}] ep{ep:02d} [{batch_idx}/{total_batches}] "
                      f"loss={avg_loss:.4f}  page={page_free:.1f}GB  "
                      f"{elapsed:.0f}s/{eta:.0f}s",
                      flush=True)
                last_log_time = time.time()
                # 资源监控：页面文件不足时自动清理
                _check_resources()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # 中间断点保存（每 log_interval 批次，便于进程被杀死后恢复）
                _save_checkpoint(out_dir, model, opt, ep - 1, best_val, rows, batch_idx + 1)
            elif time.time() - last_log_time > stall_limit:
                elapsed = time.time() - ep_start
                print(f"  [{_ts()}] ep{ep:02d} [心跳] 仍在运行 batch {batch_idx}/{total_batches}"
                      f"  耗时 {elapsed:.0f}s", flush=True)
                last_log_time = time.time()
        train_loss = loss_sum / max(n_batches, 1)
        val_acc, val_f1 = evaluate(
            model, val_loader, mode, face_size, yolo_weights, device,
            num_classes=num_classes,
        )
        rows.append({
            "epoch": ep,
            "train_loss": train_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
        })
        print(f"  [{_ts()}] epoch {ep:02d}/{epochs}: loss={train_loss:.4f}  "
              f"val_acc={val_acc:.4f}  val_f1={val_f1:.4f}  "
              f"{'*' if val_f1 > best_val else ' '}",
              flush=True)
        score = val_f1
        if score > best_val:
            best_val = score
            torch.save({"model": model.state_dict(), "mode": mode}, best_path)

        # ReduceLROnPlateau：val_f1 停滞 2 个 epoch 后 LR 减半
        old_lr = opt.param_groups[0]["lr"]
        scheduler.step(val_f1)
        new_lr = opt.param_groups[0]["lr"]
        if new_lr < old_lr:
            print(f"  [{_ts()}] [scheduler] val_f1 停滞，学习率 {old_lr:.2e} → {new_lr:.2e}", flush=True)

        # 每 epoch 保存 checkpoint（断点续训用）
        _save_checkpoint(out_dir, model, opt, ep, best_val, rows)
        cool_down(2)

        # 早停检查：连续 patience 轮未改善则停止
        if not single_epoch and len(rows) >= patience * 2:
            best_in_rows = max(r["val_macro_f1"] for r in rows)
            best_idx = next(i for i, r in enumerate(rows) if r["val_macro_f1"] == best_in_rows)
            epochs_since_best = len(rows) - 1 - best_idx
            if epochs_since_best >= patience:
                print(f"  [{_ts()}] [早停] val_f1 连续 {patience} 个 epoch 未改善"
                      f"（最佳 epoch {rows[best_idx]['epoch']}, val_f1={best_in_rows:.4f}），停止训练",
                      flush=True)
                _finalize_run(out_dir, run_meta_path, rows, best_val,
                              early_stopped=True, stopped_epoch=ep)
                return {"best_val_macro_f1": best_val, "rows": rows, "run_dir": str(out_dir)}

        # 单 epoch 模式：完成一个 epoch 就退出进程，释放系统资源
        if single_epoch and ep < epochs:
            print(f"  [{_ts()}] [单epoch] epoch {ep} 完成，退出进程以释放系统资源。"
                  f"看门狗将自动重启进行下一轮。", flush=True)
            break

    # 训练完成（全部 epoch 或单 epoch 提前退出）
    all_done = ep >= epochs if 'ep' in dir() else False
    if all_done:
        _finalize_run(out_dir, run_meta_path, rows, best_val)
    else:
        _rm = json.loads(run_meta_path.read_text(encoding="utf-8"))
        _rm["incomplete_epoch"] = ep
        run_meta_path.write_text(json.dumps(_rm, indent=2), encoding="utf-8")
    return {"best_val_macro_f1": best_val, "rows": rows, "run_dir": str(out_dir)}


# ---------------------------------------------------------------------------
# 跨域评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def cross_domain_evaluate(
    model_path: Path,
    target_loader: DataLoader,
    mode: AblationMode,
    face_size: int,
    yolo_weights: Optional[str],
    device: torch.device,
    num_classes: int = 7,
) -> dict[str, Any]:
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    saved_mode = ckpt.get("mode", mode)
    model = FERWaveletModel(
        mode=saved_mode, num_classes=num_classes, pretrained=False
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    correct = 0
    total = 0
    tp = torch.zeros(num_classes, device=device)
    fp = torch.zeros(num_classes, device=device)
    fn = torch.zeros(num_classes, device=device)
    class_correct = torch.zeros(num_classes, device=device)
    class_total = torch.zeros(num_classes, device=device)

    for batch_data, y in target_loader:
        y = y.to(device)
        bs = y.size(0)
        if isinstance(batch_data, torch.Tensor):
            rgb_batch = batch_data.to(device)
        elif yolo_weights:
            rgb_list = []
            for pil in batch_data:
                crop = yolo_crop_if_available(
                    pil.convert("RGB"), yolo_weights, face_size
                )
                rgb_list.append(pil_to_tensor01(crop))
            rgb_batch = torch.stack(rgb_list, dim=0).to(device)
        else:
            rgb_batch = _pil_batch_to_tensor(batch_data, face_size, device)

        if saved_mode == "rgb":
            logits = model(rgb_batch, None, None)
        else:
            gray = rgb_batch.mean(dim=1, keepdim=True)
            low, high = batch_dwt_torch(gray)
            logits = model(None, low, high)
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += bs

        for c in range(num_classes):
            tp[c] += ((pred == c) & (y == c)).sum()
            fp[c] += ((pred == c) & (y != c)).sum()
            fn[c] += ((pred != c) & (y == c)).sum()
            class_correct[c] += ((pred == c) & (y == c)).sum()
            class_total[c] += (y == c).sum()

    acc = correct / max(total, 1)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    macro_f1 = f1.mean().item()

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "per_class_acc": (class_correct / (class_total + 1e-8)).tolist(),
        "per_class_f1": f1.tolist(),
        "total_samples": total,
    }


def compute_class_weights_from_subset(
    dataset: torch.utils.data.Dataset, indices: list[int], num_classes: int = 7
) -> torch.Tensor:
    labels = [int(dataset[i][1]) for i in indices]
    counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float()
    w = counts.sum() / (counts + 1.0)
    w = w / w.mean()
    return w


def random_train_val_indices(n: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(n * val_fraction))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx
