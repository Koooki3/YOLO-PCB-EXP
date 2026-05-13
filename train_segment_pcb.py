#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SegmentData 分割训练启动脚本。

功能：
- 为 SegmentData 数据集集中维护分割训练路径、模型和超参数。
- 默认自动扫描数据集尺寸，推断最小可用 base `imgsz`，避免把小图直接放大到 640/960。
- 使用 `rect=True` + `multi_scale=True` 进行更接近动态分辨率的训练。

使用：
- `python train_segment_pcb.py`
- `python train_segment_pcb.py --help`
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


@dataclass
class UserConfig:
    """用户优先关注和常改的配置。"""

    # === 路径配置 ===
    project_root: Path = Path(r"D:\YOLO_PCB")
    data_yaml: Path = Path(r"D:\YOLO_PCB\SegmentData\SegmentData.yaml")
    model_cfg: str = r"ultralytics/cfg/models/12/yolo12_pcb_final.yaml"
    project: str = "train_segment"
    name: str = "PCBFINAL_SegmentData_12s_standardLoss_DINOv3_P2P3P4_seg_ehanced"

    # === 核心训练配置 ===
    epochs: int = 500
    imgsz: Optional[int] = None
    batch: int = 8
    device: str = "0"
    workers: int = 4
    pretrained: bool = False
    optimizer: str = "AdamW"
    resume: bool = False

    # === 动态分辨率相关 ===
    rect: bool = True
    multi_scale: bool = True
    stride: int = 32
    mask_ratio: int = 2

    # === 训练稳定性与可复现 ===
    seed: int = 42
    deterministic: bool = True
    patience: int = 50


def parse_args() -> UserConfig:
    """解析少量常用 CLI 覆盖项。"""
    parser = argparse.ArgumentParser(description="Train a YOLO segmentation model on SegmentData.")
    parser.add_argument("--data-yaml", type=Path, help="Dataset yaml path.")
    parser.add_argument("--model-cfg", type=str, help="Model cfg/weights path.")
    parser.add_argument("--project", type=str, help="Output project directory.")
    parser.add_argument("--name", type=str, help="Run name.")
    parser.add_argument("--epochs", type=int, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, help="Base train image size. Default: auto infer from dataset.")
    parser.add_argument("--batch", type=int, help="Batch size.")
    parser.add_argument("--device", type=str, help="CUDA device, multi-GPU list, or cpu.")
    parser.add_argument("--workers", type=int, help="Dataloader workers.")
    parser.add_argument("--mask-ratio", type=int, help="GT mask downsample ratio for segment training.")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint.")
    parser.add_argument("--disable-rect", action="store_true", help="Disable rectangular batches.")
    parser.add_argument("--disable-multi-scale", action="store_true", help="Disable multi-scale training.")
    args = parser.parse_args()

    cfg = UserConfig()
    if args.data_yaml is not None:
        cfg.data_yaml = args.data_yaml
    if args.model_cfg is not None:
        cfg.model_cfg = args.model_cfg
    if args.project is not None:
        cfg.project = args.project
    if args.name is not None:
        cfg.name = args.name
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.imgsz is not None:
        cfg.imgsz = args.imgsz
    if args.batch is not None:
        cfg.batch = args.batch
    if args.device is not None:
        cfg.device = args.device
    if args.workers is not None:
        cfg.workers = args.workers
    if args.mask_ratio is not None:
        cfg.mask_ratio = args.mask_ratio
    if args.resume:
        cfg.resume = True
    if args.disable_rect:
        cfg.rect = False
    if args.disable_multi_scale:
        cfg.multi_scale = False
    return cfg


def validate_paths(cfg: UserConfig) -> None:
    """启动前做路径检查，减少无效运行。"""
    if not cfg.project_root.exists():
        raise FileNotFoundError(f"project_root 不存在: {cfg.project_root}")
    if not cfg.data_yaml.exists():
        raise FileNotFoundError(f"data_yaml 不存在: {cfg.data_yaml}")

    model_path = Path(cfg.model_cfg)
    if model_path.suffix == ".yaml":
        if not model_path.exists() and not (cfg.project_root / model_path).exists():
            raise FileNotFoundError(f"model_cfg 不存在: {cfg.model_cfg}")


def _collect_image_dirs(data_yaml: Path) -> list[Path]:
    """从数据集 YAML 中提取 train/val 图像目录。"""
    image_dirs: list[Path] = []
    for raw_line in data_yaml.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key not in {"train", "val"}:
            continue
        value = value.strip().strip("'\"")
        if value:
            image_dirs.append(Path(value))
    return image_dirs


def infer_dataset_base_imgsz(cfg: UserConfig) -> int:
    """自动推断最接近原图尺寸的 base imgsz，并向上对齐到 stride。"""
    from PIL import Image

    if cfg.imgsz is not None:
        return cfg.imgsz

    image_paths: list[Path] = []
    for image_dir in _collect_image_dirs(cfg.data_yaml):
        if not image_dir.exists():
            continue
        for path in image_dir.rglob("*"):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                image_paths.append(path)

    if not image_paths:
        raise FileNotFoundError(f"未在 {cfg.data_yaml} 对应的 train/val 目录中找到图像")

    max_side = 0
    for image_path in image_paths:
        with Image.open(image_path) as image:
            max_side = max(max_side, *image.size)

    stride = max(1, cfg.stride)
    return ((max_side + stride - 1) // stride) * stride


def build_train_args(cfg: UserConfig, resolved_imgsz: int) -> Dict[str, Any]:
    """组装训练参数。除 imgsz 与分割相关项外，其余尽量贴近现有 PCB 训练默认值。"""
    return {
        "model": cfg.model_cfg,
        "data": str(cfg.data_yaml),
        "project": cfg.project,
        "name": cfg.name,
        "task": "segment",
        "exist_ok": True,
        "epochs": cfg.epochs,
        "patience": cfg.patience,
        "batch": cfg.batch,
        "imgsz": resolved_imgsz,
        "device": cfg.device,
        "workers": cfg.workers,
        "pretrained": cfg.pretrained,
        "optimizer": cfg.optimizer,
        "resume": cfg.resume,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "lr0": 0.001,
        "lrf": 0.05,
        "cos_lr": True,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "hsv_h": 0.01,
        "hsv_s": 0.4,
        "hsv_v": 0.2,
        "degrees": 1.0,
        "translate": 0.05,
        "scale": 0.4,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.3,
        "mosaic": 1.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "auto_augment": None,
        "erasing": 0.0,
        "close_mosaic": 10,
        "rect": cfg.rect,
        "multi_scale": cfg.multi_scale,
        "mask_ratio": cfg.mask_ratio,
        "val": True,
        "nbs": 64,
        "deterministic": cfg.deterministic,
        "seed": cfg.seed,
        "cache": False,
        "save_json": False,
        "plots": False,
        "amp": True,
        "save": True,
        "save_period": -1,
        "verbose": True,
    }


def print_user_guide(cfg: UserConfig, resolved_imgsz: int, train_args: Dict[str, Any]) -> None:
    """打印关键配置，方便用户二次确认。"""
    print("=" * 72)
    print("SegmentData 分割训练配置确认")
    print("=" * 72)
    print(f"data_yaml   : {cfg.data_yaml}")
    print(f"model_cfg   : {cfg.model_cfg}")
    print(f"project/name: {cfg.project}/{cfg.name}")
    print(f"device      : {cfg.device}")
    print(f"epochs      : {cfg.epochs}")
    print(f"imgsz(base) : {resolved_imgsz}")
    print(f"batch       : {cfg.batch}")
    print(f"optimizer   : {cfg.optimizer}")
    print(f"resume      : {cfg.resume}")
    print(f"rect        : {cfg.rect}")
    print(f"multi_scale : {cfg.multi_scale}")
    print(f"mask_ratio  : {cfg.mask_ratio}")
    print("-" * 72)
    print("说明：当前本地 Ultralytics 仍会按 batch 尺寸做 letterbox/缩放。")
    print("这里通过自动推断 base imgsz + rect=True + multi_scale=True，尽量减少固定方形缩放。")
    print("当前默认 mask_ratio=2，用于匹配 P2 分割头更高的 prototype 分辨率。")
    print("=" * 72)
    _ = train_args


def main() -> None:
    """主入口：校验配置、推断尺寸并启动训练。"""
    cfg = parse_args()
    validate_paths(cfg)
    os.chdir(cfg.project_root)

    resolved_imgsz = infer_dataset_base_imgsz(cfg)
    train_args = build_train_args(cfg, resolved_imgsz)
    print_user_guide(cfg, resolved_imgsz, train_args)

    from ultralytics import YOLO

    model = YOLO(cfg.model_cfg)
    model.train(**train_args)


if __name__ == "__main__":
    main()
