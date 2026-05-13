#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多模型中间特征图对比分析脚本。

功能：
- 比较多个 YOLO 模型在指定层上的特征增强与抑制效果。
- 可视化通道均值热力图，并计算前景背景分离指标与线性 CKA。
- 导出逐图统计、汇总统计和成对对比 CSV。

使用：
- `python test_feature_compare.py --help`
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from chart_style import get_visual_theme


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EPS = 1e-8
THEME = get_visual_theme()


def require_cv2():
    """按需导入 OpenCV，并在缺失时抛出明确错误。"""
    try:
        import cv2  # type: ignore

        return cv2
    except Exception as e:
        raise RuntimeError(
            "Visualization requires OpenCV. Please install: pip install opencv-python"
        ) from e


@dataclass
class Experiment:
    """描述一个待比较实验的目录、显示名称和关键文件路径。"""

    name: str
    exp_dir: Path
    weights: Path
    train_results_csv: Optional[Path]


class FeatureHookCollector:
    """收集中间层特征输出，供可视化和统计分析复用。"""

    def __init__(self, torch_model: Any, layer_indices: Sequence[int]) -> None:
        self.layer_indices = sorted(set(layer_indices))
        self.handles: List[Any] = []
        self.current: Dict[int, Any] = {}

        model_layers = getattr(torch_model, "model", None)
        if model_layers is None:
            raise RuntimeError(
                "Unexpected model structure: missing `model.model` layers."
            )

        for idx in self.layer_indices:
            if idx < 0 or idx >= len(model_layers):
                continue
            module = model_layers[idx]
            handle = module.register_forward_hook(self._hook_factory(idx))
            self.handles.append(handle)

    def _hook_factory(self, idx: int):
        def _hook(_module, _inputs, output):
            tensor = None
            # 局部导入，避免在仅查看 `--help` 时强依赖 torch。
            import torch

            if isinstance(output, torch.Tensor):
                tensor = output
            elif isinstance(output, (list, tuple)):
                for item in output:
                    if isinstance(item, torch.Tensor):
                        tensor = item
                        break
            if tensor is None:
                return
            if tensor.dim() == 4:
                self.current[idx] = tensor.detach().cpu()[0]
            elif tensor.dim() == 3:
                self.current[idx] = tensor.detach().cpu()

        return _hook

    def reset(self) -> None:
        self.current = {}

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="对多个 YOLO 模型的中间特征图和量化指标进行比较。"
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=(
            r"D:\YOLO_PCB\train_ex\Ex_12s_960_WassersteinLoss0.7_DINOP2,"
            r"D:\YOLO_PCB\train_ex\Ex_12s_960_WassersteinLoss_0.7,"
            r"D:\YOLO_PCB\train_ex\Ex_12s_960_standardLoss"
        ),
        help="按比较顺序填写实验目录，多个目录用逗号分隔。",
    )
    parser.add_argument(
        "--model-names",
        type=str,
        default="DINO+NWD,NWD,Baseline",
        help="实验显示名称，多个名称用逗号分隔，顺序需与 --experiments 一致。",
    )
    parser.add_argument(
        "--dataset-yaml",
        type=str,
        default=r"D:\YOLO_PCB\PKU-Market-PCB-ex\pku_market_pcb_ex.yaml",
        help="用于枚举数据集图片和标签的 YAML 配置路径。",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="detect",
        choices=["detect", "segment"],
        help="任务类型，决定标签解析方式与训练指标列名。",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="参与分析的数据集划分。",
    )
    parser.add_argument("--imgsz", type=int, default=960, help="推理时使用的图像尺寸。")
    parser.add_argument(
        "--device", type=str, default="0", help="推理设备，例如 0、0,1 或 cpu。"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="前向推理时使用的置信度阈值。",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="前向推理时使用的 IoU 阈值。",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="单张图片允许保留的最大检测数。",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=60,
        help="用于量化评估的最大图片数量。",
    )
    parser.add_argument(
        "--vis-images",
        type=int,
        default=12,
        help="用于特征图可视化的最大图片数量。",
    )
    parser.add_argument(
        "--layer-spec",
        type=str,
        default="P3=21,14,14;P4=24,17,17;P5=27,20,20;P2_DINO=18,-1,-1",
        help="层映射配置，格式为 名称=模型1索引,模型2索引,模型3索引；使用 -1 表示该模型跳过该层。",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=r"D:\YOLO_PCB\results_analyse\feature_compare_dino_nwd",
        help="可视化图片和 CSV 结果的输出目录。",
    )
    parser.add_argument(
        "--run-val",
        action="store_true",
        help="额外执行 model.val() 并导出当前验证集 mAP 指标，耗时较长。",
    )
    parser.add_argument(
        "--val-batch", type=int, default=4, help="可选验证阶段使用的 batch size。"
    )
    return parser.parse_args()


def parse_layer_spec(layer_spec: str, num_models: int) -> Dict[str, List[int]]:
    """将层映射字符串解析为“层名 -> 各模型层索引列表”的字典。"""
    mapping: Dict[str, List[int]] = {}
    if not layer_spec.strip():
        return mapping
    items = [item.strip() for item in layer_spec.split(";") if item.strip()]
    for item in items:
        if "=" not in item:
            continue
        label, raw = item.split("=", 1)
        values = [v.strip() for v in raw.split(",")]
        if len(values) != num_models:
            raise ValueError(
                f"Layer '{label}' has {len(values)} values, expected {num_models}."
            )
        parsed = [int(v) for v in values]
        mapping[label.strip()] = parsed
    return mapping


def collect_experiments(
    exp_dirs: Sequence[str], model_names: Sequence[str]
) -> List[Experiment]:
    """根据实验目录和显示名称构造实验对象列表。"""
    if len(exp_dirs) != len(model_names):
        raise ValueError(
            "The number of experiment directories and model names must match."
        )
    experiments: List[Experiment] = []
    for name, exp in zip(model_names, exp_dirs):
        exp_dir = Path(exp).resolve()
        weights = exp_dir / "weights" / "best.pt"
        train_csv = exp_dir / "results.csv"
        if not weights.exists():
            raise FileNotFoundError(f"Missing weights: {weights}")
        experiments.append(
            Experiment(
                name=name.strip(),
                exp_dir=exp_dir,
                weights=weights,
                train_results_csv=train_csv if train_csv.exists() else None,
            )
        )
    return experiments


def load_dataset_yaml(dataset_yaml: Path) -> dict:
    """读取数据集 YAML 配置。"""
    with dataset_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid dataset yaml format: {dataset_yaml}")
    return data


def resolve_split_dirs(
    dataset_yaml: Path, split_value: str, base_root: Optional[str]
) -> List[Path]:
    """解析指定数据集划分对应的图片目录列表。"""
    root_from_yaml = (
        Path(base_root).resolve() if base_root else dataset_yaml.parent.resolve()
    )
    split_paths = split_value if isinstance(split_value, list) else [split_value]
    out: List[Path] = []
    for p in split_paths:
        candidate = Path(p)
        if candidate.is_absolute():
            out.append(candidate)
        else:
            out.append((root_from_yaml / candidate).resolve())
    return out


def list_split_images(dataset_yaml: Path, split: str) -> List[Path]:
    """列出指定划分目录下的全部图片文件。"""
    d = load_dataset_yaml(dataset_yaml)
    if split not in d:
        raise KeyError(f"Split '{split}' not found in {dataset_yaml}")
    split_dirs = resolve_split_dirs(dataset_yaml, d[split], d.get("path"))
    images: List[Path] = []
    for split_dir in split_dirs:
        if split_dir.is_file() and split_dir.suffix.lower() in IMG_EXTS:
            images.append(split_dir)
            continue
        if not split_dir.exists():
            continue
        images.extend([x for x in split_dir.rglob("*") if x.suffix.lower() in IMG_EXTS])
    images = sorted(set(images))
    return images


def image_to_label_path(img_path: Path) -> Path:
    """将图片路径映射为对应的标签文件路径。"""
    p = img_path.as_posix()
    if "/images/" in p:
        p = p.replace("/images/", "/labels/")
        return Path(p).with_suffix(".txt")
    # fallback: sibling labels folder
    return img_path.parent.parent / "labels" / f"{img_path.stem}.txt"


def read_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """读取单个 YOLO 检测标签文件。"""
    if not label_path.exists():
        return []
    labels = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = parts
            labels.append((int(float(cls)), float(x), float(y), float(w), float(h)))
    return labels


def read_yolo_segments(label_path: Path) -> List[Tuple[int, List[Tuple[float, float]]]]:
    """读取单个 YOLO 分割标签文件。"""
    if not label_path.exists():
        return []
    labels: List[Tuple[int, List[Tuple[float, float]]]] = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
                continue
            cls = int(float(parts[0]))
            points: List[Tuple[float, float]] = []
            ok = True
            for i in range(1, len(parts), 2):
                try:
                    points.append((float(parts[i]), float(parts[i + 1])))
                except Exception:
                    ok = False
                    break
            if ok and points:
                labels.append((cls, points))
    return labels


def feature_to_map(feature: Any) -> np.ndarray:
    """将特征张量转换为二维特征图，便于可视化和统计。"""
    if feature.dim() == 3:
        fmap = feature.abs().mean(dim=0).numpy()
    elif feature.dim() == 2:
        fmap = feature.abs().numpy()
    else:
        raise ValueError(f"Unsupported feature dimension: {feature.shape}")
    return fmap


def feature_vector_for_cka(feature: Any) -> np.ndarray:
    """将特征转换为用于 CKA 计算的一维表示向量。"""
    if feature.dim() == 3:
        # spatial-average each channel -> C
        vec = feature.float().abs().mean(dim=(1, 2)).numpy()
    elif feature.dim() == 2:
        vec = feature.float().abs().reshape(-1).numpy()
    else:
        raise ValueError(f"Unsupported feature dimension: {feature.shape}")
    return vec.astype(np.float64)


def build_fg_mask_detect(
    fmap_shape: Tuple[int, int], labels: Sequence[Tuple[int, float, float, float, float]]
) -> np.ndarray:
    """将检测框标签转换为前景掩码。"""
    h, w = fmap_shape
    fg_mask = np.zeros((h, w), dtype=bool)
    for _cls, cx, cy, bw, bh in labels:
        x1 = int(max(0, min(w - 1, (cx - bw / 2.0) * w)))
        y1 = int(max(0, min(h - 1, (cy - bh / 2.0) * h)))
        x2 = int(max(0, min(w, (cx + bw / 2.0) * w)))
        y2 = int(max(0, min(h, (cy + bh / 2.0) * h)))
        if x2 <= x1 or y2 <= y1:
            continue
        fg_mask[y1:y2, x1:x2] = True
    return fg_mask


def build_fg_mask_segment(
    fmap_shape: Tuple[int, int], labels: Sequence[Tuple[int, List[Tuple[float, float]]]]
) -> np.ndarray:
    """将分割多边形标签转换为前景掩码。"""
    from PIL import Image, ImageDraw

    h, w = fmap_shape
    image = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(image)
    for _cls, points in labels:
        xy = []
        for x, y in points:
            px = max(0.0, min(w - 1, x * w))
            py = max(0.0, min(h - 1, y * h))
            xy.append((px, py))
        if len(xy) >= 3:
            draw.polygon(xy, outline=1, fill=1)
    return np.array(image, dtype=bool)


def calc_fg_bg_metrics(fmap: np.ndarray, fg_mask: np.ndarray) -> Dict[str, float]:
    """计算前景与背景区域的特征分离指标。"""
    h, w = fmap.shape
    if fg_mask.shape != (h, w):
        raise ValueError(f"Foreground mask shape {fg_mask.shape} != fmap shape {(h, w)}")

    if fg_mask.any():
        fg_values = fmap[fg_mask]
    else:
        fg_values = np.array([0.0], dtype=np.float32)
    bg_values = (
        fmap[~fg_mask] if (~fg_mask).any() else np.array([0.0], dtype=np.float32)
    )

    fg_mean = float(np.mean(fg_values))
    bg_mean = float(np.mean(bg_values))
    all_std = float(np.std(fmap) + EPS)
    sep_score = float((fg_mean - bg_mean) / all_std)
    ratio = float((fg_mean + EPS) / (bg_mean + EPS))
    sparsity = float(np.mean(np.abs(fmap) < 1e-6))

    return {
        "fg_mean": fg_mean,
        "bg_mean": bg_mean,
        "sep_score": sep_score,
        "fg_bg_ratio": ratio,
        "sparsity": sparsity,
        "fg_pixels": int(fg_mask.sum()),
        "bg_pixels": int((~fg_mask).sum()),
    }


def center_gram(K: np.ndarray) -> np.ndarray:
    """对 Gram 矩阵做中心化处理。"""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """计算两个样本特征矩阵之间的线性 CKA。"""
    K = X @ X.T
    L = Y @ Y.T
    Kc = center_gram(K)
    Lc = center_gram(L)
    hsic = np.sum(Kc * Lc)
    norm = math.sqrt(np.sum(Kc * Kc) * np.sum(Lc * Lc)) + EPS
    return float(hsic / norm)


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """将浮点数组归一化到 `uint8` 图像范围。"""
    arr = arr.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < EPS:
        return np.zeros_like(arr, dtype=np.uint8)
    out = (arr - mn) / (mx - mn)
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out


def overlay_heatmap(
    image_bgr: np.ndarray, fmap: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    """将热力图叠加到原图上。"""
    cv2 = require_cv2()
    hm = normalize_to_uint8(fmap)
    hm = cv2.resize(
        hm, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_CUBIC
    )
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1.0 - alpha, hm, alpha, 0)


def load_image_bgr(path: Path) -> np.ndarray:
    """以 BGR 格式读取图片，并在失败时抛出异常。"""
    cv2 = require_cv2()
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def write_csv(path: Path, rows: List[dict], fieldnames: Sequence[str]) -> None:
    """将字典列表写入 CSV 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_train_metrics(train_csv: Path, task: str) -> Dict[str, float]:
    """从训练结果 CSV 中读取关键指标的最终值。"""
    rows = []
    with train_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return {}

    def _safe_float(v: str) -> float:
        try:
            return float(v)
        except Exception:
            return float("nan")

    suffix = "M" if task == "segment" else "B"
    key = f"metrics/mAP50-95({suffix})"
    best = max(rows, key=lambda r: _safe_float(r.get(key, "nan")))
    last = rows[-1]
    out = {
        "best_epoch": _safe_float(best.get("epoch", "nan")),
        "best_mAP50": _safe_float(best.get(f"metrics/mAP50({suffix})", "nan")),
        "best_mAP50_95": _safe_float(best.get(f"metrics/mAP50-95({suffix})", "nan")),
        "best_precision": _safe_float(best.get(f"metrics/precision({suffix})", "nan")),
        "best_recall": _safe_float(best.get(f"metrics/recall({suffix})", "nan")),
        "last_mAP50": _safe_float(last.get(f"metrics/mAP50({suffix})", "nan")),
        "last_mAP50_95": _safe_float(last.get(f"metrics/mAP50-95({suffix})", "nan")),
    }
    return out


def maybe_run_val_metrics(
    yolo_model: Any,
    dataset_yaml: Path,
    split: str,
    imgsz: int,
    batch: int,
    device: str,
    task: str,
) -> Dict[str, float]:
    """执行 `model.val()` 并返回关键验证指标。"""
    val_result = yolo_model.val(
        data=str(dataset_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        verbose=False,
        plots=False,
        save_json=False,
    )
    metric_obj = val_result.seg if task == "segment" else val_result.box
    metrics = {
        "val_mAP50": float(getattr(metric_obj, "map50", np.nan)),
        "val_mAP50_95": float(getattr(metric_obj, "map", np.nan)),
    }
    return metrics


def main() -> None:
    """脚本入口：执行多模型特征图对比、可视化和统计导出。"""
    args = parse_args()
    try:
        import torch  # noqa: F401
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(
            "This script requires torch and ultralytics. Install them before running."
        ) from e

    cv2 = require_cv2() if args.vis_images > 0 else None
    save_dir = Path(args.save_dir).resolve()
    vis_dir = save_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    exp_dirs = [x.strip() for x in args.experiments.split(",") if x.strip()]
    model_names = [x.strip() for x in args.model_names.split(",") if x.strip()]
    experiments = collect_experiments(exp_dirs, model_names)
    layer_map = parse_layer_spec(args.layer_spec, len(experiments))
    if not layer_map:
        raise RuntimeError("No valid layer mapping found. Please check --layer-spec.")

    dataset_yaml = Path(args.dataset_yaml).resolve()
    images = list_split_images(dataset_yaml, args.split)
    if not images:
        raise RuntimeError(
            f"No images found in split={args.split}, dataset={dataset_yaml}"
        )
    images = images[: args.max_images]

    # keep images with labels for fg/bg metrics
    labeled_images = []
    for img in images:
        label_path = image_to_label_path(img)
        labels = (
            read_yolo_segments(label_path)
            if args.task == "segment"
            else read_yolo_labels(label_path)
        )
        if labels:
            labeled_images.append((img, labels))
    if not labeled_images:
        raise RuntimeError("No labeled images found for fg/bg metrics.")

    print("=" * 72)
    print("Feature Compare Start")
    print(f"Experiments: {[e.name for e in experiments]}")
    print(f"Dataset: {dataset_yaml}  split={args.split}")
    print(f"Using {len(labeled_images)} images (max_images={args.max_images})")
    print(f"Layer map: {layer_map}")
    print("=" * 72)

    # Load models and build hook collectors
    yolo_models: List[Any] = []
    collectors: List[FeatureHookCollector] = []
    per_model_layer_indices: List[List[int]] = []
    for i, exp in enumerate(experiments):
        model = YOLO(str(exp.weights))
        model.model.eval()
        layer_indices = [v[i] for v in layer_map.values() if v[i] >= 0]
        per_model_layer_indices.append(layer_indices)
        collector = FeatureHookCollector(model.model, layer_indices)
        yolo_models.append(model)
        collectors.append(collector)
        print(f"[Model] {exp.name}: {exp.weights}")
        print(f"        Hook layers: {sorted(set(layer_indices))}")

    per_image_rows: List[dict] = []
    vis_count = 0
    # cka_store[layer_label][model_name][image_stem] = vector
    cka_store: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {
        layer: {exp.name: {} for exp in experiments} for layer in layer_map
    }

    try:
        for img_idx, (img_path, labels) in enumerate(labeled_images):
            image_id = img_path.stem
            # features_by_model_layer[model_name][layer_label] = tensor
            features_by_model_layer: Dict[str, Dict[str, Any]] = {
                exp.name: {} for exp in experiments
            }

            for m_idx, (exp, model, collector) in enumerate(
                zip(experiments, yolo_models, collectors)
            ):
                collector.reset()
                _ = model.predict(
                    source=str(img_path),
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    max_det=args.max_det,
                    device=args.device,
                    verbose=False,
                    save=False,
                )

                for layer_label, layer_idxs in layer_map.items():
                    lidx = layer_idxs[m_idx]
                    if lidx < 0:
                        continue
                    feat = collector.current.get(lidx)
                    if feat is None:
                        continue
                    features_by_model_layer[exp.name][layer_label] = feat

            # Compute per-image quantitative metrics
            for exp in experiments:
                for layer_label in layer_map:
                    feat = features_by_model_layer[exp.name].get(layer_label)
                    if feat is None:
                        continue
                    fmap = feature_to_map(feat)
                    fg_mask = (
                        build_fg_mask_segment(fmap.shape, labels)
                        if args.task == "segment"
                        else build_fg_mask_detect(fmap.shape, labels)
                    )
                    stats = calc_fg_bg_metrics(fmap, fg_mask)
                    row = {
                        "image_id": image_id,
                        "image_path": str(img_path),
                        "model": exp.name,
                        "layer": layer_label,
                    }
                    row.update(stats)
                    per_image_rows.append(row)
                    cka_store[layer_label][exp.name][image_id] = feature_vector_for_cka(
                        feat
                    )

            # Save visualization boards for first vis_images samples
            if vis_count < args.vis_images:
                raw_img = load_image_bgr(img_path)
                for layer_label in layer_map:
                    panels = [raw_img]
                    titles = ["Input"]
                    for exp in experiments:
                        feat = features_by_model_layer[exp.name].get(layer_label)
                        if feat is None:
                            continue
                        fmap = feature_to_map(feat)
                        ov = overlay_heatmap(raw_img, fmap, alpha=0.45)
                        panels.append(ov)
                        titles.append(f"{exp.name}-{layer_label}")

                    if len(panels) <= 1:
                        continue
                    header_h = 42
                    panel_gap = 6
                    board_h = raw_img.shape[0] + header_h
                    board_w = sum(panel.shape[1] for panel in panels) + panel_gap * (
                        len(panels) - 1
                    )
                    board = np.full((board_h, board_w, 3), 255, dtype=np.uint8)
                    header_color = tuple(
                        int(THEME["header_fill"].lstrip("#")[i : i + 2], 16)
                        for i in (4, 2, 0)
                    )
                    edge_color = tuple(
                        int(THEME["header_edge"].lstrip("#")[i : i + 2], 16)
                        for i in (4, 2, 0)
                    )
                    text_color = tuple(
                        int(THEME["text"].lstrip("#")[i : i + 2], 16)
                        for i in (4, 2, 0)
                    )
                    x_cursor = 0
                    for title, panel in zip(titles, panels):
                        x_end = x_cursor + panel.shape[1]
                        board[0:header_h, x_cursor:x_end] = header_color
                        board[header_h:, x_cursor:x_end] = panel
                        cv2.rectangle(  # type: ignore[union-attr]
                            board,
                            (x_cursor, 0),
                            (x_end - 1, board_h - 1),
                            edge_color,
                            1,
                        )
                        cv2.putText(  # type: ignore[union-attr]
                            board,
                            title,
                            (x_cursor + 10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX,  # type: ignore[union-attr]
                            0.72,
                            text_color,
                            2,
                            lineType=cv2.LINE_AA,  # type: ignore[union-attr]
                        )
                        x_cursor = x_end + panel_gap
                    out = vis_dir / f"{img_path.stem}_{layer_label}.jpg"
                    cv2.imwrite(str(out), board)  # type: ignore[union-attr]
                vis_count += 1

            if (img_idx + 1) % 10 == 0 or (img_idx + 1) == len(labeled_images):
                print(f"Processed {img_idx + 1}/{len(labeled_images)} images")

    finally:
        for c in collectors:
            c.close()

    # Aggregate per-layer/model metrics
    grouped: Dict[Tuple[str, str], List[dict]] = {}
    for row in per_image_rows:
        key = (row["model"], row["layer"])
        grouped.setdefault(key, []).append(row)

    layer_summary_rows: List[dict] = []
    for (model_name, layer_label), rows in grouped.items():

        def _mean(k: str) -> float:
            return float(np.mean([float(r[k]) for r in rows])) if rows else float("nan")

        def _std(k: str) -> float:
            return float(np.std([float(r[k]) for r in rows])) if rows else float("nan")

        layer_summary_rows.append(
            {
                "model": model_name,
                "layer": layer_label,
                "num_images": len(rows),
                "fg_mean_mean": _mean("fg_mean"),
                "bg_mean_mean": _mean("bg_mean"),
                "sep_score_mean": _mean("sep_score"),
                "sep_score_std": _std("sep_score"),
                "fg_bg_ratio_mean": _mean("fg_bg_ratio"),
                "fg_bg_ratio_std": _std("fg_bg_ratio"),
                "sparsity_mean": _mean("sparsity"),
            }
        )

    # Compute pairwise CKA
    cka_rows: List[dict] = []
    for layer_label in layer_map:
        for exp_a, exp_b in itertools.combinations(experiments, 2):
            data_a = cka_store[layer_label][exp_a.name]
            data_b = cka_store[layer_label][exp_b.name]
            common_ids = sorted(set(data_a.keys()) & set(data_b.keys()))
            if len(common_ids) < 2:
                cka = float("nan")
                n = len(common_ids)
            else:
                X = np.stack([data_a[k] for k in common_ids], axis=0)
                Y = np.stack([data_b[k] for k in common_ids], axis=0)
                cka = linear_cka(X, Y)
                n = len(common_ids)
            cka_rows.append(
                {
                    "layer": layer_label,
                    "model_a": exp_a.name,
                    "model_b": exp_b.name,
                    "num_images": n,
                    "linear_cka": cka,
                }
            )

    # Detection metrics from training CSV and optional runtime val
    det_rows: List[dict] = []
    for exp, model in zip(experiments, yolo_models):
        row = {
            "model": exp.name,
            "experiment_dir": str(exp.exp_dir),
            "weights": str(exp.weights),
        }
        if exp.train_results_csv is not None:
            row.update(read_train_metrics(exp.train_results_csv, args.task))
        if args.run_val:
            row.update(
                maybe_run_val_metrics(
                    yolo_model=model,
                    dataset_yaml=dataset_yaml,
                    split=args.split,
                    imgsz=args.imgsz,
                    batch=args.val_batch,
                    device=args.device,
                    task=args.task,
                )
            )
        det_rows.append(row)

    # Export CSV files
    write_csv(
        save_dir / "feature_metrics_per_image.csv",
        per_image_rows,
        fieldnames=[
            "image_id",
            "image_path",
            "model",
            "layer",
            "fg_mean",
            "bg_mean",
            "sep_score",
            "fg_bg_ratio",
            "sparsity",
            "fg_pixels",
            "bg_pixels",
        ],
    )
    write_csv(
        save_dir / "feature_metrics_summary.csv",
        sorted(layer_summary_rows, key=lambda x: (x["layer"], x["model"])),
        fieldnames=[
            "model",
            "layer",
            "num_images",
            "fg_mean_mean",
            "bg_mean_mean",
            "sep_score_mean",
            "sep_score_std",
            "fg_bg_ratio_mean",
            "fg_bg_ratio_std",
            "sparsity_mean",
        ],
    )
    write_csv(
        save_dir / "cka_pairwise.csv",
        cka_rows,
        fieldnames=["layer", "model_a", "model_b", "num_images", "linear_cka"],
    )

    # Collect all keys to keep detection csv flexible.
    det_fields = sorted(set().union(*(row.keys() for row in det_rows)))
    write_csv(save_dir / "detection_metrics.csv", det_rows, fieldnames=det_fields)

    print("\nDone.")
    print(f"Visualizations: {vis_dir}")
    print(f"CSV: {save_dir / 'feature_metrics_per_image.csv'}")
    print(f"CSV: {save_dir / 'feature_metrics_summary.csv'}")
    print(f"CSV: {save_dir / 'cka_pairwise.csv'}")
    print(f"CSV: {save_dir / 'detection_metrics.csv'}")


if __name__ == "__main__":
    main()
