#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gradio 推理对比界面。

功能：
- 提供多模型、多图片的 YOLO 推理可视化与结果对比界面。
- 支持推理耗时统计、结果缓存、配置加载，以及可选的 Transformer 组件。
- 面向当前工作区实验模型的交互式验证与展示。

使用：
- `python test_infer_ui.py --workspace D:/YOLO_PCB`
- 启动后通过浏览器访问本地 Gradio 页面。
"""

from __future__ import annotations

import argparse
import base64
import html
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 按需加载较重依赖，减少查看帮助或轻量检查时的启动开销。
_cv2 = None
_gr = None
_torch = None
_yolo_cls = None
_transformers_ready = None
_transformers_error = ""


def require_cv2():
    """按需导入 OpenCV，并在缺失时抛出清晰错误。"""
    global _cv2
    if _cv2 is None:
        try:
            import cv2 as cv2_mod  # type: ignore

            _cv2 = cv2_mod
        except Exception as e:
            raise RuntimeError(
                "OpenCV is required. Install with: pip install opencv-python"
            ) from e
    return _cv2


def require_gradio():
    """按需导入 Gradio，并在缺失时抛出清晰错误。"""
    global _gr
    if _gr is None:
        try:
            import gradio as gr_mod  # type: ignore

            _gr = gr_mod
        except Exception as e:
            raise RuntimeError(
                "Gradio is required. Install with: pip install gradio"
            ) from e
    return _gr


def require_torch():
    """按需导入 PyTorch，并在缺失时抛出清晰错误。"""
    global _torch
    if _torch is None:
        try:
            import torch as torch_mod  # type: ignore

            _torch = torch_mod
        except Exception as e:
            raise RuntimeError(
                "PyTorch is required. Install with: pip install torch"
            ) from e
    return _torch


def require_yolo():
    """按需导入 Ultralytics YOLO 类。"""
    global _yolo_cls
    if _yolo_cls is None:
        try:
            from ultralytics import YOLO as yolo_cls  # type: ignore

            _yolo_cls = yolo_cls
        except Exception as e:
            raise RuntimeError(
                "Ultralytics is required. Install with: pip install ultralytics"
            ) from e
    return _yolo_cls


def transformers_status() -> Tuple[bool, str]:
    """检查 `transformers` 是否可用，并缓存检查结果。"""
    global _transformers_ready, _transformers_error
    if _transformers_ready is not None:
        return _transformers_ready, _transformers_error
    try:
        import transformers  # type: ignore  # noqa: F401

        _transformers_ready = True
        _transformers_error = ""
    except Exception as e:
        _transformers_ready = False
        _transformers_error = str(e)
    return _transformers_ready, _transformers_error


def model_may_require_transformers(model: ModelInfo) -> bool:
    """根据模型路径特征粗略判断是否可能依赖 Transformers。"""
    p = str(model.path).lower()
    return "dino" in p or "dinov" in p


def short_error_text(e: Exception, limit: int = 260) -> str:
    """压缩异常文本，便于在界面中展示。"""
    msg = str(e).strip().replace("\n", " ")
    return msg if len(msg) <= limit else (msg[: limit - 3] + "...")


def image_to_label_path(img_path: Path) -> Path:
    """将图片路径映射到同名标签路径。"""
    p = img_path.as_posix()
    if "/images/" in p:
        return Path(p.replace("/images/", "/labels/")).with_suffix(".txt")
    return img_path.parent.parent / "labels" / f"{img_path.stem}.txt"


def box_iou_xyxy(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> float:
    """计算两个 `xyxy` 边界框的 IoU。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def parse_gt_labels_normalized(
    label_path: Path,
) -> List[Tuple[int, float, float, float, float]]:
    """读取归一化格式的 YOLO 真值标签。"""
    out: List[Tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return out
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return out
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
        except Exception:
            continue
        if len(parts) == 5:
            try:
                xc, yc, w, h = map(float, parts[1:5])
            except Exception:
                continue
            x1 = max(0.0, xc - w / 2.0)
            y1 = max(0.0, yc - h / 2.0)
            x2 = min(1.0, xc + w / 2.0)
            y2 = min(1.0, yc + h / 2.0)
            out.append((cls, x1, y1, x2, y2))
        elif (len(parts) - 1) % 2 == 0:
            coords = []
            ok = True
            for i in range(1, len(parts), 2):
                try:
                    coords.append((float(parts[i]), float(parts[i + 1])))
                except Exception:
                    ok = False
                    break
            if not ok or not coords:
                continue
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            out.append(
                (
                    cls,
                    max(0.0, min(xs)),
                    max(0.0, min(ys)),
                    min(1.0, max(xs)),
                    min(1.0, max(ys)),
                )
            )
    return out


def polygon_to_mask(
    polygon: Sequence[Tuple[float, float]], image_shape: Tuple[int, int]
) -> np.ndarray:
    """将归一化多边形转为原图尺寸二值 mask。"""
    cv2 = require_cv2()
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    if not polygon:
        return mask
    pts = []
    for x, y in polygon:
        px = int(round(max(0.0, min(1.0, x)) * max(0, width - 1)))
        py = int(round(max(0.0, min(1.0, y)) * max(0, height - 1)))
        pts.append([px, py])
    if len(pts) >= 3:
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 1)
    return mask


def parse_gt_instances(
    label_path: Path,
    image_shape: Tuple[int, int],
) -> List[Dict[str, object]]:
    """读取 YOLO detect/segment 标签并构建统一 GT 结构。"""
    out: List[Dict[str, object]] = []
    if not label_path.exists():
        return out
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return out

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
        except Exception:
            continue

        instance: Dict[str, object] = {"cls": cls, "mask": None}
        if len(parts) == 5:
            try:
                xc, yc, w, h = map(float, parts[1:5])
            except Exception:
                continue
            x1 = max(0.0, xc - w / 2.0)
            y1 = max(0.0, yc - h / 2.0)
            x2 = min(1.0, xc + w / 2.0)
            y2 = min(1.0, yc + h / 2.0)
            instance["bbox"] = (x1, y1, x2, y2)
            out.append(instance)
            continue

        if (len(parts) - 1) % 2 != 0:
            continue

        polygon: List[Tuple[float, float]] = []
        ok = True
        for i in range(1, len(parts), 2):
            try:
                polygon.append((float(parts[i]), float(parts[i + 1])))
            except Exception:
                ok = False
                break
        if not ok or not polygon:
            continue
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        instance["bbox"] = (
            max(0.0, min(xs)),
            max(0.0, min(ys)),
            min(1.0, max(xs)),
            min(1.0, max(ys)),
        )
        instance["mask"] = polygon_to_mask(polygon, image_shape)
        out.append(instance)
    return out


def mask_iou_np(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """计算两个二值 mask 的 IoU。"""
    if mask_a.shape != mask_b.shape:
        return 0.0
    inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
    if inter <= 0:
        return 0.0
    union = np.logical_or(mask_a > 0, mask_b > 0).sum()
    return float(inter / union) if union > 0 else 0.0


def prediction_overlap(
    pred: Dict[str, object], gt: Dict[str, object], effective_task: str
) -> float:
    """按任务类型计算预测与 GT 的匹配 IoU。"""
    if effective_task == "segment":
        pred_mask = pred.get("mask")
        gt_mask = gt.get("mask")
        if isinstance(pred_mask, np.ndarray) and isinstance(gt_mask, np.ndarray):
            return mask_iou_np(pred_mask, gt_mask)
        return 0.0
    pred_box = pred.get("bbox")
    gt_box = gt.get("bbox")
    if pred_box is None or gt_box is None:
        return 0.0
    return box_iou_xyxy(pred_box, gt_box)  # type: ignore[arg-type]


def classwise_nms(
    preds: List[Dict[str, object]],
    conf_thres: float,
    iou_thres: float,
    effective_task: str,
) -> List[Dict[str, object]]:
    """按类别执行简单 NMS，返回过滤后的预测结果。"""
    by_cls: Dict[int, List[Dict[str, object]]] = {}
    for p in preds:
        if effective_task == "segment" and not isinstance(p.get("mask"), np.ndarray):
            continue
        cls = int(p["cls"])
        conf = float(p["conf"])
        if conf < conf_thres:
            continue
        by_cls.setdefault(cls, []).append(p)

    kept: List[Dict[str, object]] = []
    for cls, items in by_cls.items():
        items_sorted = sorted(items, key=lambda x: float(x["conf"]), reverse=True)
        while items_sorted:
            best = items_sorted.pop(0)
            kept.append(best)
            remain = []
            for cand in items_sorted:
                if prediction_overlap(best, cand, effective_task) <= iou_thres:
                    remain.append(cand)
            items_sorted = remain
    kept.sort(key=lambda x: float(x["conf"]), reverse=True)
    return kept


def filter_gt_instances_for_task(
    gts: List[Dict[str, object]], effective_task: str
) -> List[Dict[str, object]]:
    """按任务过滤 GT；segment 只保留带有效 mask 的实例。"""
    if effective_task != "segment":
        return gts
    return [gt for gt in gts if isinstance(gt.get("mask"), np.ndarray)]


def build_prediction_cache_items(
    result, effective_task: str
) -> Tuple[List[Dict[str, object]], bool]:
    """从一次推理结果构建阈值优化缓存项，并返回是否满足任务要求。"""
    boxes = getattr(result, "boxes", None)
    box_count = int(len(boxes)) if boxes is not None else 0
    if boxes is None or boxes.xyxy is None or box_count == 0:
        return [], True

    h, w = result.orig_shape
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
    clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []

    mask_array = None
    if effective_task == "segment":
        masks = getattr(result, "masks", None)
        if masks is None or getattr(masks, "data", None) is None:
            return [], False
        try:
            mask_array = masks.data.cpu().numpy()
        except Exception:
            return [], False
        if mask_array is None or len(mask_array) == 0:
            return [], False

    n = min(len(xyxy), len(confs), len(clss))
    if effective_task == "segment":
        n = min(n, len(mask_array))
    if n <= 0 and box_count > 0:
        return [], effective_task != "segment"

    preds_norm: List[Dict[str, object]] = []
    for i in range(n):
        x1, y1, x2, y2 = xyxy[i].tolist()
        pred_item: Dict[str, object] = {
            "cls": int(clss[i]),
            "conf": float(confs[i]),
            "bbox": (
                float(max(0.0, min(1.0, x1 / max(1, w)))),
                float(max(0.0, min(1.0, y1 / max(1, h)))),
                float(max(0.0, min(1.0, x2 / max(1, w)))),
                float(max(0.0, min(1.0, y2 / max(1, h)))),
            ),
            "mask": None,
        }
        if effective_task == "segment" and mask_array is not None:
            pred_item["mask"] = (mask_array[i] > 0.5).astype(np.uint8)
        preds_norm.append(pred_item)
    return preds_norm, True


def eval_f1_from_cache(
    cache: List[Dict[str, object]],
    conf_thres: float,
    iou_thres: float,
    effective_task: str,
    match_iou: float = 0.5,
) -> Tuple[float, float, float, int, int, int]:
    """基于缓存预测结果快速评估 Precision、Recall 和 F1。"""
    tp = 0
    fp = 0
    fn = 0
    for item in cache:
        raw_preds = item["preds"]  # type: ignore[index]
        gts = item["gts"]  # type: ignore[index]
        preds = classwise_nms(
            raw_preds,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            effective_task=effective_task,
        )
        gt_used = [False] * len(gts)
        for pred in preds:
            pcls = int(pred["cls"])
            best_j = -1
            best_iou = 0.0
            for j, gt in enumerate(gts):
                if gt_used[j]:
                    continue
                gcls = int(gt["cls"])
                if gcls != pcls:
                    continue
                iou = prediction_overlap(pred, gt, effective_task)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0 and best_iou >= match_iou:
                gt_used[best_j] = True
                tp += 1
            else:
                fp += 1
        fn += sum(0 if u else 1 for u in gt_used)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-9)
    return precision, recall, f1, tp, fp, fn


def _sparkline(values: List[float]) -> str:
    """将数值序列编码成适合终端显示的简易火花线。"""
    if not values:
        return ""
    # ASCII ramp for robust rendering across fonts/terminals.
    blocks = " .:-=+*#%@"
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-12:
        return "-" * len(values)
    chars = []
    for v in values:
        norm = (v - vmin) / (vmax - vmin)
        idx = max(0, min(len(blocks) - 1, int(round(norm * (len(blocks) - 1)))))
        chars.append(blocks[idx])
    return "".join(chars)


def _downsample(values: List[float], max_points: int = 48) -> List[float]:
    """将长序列下采样到固定点数，便于展示。"""
    if len(values) <= max_points:
        return values
    step = (len(values) - 1) / (max_points - 1)
    out = []
    for i in range(max_points):
        idx = int(round(i * step))
        out.append(values[idx])
    return out


def build_opt_history_markdown(history: List[Tuple[int, float, float, float]]) -> str:
    """把参数搜索历史转换为 Markdown 摘要。"""
    if not history:
        return "### 优化历史\n- no trials"
    history_sorted = sorted(history, key=lambda x: x[3], reverse=True)
    f1_values = [h[3] for h in history]
    best_so_far = []
    cur = -1.0
    for v in f1_values:
        cur = max(cur, v)
        best_so_far.append(cur)
    f1_sample = _downsample(f1_values, max_points=48)
    best_sample = _downsample(best_so_far, max_points=48)
    lines = [
        "### 优化历史",
        f"- trials: {len(history)}",
        f"- F1 curve: {_sparkline(f1_sample)}",
        f"- F1 stats: start {f1_values[0]:.4f}, end {f1_values[-1]:.4f}, best {max(f1_values):.4f}",
        f"- best-so-far: {_sparkline(best_sample)}",
        f"- best progression: start {best_so_far[0]:.4f}, end {best_so_far[-1]:.4f}",
        "",
        "| trial | conf | iou | f1 |",
        "|---:|---:|---:|---:|",
    ]
    for t, c, n, f1 in history_sorted[:8]:
        lines.append(f"| {t} | {c:.4f} | {n:.4f} | {f1:.4f} |")
    return "\n".join(lines)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MODEL_EXTS = {".pt", ".onnx"}
DEFAULT_WORKSPACE = Path(r"D:\YOLO_PCB")
DEFAULT_SEGMENT_SAVE_DIR = Path(r"D:\YOLO_PCB\SegmentData\images")
MAX_IMAGE_CANDIDATES = 800

MODERN_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Manrope:wght@500;700;800&display=swap');

:root {
  --bg-1: #f4f8f4;
  --bg-2: #eaf4ef;
  --panel: #1b2b45;
  --ink: #10231a;
  --muted: #365246;
  --accent: #0d8f5b;
  --accent-2: #e68a2e;
  --border: rgba(157, 198, 255, 0.2);
}

.gradio-container {
  font-family: 'Space Grotesk', 'Microsoft YaHei UI', sans-serif !important;
  color: var(--ink) !important;
  background:
    radial-gradient(1200px 800px at -5% -10%, #d6efe2 0%, transparent 62%),
    radial-gradient(900px 700px at 110% 0%, #f5deb8 0%, transparent 62%),
    linear-gradient(140deg, var(--bg-1), var(--bg-2));
}

#hero {
  background: linear-gradient(130deg, rgba(12, 122, 80, 0.96), rgba(33, 146, 107, 0.9));
  color: #f3fff8;
  border-radius: 18px;
  padding: 20px 22px;
  border: 1px solid rgba(255, 255, 255, 0.28);
  box-shadow: 0 22px 44px rgba(14, 52, 35, 0.2);
  animation: fadeSlide 420ms ease;
}

#hero h2, #hero p {
  margin: 0;
}

#hero p {
  margin-top: 8px;
  opacity: 0.96;
}

.panel {
  background: linear-gradient(160deg, #172740 0%, #1e2f4c 100%) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px;
  box-shadow: 0 14px 28px rgba(8, 20, 40, 0.24);
  padding: 12px !important;
  animation: fadeSlide 520ms ease;
}

.panel .gr-markdown,
.panel .gr-markdown p,
.panel .gr-markdown li,
.panel .gr-markdown h1,
.panel .gr-markdown h2,
.panel .gr-markdown h3,
.panel .gr-markdown h4 {
  color: #e9f1ff !important;
}

.model-scroll {
  max-height: 320px;
  overflow: auto;
  border: 1px solid rgba(157, 198, 255, 0.25);
  border-radius: 12px;
  background: rgba(13, 23, 38, 0.55);
  padding: 8px;
}

.summary-scroll {
  max-height: 420px;
  overflow: auto;
}

.control-compact .gr-form,
.control-compact .gr-box,
.control-compact .gr-group {
  gap: 8px !important;
}

.result-image img {
  border-radius: 10px;
}

#left-control {
  align-self: start;
}

#right-workbench {
  align-self: start;
}

#result-panel {
  background: rgba(10, 20, 36, 0.35);
  border: 1px solid rgba(157, 198, 255, 0.2);
  border-radius: 12px;
  padding: 8px;
  min-height: 420px;
}

#summary-panel {
  background: rgba(17, 31, 52, 0.92);
  border: 1px solid rgba(157, 198, 255, 0.2);
  border-radius: 12px;
  padding: 8px;
  min-height: 420px;
}

#summary-panel .gr-markdown,
#summary-panel .gr-markdown p,
#summary-panel .gr-markdown li,
#summary-panel .gr-markdown h1,
#summary-panel .gr-markdown h2,
#summary-panel .gr-markdown h3 {
  color: #e9f1ff !important;
}

#summary-panel .tabs,
#summary-panel .tab-nav,
#summary-panel .tabitem {
  background: transparent !important;
}

.det-result-root {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.det-image-stage {
  position: relative;
  width: 100%;
  min-height: 320px;
  border-radius: 10px;
  overflow: hidden;
  background: rgba(6, 15, 28, 0.9);
}

.det-base-image {
  width: 100%;
  height: 100%;
  display: block;
  object-fit: contain;
}

.det-base-image-seg {
  image-rendering: pixelated;
}

.det-click-box {
  position: absolute;
  background: rgba(0, 0, 0, 0.02);
  border: 2px solid #22c55e;
  border-radius: 6px;
  cursor: pointer;
  transition: transform 120ms ease, box-shadow 120ms ease, background 120ms ease;
}

.det-click-box:hover,
.det-click-box:focus {
  background: rgba(34, 197, 94, 0.12);
  box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.28);
  transform: scale(1.01);
}

.det-click-box span {
  position: absolute;
  left: 0;
  top: 0;
  transform: translateY(-100%);
  background: rgba(15, 23, 42, 0.92);
  color: #f8fafc;
  padding: 2px 6px;
  border-radius: 6px 6px 6px 0;
  font-size: 11px;
  white-space: nowrap;
}

.det-chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.det-chip {
  border: none;
  border-radius: 8px;
  padding: 6px 10px;
  font-size: 12px;
  cursor: pointer;
}

.det-chip-empty {
  background: rgba(71, 85, 105, 0.8);
  color: #e2e8f0;
  cursor: default;
}

#save-modal-wrap {
  position: fixed;
  inset: 0;
  z-index: 1200;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(3, 7, 18, 0.62);
  backdrop-filter: blur(4px);
}

#save-modal {
  width: min(760px, calc(100vw - 32px));
  max-height: calc(100vh - 40px);
  overflow: auto;
  background: linear-gradient(160deg, #172740 0%, #1e2f4c 100%);
  border: 1px solid rgba(157, 198, 255, 0.25);
  border-radius: 18px;
  box-shadow: 0 24px 56px rgba(8, 20, 40, 0.38);
  padding: 16px;
}

#summary-panel button[role="tab"] {
  color: #dbe7ff !important;
}

#summary-panel button[role="tab"][aria-selected="true"] {
  color: #23d0a4 !important;
}

.model-scroll .gr-markdown table td,
.model-scroll .gr-markdown table th {
  font-size: 12px !important;
  line-height: 1.35 !important;
}

footer { display: none !important; }

.gr-button {
  border-radius: 12px !important;
  border: 1px solid rgba(12, 95, 63, 0.22) !important;
}

.gr-button-primary {
  background: linear-gradient(135deg, #0e8a59, #18a56d) !important;
  color: #f6fff9 !important;
}

.gr-button-secondary {
  background: linear-gradient(135deg, #31435f, #42526f) !important;
  color: #edf3ff !important;
}

.gr-markdown table {
  border-radius: 12px;
  overflow: hidden;
}

.gr-markdown td, .gr-markdown th {
  font-size: 12.5px;
}

@keyframes fadeSlide {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 900px) {
  #hero { padding: 16px; border-radius: 14px; }
  .panel { border-radius: 12px; }
}
"""

@dataclass
class DatasetInfo:
    """描述一个数据集配置及其可供界面展示的摘要信息。"""

    name: str
    yaml_path: Path
    task: str
    nc: Optional[int]
    names: List[str]
    image_paths: List[Path]
    image_count: int
    split_dirs: Dict[str, List[Path]]
    image_classes: Dict[str, List[int]]
    tag: str


@dataclass
class ModelInfo:
    """描述一个模型文件及其在界面中的展示元数据。"""

    path: Path
    fmt: str
    size_mb: float
    mtime: float
    inferred_task: str
    dataset_hint: str
    rank_hint: str
    train_args_path: Optional[Path]
    train_data_path: Optional[Path]
    train_nc: Optional[int]
    train_names: List[str]


def clamp_box_to_image(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> Optional[Tuple[int, int, int, int]]:
    """将浮点框坐标裁剪到图像边界内。"""
    ix1 = max(0, min(int(round(x1)), max(0, width - 1)))
    iy1 = max(0, min(int(round(y1)), max(0, height - 1)))
    ix2 = max(ix1 + 1, min(int(round(x2)), width))
    iy2 = max(iy1 + 1, min(int(round(y2)), height))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1, iy1, ix2, iy2


def serialize_detections(result, label_map: Dict[int, str]) -> List[Dict[str, object]]:
    """提取当前结果中的检测框，供 ROI 对比和裁剪保存使用。"""
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
    clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []
    height, width = result.orig_shape
    detections: List[Dict[str, object]] = []
    for idx in range(min(len(xyxy), len(confs), len(clss))):
        clamped = clamp_box_to_image(*xyxy[idx].tolist(), width=width, height=height)
        if clamped is None:
            continue
        x1, y1, x2, y2 = clamped
        detections.append(
            {
                "index": idx + 1,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": x2 - x1,
                "height": y2 - y1,
                "conf": float(confs[idx]),
                "cls": int(clss[idx]),
                "label": label_map.get(int(clss[idx]), str(int(clss[idx]))),
                "area": int((x2 - x1) * (y2 - y1)),
            }
        )
    return detections


def next_save_filename(source_path: str, counter: int) -> str:
    """生成默认保存文件名。"""
    return f"{Path(source_path).stem}_{counter}.png"


def build_detection_list_markdown(
    detections: List[Dict[str, object]], effective_task: str
) -> str:
    """构建批量保存弹窗中的结果列表。"""
    if not detections:
        noun = "分割结果" if effective_task == "segment" else "检测框"
        return f"### 结果列表\n- 当前推理没有{noun}。"
    size_title = "包围框像素尺寸" if effective_task == "segment" else "像素尺寸"
    lines = [
        "### 结果列表",
        f"- 结果数量: `{len(detections)}`",
        "",
        f"| 序号 | 类别 | 置信度 | {size_title} | 坐标(xyxy) |",
        "|---:|---|---:|---:|---|",
    ]
    for det in detections:
        lines.append(
            f"| {int(det['index'])} | {det['label']} | {float(det['conf']):.4f} | "
            f"{int(det['width'])}x{int(det['height'])} | "
            f"({int(det['x1'])}, {int(det['y1'])}, {int(det['x2'])}, {int(det['y2'])}) |"
        )
    return "\n".join(lines)


def image_to_data_url(image) -> str:
    """将 RGB numpy 图像编码为可嵌入 HTML 的 data URL。"""
    cv2 = require_cv2()
    ok, buffer = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Failed to encode image for HTML preview.")
    return f"data:image/png;base64,{base64.b64encode(buffer.tobytes()).decode('ascii')}"


def segment_palette_rgb() -> List[Tuple[int, int, int]]:
    """返回分割可视化使用的高对比度颜色板。"""
    return [
        (239, 68, 68),
        (16, 185, 129),
        (59, 130, 246),
        (245, 158, 11),
        (168, 85, 247),
        (20, 184, 166),
    ]


def segment_palette_css() -> List[Tuple[str, str]]:
    """返回与分割掩码一致的标签颜色。"""
    return [
        ("#ef4444", "#fee2e2"),
        ("#10b981", "#d1fae5"),
        ("#3b82f6", "#dbeafe"),
        ("#f59e0b", "#fef3c7"),
        ("#a855f7", "#f3e8ff"),
        ("#14b8a6", "#ccfbf1"),
    ]


def upscale_for_preview(display_image, effective_task: str, min_side: int = 520):
    """对小图做预览放大，segment 使用最近邻保持边缘清晰。"""
    cv2 = require_cv2()
    h, w = display_image.shape[:2]
    short_side = min(h, w)
    if short_side >= min_side:
        return display_image
    scale = float(min_side) / max(1, short_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_NEAREST if effective_task == "segment" else cv2.INTER_LINEAR
    return cv2.resize(display_image, (new_w, new_h), interpolation=interpolation)


def render_segment_preview(result):
    """为 segment 任务生成更清晰的掩码叠加图。"""
    cv2 = require_cv2()
    base_image = result.orig_img.copy()
    if base_image.ndim == 2:
        base_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2RGB)
    elif base_image.shape[2] == 4:
        base_image = cv2.cvtColor(base_image, cv2.COLOR_BGRA2RGB)
    else:
        base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
    display_image = base_image.copy()
    masks = getattr(result, "masks", None)
    if masks is None or getattr(masks, "data", None) is None:
        return upscale_for_preview(display_image, "segment")

    palette = segment_palette_rgb()
    overlay = display_image.copy()
    img_h, img_w = overlay.shape[:2]
    line_width = max(2, int(round(min(img_h, img_w) / 120)))
    try:
        mask_array = masks.data.detach().cpu().numpy()
    except Exception:
        return upscale_for_preview(display_image, "segment")

    for idx, mask in enumerate(mask_array):
        if getattr(mask, "ndim", 0) != 2:
            continue
        if mask.shape != (img_h, img_w):
            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        mask_u8 = (mask > 0.5).astype("uint8")
        if int(mask_u8.sum()) == 0:
            continue
        color = palette[idx % len(palette)]
        color_arr = np.array(color, dtype=np.uint8)
        overlay[mask_u8.astype(bool)] = (
            0.62 * overlay[mask_u8.astype(bool)] + 0.38 * color_arr
        ).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(overlay, contours, -1, color, thickness=line_width)

    return upscale_for_preview(overlay, "segment")


def render_result_image(result, effective_task: str):
    """根据任务类型生成更适合界面展示的结果图。"""
    cv2 = require_cv2()
    if effective_task == "segment":
        return render_segment_preview(result)
    plotted = result.plot(line_width=2, font_size=14, color_mode="class")
    display_image = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
    return upscale_for_preview(display_image, effective_task)


def build_result_html(
    display_image, detections: List[Dict[str, object]], effective_task: str
) -> str:
    """构建 detect/segment 通用结果展示 HTML。"""
    img_h, img_w = display_image.shape[:2]
    image_url = image_to_data_url(display_image)
    chip_parts: List[str] = []
    palette = segment_palette_css() if effective_task == "segment" else [
        ("#7f1d1d", "#fecaca"),
        ("#14532d", "#bbf7d0"),
        ("#1e3a8a", "#bfdbfe"),
        ("#78350f", "#fde68a"),
        ("#581c87", "#e9d5ff"),
        ("#0f766e", "#99f6e4"),
    ]
    for idx, det in enumerate(detections):
        border_color, chip_bg = palette[idx % len(palette)]
        label_text = (
            f"#{int(det['index'])} {det['label']}"
            if effective_task == "segment"
            else f"#{int(det['index'])} {det['label']} {float(det['conf']):.2f}"
        )
        label = html.escape(label_text)
        chip_parts.append(
            "<span "
            "class='det-chip' "
            f"style='background:{border_color};color:{chip_bg};'>"
            f"{label}"
            "</span>"
        )

    if not chip_parts:
        empty_text = "无分割结果" if effective_task == "segment" else "无检测框"
        chip_parts.append(f"<span class='det-chip det-chip-empty'>{empty_text}</span>")

    return (
        "<div class='det-result-root'>"
        f"<div class='det-image-stage' style='aspect-ratio:{img_w} / {img_h};'>"
        f"<img src='{image_url}' alt='detection result' class='det-base-image "
        f"{'det-base-image-seg' if effective_task == 'segment' else ''}' />"
        "</div>"
        f"<div class='det-chip-row'>{''.join(chip_parts)}</div>"
        "</div>"
    )


def resolve_origin_image_path(source_path: str) -> Optional[Path]:
    """为 PKU-Market-PCB-ex 检测图解析 origin 原图路径。"""
    img_path = Path(source_path).resolve()
    path_lower = str(img_path).lower()
    if "pku-market-pcb-ex" not in path_lower:
        return None

    images_dir = None
    for parent in img_path.parents:
        if parent.name.lower() == "images":
            images_dir = parent
            break
    if images_dir is None:
        return None

    origin_dir = images_dir.parent / "origin"
    if not origin_dir.exists():
        return None

    direct_match = origin_dir / img_path.name
    if direct_match.exists():
        return direct_match.resolve()

    prefix = img_path.stem.split("_", 1)[0]
    fallback_match = origin_dir / f"{prefix}.jpg"
    if fallback_match.exists():
        return fallback_match.resolve()
    return None


def load_rgb_image(image_path: Path):
    """读取本地图像并统一转为 RGB。"""
    cv2 = require_cv2()
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def build_roi_compare_choices(
    detections: List[Dict[str, object]],
) -> List[Tuple[str, str]]:
    """构建 ROI 对比下拉选项。"""
    choices: List[Tuple[str, str]] = []
    for det in detections:
        label = (
            f"#{int(det['index'])} {det['label']} {float(det['conf']):.2f} | "
            f"bbox=({int(det['x1'])}, {int(det['y1'])}, {int(det['x2'])}, {int(det['y2'])}) | "
            f"size={int(det['width'])}x{int(det['height'])}"
        )
        choices.append((label, str(int(det["index"]))))
    return choices


def find_detection_by_index(
    detections: List[Dict[str, object]], detection_index: int
) -> Optional[Dict[str, object]]:
    """按编号查找检测结果。"""
    for det in detections:
        if int(det["index"]) == detection_index:
            return det
    return None


def generate_roi_diff_outputs(
    detect_roi, origin_roi
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对两张 ROI 做滤波、差分和高亮叠加。"""
    cv2 = require_cv2()
    detect_filtered = cv2.bilateralFilter(detect_roi, d=9, sigmaColor=75, sigmaSpace=75)
    origin_filtered = cv2.bilateralFilter(origin_roi, d=9, sigmaColor=75, sigmaSpace=75)
    diff = cv2.absdiff(detect_filtered, origin_filtered)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    _thr, diff_mask = cv2.threshold(
        diff_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    overlay = detect_roi.copy()
    mask_bool = diff_mask > 0
    if np.any(mask_bool):
        red = np.array([255, 0, 0], dtype=np.uint8)
        overlay[mask_bool] = (
            0.6 * overlay[mask_bool] + 0.4 * red
        ).astype(np.uint8)
    diff_mask_rgb = np.stack([diff_mask, diff_mask, diff_mask], axis=-1)
    return diff_mask_rgb, overlay, diff_mask


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="启动 detect/segment 模型的交互式推理界面。"
    )
    parser.add_argument(
        "--workspace", type=str, default=str(DEFAULT_WORKSPACE), help="工作区根目录。"
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Gradio 监听地址。"
    )
    parser.add_argument("--port", type=int, default=7860, help="Gradio 监听端口。")
    parser.add_argument(
        "--share", action="store_true", help="启用 Gradio 的外网分享链接。"
    )
    parser.add_argument(
        "--imgsz", type=int, default=960, help="界面默认使用的推理图像尺寸。"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5, help="界面默认使用的置信度阈值。"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45, help="界面默认使用的 NMS IoU 阈值。"
    )
    parser.add_argument(
        "--max-det", type=int, default=300, help="界面默认允许的最大检测数。"
    )
    return parser.parse_args()


def is_image(path: Path) -> bool:
    """判断路径是否为支持的图片文件。"""
    return path.suffix.lower() in IMG_EXTS


def detect_tag_from_path(path_str: str) -> str:
    """根据路径关键词推断数据集或实验标签。"""
    p = path_str.lower()
    if "riva" in p:
        return "riva"
    if "pku-market-pcb-ex" in p or "\\ex_" in p or "/ex_" in p:
        return "pku_ex"
    if "pku-market-pcb-raw" in p or "raw" in p:
        return "pku_raw"
    if "pku-market-pcb" in p or "pcb" in p:
        return "pku"
    return "general"


def to_names_list(raw_names: object) -> List[str]:
    """将类别名称配置统一转换为字符串列表。"""
    if isinstance(raw_names, list):
        return [str(x) for x in raw_names]
    if isinstance(raw_names, dict):
        out: List[Tuple[int, str]] = []
        for k, v in raw_names.items():
            try:
                out.append((int(k), str(v)))
            except Exception:
                continue
        out.sort(key=lambda x: x[0])
        return [v for _, v in out]
    return []


def resolve_split_dirs(
    dataset_yaml: Path, split_value: object, root_value: object
) -> List[Path]:
    """解析 YAML 中某个数据集划分对应的目录列表。"""
    if split_value is None:
        return []
    values = split_value if isinstance(split_value, list) else [split_value]
    root = (
        Path(str(root_value)).resolve() if root_value else dataset_yaml.parent.resolve()
    )
    dirs: List[Path] = []
    for v in values:
        p = Path(str(v))
        dirs.append(p.resolve() if p.is_absolute() else (root / p).resolve())
    return dirs


def infer_task_from_labels(split_dirs: Dict[str, List[Path]]) -> str:
    """根据标签文件格式推断任务类型。"""
    checked = 0
    saw_seg = False
    saw_det = False
    for dirs in split_dirs.values():
        for split_dir in dirs:
            if not split_dir.exists():
                continue
            for txt in split_dir.rglob("*.txt"):
                checked += 1
                if checked > 200:
                    break
                try:
                    lines = txt.read_text(encoding="utf-8").splitlines()
                except Exception:
                    continue
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) == 5:
                        saw_det = True
                    elif len(parts) > 5 and (len(parts) - 1) % 2 == 0:
                        saw_seg = True
                    if saw_det and saw_seg:
                        return "segment"
            if checked > 200:
                break
    if saw_seg:
        return "segment"
    if saw_det:
        return "detect"
    return "detect"


def list_images_from_split_dirs(
    split_dirs: Dict[str, List[Path]], limit: Optional[int] = MAX_IMAGE_CANDIDATES
) -> Tuple[List[Path], int]:
    """从多个划分目录中收集图片路径，并限制最大候选数量。"""
    images: List[Path] = []
    count = 0
    for dirs in split_dirs.values():
        for split_dir in dirs:
            if not split_dir.exists():
                continue
            if split_dir.is_file() and is_image(split_dir):
                count += 1
                if limit is None or len(images) < limit:
                    images.append(split_dir)
                continue
            for p in split_dir.rglob("*"):
                if not p.is_file() or not is_image(p):
                    continue
                count += 1
                if limit is None or len(images) < limit:
                    images.append(p)
    images = sorted(set(images))
    return images, count


def collect_image_class_index(
    image_paths: Sequence[Path],
) -> Dict[str, List[int]]:
    """为图片建立类别索引，便于按类别过滤候选图像。"""
    image_classes: Dict[str, List[int]] = {}
    for img_path in image_paths:
        class_ids: set[int] = set()
        label_path = image_to_label_path(img_path)
        if label_path.exists():
            try:
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        class_ids.add(int(float(parts[0])))
                    except Exception:
                        continue
            except Exception:
                pass
        image_classes[str(img_path)] = sorted(class_ids)
    return image_classes


def filter_images_by_classes(
    dataset: DatasetInfo, selected_class_names: Sequence[str]
) -> List[str]:
    """根据所选类别过滤数据集图片；未选择时返回整个数据集。"""
    if not selected_class_names:
        return [str(p) for p in dataset.image_paths]
    selected_ids = {
        idx for idx, name in enumerate(dataset.names) if name in set(selected_class_names)
    }
    if not selected_ids:
        return [str(p) for p in dataset.image_paths]
    filtered: List[str] = []
    for img_path in dataset.image_paths:
        class_ids = dataset.image_classes.get(str(img_path), [])
        if any(class_id in selected_ids for class_id in class_ids):
            filtered.append(str(img_path))
    return filtered


def dataset_class_choices(dataset: DatasetInfo) -> List[str]:
    """返回可用于筛图的类别列表。"""
    if dataset.names:
        return dataset.names
    class_ids = sorted(
        {
            class_id
            for class_ids in dataset.image_classes.values()
            for class_id in class_ids
        }
    )
    return [str(class_id) for class_id in class_ids]


def scan_datasets(workspace: Path) -> Dict[str, DatasetInfo]:
    """扫描工作区内的数据集 YAML，并构建数据集信息索引。"""
    datasets: Dict[str, DatasetInfo] = {}
    for yaml_path in workspace.rglob("*.yaml"):
        p = str(yaml_path).lower()
        if "ultralytics\\cfg\\" in p or "dinov3_yolo\\ultralytics\\cfg\\" in p:
            continue
        try:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        has_split = any(k in data for k in ("train", "val", "test"))
        if not has_split:
            continue
        names = to_names_list(data.get("names"))
        nc = (
            int(data["nc"])
            if isinstance(data.get("nc"), int)
            else (len(names) if names else None)
        )

        split_image_dirs = {
            "train": resolve_split_dirs(yaml_path, data.get("train"), data.get("path")),
            "val": resolve_split_dirs(yaml_path, data.get("val"), data.get("path")),
            "test": resolve_split_dirs(yaml_path, data.get("test"), data.get("path")),
        }
        split_label_dirs: Dict[str, List[Path]] = {}
        for split, dirs in split_image_dirs.items():
            labels_for_split: List[Path] = []
            for d in dirs:
                as_posix = d.as_posix()
                if "/images/" in as_posix:
                    labels_for_split.append(
                        Path(as_posix.replace("/images/", "/labels/"))
                    )
                elif d.name == "images":
                    labels_for_split.append(d.parent / "labels")
                else:
                    labels_for_split.append(d.parent / "labels" / d.name)
            split_label_dirs[split] = labels_for_split

        task = infer_task_from_labels(split_label_dirs)
        imgs, img_count = list_images_from_split_dirs(split_image_dirs, limit=None)
        if not imgs and img_count == 0:
            continue
        image_classes = collect_image_class_index(imgs)

        ds = DatasetInfo(
            name=yaml_path.stem,
            yaml_path=yaml_path.resolve(),
            task=task,
            nc=nc,
            names=names,
            image_paths=imgs,
            image_count=img_count,
            split_dirs=split_image_dirs,
            image_classes=image_classes,
            tag=detect_tag_from_path(str(yaml_path)),
        )
        key = str(ds.yaml_path)
        datasets[key] = ds
    return datasets


def infer_model_task(path: Path) -> str:
    """根据模型文件名粗略推断任务类型。"""
    lower = str(path).lower()
    if "seg" in lower or "segment" in lower:
        return "segment"
    return "detect"


def extract_train_info(
    model_path: Path,
) -> Tuple[Optional[Path], Optional[Path], Optional[int], List[str]]:
    """从训练目录中提取数据集配置、类别数和类别名称等信息。"""
    train_args = model_path.parent.parent / "args.yaml"
    if not train_args.exists():
        return None, None, None, []
    try:
        args_data = yaml.safe_load(train_args.read_text(encoding="utf-8"))
    except Exception:
        return train_args, None, None, []
    if not isinstance(args_data, dict):
        return train_args, None, None, []
    raw_data = args_data.get("data")
    data_path = Path(str(raw_data)).resolve() if isinstance(raw_data, str) else None
    if not data_path or not data_path.exists():
        return train_args, data_path, None, []
    try:
        dataset_data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    except Exception:
        return train_args, data_path, None, []
    if not isinstance(dataset_data, dict):
        return train_args, data_path, None, []
    names = to_names_list(dataset_data.get("names"))
    nc = (
        int(dataset_data["nc"])
        if isinstance(dataset_data.get("nc"), int)
        else (len(names) if names else None)
    )
    return train_args, data_path, nc, names


def scan_models(workspace: Path) -> Dict[str, ModelInfo]:
    """扫描工作区内可用模型文件，并生成模型信息索引。"""
    models: Dict[str, ModelInfo] = {}
    for path in workspace.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in MODEL_EXTS:
            continue
        stat = path.stat()
        rank_hint = (
            "best"
            if path.stem.lower() == "best"
            else ("last" if path.stem.lower() == "last" else "other")
        )
        train_args_path, train_data_path, train_nc, train_names = extract_train_info(
            path
        )
        info = ModelInfo(
            path=path.resolve(),
            fmt=path.suffix.lower().replace(".", ""),
            size_mb=stat.st_size / (1024 * 1024),
            mtime=stat.st_mtime,
            inferred_task=infer_model_task(path),
            dataset_hint=detect_tag_from_path(str(path)),
            rank_hint=rank_hint,
            train_args_path=train_args_path,
            train_data_path=train_data_path,
            train_nc=train_nc,
            train_names=train_names,
        )
        models[str(info.path)] = info
    return models


def score_model_for_dataset(model: ModelInfo, dataset: DatasetInfo) -> float:
    """为数据集选择默认模型时计算简单排序分数。"""
    score = 0.0
    if model.fmt == "pt":
        score += 2.0
    if model.rank_hint == "best":
        score += 3.0
    elif model.rank_hint == "last":
        score += 1.0
    if model.inferred_task == dataset.task:
        score += 3.0
    if model.dataset_hint == dataset.tag and dataset.tag != "general":
        score += 4.0
    if (
        model.train_nc is not None
        and dataset.nc is not None
        and model.train_nc == dataset.nc
    ):
        score += 4.0
    if model.train_data_path and dataset.yaml_path.name == model.train_data_path.name:
        score += 3.0
    tf_ready, _ = transformers_status()
    if model_may_require_transformers(model) and not tf_ready:
        score -= 100.0
    score += min((time.time() - model.mtime) / 86400.0, 365.0) * (-0.002)
    return score


def format_dataset_info(dataset: DatasetInfo) -> str:
    """将数据集摘要格式化为 Markdown。"""
    names_preview = ", ".join(dataset.names[:20]) if dataset.names else "(unknown)"
    if len(dataset.names) > 20:
        names_preview += " ..."
    image_preview = str(dataset.image_paths[0]) if dataset.image_paths else "(none)"
    return (
        f"### Dataset Info\n"
        f"- name: {dataset.name}\n"
        f"- yaml: {dataset.yaml_path}\n"
        f"- task(auto): {dataset.task}\n"
        f"- classes(nc): {dataset.nc if dataset.nc is not None else 'unknown'}\n"
        f"- images(found): {dataset.image_count} (loaded candidates: {len(dataset.image_paths)})\n"
        f"- labels(auto): {names_preview}\n"
        f"- sample image: {image_preview}\n"
    )


def build_model_markdown(models: Sequence[ModelInfo], dataset: DatasetInfo) -> str:
    """生成模型扫描结果和推荐信息的 Markdown 文本。"""
    tf_ready, tf_err = transformers_status()
    lines = [
        "### Model Scan & Recommendation",
        f"- scanned models: {len(models)} (.pt + .onnx)",
    ]
    if not tf_ready:
        lines.append(
            f"- dependency note: transformers unavailable/incompatible, dino-like .pt models are deprioritized. ({tf_err[:120]})"
        )
    lines.extend(
        [
            "",
            "| rank | model | fmt | task | size(MB) | hint | score |",
            "|---:|---|---|---|---:|---|---:|",
        ]
    )
    ranked = sorted(
        ((score_model_for_dataset(m, dataset), m) for m in models),
        key=lambda x: x[0],
        reverse=True,
    )
    for idx, (score, model) in enumerate(ranked[:12], start=1):
        lines.append(
            f"| {idx} | {model.path.name} | {model.fmt} | {model.inferred_task} | "
            f"{model.size_mb:.1f} | {model.dataset_hint}/{model.rank_hint} | {score:.2f} |"
        )
    return "\n".join(lines)


def summarize_result(result, label_names: Dict[int, str], effective_task: str) -> str:
    """将单次推理结果整理为简洁的文本摘要。"""
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    total = int(len(boxes)) if boxes is not None else 0
    cls_counter: Dict[int, int] = {}
    if boxes is not None and boxes.cls is not None:
        cls_np = boxes.cls.cpu().numpy().astype(int).tolist()
        for c in cls_np:
            cls_counter[c] = cls_counter.get(c, 0) + 1
    if cls_counter:
        stats_text = ", ".join(
            [
                f"{label_names.get(k, str(k))}:{v}"
                for k, v in sorted(cls_counter.items())
            ]
        )
    else:
        stats_text = "(no detections)"

    speed = getattr(result, "speed", {}) or {}
    pre = float(speed.get("preprocess", 0.0))
    inf = float(speed.get("inference", 0.0))
    post = float(speed.get("postprocess", 0.0))
    total_ms = pre + inf + post
    mask_count = 0
    if masks is not None and getattr(masks, "data", None) is not None:
        try:
            mask_count = int(len(masks.data))
        except Exception:
            mask_count = 0
    result_title = "segments" if effective_task == "segment" else "detections"
    extra_line = (
        f"- masks: `{mask_count}`\n" if effective_task == "segment" else ""
    )

    return (
        "### Inference Summary\n"
        f"- task: `{effective_task}`\n"
        f"- {result_title}: `{total}`\n"
        f"{extra_line}"
        f"- class stats: `{stats_text}`\n"
        f"- speed(current): preprocess `{pre:.2f} ms`, inference `{inf:.2f} ms`, postprocess `{post:.2f} ms`, total `{total_ms:.2f} ms`\n"
    )


def run_latency_benchmark(
    model_path: str,
    image_path: str,
    effective_task: str,
    conf: float,
    iou: float,
    imgsz: int,
    max_det: int,
    rounds: int = 3,
) -> str:
    """对单模型单图片执行多轮推理并输出延迟统计。"""
    torch = require_torch()
    YOLO = require_yolo()
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("0")

    lines = [
        "### CPU/GPU Latency",
        f"- rounds(each): `{rounds}` (+1 warmup)",
        "",
        "| device | mean(ms) | median(ms) | p95(ms) |",
        "|---|---:|---:|---:|",
    ]

    for device in devices:
        try:
            model = YOLO(model_path)
            _ = model.predict(
                source=image_path,
                task=effective_task,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                max_det=max_det,
                verbose=False,
            )
            samples: List[float] = []
            for _i in range(rounds):
                t1 = time.perf_counter()
                _ = model.predict(
                    source=image_path,
                    task=effective_task,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    device=device,
                    max_det=max_det,
                    verbose=False,
                )
                t2 = time.perf_counter()
                samples.append((t2 - t1) * 1000.0)
            samples_sorted = sorted(samples)
            median = statistics.median(samples_sorted)
            p95 = samples_sorted[
                max(
                    0, min(len(samples_sorted) - 1, int(len(samples_sorted) * 0.95) - 1)
                )
            ]
            lines.append(
                f"| `{device}` | {statistics.mean(samples_sorted):.2f} | {median:.2f} | {p95:.2f} |"
            )
        except Exception as e:
            lines.append(f"| `{device}` | N/A | N/A | N/A |")
            lines.append(f"- note({device}): `{short_error_text(e)}`")
    return "\n".join(lines)


class AppState:
    """缓存工作区扫描结果、模型实例和推理中间状态。"""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.datasets = scan_datasets(workspace)
        self.models = scan_models(workspace)
        self.model_cache: Dict[str, object] = {}
        self.model_load_errors: Dict[str, str] = {}
        self.save_counters: Dict[str, int] = {}

    def dataset_keys(self) -> List[str]:
        return sorted(self.datasets.keys())

    def model_keys(self) -> List[str]:
        return sorted(self.models.keys())

    def get_or_load_model(self, model_path: str):
        if model_path in self.model_load_errors:
            raise RuntimeError(self.model_load_errors[model_path])
        YOLO = require_yolo()
        if model_path not in self.model_cache:
            try:
                self.model_cache[model_path] = YOLO(model_path)
            except Exception as e:
                msg = (
                    f"Failed to load model `{model_path}`: {short_error_text(e)}. "
                    "If this is a DINO/custom model, check transformers + huggingface-hub compatibility."
                )
                self.model_load_errors[model_path] = msg
                raise RuntimeError(msg) from e
        return self.model_cache[model_path]

    def peek_next_save_index(self, source_path: str) -> int:
        """返回当前图片下一次建议保存编号。"""
        return self.save_counters.get(source_path, 0) + 1

    def mark_saved(self, source_path: str) -> int:
        """记录一次保存并返回新的计数。"""
        current = self.save_counters.get(source_path, 0) + 1
        self.save_counters[source_path] = current
        return current


def make_label_map(dataset: DatasetInfo, model) -> Dict[int, str]:
    """优先使用数据集定义，其次使用模型定义，构建类别映射表。"""
    if dataset.names:
        return {i: name for i, name in enumerate(dataset.names)}
    names = getattr(model, "names", None)
    names_list = to_names_list(names)
    if names_list:
        return {i: name for i, name in enumerate(names_list)}
    return {}


def choose_default_model(
    models: Sequence[ModelInfo], dataset: DatasetInfo
) -> Optional[str]:
    """为当前数据集选择推荐的默认模型路径。"""
    if not models:
        return None
    ranked = sorted(
        models, key=lambda m: score_model_for_dataset(m, dataset), reverse=True
    )
    return str(ranked[0].path)


def available_task_choices() -> List[str]:
    """返回界面允许切换的任务类型列表。"""
    return ["auto", "detect", "segment"]


def launch_ui(
    app_state: AppState,
    host: str,
    port: int,
    share: bool,
    default_imgsz: int,
    default_conf: float,
    default_iou: float,
    default_max_det: int,
) -> None:
    """构建并启动 Gradio 推理对比界面。"""
    gr = require_gradio()
    torch = require_torch()
    cv2 = require_cv2()
    dataset_keys = app_state.dataset_keys()
    model_keys = app_state.model_keys()
    if not dataset_keys:
        raise RuntimeError(
            f"No dataset yaml found under workspace: {app_state.workspace}"
        )
    if not model_keys:
        raise RuntimeError(
            f"No .pt/.onnx models found under workspace: {app_state.workspace}"
        )

    default_dataset_key = dataset_keys[0]
    default_dataset = app_state.datasets[default_dataset_key]
    model_infos = [app_state.models[k] for k in model_keys]
    default_model_key = (
        choose_default_model(model_infos, default_dataset) or model_keys[0]
    )
    default_class_choices = dataset_class_choices(default_dataset)
    default_images = filter_images_by_classes(default_dataset, []) or [""]
    default_save_dir = str(DEFAULT_SEGMENT_SAVE_DIR)
    roi_compare_hint = (
        "### 差异统计\n"
        "- 选择检测结果编号后点击“生成原图对比”\n"
        "- 处理流程: bilateralFilter + absdiff + Otsu + open/close"
    )

    def hidden_save_modal_outputs():
        return (
            gr.update(value="### 检测结果列表\n- waiting", visible=False),
            gr.update(value=default_save_dir, visible=False),
            gr.update(value="### 保存状态\n- waiting", visible=False),
            gr.update(visible=False),
        )

    def hidden_inference_outputs(summary_text: str, latency_text: str):
        return (
            None,
            summary_text,
            latency_text,
            None,
            *hidden_save_modal_outputs(),
            *hidden_roi_compare_outputs(),
        )

    def empty_optimize_outputs(report_text: str):
        return (
            gr.update(),
            gr.update(),
            report_text,
            "### 优化历史\n- waiting",
            *hidden_inference_outputs(
                "### Inference Summary\n- waiting",
                "### CPU/GPU Latency\n- waiting",
            ),
        )

    def resolve_valid_image_key(dataset: DatasetInfo, image_key: str) -> Optional[str]:
        """确保当前图片值属于当前数据集候选列表；否则回退到首个合法值。"""
        image_choices = filter_images_by_classes(dataset, [])
        if image_key and image_key in image_choices:
            return image_key
        return image_choices[0] if image_choices else None

    def shown_save_modal_outputs(inference_state):
        detections = inference_state.get("detections", []) if inference_state else []
        effective_task = (
            str(inference_state.get("effective_task", "detect"))
            if inference_state
            else "detect"
        )
        if effective_task != "detect":
            return hidden_save_modal_outputs()
        status = (
            "### 保存状态\n- 点击“保存确认”后会将所有检测框按原始像素无损裁切保存"
            if detections
            else "### 保存状态\n- 当前推理没有检测框，无需保存"
        )
        return (
            gr.update(
                value=build_detection_list_markdown(detections, effective_task),
                visible=True,
            ),
            gr.update(value=default_save_dir, visible=True),
            gr.update(value=status, visible=True),
            gr.update(visible=True),
        )

    def hidden_roi_compare_outputs():
        return (
            gr.update(visible=False),
            gr.update(choices=[], value=None, visible=False),
            gr.update(value="### ROI 原图对比\n- unavailable", visible=False),
            gr.update(value="### 差异统计\n- waiting", visible=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
        )

    def shown_roi_compare_outputs(inference_state):
        if not inference_state:
            return hidden_roi_compare_outputs()
        effective_task = str(inference_state.get("effective_task", "detect"))
        detections = inference_state.get("detections", [])
        origin_path = inference_state.get("origin_path")
        if effective_task != "detect" or not detections or not origin_path:
            return hidden_roi_compare_outputs()
        roi_choices = build_roi_compare_choices(detections)
        default_choice = roi_choices[0][1] if roi_choices else None
        return (
            gr.update(visible=True),
            gr.update(choices=roi_choices, value=default_choice, visible=True),
            gr.update(value=f"### ROI 原图对比\n- origin: `{origin_path}`", visible=True),
            gr.update(value=roi_compare_hint, visible=True),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
        )

    def build_inference_state(
        result,
        source_path: str,
        detections: List[Dict[str, object]],
        effective_task: str,
        origin_path: Optional[str],
    ) -> Dict[str, object]:
        orig_image = result.orig_img.copy()
        if len(orig_image.shape) == 3 and orig_image.shape[2] == 3:
            orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        return {
            "source_path": source_path,
            "orig_image": orig_image,
            "detections": detections,
            "effective_task": effective_task,
            "origin_path": origin_path,
        }

    def refresh_workspace():
        app_state.datasets = scan_datasets(app_state.workspace)
        app_state.models = scan_models(app_state.workspace)
        ds_keys = sorted(app_state.datasets.keys())
        md_keys = sorted(app_state.models.keys())
        if not ds_keys or not md_keys:
            return (
                gr.update(choices=ds_keys, value=None),
                "### Dataset Info\n- no dataset found",
                gr.update(choices=md_keys, value=None),
                "### Model Scan & Recommendation\n- no model found",
                gr.update(choices=[], value=[]),
                gr.update(choices=[], value=None),
            )
        ds = app_state.datasets[ds_keys[0]]
        models_local = [app_state.models[k] for k in md_keys]
        model_default = choose_default_model(models_local, ds) or md_keys[0]
        image_choices = filter_images_by_classes(ds, [])
        return (
            gr.update(choices=ds_keys, value=ds_keys[0]),
            format_dataset_info(ds),
            gr.update(choices=md_keys, value=model_default),
            build_model_markdown(models_local, ds),
            gr.update(choices=dataset_class_choices(ds), value=[]),
            gr.update(
                choices=image_choices, value=image_choices[0] if image_choices else None
            ),
        )

    def on_dataset_change(dataset_key: str):
        if not dataset_key or dataset_key not in app_state.datasets:
            return (
                "### Dataset Info\n- invalid dataset",
                gr.update(),
                "### Model Scan & Recommendation\n- invalid dataset",
                gr.update(choices=[], value=[]),
                gr.update(choices=[], value=None),
                gr.update(value=None),
            )
        ds = app_state.datasets[dataset_key]
        models_local = [app_state.models[k] for k in sorted(app_state.models.keys())]
        default_model = choose_default_model(models_local, ds) or (
            str(models_local[0].path) if models_local else None
        )
        image_choices = filter_images_by_classes(ds, [])
        return (
            format_dataset_info(ds),
            gr.update(value=ds.task),
            build_model_markdown(models_local, ds),
            gr.update(choices=dataset_class_choices(ds), value=[]),
            gr.update(
                choices=image_choices, value=image_choices[0] if image_choices else None
            ),
            gr.update(value=default_model),
        )

    def on_class_filter_change(dataset_key: str, selected_class_names: Sequence[str]):
        if not dataset_key or dataset_key not in app_state.datasets:
            return gr.update(choices=[], value=None)
        ds = app_state.datasets[dataset_key]
        image_choices = filter_images_by_classes(ds, selected_class_names)
        return gr.update(
            choices=image_choices,
            value=image_choices[0] if image_choices else None,
        )

    def run_once(
        dataset_key: str,
        model_key: str,
        image_key: str,
        uploaded_image,
        task_choice: str,
        conf: float,
        iou: float,
        imgsz: int,
        max_det: int,
        device: str,
        compare_cpu_gpu: bool,
    ):
        if dataset_key not in app_state.datasets:
            return hidden_inference_outputs(
                "### Inference Summary\n- invalid dataset selection",
                "### CPU/GPU Latency\n- unavailable",
            )
        if model_key not in app_state.models:
            return hidden_inference_outputs(
                "### Inference Summary\n- invalid model selection",
                "### CPU/GPU Latency\n- unavailable",
            )
        ds = app_state.datasets[dataset_key]
        model_info = app_state.models[model_key]
        source_path = ""
        if uploaded_image is not None:
            source_path = str(uploaded_image)
        elif image_key:
            valid_image_key = resolve_valid_image_key(ds, image_key)
            source_path = valid_image_key or ""
        if not source_path or not Path(source_path).exists():
            return hidden_inference_outputs(
                "### Inference Summary\n- please select a valid test image",
                "### CPU/GPU Latency\n- unavailable",
            )

        effective_task = ds.task if task_choice == "auto" else task_choice
        if effective_task not in ("detect", "segment"):
            effective_task = model_info.inferred_task

        try:
            model = app_state.get_or_load_model(model_key)
            predict_kwargs = dict(
                source=source_path,
                task=effective_task,
                conf=float(conf),
                iou=float(iou),
                imgsz=int(imgsz),
                max_det=int(max_det),
                device=device,
                verbose=False,
            )
            if effective_task == "segment":
                predict_kwargs["retina_masks"] = True
            results = model.predict(**predict_kwargs)
        except Exception as e:
            msg = short_error_text(e)
            recovery = (
                "建议：切换到非 DINO 模型或 .onnx；"
                '若需当前模型，安装兼容依赖如 `pip install "huggingface-hub<1.0,>=0.30" -U`。'
            )
            return hidden_inference_outputs(
                f"### Inference Summary\n- model load/infer failed: `{msg}`\n- {recovery}",
                "### CPU/GPU Latency\n- unavailable (model load failed)",
            )
        if not results:
            return hidden_inference_outputs(
                "### Inference Summary\n- no result returned",
                "### CPU/GPU Latency\n- unavailable",
            )

        result = results[0]
        plotted = render_result_image(result, effective_task)
        label_map = make_label_map(ds, model)
        detections = serialize_detections(result, label_map)
        resolved_origin = (
            resolve_origin_image_path(source_path) if effective_task == "detect" else None
        )
        origin_path = str(resolved_origin) if resolved_origin is not None else None
        inference_state = build_inference_state(
            result, source_path, detections, effective_task, origin_path
        )
        result_html = build_result_html(plotted, detections, effective_task)
        summary = summarize_result(result, label_map, effective_task)
        roi_note = (
            "- 操作: 推理完成后会弹出保存确认层，可一次性保存全部检测框裁切结果。\n"
            if effective_task == "detect"
            else "- 操作: 当前为 segment 结果显示模式，展示掩码可视化，不启用 ROI 裁切保存。\n"
        )
        if effective_task == "detect":
            origin_note = (
                f"- origin linked: `{origin_path}`\n- ROI diff: available\n"
                if origin_path
                else "- origin linked: unavailable\n- ROI diff: hidden (origin image not found)\n"
            )
        else:
            origin_note = ""
        summary += (
            f"- model: `{model_info.path}`\n"
            f"- format: `{model_info.fmt}`\n"
            f"- input image: `{source_path}`\n"
            f"- params: conf `{conf:.2f}`, iou `{iou:.2f}`, imgsz `{imgsz}`, max_det `{max_det}`, device `{device}`\n"
            f"- results parsed: `{len(detections)}`\n"
            f"{origin_note}"
            f"{roi_note}"
        )

        latency_md = "### CPU/GPU Latency\n- disabled (check benchmark option to measure both CPU/GPU)"
        if compare_cpu_gpu:
            latency_md = run_latency_benchmark(
                model_path=model_key,
                image_path=source_path,
                effective_task=effective_task,
                conf=float(conf),
                iou=float(iou),
                imgsz=int(imgsz),
                max_det=int(max_det),
            )
        return (
            result_html,
            summary,
            latency_md,
            inference_state,
            *shown_save_modal_outputs(inference_state),
            *shown_roi_compare_outputs(inference_state),
        )

    def save_all_detections(inference_state, save_dir: str):
        if not inference_state:
            return "### 保存状态\n- 当前没有可保存的推理结果", gr.update(visible=False)
        effective_task = str(inference_state.get("effective_task", "detect"))
        if effective_task != "detect":
            return "### 保存状态\n- 仅 detect 模式支持 ROI 裁切保存", gr.update(
                visible=False
            )
        detections = inference_state.get("detections", [])
        if not detections:
            return "### 保存状态\n- 当前推理没有检测框，无需保存", gr.update(visible=False)
        orig_image = inference_state.get("orig_image")
        source_path = str(inference_state.get("source_path", ""))
        if orig_image is None:
            return "### 保存状态\n- 原图缓存缺失，无法保存", gr.update(visible=True)
        target_dir = Path((save_dir or default_save_dir).strip() or default_save_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: List[str] = []
        for det in detections:
            x1, y1, x2, y2 = (
                int(det["x1"]),
                int(det["y1"]),
                int(det["x2"]),
                int(det["y2"]),
            )
            crop = orig_image[y1:y2, x1:x2]
            if crop is None or getattr(crop, "size", 0) == 0:
                continue
            save_index = app_state.mark_saved(source_path)
            filename = next_save_filename(source_path, save_index)
            target_path = target_dir / filename
            cv2.imwrite(str(target_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            saved_paths.append(str(target_path))
        if not saved_paths:
            return "### 保存状态\n- 没有成功写入任何裁切图像", gr.update(visible=True)
        preview_lines = "\n".join([f"- `{p}`" for p in saved_paths[:8]])
        if len(saved_paths) > 8:
            preview_lines += f"\n- ... 共 `{len(saved_paths)}` 个文件"
        status = (
            "### 保存状态\n"
            f"- 已保存 `{len(saved_paths)}` 个检测框裁切图像到 `{target_dir}`\n"
            f"{preview_lines}"
        )
        return status, gr.update(visible=False)

    def cancel_save_modal():
        return hidden_save_modal_outputs()

    def reset_roi_compare_preview():
        return (
            gr.update(value=roi_compare_hint, visible=True),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
        )

    def generate_roi_compare_preview(inference_state, roi_choice_value: str):
        if not inference_state:
            return (
                gr.update(value="### 差异统计\n- 当前没有可对比的推理结果", visible=True),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
            )
        if str(inference_state.get("effective_task", "detect")) != "detect":
            return (
                gr.update(value="### 差异统计\n- 仅 detect 模式支持 ROI 原图对比", visible=True),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
            )
        origin_path = inference_state.get("origin_path")
        if not origin_path:
            return (
                gr.update(value="### 差异统计\n- 未找到可联动的 origin 原图", visible=True),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
            )
        if not roi_choice_value or not str(roi_choice_value).isdigit():
            return (
                gr.update(value="### 差异统计\n- 请选择一个检测结果编号", visible=True),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
            )

        detection = find_detection_by_index(
            inference_state.get("detections", []), int(str(roi_choice_value))
        )
        if detection is None:
            return (
                gr.update(value="### 差异统计\n- 检测结果编号不存在", visible=True),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
            )

        detect_image = inference_state.get("orig_image")
        if detect_image is None:
            return (
                gr.update(value="### 差异统计\n- 当前检测图缓存缺失", visible=True),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
            )

        try:
            origin_image = load_rgb_image(Path(str(origin_path)))
        except Exception as e:
            return (
                gr.update(
                    value=f"### 差异统计\n- 原图读取失败: `{short_error_text(e)}`",
                    visible=True,
                ),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
            )

        if detect_image.shape[:2] != origin_image.shape[:2]:
            return (
                gr.update(
                    value=(
                        "### 差异统计\n"
                        f"- 图像尺寸不一致: detect `{detect_image.shape[1]}x{detect_image.shape[0]}` "
                        f"vs origin `{origin_image.shape[1]}x{origin_image.shape[0]}`"
                    ),
                    visible=True,
                ),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
            )

        x1, y1, x2, y2 = (
            int(detection["x1"]),
            int(detection["y1"]),
            int(detection["x2"]),
            int(detection["y2"]),
        )
        detect_roi = detect_image[y1:y2, x1:x2].copy()
        origin_roi = origin_image[y1:y2, x1:x2].copy()
        if getattr(detect_roi, "size", 0) == 0 or getattr(origin_roi, "size", 0) == 0:
            return (
                gr.update(value="### 差异统计\n- ROI 裁切为空，无法对比", visible=True),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
            )

        diff_mask_rgb, overlay, diff_mask = generate_roi_diff_outputs(
            detect_roi, origin_roi
        )
        diff_pixels = int(np.count_nonzero(diff_mask))
        total_pixels = int(diff_mask.size)
        diff_ratio = (diff_pixels / max(1, total_pixels)) * 100.0
        status = (
            "### 差异统计\n"
            f"- 检测结果: `#{int(detection['index'])} {detection['label']} ({float(detection['conf']):.4f})`\n"
            f"- ROI 像素尺寸: `{detect_roi.shape[1]} x {detect_roi.shape[0]}`\n"
            f"- 差异像素数: `{diff_pixels}` / `{total_pixels}`\n"
            f"- 差异面积占比: `{diff_ratio:.4f}%`\n"
            "- 处理流程: bilateralFilter + absdiff + Otsu + morphology(open+close)\n"
        )
        return (
            gr.update(value=status, visible=True),
            gr.update(value=detect_roi, visible=True),
            gr.update(value=origin_roi, visible=True),
            gr.update(value=diff_mask_rgb, visible=True),
            gr.update(value=overlay, visible=True),
        )

    def optimize_thresholds(
        dataset_key: str,
        model_key: str,
        image_key: str,
        uploaded_image,
        task_choice: str,
        imgsz: int,
        max_det: int,
        device: str,
        compare_cpu_gpu: bool,
        n_trials: int,
        max_images: int,
    ):
        if dataset_key not in app_state.datasets:
            return empty_optimize_outputs("### 阈值优化\n- invalid dataset")
        if model_key not in app_state.models:
            return empty_optimize_outputs("### 阈值优化\n- invalid model")

        ds = app_state.datasets[dataset_key]
        model_info = app_state.models[model_key]
        effective_task = ds.task if task_choice == "auto" else task_choice
        if effective_task not in ("detect", "segment"):
            effective_task = model_info.inferred_task

        val_dirs = ds.split_dirs.get("val", [])
        val_candidates, _count = list_images_from_split_dirs(
            {"val": val_dirs}, limit=max(1, int(max_images))
        )
        if not val_candidates:
            val_candidates = ds.image_paths[: max(1, int(max_images))]
        if not val_candidates:
            return empty_optimize_outputs("### 阈值优化\n- 未找到可用验证图像（val）")

        try:
            model = app_state.get_or_load_model(model_key)
        except Exception as e:
            return empty_optimize_outputs(
                f"### 阈值优化\n- 模型加载失败: `{short_error_text(e)}`"
            )

        t0 = time.perf_counter()
        cache: List[Dict[str, object]] = []
        total_imgs = len(val_candidates)
        valid_mask_samples = 0
        skipped_no_pred_mask = 0
        skipped_no_gt_mask = 0
        for idx, img_path in enumerate(val_candidates, start=1):
            try:
                predict_kwargs = dict(
                    source=str(img_path),
                    task=effective_task,
                    conf=0.001,
                    iou=0.99,
                    imgsz=int(imgsz),
                    max_det=3000,
                    device=device,
                    verbose=False,
                )
                if effective_task == "segment":
                    predict_kwargs["retina_masks"] = True
                results = model.predict(**predict_kwargs)
            except Exception:
                continue
            if not results:
                continue
            r0 = results[0]
            h, w = r0.orig_shape
            preds_norm, pred_valid = build_prediction_cache_items(r0, effective_task)
            gts = filter_gt_instances_for_task(
                parse_gt_instances(image_to_label_path(Path(img_path)), (h, w)),
                effective_task,
            )
            if effective_task == "segment":
                if not pred_valid:
                    skipped_no_pred_mask += 1
                    continue
                if not gts:
                    skipped_no_gt_mask += 1
                    continue
                valid_mask_samples += 1
            cache.append({"preds": preds_norm, "gts": gts})

        if not cache:
            if effective_task == "segment":
                return empty_optimize_outputs(
                    "### 阈值优化\n"
                    "- 严格 mask IoU 优化失败：没有可用的有效样本\n"
                    f"- total val images: `{total_imgs}`\n"
                    f"- valid mask samples: `{valid_mask_samples}`\n"
                    f"- skipped(no pred mask): `{skipped_no_pred_mask}`\n"
                    f"- skipped(no gt mask): `{skipped_no_gt_mask}`\n"
                    "- metric basis: `mask IoU (strict)`"
                )
            return empty_optimize_outputs("### 阈值优化\n- 缓存推理失败，未得到有效样本")

        if effective_task == "segment" and valid_mask_samples <= 0:
            return empty_optimize_outputs(
                "### 阈值优化\n"
                "- 严格 mask IoU 优化失败：没有可用的有效样本\n"
                f"- total val images: `{total_imgs}`\n"
                f"- valid mask samples: `{valid_mask_samples}`\n"
                f"- skipped(no pred mask): `{skipped_no_pred_mask}`\n"
                f"- skipped(no gt mask): `{skipped_no_gt_mask}`\n"
                "- metric basis: `mask IoU (strict)`"
            )

        best_conf = 0.5
        best_iou = 0.45
        best_f1 = -1.0
        best_p = 0.0
        best_r = 0.0
        history: List[Tuple[int, float, float, float]] = []

        optuna_used = False
        try:
            import optuna  # type: ignore

            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                conf = trial.suggest_float("conf", 0.001, 0.9)
                iou = trial.suggest_float("iou", 0.3, 0.8)
                _p, _r, _f1, _tp, _fp, _fn = eval_f1_from_cache(
                    cache,
                    conf_thres=conf,
                    iou_thres=iou,
                    effective_task=effective_task,
                    match_iou=0.5,
                )
                history.append((trial.number + 1, conf, iou, _f1))
                return _f1

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=max(5, int(n_trials)))
            best_conf = float(study.best_params["conf"])
            best_iou = float(study.best_params["iou"])
            best_p, best_r, best_f1, _tp, _fp, _fn = eval_f1_from_cache(
                cache,
                conf_thres=best_conf,
                iou_thres=best_iou,
                effective_task=effective_task,
                match_iou=0.5,
            )
            optuna_used = True
        except Exception:
            # Fallback to deterministic coarse-to-fine search
            conf_grid = [
                0.001,
                0.01,
                0.03,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.65,
                0.8,
            ]
            iou_grid = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]
            total = len(conf_grid) * len(iou_grid)
            cur = 0
            for c in conf_grid:
                for n in iou_grid:
                    cur += 1
                    p, r, f1, _tp, _fp, _fn = eval_f1_from_cache(
                        cache,
                        conf_thres=c,
                        iou_thres=n,
                        effective_task=effective_task,
                        match_iou=0.5,
                    )
                    history.append((cur, c, n, f1))
                    if f1 > best_f1:
                        best_conf, best_iou, best_f1, best_p, best_r = c, n, f1, p, r

        elapsed = time.perf_counter() - t0
        report = (
            "### 阈值优化结果\n"
            f"- method: {'Optuna' if optuna_used else 'Fallback Grid'}\n"
            f"- total val images: `{total_imgs}`\n"
            f"- val images used: {len(cache)}\n"
            f"- best conf: {best_conf:.4f}\n"
            f"- best nms(iou): {best_iou:.4f}\n"
            f"- best F1@IoU0.5: {best_f1:.4f}\n"
            f"- precision: {best_p:.4f}, recall: {best_r:.4f}\n"
            f"- metric basis: {'mask IoU (strict)' if effective_task == 'segment' else 'box IoU'}\n"
            f"- elapsed: {elapsed:.2f}s\n"
            "- note: 使用低阈值缓存推理 + 后处理搜索，无需重复全量前向。"
        )
        if effective_task == "segment":
            report = (
                report
                + "\n"
                + f"- valid mask samples: `{valid_mask_samples}`\n"
                + f"- skipped(no pred mask): `{skipped_no_pred_mask}`\n"
                + f"- skipped(no gt mask): `{skipped_no_gt_mask}`"
            )
        history_md = build_opt_history_markdown(history)

        (
            plotted,
            summary,
            latency,
            inference_state,
            detection_list_update,
            save_dir_update,
            save_status_update,
            save_modal_update,
            roi_compare_group_update,
            roi_select_update,
            origin_path_update,
            roi_compare_status_update,
            detect_roi_update,
            origin_roi_update,
            diff_mask_update,
            diff_overlay_update,
        ) = run_once(
            dataset_key=dataset_key,
            model_key=model_key,
            image_key=image_key,
            uploaded_image=uploaded_image,
            task_choice=task_choice,
            conf=best_conf,
            iou=best_iou,
            imgsz=imgsz,
            max_det=max_det,
            device=device,
            compare_cpu_gpu=compare_cpu_gpu,
        )
        return (
            best_conf,
            best_iou,
            report,
            history_md,
            plotted,
            summary,
            latency,
            inference_state,
            detection_list_update,
            save_dir_update,
            save_status_update,
            save_modal_update,
            roi_compare_group_update,
            roi_select_update,
            origin_path_update,
            roi_compare_status_update,
            detect_roi_update,
            origin_roi_update,
            diff_mask_update,
            diff_overlay_update,
        )

    theme = gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="amber",
        neutral_hue="slate",
        radius_size=gr.themes.sizes.radius_lg,
        text_size=gr.themes.sizes.text_md,
        font=[
            gr.themes.GoogleFont("Space Grotesk"),
            "Microsoft YaHei UI",
            "sans-serif",
        ],
    )

    with gr.Blocks(title="YOLO Inference UI (.pt/.onnx, detect/segment)") as demo:
        inference_state = gr.State(None)
        gr.Markdown(
            "<div id='hero'>"
            "<h2>YOLO 智能推理台</h2>"
            "<p>自动识别数据集与标签，自动推荐模型，实时联动阈值与可视化结果，支持 detect / segment 与 .pt / .onnx。</p>"
            "</div>"
        )

        with gr.Row():
            with gr.Column(
                scale=8,
                elem_classes=["panel", "control-compact"],
                elem_id="left-control",
            ):
                gr.Markdown("### 控制面板")
                dataset_dropdown = gr.Dropdown(
                    choices=dataset_keys,
                    value=default_dataset_key,
                    label="推理数据集（自动识别）",
                )
                task_choice = gr.Radio(
                    choices=available_task_choices(),
                    value="auto",
                    label="任务类型（auto=按数据集自动判定）",
                )
                model_dropdown = gr.Dropdown(
                    choices=model_keys,
                    value=default_model_key,
                    label="模型选择（按数据集自动推荐）",
                )
                refresh_btn = gr.Button("重新扫描工作区", variant="secondary")
                class_filter = gr.CheckboxGroup(
                    choices=default_class_choices,
                    value=[],
                    label="类别筛选（不选=整个数据集）",
                )
                with gr.Accordion("数据集信息", open=True):
                    dataset_info_md = gr.Markdown(format_dataset_info(default_dataset))
                with gr.Accordion("模型详情与推荐", open=True):
                    model_md = gr.Markdown(
                        build_model_markdown(model_infos, default_dataset),
                        elem_classes=["model-scroll"],
                    )

            with gr.Column(
                scale=14,
                elem_classes=["panel", "control-compact"],
                elem_id="right-workbench",
            ):
                gr.Markdown("### 推理工作区")
                image_dropdown = gr.Dropdown(
                    choices=default_images,
                    value=default_images[0] if default_images else None,
                    label="测试图像选择（来自数据集）",
                    allow_custom_value=True,
                )
                upload_image = gr.Image(
                    type="filepath", label="或上传单张测试图像（优先于下拉图像）"
                )

                with gr.Row():
                    device_radio = gr.Radio(
                        choices=["cpu", "0"] if torch.cuda.is_available() else ["cpu"],
                        value="0" if torch.cuda.is_available() else "cpu",
                        label="当前推理设备",
                    )
                    with gr.Column():
                        compare_ck = gr.Checkbox(value=True, label="统计 CPU/GPU 延时")
                        auto_update_ck = gr.Checkbox(
                            value=True, label="参数变化自动更新"
                        )

                with gr.Accordion("高级参数（阈值与尺寸）", open=True):
                    conf_slider = gr.Slider(
                        0.01, 0.99, value=default_conf, step=0.01, label="conf 阈值"
                    )
                    iou_slider = gr.Slider(
                        0.01, 0.99, value=default_iou, step=0.01, label="nms(iou) 阈值"
                    )
                    imgsz_slider = gr.Slider(
                        320, 2048, value=default_imgsz, step=32, label="imgsz"
                    )
                    max_det_slider = gr.Slider(
                        1, 1000, value=default_max_det, step=1, label="max_det"
                    )
                with gr.Row():
                    run_btn = gr.Button("执行推理", variant="primary")
                    optimize_btn = gr.Button("阈值优化（val）", variant="secondary")
                with gr.Row():
                    opt_trials = gr.Slider(
                        5, 120, value=40, step=1, label="优化试验次数"
                    )
                    opt_max_images = gr.Slider(
                        20, 1000, value=300, step=10, label="val样本上限"
                    )
                with gr.Accordion("阈值优化报告", open=False):
                    with gr.Tabs():
                        with gr.Tab("结果"):
                            opt_report_md = gr.Markdown("### 阈值优化结果\n- waiting")
                        with gr.Tab("历史"):
                            opt_history_md = gr.Markdown("### 优化历史\n- waiting")

                with gr.Row():
                    with gr.Column(scale=14, elem_id="result-panel"):
                        gr.Markdown("### 结果图")
                        result_image = gr.HTML(
                            label="检测结果预览",
                            elem_classes=["result-image"],
                        )
                    with gr.Column(scale=10, elem_id="summary-panel"):
                        with gr.Tabs():
                            with gr.Tab("摘要"):
                                summary_md = gr.Markdown(
                                    "### Inference Summary\n- waiting",
                                    elem_classes=["summary-scroll"],
                                )
                            with gr.Tab("延时"):
                                latency_md = gr.Markdown(
                                    "### CPU/GPU Latency\n- waiting",
                                    elem_classes=["summary-scroll"],
                                )
                with gr.Group(visible=False, elem_id="roi-compare-wrap") as roi_compare_group:
                    gr.Markdown("### ROI 原图对比")
                    with gr.Row():
                        with gr.Column(scale=7):
                            roi_select_dropdown = gr.Dropdown(
                                choices=[],
                                value=None,
                                label="检测结果编号",
                                visible=False,
                            )
                            generate_roi_compare_btn = gr.Button(
                                "生成原图对比", variant="secondary"
                            )
                            origin_path_md = gr.Markdown(
                                "### ROI 原图对比\n- unavailable",
                                visible=False,
                            )
                            roi_compare_status = gr.Markdown(
                                "### 差异统计\n- waiting",
                                visible=False,
                            )
                        with gr.Column(scale=17):
                            with gr.Row():
                                detect_roi_image = gr.Image(
                                    label="检测 ROI",
                                    type="numpy",
                                    height=280,
                                    visible=False,
                                )
                                origin_roi_image = gr.Image(
                                    label="原图 ROI",
                                    type="numpy",
                                    height=280,
                                    visible=False,
                                )
                            with gr.Row():
                                diff_mask_image = gr.Image(
                                    label="差异掩码",
                                    type="numpy",
                                    height=280,
                                    visible=False,
                                )
                                diff_overlay_image = gr.Image(
                                    label="红色叠加结果",
                                    type="numpy",
                                    height=280,
                                    visible=False,
                                )
                with gr.Group(visible=False, elem_id="save-modal-wrap") as save_modal_group:
                    with gr.Column(elem_id="save-modal"):
                        gr.Markdown("### 检测结果批量保存")
                        detection_list_md = gr.Markdown(
                            "### 检测结果列表\n- waiting",
                            visible=False,
                        )
                        save_dir_tb = gr.Textbox(
                            label="保存路径",
                            value=default_save_dir,
                            visible=False,
                        )
                        batch_save_status = gr.Markdown(
                            "### 保存状态\n- waiting",
                            visible=False,
                        )
                        with gr.Row():
                            save_all_btn = gr.Button("保存确认", variant="primary")
                            cancel_save_btn = gr.Button("取消", variant="secondary")

        run_outputs = [
            result_image,
            summary_md,
            latency_md,
            inference_state,
            detection_list_md,
            save_dir_tb,
            batch_save_status,
            save_modal_group,
            roi_compare_group,
            roi_select_dropdown,
            origin_path_md,
            roi_compare_status,
            detect_roi_image,
            origin_roi_image,
            diff_mask_image,
            diff_overlay_image,
        ]

        dataset_dropdown.change(
            fn=on_dataset_change,
            inputs=[dataset_dropdown],
            outputs=[
                dataset_info_md,
                task_choice,
                model_md,
                class_filter,
                image_dropdown,
                model_dropdown,
            ],
        )

        refresh_btn.click(
            fn=refresh_workspace,
            inputs=[],
            outputs=[
                dataset_dropdown,
                dataset_info_md,
                model_dropdown,
                model_md,
                class_filter,
                image_dropdown,
            ],
        )

        class_filter.change(
            fn=on_class_filter_change,
            inputs=[dataset_dropdown, class_filter],
            outputs=[image_dropdown],
        )

        run_inputs = [
            dataset_dropdown,
            model_dropdown,
            image_dropdown,
            upload_image,
            task_choice,
            conf_slider,
            iou_slider,
            imgsz_slider,
            max_det_slider,
            device_radio,
            compare_ck,
        ]

        run_btn.click(
            fn=run_once,
            inputs=run_inputs,
            outputs=run_outputs,
        )

        optimize_btn.click(
            fn=optimize_thresholds,
            inputs=[
                dataset_dropdown,
                model_dropdown,
                image_dropdown,
                upload_image,
                task_choice,
                imgsz_slider,
                max_det_slider,
                device_radio,
                compare_ck,
                opt_trials,
                opt_max_images,
            ],
            outputs=[
                conf_slider,
                iou_slider,
                opt_report_md,
                opt_history_md,
                *run_outputs,
            ],
        )

        def maybe_run(auto_update, *vals):
            if not auto_update:
                return tuple(gr.update() for _ in range(len(run_outputs)))
            return run_once(*vals)

        realtime_inputs = [
            auto_update_ck,
            dataset_dropdown,
            model_dropdown,
            image_dropdown,
            upload_image,
            task_choice,
            conf_slider,
            iou_slider,
            imgsz_slider,
            max_det_slider,
            device_radio,
            compare_ck,
        ]
        for control in [
            conf_slider,
            iou_slider,
            imgsz_slider,
            max_det_slider,
            model_dropdown,
            image_dropdown,
            device_radio,
            task_choice,
        ]:
            control.change(
                fn=maybe_run,
                inputs=realtime_inputs,
                outputs=run_outputs,
            )

        save_all_btn.click(
            fn=save_all_detections,
            inputs=[inference_state, save_dir_tb],
            outputs=[batch_save_status, save_modal_group],
        )

        roi_select_dropdown.change(
            fn=reset_roi_compare_preview,
            inputs=[],
            outputs=[
                roi_compare_status,
                detect_roi_image,
                origin_roi_image,
                diff_mask_image,
                diff_overlay_image,
            ],
        )

        generate_roi_compare_btn.click(
            fn=generate_roi_compare_preview,
            inputs=[inference_state, roi_select_dropdown],
            outputs=[
                roi_compare_status,
                detect_roi_image,
                origin_roi_image,
                diff_mask_image,
                diff_overlay_image,
            ],
        )

        cancel_save_btn.click(
            fn=cancel_save_modal,
            inputs=[],
            outputs=[detection_list_md, save_dir_tb, batch_save_status, save_modal_group],
        )

    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        theme=theme,
        css=MODERN_CSS,
    )


def main() -> None:
    """主入口：解析参数、校验工作区并启动界面。"""
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    if not workspace.exists():
        raise FileNotFoundError(f"Workspace does not exist: {workspace}")
    app_state = AppState(workspace=workspace)
    launch_ui(
        app_state=app_state,
        host=args.host,
        port=args.port,
        share=args.share,
        default_imgsz=args.imgsz,
        default_conf=args.conf,
        default_iou=args.iou,
        default_max_det=args.max_det,
    )


if __name__ == "__main__":
    main()
