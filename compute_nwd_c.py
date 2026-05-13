#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 YOLO 标签估计 NWD 常数 `C`。

功能：
- 读取一个标签目录或单个 `*.txt` 文件。
- 估计目标绝对尺寸分布，并给出推荐的 NWD 常数。
- 输出均值、分位数和截尾均值，便于后续调参。

使用：
- `python compute_nwd_c.py --labels PKU-Market-PCB/labels/train --imgsz 960`
"""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


def _iter_label_files(labels_root: Path) -> Iterable[Path]:
    """遍历标签根目录下的所有 YOLO `*.txt` 文件。"""
    if labels_root.is_file():
        if labels_root.suffix.lower() == ".txt":
            yield labels_root
        return
    if not labels_root.exists():
        return
    # 递归扫描所有 `.txt` 标签文件，兼容 train/val/test 或自定义划分目录。
    yield from labels_root.rglob("*.txt")


def _parse_yolo_txt(path: Path) -> List[Tuple[int, float, float, float, float]]:
    """解析单个 YOLO 标签文件并返回标准检测框列表。

    返回值格式为 `(cls, cx, cy, w, h)`，坐标均为归一化值。
    空行或非法行会被自动跳过。
    """
    rows: List[Tuple[int, float, float, float, float]] = []
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return rows

    for ln, line in enumerate(txt.splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            continue
        if not (
            math.isfinite(cx)
            and math.isfinite(cy)
            and math.isfinite(w)
            and math.isfinite(h)
        ):
            continue
        if w <= 0 or h <= 0:
            continue
        rows.append((cls, cx, cy, w, h))
    return rows


def _abs_size_from_wh_px(w_px: float, h_px: float, mode: str) -> float:
    """将像素级宽高转换为单个“绝对尺寸”标量。"""
    mode = mode.lower().strip()
    if mode in ("sqrt_area", "sqrtarea", "geom", "geomean", "geometric"):
        return math.sqrt(w_px * h_px)
    if mode in ("arith", "mean_edge", "meanedge", "avg_edge", "avgedge"):
        return 0.5 * (w_px + h_px)
    if mode in ("diag", "diagonal"):
        return math.hypot(w_px, h_px)
    if mode in ("max_edge", "maxedge", "max"):
        return max(w_px, h_px)
    if mode in ("min_edge", "minedge", "min"):
        return min(w_px, h_px)
    raise ValueError(f"Unknown size mode: {mode}")


def _trimmed_mean(xs: Sequence[float], trim_ratio: float) -> float:
    """计算去掉两端极值后的截尾均值。"""
    if not xs:
        raise ValueError("empty")
    if trim_ratio <= 0:
        return float(statistics.fmean(xs))
    if trim_ratio >= 0.5:
        raise ValueError("--trim must be in [0, 0.5)")
    s = sorted(xs)
    k = int(len(s) * trim_ratio)
    core = s[k : len(s) - k] if len(s) - 2 * k > 0 else s
    return float(statistics.fmean(core))


def _percentile(xs: Sequence[float], p: float) -> float:
    """计算给定分位点，采用线性插值方式。"""
    if not xs:
        raise ValueError("empty")
    if not (0 <= p <= 100):
        raise ValueError("--percentile must be in [0, 100]")
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    # 使用与 `numpy.percentile` 类似的线性插值策略。
    r = (p / 100) * (len(s) - 1)
    lo = int(math.floor(r))
    hi = int(math.ceil(r))
    if lo == hi:
        return float(s[lo])
    t = r - lo
    return float((1 - t) * s[lo] + t * s[hi])


@dataclass(frozen=True)
class Stats:
    """保存目标尺寸分布的汇总统计结果。"""

    n_files: int
    n_boxes: int
    min_: float
    max_: float
    mean: float
    median: float
    p10: float
    p30: float
    p50: float
    p70: float
    p90: float
    trimmed_mean_10: float


def compute_c_from_labels(
    labels_root: Path,
    imgsz: int,
    size_mode: str,
) -> Tuple[float, Stats]:
    """从标签目录计算推荐的 NWD 常数及其统计信息。"""
    label_files = list(_iter_label_files(labels_root))
    sizes: List[float] = []

    for f in label_files:
        rows = _parse_yolo_txt(f)
        for _, _, _, w_n, h_n in rows:
            w_px = w_n * float(imgsz)
            h_px = h_n * float(imgsz)
            sizes.append(_abs_size_from_wh_px(w_px, h_px, size_mode))

    if not sizes:
        raise RuntimeError(f"No valid boxes found under: {labels_root}")

    smin = min(sizes)
    smax = max(sizes)
    mean = float(statistics.fmean(sizes))
    med = float(statistics.median(sizes))
    p10 = _percentile(sizes, 10)
    p30 = _percentile(sizes, 30)
    p50 = _percentile(sizes, 50)
    p70 = _percentile(sizes, 70)
    p90 = _percentile(sizes, 90)
    t10 = _trimmed_mean(sizes, 0.10)

    # paper-style C: dataset average absolute size
    c = mean

    st = Stats(
        n_files=len(label_files),
        n_boxes=len(sizes),
        min_=smin,
        max_=smax,
        mean=mean,
        median=med,
        p10=p10,
        p30=p30,
        p50=p50,
        p70=p70,
        p90=p90,
        trimmed_mean_10=t10,
    )
    return c, st


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行入口：解析参数、计算统计量并打印推荐常数。"""
    ap = argparse.ArgumentParser(
        description="递归扫描 YOLO 标签并估计 NWD 常数 C。",
    )
    ap.add_argument(
        "--labels",
        type=str,
        default=r"D:\YOLO_PCB\PKU-Market-PCB\labels",
        help="YOLO 标签根目录，会递归扫描其中的所有 .txt 文件。",
    )
    ap.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="训练输入尺寸，用于将归一化宽高近似换算为像素宽高，默认 960。",
    )
    ap.add_argument(
        "--size-mode",
        type=str,
        default="sqrt_area",
        choices=["sqrt_area", "arith", "diag", "max_edge", "min_edge"],
        help="将目标宽高转换为单个尺寸标量的方式，可选 sqrt_area、arith、diag、max_edge、min_edge。",
    )
    args = ap.parse_args(argv)

    labels_root = Path(args.labels)
    c, st = compute_c_from_labels(
        labels_root=labels_root, imgsz=args.imgsz, size_mode=args.size_mode
    )

    print("=== NWD constant C estimation (from YOLO txt labels) ===")
    print(f"labels_root      : {labels_root}")
    print(f"recursive txt     : {st.n_files} files")
    print(f"total boxes       : {st.n_boxes}")
    print(f"imgsz assumption  : {args.imgsz} x {args.imgsz}")
    print(f"size_mode         : {args.size_mode}")
    print("")
    print("Absolute-size stats (pixels):")
    print(f"min               : {st.min_:.6g}")
    print(f"p10               : {st.p10:.6g}")
    print(f"p30               : {st.p30:.6g}")
    print(f"median (p50)      : {st.median:.6g}")
    print(f"p70               : {st.p70:.6g}")
    print(f"p90               : {st.p90:.6g}")
    print(f"max               : {st.max_:.6g}")
    print(f"trimmed_mean_10%  : {st.trimmed_mean_10:.6g}")
    print(f"mean              : {st.mean:.6g}")
    print("")
    print(f"Recommended C (paper-style: mean absolute size): {c:.6g}")
    print(
        "Tip: If your size distribution is long-tailed, consider using median or p30 as a more robust C."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
