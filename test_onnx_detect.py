#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ONNX 版 YOLO 检测推理与测速脚本。

功能：
- 基于 `onnxruntime` 执行单图或目录批量推理。
- 统计读取、预处理、推理、后处理、绘图各阶段耗时。
- 解析 `data.yaml` 类别名并保存可视化结果。

使用：
- `python test_onnx_detect.py --model best.onnx --source demo.jpg --data data.yaml`
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


def _lazy_import_ort():
    """延迟导入 onnxruntime，缺失时抛出 RuntimeError 提示安装。"""
    try:
        import onnxruntime as ort  # type: ignore

        return ort
    except Exception as e:
        raise RuntimeError(
            "未找到 onnxruntime。请先安装：pip install onnxruntime （CPU）或 "
            "pip install onnxruntime-gpu （NVIDIA GPU）"
        ) from e


def _lazy_import_ultralytics_utils():
    """延迟导入 torch、YAML、scale_boxes、xywh2xyxy、non_max_suppression。"""
    try:
        import torch
        from ultralytics.utils import YAML
        from ultralytics.utils.nms import non_max_suppression
        from ultralytics.utils.ops import scale_boxes, xywh2xyxy

        return torch, YAML, scale_boxes, xywh2xyxy, non_max_suppression
    except Exception as e:
        raise RuntimeError(
            "导入 ultralytics 工具失败。请确认可正常 import ultralytics。"
        ) from e


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(source: str) -> List[Path]:
    """收集 source 指向的单文件或目录下所有图片路径（含子目录），按路径排序。"""
    p = Path(source)
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f"source 不存在: {source}")
    files = [x for x in p.rglob("*") if x.suffix.lower() in IMG_EXTS]
    files.sort()
    return files


def load_names(data_yaml: Optional[str]) -> List[str]:
    """从 data.yaml 的 names（dict 或 list）解析类别名列表，与 id 顺序一致。"""
    if not data_yaml:
        return []
    _, YAML, *_ = _lazy_import_ultralytics_utils()
    d = YAML.load(data_yaml)
    names = d.get("names")
    if isinstance(names, dict):
        # {0: 'a', 1: 'b'} -> ['a','b']
        return [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    if isinstance(names, list):
        return names
    return []


def letterbox(
    im: np.ndarray,
    new_shape: Tuple[int, int],
    color: tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scale_fill: bool = False,
    scale_up: bool = True,
    stride: int = 32,
):
    """YOLO 风格 letterbox：等比例缩放+ padding。返回 (resize 后图, ratio, (left, top))。"""
    shape = im.shape[:2]  # h,w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up:
        r = min(r, 1.0)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # w,h
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # w,h padding

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        r = (new_shape[1] / shape[1], new_shape[0] / shape[0])

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    ratio = (r, r) if isinstance(r, float) else r
    return im, ratio, (left, top)


def draw_boxes_bgr(img: np.ndarray, dets: np.ndarray, names: list[str]):
    """在原图上绘制检测框与标签。dets: Nx6 为 x1,y1,x2,y2,conf,cls（原地修改 img）。"""
    for x1, y1, x2, y2, conf, cls in dets:
        cls_i = int(cls)
        label_name = names[cls_i] if 0 <= cls_i < len(names) else str(cls_i)
        label = f"{label_name} {float(conf):.2f}"
        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = y1i - 6 if y1i - 6 > th else y1i + th + 6
        cv2.rectangle(
            img, (x1i, y_text - th - 6), (x1i + tw + 6, y_text + 2), (0, 255, 0), -1
        )
        cv2.putText(
            img,
            label,
            (x1i + 3, y_text - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )


def infer_one(
    session,
    input_name: str,
    img_path: Path,
    imgsz: int,
    conf: float,
    iou: float,
    names: list[str],
    warmup: bool = False,
):
    """对单张图片做 ONNX 推理+NMS+坐标还原，返回 (绘制后图, dets Nx6, 各阶段耗时 dict)。"""
    torch, _, scale_boxes, xywh2xyxy, non_max_suppression = (
        _lazy_import_ultralytics_utils()
    )

    t0 = time.perf_counter()
    im0 = cv2.imread(str(img_path))
    if im0 is None:
        raise RuntimeError(f"读取图片失败: {img_path}")

    t_read = time.perf_counter()
    im, ratio, pad = letterbox(im0, (imgsz, imgsz))
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_chw = im_rgb.transpose(2, 0, 1)  # HWC -> CHW
    im_chw = np.ascontiguousarray(im_chw, dtype=np.float32) / 255.0
    inp = im_chw[None, ...]  # 1x3xHxW
    t_pre = time.perf_counter()

    # warmup 不计时：避免第一次构图/内存分配影响
    if warmup:
        _ = session.run(None, {input_name: inp})

    t1 = time.perf_counter()
    outputs = session.run(None, {input_name: inp})
    t2 = time.perf_counter()

    # 兼容常见 YOLO onnx 输出：
    # - (1,84,8400) 或 (1, nc+4, n)
    # - (1, n, 85) 或 (1, n, nc+5)
    out0 = outputs[0]
    pred = torch.from_numpy(out0)
    if pred.ndim == 3 and pred.shape[1] < pred.shape[2]:
        # BCN: (1,84,8400) -> keep
        pass
    elif pred.ndim == 3 and pred.shape[2] < pred.shape[1]:
        # BNC: (1,8400,84) -> transpose to BCN
        pred = pred.transpose(1, 2)
    else:
        # 兜底：尝试 reshape 到 (1,*,*) 不处理
        pass

    # NMS（输出：xyxy, conf, cls）
    det_list = non_max_suppression(pred, conf_thres=conf, iou_thres=iou)
    det = det_list[0]
    if det is None or len(det) == 0:
        det_np = np.zeros((0, 6), dtype=np.float32)
    else:
        # 将推理图尺度映射回原图（用 ratio/pad 与 scale_boxes 一致）
        # scale_boxes 需要 ratio_pad=((gain,gain),(padx,pady))；这里 pad 是 (left, top)
        ratio_pad = (ratio, pad)
        det[:, :4] = scale_boxes(
            (imgsz, imgsz), det[:, :4], im0.shape, ratio_pad=ratio_pad
        )
        det_np = det[:, :6].detach().cpu().numpy()

    t3 = time.perf_counter()
    out_img = im0.copy()
    if det_np.shape[0]:
        draw_boxes_bgr(out_img, det_np, names)
    t4 = time.perf_counter()

    times = {
        "read_ms": (t_read - t0) * 1000.0,
        "pre_ms": (t_pre - t_read) * 1000.0,
        "infer_ms": (t2 - t1) * 1000.0,
        "post_ms": (t3 - t2) * 1000.0,
        "draw_ms": (t4 - t3) * 1000.0,
        "total_ms": (t4 - t0) * 1000.0,
    }
    return out_img, det_np, times


def format_provider_info(session) -> str:
    """返回 onnxruntime 的 device 与 providers 信息，用于日志。"""
    ort = _lazy_import_ort()
    providers = session.get_providers()
    device = ort.get_device()
    return f"onnxruntime device={device}, providers={providers}"


def main():
    """解析参数、加载 ONNX、遍历图片推理、保存结果图并打印平均耗时。"""
    parser = argparse.ArgumentParser(
        description="运行 ONNX 版 YOLO 检测推理，并统计各阶段耗时。"
    )
    parser.add_argument("--model", type=str, required=False, help="ONNX 模型文件路径。")
    parser.add_argument(
        "--source", type=str, required=False, help="输入路径，可为单张图片或图片目录。"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=r"D:\YOLO_PCB\PKU-Market-PCB-ex\pku_market_pcb_ex.yaml",
        help="数据集 YAML 路径，用于读取类别名称；不提供时仅显示类别 ID。",
    )
    parser.add_argument(
        "--imgsz", type=int, default=960, help="推理输入尺寸，默认 960。"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5, help="预测置信度阈值，默认 0.5。"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45, help="NMS 的 IoU 阈值，默认 0.45。"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(Path("test_results") / "detect_onnx_prediction"),
        help="推理结果输出目录。",
    )
    parser.add_argument(
        "--providers",
        type=str,
        default="CPUExecutionProvider",
        help="onnxruntime 执行提供者，多个值用逗号分隔，例如 CUDAExecutionProvider,CPUExecutionProvider。",
    )
    parser.add_argument(
        "--warmup", type=int, default=1, help="正式计时前的预热次数，默认 1。"
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=0,
        help="onnxruntime CPU 线程数；0 表示按物理核心自动选择，负数表示使用 ORT 默认设置。",
    )
    parser.add_argument(
        "--rk3588_scale",
        type=float,
        default=1.0,
        help="RK3588 全流程耗时估计缩放系数；1.0 表示不估算，3.0 表示估算总耗时约为本机的 3 倍。",
    )
    args = parser.parse_args()

    # 默认配置（对齐你 test_all.py 的“未传参则跑默认示例”行为）
    if not args.model or not args.source:
        print("未提供完整命令行参数，改用默认配置进行 ONNX detect 测试。\n")
        args.model = (
            args.model
            or r"D:\YOLO_PCB\train_ex\Ex_12s_960_WassersteinLoss0.7_DINOP2\weights\best_int8.onnx"
        )
        args.source = (
            args.source
            or r"D:\YOLO_PCB\PKU-Market-PCB-ex\images\train\05_scratch_val.bmp"
        )
        # args.data 可按需填写：例如 r"D:\YOLO_PCB\PKU-Market-PCB-ex\data.yaml"

    ort = _lazy_import_ort()
    names = load_names(args.data)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 创建 SessionOptions：启用最高级图优化，并配置 CPU 线程数（intra_op_num_threads）
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if args.num_threads >= 0:
        sess_options.intra_op_num_threads = args.num_threads

    provider_list = [p.strip() for p in args.providers.split(",") if p.strip()]
    if provider_list:
        session = ort.InferenceSession(
            args.model, sess_options, providers=provider_list
        )
    else:
        # 让 ORT 自己选择（CPU 机器就 CPU；有 CUDA 则通常会选 CUDA/CPU）
        session = ort.InferenceSession(args.model, sess_options)

    input0 = session.get_inputs()[0]
    input_name = input0.name

    print(f"{'=' * 60}")
    print("开始执行 ONNX detect 测试。")
    print(f"{'=' * 60}")
    print(f"模型路径：{args.model}")
    print(f"输入路径：{args.source}")
    print(f"输出目录：{outdir}")
    print(f"imgsz: {args.imgsz}, conf: {args.conf}, iou: {args.iou}")
    print(f"intra_op_num_threads：{args.num_threads}（0=自动，<0=使用 ORT 默认）")
    print(f"推理设备信息：{format_provider_info(session)}")
    if names:
        print(f"类别数量：{len(names)}（来自 {args.data}）")
    else:
        print("类别名称：未提供，将只显示 class_id。")
    print(f"{'=' * 60}\n")

    # warmup
    if args.warmup and args.warmup > 0:
        dummy = np.zeros((1, 3, args.imgsz, args.imgsz), dtype=np.float32)
        for _ in range(args.warmup):
            _ = session.run(None, {input_name: dummy})

    img_list = list_images(args.source)
    if not img_list:
        raise RuntimeError("未找到可推理的图片文件。")

    sum_read = sum_pre = sum_infer = sum_post = sum_draw = sum_total = 0.0
    for idx, p in enumerate(img_list, 1):
        out_img, det_np, times = infer_one(
            session=session,
            input_name=input_name,
            img_path=p,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            names=names,
        )

        save_path = outdir / p.name
        cv2.imwrite(str(save_path), out_img)

        sum_read += times["read_ms"]
        sum_pre += times["pre_ms"]
        sum_infer += times["infer_ms"]
        sum_post += times["post_ms"]
        sum_draw += times["draw_ms"]
        sum_total += times["total_ms"]

        print(
            f"[{idx}/{len(img_list)}] {p.name} | det={det_np.shape[0]} | "
            f"read={times['read_ms']:.2f}ms, pre={times['pre_ms']:.2f}ms, infer={times['infer_ms']:.2f}ms, "
            f"post={times['post_ms']:.2f}ms, draw={times['draw_ms']:.2f}ms, total={times['total_ms']:.2f}ms"
        )

        if args.rk3588_scale and args.rk3588_scale != 1.0:
            est = times["total_ms"] * float(args.rk3588_scale)
            print(
                f"    RK3588 全流程耗时估计: {est:.2f}ms（基于 rk3588_scale={args.rk3588_scale}）"
            )

    n = len(img_list)
    print(f"\n{'-' * 60}")
    print("耗时统计（平均）：")
    print(f"{'-' * 60}")
    print(f"read:  {sum_read / n:.2f} ms")
    print(f"pre:   {sum_pre / n:.2f} ms")
    print(f"infer: {sum_infer / n:.2f} ms")
    print(f"post:  {sum_post / n:.2f} ms")
    print(f"draw:  {sum_draw / n:.2f} ms")
    print(f"total: {sum_total / n:.2f} ms")
    print(f"结果已保存到：{outdir}")


if __name__ == "__main__":
    # OpenCV 在多线程下偶发卡顿/耗时抖动，关闭线程更利于稳定统计
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass
    main()
