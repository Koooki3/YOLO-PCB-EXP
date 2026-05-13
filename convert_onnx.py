#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 YOLO 模型导出为 ONNX。

功能：
- 支持 `detect`、`obb`、`segment` 三类任务权重导出。
- 可配置输入尺寸、ONNX opset，并可选执行 INT8 量化。

使用：
- `python convert_onnx.py --model weights/best.pt --task detect --imgsz 640`
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

from ultralytics import YOLO


def convert_to_onnx(
    model_path,
    output_path=None,
    imgsz=640,
    opset=12,
    task="detect",
    int8: bool = False,
):
    """将 YOLO 模型导出为 ONNX 格式。

    Args:
        model_path: 原始模型路径（.pt）。
        output_path: 输出 .onnx 路径；为 None 时与 model_path 同目录、同名 .onnx。
        imgsz: 导出时的输入图像尺寸（正方形边长），默认 640。
        opset: ONNX 算子集版本，默认 12。
        task: 任务类型，'detect' | 'obb' | 'segment'，默认 'detect'。
        int8: 是否在导出后执行 ONNX Runtime 动态 INT8 量化（FP32 → INT8）。

    Returns:
        Path: 生成的 ONNX 文件路径。
    """
    # 加载模型
    model = YOLO(model_path)

    # 设置输出路径
    if output_path is None:
        output_path = Path(model_path).with_suffix(".onnx")
    else:
        output_path = Path(output_path)

    # 执行转换
    # 设置工作目录到输出路径的父目录，让export方法自动生成文件
    original_cwd = os.getcwd()
    os.chdir(str(output_path.parent))

    try:
        # INT8 路线需要 FP32 ONNX 作为输入；不做 INT8 时保持原来的 half=True。
        half = False if int8 else True

        model.export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,
            simplify=True,
            half=half,
            dynamic=False,
            batch=1,
            name=output_path.stem,
            task=task,
        )
    finally:
        os.chdir(original_cwd)

    # 可选：导出后做 INT8 量化（基于 onnxruntime.quantization）
    if int8:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
        except Exception as e:
            raise RuntimeError(
                "INT8 量化失败：缺少 onnxruntime 量化依赖。\n"
                "请先安装：pip install onnxruntime onnxruntime-tools"
            ) from e

        int8_output_path = output_path.with_name(output_path.stem + "_int8.onnx")

        print(
            f"开始对 ONNX 模型进行动态 INT8 量化...\n"
            f"  输入模型: {output_path}\n"
            f"  输出模型: {int8_output_path}"
        )

        # 注意：部分 onnxruntime 版本的 quantize_dynamic 不支持 optimize_model 参数，
        # 这里只使用通用参数，避免出现 unexpected keyword argument 错误。
        quantize_dynamic(
            model_input=str(output_path),
            model_output=str(int8_output_path),
            weight_type=QuantType.QInt8,
            per_channel=True,
        )

        print("INT8 量化完成。")
        return int8_output_path

    return output_path


def main():
    """解析命令行参数并执行 YOLO → ONNX 转换，打印结果或错误信息。"""
    parser = argparse.ArgumentParser(description="将 YOLO 权重导出为 ONNX 文件。")
    parser.add_argument(
        "--model",
        type=str,
        default=r"D:\YOLO_PCB\train_ex\Ex_12s_960_WassersteinLoss0.7_DINOP2\weights\best.pt",
        help="输入的 YOLO 权重路径。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 ONNX 文件路径，默认与模型位于同目录。",
    )
    parser.add_argument(
        "--imgsz", type=int, default=960, help="导出时使用的输入图像尺寸。"
    )
    parser.add_argument(
        "--opset", type=int, default=14, help="导出使用的 ONNX opset 版本。"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="detect",
        choices=["detect", "obb", "segment"],
        help="任务类型，可选 detect、obb、segment，默认 detect。",
    )
    parser.add_argument(
        "--int8",
        default="--int8",
        action="store_true",
        help="导出完成后自动执行 ONNX Runtime 动态 INT8 量化。",
    )

    args = parser.parse_args()

    # 确保模型路径存在
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"错误：模型文件 {model_path} 不存在")
        return

    print(f"待转换模型：{model_path}")
    print(f"输入图像尺寸：{args.imgsz}")
    print(f"ONNX opset 版本：{args.opset}")
    print(f"任务类型：{args.task}")
    print(f"是否启用 INT8 量化：{args.int8}")

    try:
        # 执行转换
        output_path = convert_to_onnx(
            model_path=model_path,
            output_path=args.output,
            imgsz=args.imgsz,
            opset=args.opset,
            task=args.task,
            int8=args.int8,
        )

        print("\n转换成功。")
        print(f"ONNX 文件已保存至：{output_path}")
        print(f"文件大小：{os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    except Exception as e:
        print("\n转换失败。")
        print(f"错误信息：{str(e)}")


if __name__ == "__main__":
    main()
