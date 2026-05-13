#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PyTorch 版 YOLO 推理测试脚本。

功能：
- 加载 Ultralytics YOLO 权重，对单张图片或目录执行预测。
- 支持 `detect`、`obb`、`segment` 三类任务。
- 未传参时可按脚本内默认配置批量运行测试任务。

使用：
- `python test_all.py --model weights/best.pt --image demo.jpg --task detect`
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def get_device():
    """检测并返回可用设备，优先 CUDA，否则 CPU。会向 stdout 打印检测结果。"""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"检测到 GPU 设备：{torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("未检测到 GPU 设备，将使用 CPU。")
    return device


def test_model(model_path, image_path, task_type, conf=0.25, iou=0.7, save=True):
    """加载 YOLO 模型并对指定图像或目录执行预测，可选保存到 test_results。

    Args:
        model_path: 模型权重路径（.pt）。
        image_path: 图像路径或目录。
        task_type: 任务类型，'detect' | 'obb' | 'segment'。
        conf: 置信度阈值，默认 0.25。
        iou: NMS IOU 阈值，默认 0.7。
        save: 是否保存可视化结果到 test_results/{task_type}_prediction。

    Returns:
        None. 结果打印到 stdout，失败时提前返回。
    """
    print(f"\n{'='*60}")
    print("开始测试模型。")
    print(f"{'='*60}")
    print(f"模型路径：{model_path}")
    print(f"测试输入：{image_path}")
    print(f"任务类型：{task_type}")
    print(f"置信度阈值：{conf}")
    print(f"IoU 阈值：{iou}")

    # 检测并设置推理设备。
    device = get_device()
    print(f"使用设备：{device}")
    print(f"{'='*60}\n")

    # 加载模型。
    try:
        model = YOLO(model_path)
        # 将模型移动到指定设备。
        model.to(device)
        print("模型加载成功。")
    except Exception as e:
        print(f"模型加载失败：{e}")
        return

    # 执行预测。
    try:
        results = model.predict(
            source=image_path,
            task=task_type,
            conf=conf,
            iou=iou,
            save=save,
            device=device,
            project="test_results",
            name=f"{task_type}_prediction",
            exist_ok=True,
        )
        print("预测完成。")
        print(f"预测结果已保存到：test_results/{task_type}_prediction")

        # 输出预测结果统计。
        print("\n预测结果统计：")
        for i, result in enumerate(results):
            print(f"  图像 {i+1}: {result.path}")
            if task_type == "detect":
                print(f"    检测到 {len(result.boxes)} 个目标")
            elif task_type == "obb":
                print(f"    检测到 {len(result.boxes)} 个旋转框")
            elif task_type == "segment":
                print(f"    检测到 {len(result.boxes)} 个分割对象")

    except Exception as e:
        print(f"预测失败：{e}")


def main():
    """解析命令行参数或使用默认配置，调用 test_model 执行 predict。"""
    parser = argparse.ArgumentParser(
        description="使用 PyTorch 版 Ultralytics YOLO 执行推理测试。"
    )
    parser.add_argument("--model", type=str, help="模型权重文件路径。")
    parser.add_argument(
        "--image", type=str, help="测试输入路径，可为单张图片或图片目录。"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["detect", "obb", "segment"],
        help="任务类型，可选 detect、obb、segment。",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="预测置信度阈值，默认 0.25。"
    )
    parser.add_argument(
        "--iou", type=float, default=0.7, help="NMS 的 IoU 阈值，默认 0.7。"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="只执行推理，不保存可视化结果。"
    )

    args = parser.parse_args()

    # 如果没有提供命令行参数，使用默认配置测试两个任务
    if not args.model or not args.image or not args.task:
        print("未提供完整命令行参数，改用默认配置测试两个任务。\n")

        # 测试配置
        test_configs = [
            {
                "model_path": r"D:\YOLO_PCB\train_ex\Ex_12s_960_WassersteinLoss0.7_DINOP2\weights\best.pt",
                "image_path": r"D:\YOLO_PCB\result_optimized_matched.jpg",  # 请修改为实际的测试图像路径
                "task_type": "detect",
                "conf": 0.5,
                "iou": 0.45,
                "save": True,
            },
        ]

        # 执行测试
        for config in test_configs:
            test_model(**config)
    else:
        # 使用命令行参数进行测试
        test_model(
            model_path=args.model,
            image_path=args.image,
            task_type=args.task,
            conf=args.conf,
            iou=args.iou,
            save=not args.no_save,
        )


if __name__ == "__main__":
    main()
