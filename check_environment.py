#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""深度学习运行环境巡检脚本。

功能：
- 检查 Python、PyTorch、CUDA、cuDNN、GPU、常用依赖和 Conda 环境状态。
- 输出当前机器的关键版本信息，便于定位训练或推理环境问题。

使用：
- 直接运行：`python check_environment.py`
- 适合在新机器部署、环境迁移或排查依赖冲突时使用。
"""

import os
import platform
import subprocess
import sys
from datetime import datetime

# 避免多份 OpenMP 运行时冲突（如 libiomp5md.dll 重复加载），若已设则沿用
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def print_section(title):
    """打印分节标题横幅。"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_python():
    """检查并打印 Python 版本、路径、平台与架构。"""
    print_section("Python 环境")
    print(f"Python 版本: {sys.version}")
    print(f"Python 路径: {sys.executable}")
    print(f"平台: {platform.system()} {platform.release()}")
    print(f"架构: {platform.machine()}")


def check_torch():
    """检查并打印 PyTorch 版本、CUDA/cuDNN、GPU 数量与显存。"""
    print_section("PyTorch 环境")
    try:
        import torch

        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
            print(f"GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(
                    f"    显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB"
                )
        else:
            print("CUDA 不可用，使用 CPU")
    except ImportError:
        print("PyTorch 未安装")


def check_cuda():
    """检查 nvcc、nvidia-smi 可用性并打印输出。"""
    print_section("CUDA 环境")
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("nvcc 命令不可用")
    except FileNotFoundError:
        print("nvcc 命令未找到")

    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("\nGPU 信息:")
            print(result.stdout)
        else:
            print("nvidia-smi 命令不可用")
    except FileNotFoundError:
        print("nvidia-smi 命令未找到")


def check_packages():
    """检查 torch、ultralytics、onnx、onnxruntime 等常用包是否已安装及版本。"""
    print_section("常用深度学习包")
    packages = [
        "torch",
        "torchvision",
        "torchaudio",
        "numpy",
        "opencv-python",
        "PIL",
        "matplotlib",
        "scipy",
        "scikit-learn",
        "pandas",
        "tqdm",
        "ultralytics",
        "onnx",
        "onnxruntime",
    ]

    for package in packages:
        try:
            if package == "opencv-python":
                import cv2

                version = cv2.__version__
            elif package == "PIL":
                from PIL import Image

                version = Image.__version__
            else:
                module = __import__(package)
                version = getattr(module, "__version__", "未知版本")
            print(f"{package:20s}: {version}")
        except ImportError:
            print(f"{package:20s}: 未安装")


def check_conda():
    """检查 conda 命令及 env list 是否可用。"""
    print_section("Conda 环境")
    try:
        result = subprocess.run(["conda", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Conda 版本: {result.stdout.strip()}")
        else:
            print("Conda 命令不可用")
    except FileNotFoundError:
        print("Conda 命令未找到")

    try:
        result = subprocess.run(
            ["conda", "env", "list"], capture_output=True, text=True
        )
        if result.returncode == 0:
            print("\nConda 环境:")
            print(result.stdout)
    except Exception:
        pass


def main():
    """依次执行 Python、PyTorch、CUDA、常用包、Conda 检查并打印报告。"""
    print(f"\n{'='*60}")
    print(f"  环境信息检查报告")
    print(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    check_python()
    check_torch()
    check_cuda()
    check_packages()
    check_conda()

    print(f"\n{'='*60}")
    print("  检查完成")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
