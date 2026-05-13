#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量重命名数据集中的图片和标签文件。

功能：
- 将 `images/` 和 `labels/` 中的文件重命名为连续编号。
- 保持图片与同名标签的一一对应，适合整理原始数据集。

使用：
- `python rename.py D:/YOLO_PCB/PKU-Market-PCB-raw`
- 不传参数时脚本会进入交互式输入模式。
"""

import os
import sys


def rename_dataset(dataset_path):
    """将数据集 images 与 labels 下文件按顺序重命名为 0001、0002、…

    Args:
        dataset_path: 数据集根目录，其下需有 images/ 与 labels/ 子目录。

    Returns:
        bool: 全部成功返回 True，路径或写文件异常时返回 False。
    """
    if not os.path.exists(dataset_path):
        print(f"错误：数据集路径 '{dataset_path}' 不存在")
        return False

    # 定义 images 和 labels 目录路径。
    images_dir = os.path.join(dataset_path, "images")
    labels_dir = os.path.join(dataset_path, "labels")

    # 检查 images 和 labels 目录是否存在。
    if not os.path.exists(images_dir):
        print(f"错误：'{images_dir}' 文件夹不存在")
        return False

    if not os.path.exists(labels_dir):
        print(f"错误：'{labels_dir}' 文件夹不存在")
        return False

    # 获取 images 目录中的所有文件。
    image_files = [
        f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))
    ]

    if not image_files:
        print(f"错误：'{images_dir}' 文件夹中没有图片文件")
        return False

    # 先按文件名排序，保证重命名结果稳定可复现。
    image_files.sort()

    # 遍历图片文件并依次重命名。
    for index, image_file in enumerate(image_files, start=1):
        # 获取图片文件的扩展名。
        image_ext = os.path.splitext(image_file)[1]

        # 生成新的文件名，使用 4 位数字，不足前面补 0。
        new_name = f"{index:04d}"

        # 计算原图片路径和新图片路径。
        old_image_path = os.path.join(images_dir, image_file)
        new_image_path = os.path.join(images_dir, f"{new_name}{image_ext}")

        # 计算原标签路径和新标签路径。
        old_label_name = os.path.splitext(image_file)[0] + ".txt"
        old_label_path = os.path.join(labels_dir, old_label_name)
        new_label_path = os.path.join(labels_dir, f"{new_name}.txt")

        try:
            # 重命名图片文件。
            os.rename(old_image_path, new_image_path)
            print(f"重命名图片：{image_file} -> {new_name}{image_ext}")

            # 检查标签文件是否存在。
            if os.path.exists(old_label_path):
                # 重命名标签文件。
                os.rename(old_label_path, new_label_path)
                print(f"重命名标签：{old_label_name} -> {new_name}.txt")
            else:
                print(f"警告：标签文件 '{old_label_name}' 不存在")
        except Exception as e:
            print(f"错误：重命名文件时发生错误 - {e}")
            return False

    print("\n重命名完成！")
    return True


def main():
    """从命令行参数或交互输入获取数据集路径，并执行重命名。"""
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = input("请输入数据集路径：")

    # 调用重命名函数。
    rename_dataset(dataset_path)


if __name__ == "__main__":
    main()
