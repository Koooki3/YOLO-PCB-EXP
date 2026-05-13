#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按训练集与验证集拆分 YOLO 数据集。

功能：
- 将根目录下的 `images/` 与 `labels/` 按固定比例划分到 `train/` 和 `val/`。
- 校验图片与标签的同名对应关系，并同步移动同名 `json` 文件。

使用：
- `python split.py D:/YOLO_PCB/PKU-Market-PCB-raw`
- 默认使用 8:2 拆分，适合快速准备训练数据目录结构。
"""

import os
import random
import shutil
import sys


def check_file_matching(images_dir, labels_dir):
    """检查图片与标签是否一一对应（按 basename，忽略扩展名）。

    Args:
        images_dir: 图片目录，仅统计 .jpg/.jpeg/.png/.bmp。
        labels_dir: 标签目录，仅统计 .txt。

    Returns:
        tuple: (is_matched: bool, missing_labels: list, missing_images: list)。
    """
    image_files = [
        f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))
    ]
    label_files = [
        f for f in os.listdir(labels_dir) if os.path.isfile(os.path.join(labels_dir, f))
    ]

    image_basenames = set()
    label_basenames = set()

    for img_file in image_files:
        name, ext = os.path.splitext(img_file)
        if ext.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            image_basenames.add(name)

    for label_file in label_files:
        name, ext = os.path.splitext(label_file)
        if ext.lower() == ".txt":
            label_basenames.add(name)

    missing_labels = []
    missing_images = []

    for basename in image_basenames:
        if basename not in label_basenames:
            missing_labels.append(basename)

    for basename in label_basenames:
        if basename not in image_basenames:
            missing_images.append(basename)

    is_matched = len(missing_labels) == 0 and len(missing_images) == 0

    return is_matched, missing_labels, missing_images


def split_dataset(dataset_path):
    """将 dataset_path 下的 images/labels 按 8:2 随机划分为 train/val。

    会先做 check_file_matching，不通过则直接返回。划分后文件移动到
    images/train、images/val、labels/train、labels/val；同名 .json 随图片移动。

    Args:
        dataset_path: 数据集根目录。

    Returns:
        bool: 划分成功返回 True，否则 False。
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

    print("\n开始检查图片与标签的对应关系...")
    is_matched, missing_labels, missing_images = check_file_matching(
        images_dir, labels_dir
    )

    if not is_matched:
        print("\n文件匹配检查失败！")

        if missing_labels:
            print(f"\n以下图片缺少对应的标签文件（共 {len(missing_labels)} 个）：")
            for i, basename in enumerate(missing_labels[:10], 1):
                print(f"  {i}. {basename}")
            if len(missing_labels) > 10:
                print(f"  ... 还有 {len(missing_labels) - 10} 个文件")

        if missing_images:
            print(f"\n以下标签文件缺少对应的图片文件（共 {len(missing_images)} 个）：")
            for i, basename in enumerate(missing_images[:10], 1):
                print(f"  {i}. {basename}")
            if len(missing_images) > 10:
                print(f"  ... 还有 {len(missing_images) - 10} 个文件")

        print("\n请确保每个图片文件都有对应的标签文件后再运行划分程序。")
        return False

    print("图片与标签匹配检查通过。")

    image_files = [
        f
        for f in os.listdir(images_dir)
        if os.path.isfile(os.path.join(images_dir, f))
        and os.path.splitext(f)[1].lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ]

    if not image_files:
        print(f"错误：'{images_dir}' 文件夹中没有图片文件")
        return False

    # 随机打乱图片文件顺序
    random.shuffle(image_files)

    # 计算划分比例（8:2）
    total_files = len(image_files)
    train_count = int(total_files * 0.8)

    # 分割训练集和验证集
    train_files = image_files[:train_count]
    val_files = image_files[train_count:]

    # 定义子文件夹路径
    train_images_dir = os.path.join(images_dir, "train")
    val_images_dir = os.path.join(images_dir, "val")
    train_labels_dir = os.path.join(labels_dir, "train")
    val_labels_dir = os.path.join(labels_dir, "val")

    # 创建子文件夹
    for dir_path in [
        train_images_dir,
        val_images_dir,
        train_labels_dir,
        val_labels_dir,
    ]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 移动训练集文件
    print("\n开始移动训练集文件...")
    for image_file in train_files:
        # 图片文件路径
        src_image = os.path.join(images_dir, image_file)
        dst_image = os.path.join(train_images_dir, image_file)

        # 标签文件路径
        label_file = os.path.splitext(image_file)[0] + ".txt"
        src_label = os.path.join(labels_dir, label_file)
        dst_label = os.path.join(train_labels_dir, label_file)

        # JSON文件路径
        json_file = os.path.splitext(image_file)[0] + ".json"
        src_json = os.path.join(images_dir, json_file)
        dst_json = os.path.join(train_images_dir, json_file)

        try:
            # 移动图片文件
            shutil.move(src_image, dst_image)
            print(f"移动图片到训练集：{image_file}")

            # 检查标签文件是否存在
            if os.path.exists(src_label):
                # 移动标签文件
                shutil.move(src_label, dst_label)
                print(f"移动标签到训练集：{label_file}")
            else:
                print(f"警告：训练集标签文件 '{label_file}' 不存在")

            # 检查JSON文件是否存在
            if os.path.exists(src_json):
                # 移动JSON文件
                shutil.move(src_json, dst_json)
                print(f"移动JSON到训练集：{json_file}")
        except Exception as e:
            print(f"错误：移动文件时发生错误 - {e}")
            return False

    # 移动验证集文件
    print("\n开始移动验证集文件...")
    for image_file in val_files:
        # 图片文件路径
        src_image = os.path.join(images_dir, image_file)
        dst_image = os.path.join(val_images_dir, image_file)

        label_file = os.path.splitext(image_file)[0] + ".txt"
        src_label = os.path.join(labels_dir, label_file)
        dst_label = os.path.join(val_labels_dir, label_file)

        json_file = os.path.splitext(image_file)[0] + ".json"
        src_json = os.path.join(images_dir, json_file)
        dst_json = os.path.join(val_images_dir, json_file)

        try:
            # 移动图片文件
            shutil.move(src_image, dst_image)
            print(f"移动图片到验证集：{image_file}")

            if os.path.exists(src_label):
                shutil.move(src_label, dst_label)
                print(f"移动标签到验证集：{label_file}")
            else:
                print(f"警告：验证集标签文件 '{label_file}' 不存在")

            if os.path.exists(src_json):
                shutil.move(src_json, dst_json)
                print(f"移动JSON到验证集：{json_file}")
        except Exception as e:
            print(f"错误：移动文件时发生错误 - {e}")
            return False

    print(f"\n数据集划分完成！")
    print(f"总文件数：{total_files}")
    print(f"训练集数量：{len(train_files)}")
    print(f"验证集数量：{len(val_files)}")
    return True


def main():
    """从命令行参数或交互输入获取数据集路径，并执行划分。"""
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = input("请输入数据集路径：")

    # 调用划分函数
    split_dataset(dataset_path)


if __name__ == "__main__":
    main()
