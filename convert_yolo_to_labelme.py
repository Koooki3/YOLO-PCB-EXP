#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 YOLO 标签回写为 Labelme JSON。

功能：
- 将 `detect`、`obb`、`segment` 三种 YOLO 标签转换为 Labelme JSON。
- 根据图片目录读取图像尺寸，并按 `CLASS_NAMES` 恢复类别名称。

使用：
- 交互式运行：`python convert_yolo_to_labelme.py`
"""

import glob
import json
import os

from PIL import Image

# 类别 ID 到名称的映射（适用于 PKU-Market-PCB 等数据集，可按需修改）
CLASS_NAMES = {
    0: "missing_hole",
    1: "mouse_bite",
    2: "open_circuit",
    3: "short",
    4: "spur",
    5: "spurious_copper",
    6: "module1",
    7: "module2",
    8: "module3",
    9: "module4",
    10: "module5",
    11: "defect1",
    12: "defect2",
}


def convert_yolo_to_labelme(image_folder, label_folder, output_folder, mode="detect"):
    """将 YOLO 标签目录中的 .txt 转为 Labelme .json，按 mode 解析行格式。

    Args:
        image_folder: 图像所在目录，用于读取宽高（仅支持与 label 同名的 .jpg）。
        label_folder: YOLO .txt 标签所在目录。
        output_folder: 输出 .json 的目录，已存在则写入。
        mode: 解析模式，'detect' | 'obb' | 'segment'。

    Returns:
        None. 每个 .txt 生成同名 .json，缺失对应图片时跳过并打印提示。
    """
    os.makedirs(output_folder, exist_ok=True)

    label_files = glob.glob(os.path.join(label_folder, "*.txt"))

    for label_file in label_files:
        image_name = os.path.basename(label_file).replace(".txt", ".jpg")
        image_path = os.path.join(image_folder, image_name)

        if not os.path.exists(image_path):
            print(f"图像文件不存在: {image_path}")
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        shapes = []
        with open(label_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if mode == "detect":
                shape = parse_yolo_detect_line(line, width, height)
            elif mode == "obb":
                shape = parse_yolo_obb_line(line, width, height)
            elif mode == "segment":
                shape = parse_yolo_segment_line(line, width, height)
            else:
                continue

            if shape:
                shapes.append(shape)

        labelme_data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": image_name,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
        }

        output_file = os.path.join(
            output_folder, os.path.basename(label_file).replace(".txt", ".json")
        )
        with open(output_file, "w") as f:
            json.dump(labelme_data, f, indent=2)

        print(f"已转换 ({mode}): {label_file} -> {output_file}")


def parse_yolo_detect_line(line, width, height):
    """解析 YOLO detect 一行：class_id x_center y_center width height（归一化）。

    Returns:
        dict | None: Labelme shape（rectangle）或格式错误时 None。
    """
    parts = line.split()
    if len(parts) != 5:
        return None

    class_id = int(parts[0])
    x_center = float(parts[1])
    y_center = float(parts[2])
    w = float(parts[3])
    h = float(parts[4])

    x1 = (x_center - w / 2) * width
    y1 = (y_center - h / 2) * height
    x2 = (x_center + w / 2) * width
    y2 = (y_center + h / 2) * height

    return {
        "label": CLASS_NAMES.get(class_id, f"class_{class_id}"),
        "points": [[x1, y1], [x2, y2]],
        "group_id": None,
        "shape_type": "rectangle",
        "flags": {},
    }


def parse_yolo_obb_line(line, width, height):
    """解析 YOLO OBB 一行：class_id x1 y1 x2 y2 x3 y3 x4 y4（归一化）。

    Returns:
        dict | None: Labelme shape（polygon）或格式错误时 None。
    """
    parts = line.split()
    if len(parts) != 9:
        return None

    class_id = int(parts[0])

    points = []
    for i in range(1, 9, 2):
        x = float(parts[i]) * width
        y = float(parts[i + 1]) * height
        points.append([x, y])

    return {
        "label": CLASS_NAMES.get(class_id, f"class_{class_id}"),
        "points": points,
        "group_id": None,
        "shape_type": "polygon",
        "flags": {},
    }


def parse_yolo_segment_line(line, width, height):
    """解析 YOLO segment 一行：class_id x1 y1 x2 y2 ... xn yn（归一化）。

    Returns:
        dict | None: Labelme shape（polygon）或格式错误时 None。
    """
    parts = line.split()
    if len(parts) < 3 or len(parts) % 2 != 1:
        return None

    class_id = int(parts[0])

    points = []
    for i in range(1, len(parts), 2):
        x = float(parts[i]) * width
        y = float(parts[i + 1]) * height
        points.append([x, y])

    return {
        "label": CLASS_NAMES.get(class_id, f"class_{class_id}"),
        "points": points,
        "group_id": None,
        "shape_type": "polygon",
        "flags": {},
    }


def main():
    """交互式选择 detect/obb/segment，对 train 与 val 执行 YOLO→Labelme 转换。"""
    dataset_root = "D:\\YOLO_PCB\\PKU-Market-PCB"

    print("请选择转换模式：")
    print("1. Detect (目标检测) - 边界框标注")
    print("2. OBB (有向包围盒) - 旋转目标检测")
    print("3. Segment (实例分割) - 多边形标注")

    choice = input("请输入选项（1/2/3）：").strip()

    mode_map = {"1": "detect", "2": "obb", "3": "segment"}

    if choice not in mode_map:
        print("输入无效，请重新运行并选择 1、2 或 3。")
        return

    mode = mode_map[choice]
    print(f"\n已选择模式：{mode}")

    print("\n开始处理 train 集...")
    train_image_folder = os.path.join(dataset_root, "images", "train")
    train_label_folder = os.path.join(dataset_root, "labels", "train")
    train_output_folder = os.path.join(dataset_root, f"labelme_labels_{mode}", "train")
    convert_yolo_to_labelme(
        train_image_folder, train_label_folder, train_output_folder, mode
    )

    print("\n开始处理 val 集...")
    val_image_folder = os.path.join(dataset_root, "images", "val")
    val_label_folder = os.path.join(dataset_root, "labels", "val")
    val_output_folder = os.path.join(dataset_root, f"labelme_labels_{mode}", "val")
    convert_yolo_to_labelme(val_image_folder, val_label_folder, val_output_folder, mode)

    print("\n转换完成。")


if __name__ == "__main__":
    main()
