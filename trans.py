#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Labelme 与 YOLO 标注互转核心模块。

功能：
- 读写 Labelme JSON 与 YOLO `obb`、`segment`、`detect` 标签文本。
- 提供矩形、多边形、OBB、bbox 之间的几何转换工具。
- 支持类别扫描、类别映射生成和目录级转换流程。

使用：
- 作为库模块供 `batch_trans.py` 及其他脚本复用。
- 也可直接运行以执行单文件或目录转换。
"""

import json
import math
import os
from collections import Counter

import numpy as np

# 类别名 -> 索引，由 scan + generate_class_mapping 或调用方注入
CLASS_MAPPING = {}

# 索引 -> 类别名，由 REVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()} 维护
REVERSE_CLASS_MAPPING = {}


def scan_classes_from_directory(input_dir):
    """扫描目录或单个 JSON 中所有 shape 的 label，返回排序后的类别列表及出现次数。

    Args:
        input_dir: 目录路径或单个 .json 路径。

    Returns:
        tuple: (sorted_class_names, {class_name: count})，无数据时为 ([], {})。
    """
    class_counter = Counter()
    json_files = []

    # 如果是文件，直接处理
    if os.path.isfile(input_dir) and input_dir.endswith(".json"):
        json_files = [input_dir]
    # 如果是目录，扫描所有JSON文件
    elif os.path.isdir(input_dir):
        json_files = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.endswith(".json")
        ]
    else:
        return [], {}

    if not json_files:
        return [], {}

    print(f"正在扫描 {len(json_files)} 个 JSON 文件...")

    for json_path in json_files:
        try:
            data = read_labelme_json(json_path)
            shapes = data.get("shapes", [])
            for shape in shapes:
                label = shape.get("label", "unknown")
                if label and label != "unknown":
                    class_counter[label] += 1
        except Exception as e:
            print(f"  警告: 读取文件 {os.path.basename(json_path)} 时出错: {e}")
            continue

    # 按类别名称排序，确保映射关系的一致性
    sorted_classes = sorted(class_counter.keys())

    return sorted_classes, dict(class_counter)


def generate_class_mapping(classes):
    """按类别列表顺序生成 {class_name: index}，索引从 0 起始。"""
    mapping = {}
    for idx, class_name in enumerate(classes):
        mapping[class_name] = idx
    return mapping


def display_class_mapping_and_confirm(class_mapping, class_stats):
    """打印映射表，提示 y/n/其他；y 返回 True，n 返回 False，其他返回 None。"""
    print("\n" + "=" * 60)
    print("检测到的类别映射关系：")
    print("=" * 60)
    print(f"{'类别名称':<30} {'索引':<10} {'出现次数':<10}")
    print("-" * 60)

    for class_name in sorted(class_mapping.keys()):
        idx = class_mapping[class_name]
        count = class_stats.get(class_name, 0)
        print(f"{class_name:<30} {idx:<10} {count:<10}")

    print("=" * 60)
    print("\n请确认是否使用此映射关系：")
    print("  'y' 或 'Y'：继续执行转换")
    print("  'n' 或 'N'：停止运行")
    print("  其他按键：中断程序运行")

    user_input = input("\n请输入您的选择：").strip().lower()

    if user_input == "y":
        return True
    elif user_input == "n":
        return False
    else:
        return None


def read_labelme_json(json_path):
    """读取 Labelme 格式 JSON，UTF-8。返回解析后的 dict。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_labelme_json(data, output_path):
    """将 dict 按 Labelme 格式写入 JSON，indent=2、ensure_ascii=False。"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_yolo_obb_txt(obb_results, output_path, image_size=None):
    """保存 YOLO OBB 格式：每行 class_index x1 y1 x2 y2 x3 y3 x4 y4；image_size 存在则归一化。"""
    if os.path.exists(output_path):
        os.remove(output_path)

    if not obb_results:
        open(output_path, "w").close()
        return

    with open(output_path, "w") as f:
        for class_index, obb_points in obb_results:
            if image_size:
                img_width, img_height = image_size
                normalized_points = []
                for point in obb_points:
                    x_norm = point[0] / img_width
                    y_norm = point[1] / img_height
                    normalized_points.extend([x_norm, y_norm])
            else:
                normalized_points = []
                for point in obb_points:
                    normalized_points.extend([point[0], point[1]])

            line = f"{class_index} " + " ".join(
                [f"{coord:.6f}" for coord in normalized_points]
            )
            f.write(line + "\n")


def save_yolo_segment_txt(segment_results, output_path, image_size=None):
    """保存 YOLO segment 格式：每行 class_index x1 y1 x2 y2 ... xn yn；image_size 存在则归一化。"""
    if os.path.exists(output_path):
        os.remove(output_path)

    if not segment_results:
        open(output_path, "w").close()
        return

    with open(output_path, "w") as f:
        for class_index, polygon_points in segment_results:
            if image_size:
                img_width, img_height = image_size
                normalized_points = []
                for point in polygon_points:
                    x_norm = point[0] / img_width
                    y_norm = point[1] / img_height
                    normalized_points.extend([x_norm, y_norm])
            else:
                normalized_points = []
                for point in polygon_points:
                    normalized_points.extend([point[0], point[1]])

            line = f"{class_index} " + " ".join(
                [f"{coord:.6f}" for coord in normalized_points]
            )
            f.write(line + "\n")


def save_yolo_detect_txt(detect_results, output_path, image_size=None):
    """保存 YOLO detect 格式：每行 class_index x_center y_center width height；bbox 为 [x_min,y_min,x_max,y_max]。"""
    if os.path.exists(output_path):
        os.remove(output_path)

    if not detect_results:
        open(output_path, "w").close()
        return

    with open(output_path, "w") as f:
        for class_index, bbox in detect_results:
            x_min, y_min, x_max, y_max = bbox

            if image_size:
                img_width, img_height = image_size
                x_center = (x_min + x_max) / 2 / img_width
                y_center = (y_min + y_max) / 2 / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height
            else:
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min

            line = (
                f"{class_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )
            f.write(line + "\n")


def rectangle_to_polygon(points):
    """将矩形两点 [[x1,y1],[x2,y2]] 转为四点多边形，顺时针：左上、右上、右下、左下。"""
    if len(points) != 2:
        raise ValueError(f"rectangle需要2个点，当前有{len(points)}个点")

    x1, y1 = points[0]
    x2, y2 = points[1]

    # 确保正确的顺序（左上角和右下角）
    x_min = min(x1, x2)
    x_max = max(x1, x2)
    y_min = min(y1, y2)
    y_max = max(y1, y2)

    # 返回顺时针顺序的4个点：左上、右上、右下、左下
    return [
        [float(x_min), float(y_min)],
        [float(x_max), float(y_min)],
        [float(x_max), float(y_max)],
        [float(x_min), float(y_max)],
    ]


def calculate_bbox_from_points(points):
    """从多边形顶点列表计算轴对齐边界框 [x_min, y_min, x_max, y_max]。"""
    if len(points) < 1:
        raise ValueError("至少需要1个点来计算边界框")

    points_array = np.array(points)
    x_min = np.min(points_array[:, 0])
    y_min = np.min(points_array[:, 1])
    x_max = np.max(points_array[:, 0])
    y_max = np.max(points_array[:, 1])

    return [float(x_min), float(y_min), float(x_max), float(y_max)]


def calculate_obb_from_points(points):
    """从多边形前 4 点计算 OBB：点 1–2 作主轴，点 3–4 作近似正交轴，返回 (四点列表, 旋转角弧度)。"""
    if len(points) < 4:
        raise ValueError(f"至少需要4个点来计算OBB，当前只有{len(points)}个点")

    # 提取前4个点
    p1 = np.array(points[0])
    p2 = np.array(points[1])
    p3 = np.array(points[2])
    p4 = np.array(points[3])

    # 计算标准轴（点1->点2）
    axis1 = p2 - p1
    axis1_len = np.linalg.norm(axis1)
    if axis1_len < 1e-6:
        raise ValueError("轴1长度为零，点1和点2重合")
    axis1_unit = axis1 / axis1_len

    # 计算近似正交轴（点3->点4）
    axis2 = p4 - p3
    axis2_len = np.linalg.norm(axis2)
    if axis2_len < 1e-6:
        raise ValueError("轴2长度为零，点3和点4重合")
    axis2_unit = axis2 / axis2_len

    # 拟合标准正交轴：将axis2投影到与axis1正交的方向上
    # 计算axis2在axis1上的投影
    proj = np.dot(axis2_unit, axis1_unit)

    # 计算正交分量
    axis2_ortho = axis2_unit - proj * axis1_unit
    axis2_ortho_len = np.linalg.norm(axis2_ortho)

    # 如果正交分量太小，使用垂直向量
    if axis2_ortho_len < 1e-6:
        # 创建垂直向量
        axis2_ortho = np.array([-axis1_unit[1], axis1_unit[0]])
        axis2_ortho_len = np.linalg.norm(axis2_ortho)

    axis2_ortho_unit = axis2_ortho / axis2_ortho_len

    # 现在我们有了一组正交基：axis1_unit 和 axis2_ortho_unit
    # 计算所有点在正交基下的投影
    all_points = np.array(points)

    # 将原点移到p1
    centered_points = all_points - p1

    # 计算在每个轴上的投影
    proj_axis1 = np.dot(centered_points, axis1_unit)
    proj_axis2 = np.dot(centered_points, axis2_ortho_unit)

    # 找到投影的最小值和最大值
    min_a1, max_a1 = np.min(proj_axis1), np.max(proj_axis1)
    min_a2, max_a2 = np.min(proj_axis2), np.max(proj_axis2)

    # 计算OBB的四个顶点（在正交坐标系中）
    vertices_local = np.array(
        [[min_a1, min_a2], [max_a1, min_a2], [max_a1, max_a2], [min_a1, max_a2]]
    )

    # 将顶点转换回原始坐标系
    vertices_world = []
    for vertex in vertices_local:
        # 从局部坐标转换回世界坐标
        world_vertex = p1 + vertex[0] * axis1_unit + vertex[1] * axis2_ortho_unit
        vertices_world.append([float(world_vertex[0]), float(world_vertex[1])])

    # 计算旋转角度（轴1与x轴的夹角）
    angle = math.atan2(axis1_unit[1], axis1_unit[0])

    return vertices_world, angle


def create_labelme_obb_json(original_data, obb_shapes):
    """用 obb_shapes 中的 (label, points, original_shape) 替换 original_data['shapes']，生成新 Labelme dict。"""
    # 深拷贝原始数据
    new_data = original_data.copy()

    # 清除原始的形状，用OBB形状替换
    new_data["shapes"] = []

    for label, obb_points, original_shape in obb_shapes:
        # 创建OBB多边形
        obb_shape = {
            "label": label,
            "points": obb_points,
            "group_id": original_shape.get("group_id"),
            "description": original_shape.get("description", ""),
            "shape_type": "polygon",
            "flags": original_shape.get("flags", {}),
            "mask": original_shape.get("mask"),
        }

        new_data["shapes"].append(obb_shape)

    return new_data


def process_single_json(json_path, output_dir, image_size=None, mode="obb"):
    """处理单个 Labelme JSON：按 mode 转为 YOLO TXT，OBB 时同时写 labelme_labels 子目录。

    Args:
        json_path: Labelme .json 路径。
        output_dir: 输出目录；OBB 模式会在其下建 labelme_labels。
        image_size: (width, height)，None 时从 JSON 的 imageWidth/imageHeight 读取。
        mode: 'obb' | 'segment' | 'detect'。

    Returns:
        tuple: (结果列表, shapes 列表)，无有效目标时为 ([], [])。
    """
    data = read_labelme_json(json_path)

    image_path = data.get("imagePath", "")

    if image_size:
        img_width, img_height = image_size
    elif "imageWidth" in data and "imageHeight" in data:
        img_width = data["imageWidth"]
        img_height = data["imageHeight"]
        image_size = (img_width, img_height)
    else:
        print("警告: 未提供图像尺寸，坐标将不会归一化")
        image_size = None

    shapes = data.get("shapes", [])
    if not shapes:
        print(f"警告: {json_path} 中没有找到标注形状")
        return [], []

    all_obb_results = []
    all_obb_shapes = []
    all_segment_results = []
    all_detect_results = []

    print(f"共找到 {len(shapes)} 个标注形状：")

    for i, shape in enumerate(shapes):
        label = shape.get("label", "unknown")
        points = shape.get("points", [])
        shape_type = shape.get("shape_type", "polygon")

        if label not in CLASS_MAPPING:
            print(f"  形状{i+1}: 跳过未知类别 '{label}'")
            continue

        class_index = CLASS_MAPPING[label]

        print(
            f"  形状{i+1}: 类别={label}({class_index}), 点数={len(points)}, 类型={shape_type}"
        )

        if len(points) < 1:
            print(f"  形状{i+1}: 点数不足，无法处理")
            continue

        # 如果是rectangle类型，转换为polygon
        if shape_type == "rectangle":
            try:
                points = rectangle_to_polygon(points)
                print(f"  形状{i+1}: 将rectangle转换为polygon (4个点)")
            except Exception as e:
                print(f"  形状{i+1}: 转换rectangle时出错 - {e}")
                continue
        elif shape_type != "polygon":
            print(f"  形状{i+1}: 跳过不支持的形状类型 (类型: {shape_type})")
            continue

        if mode == "obb":
            if len(points) < 4:
                print(f"  形状{i+1}: 点数不足4个，无法计算OBB")
                continue
            try:
                obb_points, angle = calculate_obb_from_points(points)
                print(f"    成功计算OBB，旋转角度: {math.degrees(angle):.2f}度")
                all_obb_results.append((class_index, obb_points))
                all_obb_shapes.append((label, obb_points, shape))
            except Exception as e:
                print(f"  形状{i+1}: 计算OBB时出错 - {e}")
                continue

        elif mode == "segment":
            all_segment_results.append((class_index, points))
            all_obb_shapes.append((label, points, shape))
            print(f"    成功处理segment，点数: {len(points)}")

        elif mode == "detect":
            try:
                bbox = calculate_bbox_from_points(points)
                all_detect_results.append((class_index, bbox))
                all_obb_shapes.append((label, points, shape))
                print(f"    成功计算detect，边界框: {bbox}")
            except Exception as e:
                print(f"  形状{i+1}: 计算边界框时出错 - {e}")
                continue

    base_name = os.path.splitext(os.path.basename(json_path))[0]

    # 获取图像文件名（用于labelme JSON文件名）
    if image_path:
        image_base_name = os.path.splitext(os.path.basename(image_path))[0]
    else:
        # 如果没有imagePath，使用JSON文件名
        image_base_name = base_name

    if mode == "obb":
        if not all_obb_results:
            print("没有生成任何OBB结果")
            return [], []

        output_txt_path = os.path.join(output_dir, f"{base_name}.txt")
        save_yolo_obb_txt(all_obb_results, output_txt_path, image_size)
        print(f"\n已保存yolo_obb TXT: {output_txt_path}")
        print(f"共保存了 {len(all_obb_results)} 个目标的OBB标注")

        # 保存labelme格式的JSON文件到labelme_labels子文件夹，使用原图文件名
        labelme_output_dir = os.path.join(output_dir, "labelme_labels")
        os.makedirs(labelme_output_dir, exist_ok=True)
        labelme_json_data = create_labelme_obb_json(data, all_obb_shapes)
        labelme_json_path = os.path.join(labelme_output_dir, f"{image_base_name}.json")
        save_labelme_json(labelme_json_data, labelme_json_path)
        print(f"已保存labelme JSON: {labelme_json_path}")

        return all_obb_results, all_obb_shapes

    elif mode == "segment":
        if not all_segment_results:
            print("没有生成任何segment结果")
            return [], []

        output_txt_path = os.path.join(output_dir, f"{base_name}.txt")
        save_yolo_segment_txt(all_segment_results, output_txt_path, image_size)
        print(f"\n已保存yolo_segment TXT: {output_txt_path}")
        print(f"共保存了 {len(all_segment_results)} 个目标的segment标注")

        return all_segment_results, all_obb_shapes

    elif mode == "detect":
        if not all_detect_results:
            print("没有生成任何detect结果")
            return [], []

        output_txt_path = os.path.join(output_dir, f"{base_name}.txt")
        save_yolo_detect_txt(all_detect_results, output_txt_path, image_size)
        print(f"\n已保存yolo_detect TXT: {output_txt_path}")
        print(f"共保存了 {len(all_detect_results)} 个目标的detect标注")

        return all_detect_results, all_obb_shapes


def main():
    """交互式入口：扫描类别、确认映射，选择 OBB/Segment/Detect 后对单文件执行转换并打印统计。"""
    input_json_path = "0002.json"
    output_dir = "output"

    # 扫描类别并生成映射
    # 如果输入是单个文件，直接扫描该文件；如果是目录，扫描目录
    if os.path.isfile(input_json_path):
        # 扫描包含该文件的目录，以便获取所有相关类别
        input_dir = (
            os.path.dirname(input_json_path)
            if os.path.dirname(input_json_path)
            else "."
        )
        if not input_dir or input_dir == ".":
            # 如果文件在当前目录，扫描当前目录
            classes, class_stats = scan_classes_from_directory(".")
        else:
            classes, class_stats = scan_classes_from_directory(input_dir)
    else:
        # 如果输入是目录
        input_dir = input_json_path if os.path.isdir(input_json_path) else "."
        classes, class_stats = scan_classes_from_directory(input_dir)

    if not classes:
        print("错误: 未找到任何类别，请检查输入目录")
        return

    # 生成类别映射
    global CLASS_MAPPING, REVERSE_CLASS_MAPPING
    CLASS_MAPPING = generate_class_mapping(classes)
    REVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}

    # 显示映射并等待用户确认
    confirm_result = display_class_mapping_and_confirm(CLASS_MAPPING, class_stats)

    if confirm_result is None:
        print("\n程序已中断")
        return
    elif confirm_result is False:
        print("\n用户选择停止运行")
        return

    print("\n继续执行转换...")
    os.makedirs(output_dir, exist_ok=True)

    print("请选择转换模式：")
    print("1. OBB (有向包围盒) - 旋转目标检测")
    print("2. Segment (实例分割) - 多边形标注")
    print("3. Detect (目标检测) - 边界框标注")

    choice = input("请输入选项（1/2/3）：").strip()

    mode_map = {"1": "obb", "2": "segment", "3": "detect"}

    if choice not in mode_map:
        print("输入无效，请重新运行并选择 1、2 或 3。")
        return

    mode = mode_map[choice]
    print(f"\n已选择模式：{mode}")
    print(f"正在处理文件：{input_json_path}")

    results, shapes = process_single_json(input_json_path, output_dir, mode=mode)

    if results:
        print("\n处理完成。")
        print(f"共处理 {len(results)} 个目标：")

        class_counts = {}
        for class_idx, _ in results:
            class_name = REVERSE_CLASS_MAPPING[class_idx]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count}个")

        if mode == "obb" and results:
            print("\n第一个OBB的详细信息:")
            class_idx, obb_points = results[0]
            class_name = REVERSE_CLASS_MAPPING[class_idx]
            print(f"  类别: {class_name}({class_idx})")
            print(f"  顶点坐标:")
            for i, point in enumerate(obb_points):
                print(f"    点{i+1}: ({point[0]:.2f}, {point[1]:.2f})")

            if len(obb_points) >= 4:
                p1, p2, p3, p4 = obb_points
                side1 = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
                side2 = math.sqrt((p3[0] - p2[0]) ** 2 + (p3[1] - p2[1]) ** 2)
                print(f"  边长: {side1:.2f} x {side2:.2f}")

        elif mode == "detect" and results:
            print("\n第一个detect的详细信息:")
            class_idx, bbox = results[0]
            class_name = REVERSE_CLASS_MAPPING[class_idx]
            print(f"  类别: {class_name}({class_idx})")
            print(f"  边界框: {bbox}")
            x_min, y_min, x_max, y_max = bbox
            width = x_max - x_min
            height = y_max - y_min
            print(f"  尺寸: {width:.2f} x {height:.2f}")


if __name__ == "__main__":
    main()
