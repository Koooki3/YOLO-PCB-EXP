#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量将 Labelme 标注转换为 YOLO 标注。

功能：
- 扫描目录中的 Labelme `*.json` 标注文件。
- 按 `obb`、`segment` 或 `detect` 模式输出 YOLO `*.txt` 标签。
- 支持递归处理、类别扫描和类别映射确认。

使用：
- 交互式运行：`python batch_trans.py`
- 依赖同目录下的 `trans.py` 提供核心转换逻辑。
"""

import os

from trans import (
    CLASS_MAPPING,
    REVERSE_CLASS_MAPPING,
    calculate_bbox_from_points,
    calculate_obb_from_points,
    create_labelme_obb_json,
    display_class_mapping_and_confirm,
    generate_class_mapping,
    read_labelme_json,
    rectangle_to_polygon,
    save_labelme_json,
    save_yolo_detect_txt,
    save_yolo_obb_txt,
    save_yolo_segment_txt,
    scan_classes_from_directory,
)


def batch_process_jsons(
    input_dir, output_dir, image_size=None, mode="obb", class_mapping=None
):
    """批量处理目录中所有 Labelme JSON，转为 YOLO TXT（及可选 labelme_labels）。

    Args:
        input_dir: 输入目录（存放 .json）。
        output_dir: 输出目录（存放 .txt，mode=obb 时含 labelme_labels 子目录）。
        image_size: 图像尺寸 (width, height)，为 None 时从 JSON 的 imageWidth/imageHeight 读取。
        mode: 转换模式，'obb' | 'segment' | 'detect'。
        class_mapping: 类别名到索引的映射；为 None 时使用 trans 模块的 CLASS_MAPPING.

    Returns:
        None. 处理结果打印到 stdout，文件写入 output_dir。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 使用传入的映射或全局映射
    mapping = class_mapping if class_mapping is not None else CLASS_MAPPING

    json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

    if not json_files:
        print(f"在目录 {input_dir} 中没有找到 JSON 文件。")
        return

    print(f"共找到 {len(json_files)} 个 JSON 文件。")
    print("类别映射关系：")
    for label, idx in sorted(mapping.items(), key=lambda x: x[1]):
        print(f"  {label} -> {idx}")
    print()
    print(f"转换模式：{mode}")
    print()

    total_files = 0
    total_targets = 0
    class_stats = {}

    for json_file in json_files:
        json_path = os.path.join(input_dir, json_file)
        print(f"\n正在处理文件：{json_file}")

        try:
            data = read_labelme_json(json_path)

            image_path = data.get("imagePath", "")

            if image_size:
                img_width, img_height = image_size
            elif "imageWidth" in data and "imageHeight" in data:
                img_width = data["imageWidth"]
                img_height = data["imageHeight"]
                current_image_size = (img_width, img_height)
            else:
                print("  警告: 未提供图像尺寸，坐标将不会归一化")
                current_image_size = None

            shapes = data.get("shapes", [])
            if not shapes:
                print(f"  警告: {json_file} 中没有找到标注形状")
                continue

            all_obb_results = []
            all_segment_results = []
            all_detect_results = []
            all_obb_shapes = []

            for i, shape in enumerate(shapes):
                label = shape.get("label", "unknown")
                points = shape.get("points", [])
                shape_type = shape.get("shape_type", "polygon")

                if label not in mapping:
                    print(f"  形状{i+1}: 跳过未知类别 '{label}'")
                    continue

                class_index = mapping[label]

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
                        all_obb_results.append((class_index, obb_points))
                        all_obb_shapes.append((label, obb_points, shape))
                        class_stats[label] = class_stats.get(label, 0) + 1
                    except Exception as e:
                        print(f"  形状{i+1}: 计算OBB时出错 - {e}")
                        continue

                elif mode == "segment":
                    all_segment_results.append((class_index, points))
                    all_obb_shapes.append((label, points, shape))
                    class_stats[label] = class_stats.get(label, 0) + 1

                elif mode == "detect":
                    try:
                        bbox = calculate_bbox_from_points(points)
                        all_detect_results.append((class_index, bbox))
                        all_obb_shapes.append((label, points, shape))
                        class_stats[label] = class_stats.get(label, 0) + 1
                    except Exception as e:
                        print(f"  形状{i+1}: 计算边界框时出错 - {e}")
                        continue

            if mode == "obb" and not all_obb_results:
                print(f"  {json_file}: 没有生成任何OBB结果")
                continue
            elif mode == "segment" and not all_segment_results:
                print(f"  {json_file}: 没有生成任何segment结果")
                continue
            elif mode == "detect" and not all_detect_results:
                print(f"  {json_file}: 没有生成任何detect结果")
                continue

            total_files += 1

            base_name = os.path.splitext(json_file)[0]

            # 获取图像文件名（用于labelme JSON文件名）
            if image_path:
                image_base_name = os.path.splitext(os.path.basename(image_path))[0]
            else:
                # 如果没有imagePath，使用JSON文件名
                image_base_name = base_name

            if mode == "obb":
                total_targets += len(all_obb_results)
                output_txt_path = os.path.join(output_dir, f"{base_name}.txt")
                save_yolo_obb_txt(all_obb_results, output_txt_path, current_image_size)
                print(f"  已保存yolo_obb TXT: {os.path.basename(output_txt_path)}")
                print(f"  共处理了 {len(all_obb_results)} 个目标")

                # 保存labelme格式的JSON文件到labelme_labels子文件夹，使用原图文件名
                labelme_output_dir = os.path.join(output_dir, "labelme_labels")
                os.makedirs(labelme_output_dir, exist_ok=True)
                labelme_json_data = create_labelme_obb_json(data, all_obb_shapes)
                labelme_json_path = os.path.join(
                    labelme_output_dir, f"{image_base_name}.json"
                )
                save_labelme_json(labelme_json_data, labelme_json_path)
                print(f"  已保存labelme JSON: {os.path.basename(labelme_json_path)}")

            elif mode == "segment":
                total_targets += len(all_segment_results)
                output_txt_path = os.path.join(output_dir, f"{base_name}.txt")
                save_yolo_segment_txt(
                    all_segment_results, output_txt_path, current_image_size
                )
                print(f"  已保存yolo_segment TXT: {os.path.basename(output_txt_path)}")
                print(f"  共处理了 {len(all_segment_results)} 个目标")

            elif mode == "detect":
                total_targets += len(all_detect_results)
                output_txt_path = os.path.join(output_dir, f"{base_name}.txt")
                save_yolo_detect_txt(
                    all_detect_results, output_txt_path, current_image_size
                )
                print(f"  已保存yolo_detect TXT: {os.path.basename(output_txt_path)}")
                print(f"  共处理了 {len(all_detect_results)} 个目标")

        except Exception as e:
            print(f"  处理文件时出错: {e}")
            continue

    print("\n" + "=" * 50)
    print("处理完成。")
    print(f"成功处理文件数：{total_files}/{len(json_files)}")
    print(f"共生成 {total_targets} 个目标的 {mode} 标注。")
    print("\n类别统计：")
    for label, count in class_stats.items():
        print(f"  {label}: {count}个")

    print(f"\n结果保存在：{output_dir}")


def process_directory_recursive(
    input_dir, output_base_dir, image_size=None, mode="obb", class_mapping=None
):
    """递归处理目录及子目录中所有 Labelme JSON，保持相对目录结构。

    Args:
        input_dir: 输入根目录。
        output_base_dir: 输出根目录，子目录结构与 input_dir 对应。
        image_size: 图像尺寸 (width, height)，None 时从各 JSON 读取。
        mode: 转换模式，'obb' | 'segment' | 'detect'。
        class_mapping: 类别映射；None 时使用 trans.CLASS_MAPPING。

    Returns:
        None. 结果写入 output_base_dir，统计信息打印到 stdout。
    """
    os.makedirs(output_base_dir, exist_ok=True)

    # 使用传入的映射或全局映射
    mapping = class_mapping if class_mapping is not None else CLASS_MAPPING

    total_processed = 0
    for root, dirs, files in os.walk(input_dir):
        if output_base_dir in root:
            continue

        rel_path = os.path.relpath(root, input_dir)
        if rel_path == ".":
            output_dir = output_base_dir
        else:
            output_dir = os.path.join(output_base_dir, rel_path)

        os.makedirs(output_dir, exist_ok=True)

        json_files = [f for f in files if f.endswith(".json")]

        if json_files:
            print(f"\n正在处理目录：{root}")
            print(f"输出目录：{output_dir}")

            for json_file in json_files:
                json_path = os.path.join(root, json_file)

                try:
                    data = read_labelme_json(json_path)

                    image_path = data.get("imagePath", "")

                    if image_size:
                        current_image_size = image_size
                    elif "imageWidth" in data and "imageHeight" in data:
                        img_width = data["imageWidth"]
                        img_height = data["imageHeight"]
                        current_image_size = (img_width, img_height)
                    else:
                        current_image_size = None

                    shapes = data.get("shapes", [])
                    if not shapes:
                        continue

                    all_obb_results = []
                    all_segment_results = []
                    all_detect_results = []
                    all_obb_shapes = []

                    for shape in shapes:
                        label = shape.get("label", "unknown")
                        points = shape.get("points", [])
                        shape_type = shape.get("shape_type", "polygon")

                        if label not in mapping or len(points) < 1:
                            continue

                        # 如果是rectangle类型，转换为polygon
                        if shape_type == "rectangle":
                            try:
                                points = rectangle_to_polygon(points)
                            except Exception:
                                continue
                        elif shape_type != "polygon":
                            continue

                        class_index = mapping[label]

                        if mode == "obb":
                            if len(points) < 4:
                                continue
                            try:
                                obb_points, _ = calculate_obb_from_points(points)
                                all_obb_results.append((class_index, obb_points))
                                all_obb_shapes.append((label, obb_points, shape))
                            except Exception:
                                continue
                        elif mode == "segment":
                            all_segment_results.append((class_index, points))
                            all_obb_shapes.append((label, points, shape))
                        elif mode == "detect":
                            try:
                                bbox = calculate_bbox_from_points(points)
                                all_detect_results.append((class_index, bbox))
                                all_obb_shapes.append((label, points, shape))
                            except Exception:
                                continue

                    if mode == "obb" and not all_obb_results:
                        continue
                    elif mode == "segment" and not all_segment_results:
                        continue
                    elif mode == "detect" and not all_detect_results:
                        continue

                    total_processed += 1

                    base_name = os.path.splitext(json_file)[0]

                    # 获取图像文件名（用于labelme JSON文件名）
                    if image_path:
                        image_base_name = os.path.splitext(
                            os.path.basename(image_path)
                        )[0]
                    else:
                        # 如果没有imagePath，使用JSON文件名
                        image_base_name = base_name

                    if mode == "obb":
                        output_txt_path = os.path.join(output_dir, f"{base_name}.txt")
                        save_yolo_obb_txt(
                            all_obb_results, output_txt_path, current_image_size
                        )

                        # 保存labelme格式的JSON文件到labelme_labels子文件夹，使用原图文件名
                        labelme_output_dir = os.path.join(output_dir, "labelme_labels")
                        os.makedirs(labelme_output_dir, exist_ok=True)
                        labelme_json_data = create_labelme_obb_json(
                            data, all_obb_shapes
                        )
                        labelme_json_path = os.path.join(
                            labelme_output_dir, f"{image_base_name}.json"
                        )
                        save_labelme_json(labelme_json_data, labelme_json_path)

                    elif mode == "segment":
                        output_txt_path = os.path.join(output_dir, f"{base_name}.txt")
                        save_yolo_segment_txt(
                            all_segment_results, output_txt_path, current_image_size
                        )

                    elif mode == "detect":
                        output_txt_path = os.path.join(output_dir, f"{base_name}.txt")
                        save_yolo_detect_txt(
                            all_detect_results, output_txt_path, current_image_size
                        )

                except Exception as e:
                    print(f"  处理文件 {json_file} 时出错: {e}")
                    continue

    print(f"\n递归处理完成，共处理 {total_processed} 个文件。")
    print(f"输出目录：{output_base_dir}")


def main():
    """交互式入口：扫描类别、确认映射，选择 OBB/Segment/Detect 或递归模式并执行转换。"""
    input_dir = r"D:\YOLO_PCB\SegmentData\images"
    output_dir = r"D:\YOLO_PCB\SegmentData\labels"

    image_size = None

    # 扫描类别并生成映射
    print("正在扫描输入目录中的类别...")
    classes, class_stats = scan_classes_from_directory(input_dir)

    if not classes:
        print("错误: 未找到任何类别，请检查输入目录")
        return

    # 生成类别映射
    class_mapping = generate_class_mapping(classes)

    # 显示映射并等待用户确认
    confirm_result = display_class_mapping_and_confirm(class_mapping, class_stats)

    if confirm_result is None:
        print("\n程序已中断")
        return
    elif confirm_result is False:
        print("\n用户选择停止运行")
        return

    print("\n继续执行转换。")

    # 更新全局映射
    global CLASS_MAPPING, REVERSE_CLASS_MAPPING
    CLASS_MAPPING = class_mapping
    REVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}

    print("请选择处理模式：")
    print("1. OBB (有向包围盒) - 旋转目标检测")
    print("2. Segment (实例分割) - 多边形标注")
    print("3. Detect (目标检测) - 边界框标注")
    print("4. 递归处理目录及其子目录")

    choice = input("请输入选项（1/2/3/4）：").strip()

    mode_map = {"1": "obb", "2": "segment", "3": "detect"}

    if choice == "4":
        print("\n请选择递归处理模式：")
        print("1. OBB (有向包围盒) - 旋转目标检测")
        print("2. Segment (实例分割) - 多边形标注")
        print("3. Detect (目标检测) - 边界框标注")

        mode_choice = input("请输入模式选项（1/2/3）：").strip()

        if mode_choice not in mode_map:
            print("输入无效，请重新运行并选择 1、2 或 3。")
            return

        mode = mode_map[mode_choice]
        recursive_output_dir = "output_recursive"
        print(f"\n已选择模式：{mode}")
        print("将递归处理当前目录及其所有子目录。")
        process_directory_recursive(
            input_dir, recursive_output_dir, image_size, mode, class_mapping
        )

    elif choice in mode_map:
        mode = mode_map[choice]
        print(f"\n已选择模式：{mode}")
        print("将批量处理当前目录中的 JSON 文件。")
        batch_process_jsons(input_dir, output_dir, image_size, mode, class_mapping)

    else:
        print("输入无效，请重新运行并选择 1、2、3 或 4。")


if __name__ == "__main__":
    main()
