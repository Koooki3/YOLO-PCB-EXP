#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练结果分析脚本。

功能：
- 汇总 `train_origin/` 与 `train_ex/` 下多个训练实验的结果文件。
- 生成主指标对比图、Loss 对比图、最优值图表和文字分析报告。
- 可选补充推理评估结果，并统一输出到带时间戳的分析目录。

使用：
- 交互式运行：`python analyze_training_results.py`
- 查看帮助：`python analyze_training_results.py --help`
"""

import argparse
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import warnings
from datetime import datetime
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from chart_style import apply_unified_mpl_style, build_series_palette, get_visual_theme

warnings.filterwarnings("ignore")

# 支持的实验组目录
SUPPORTED_SOURCE_DIRS = ("train_origin", "train_ex", "train_segment")

# 分析结果统一根目录
RESULTS_ANALYSE_ROOT = "results_analyse"

THEME = apply_unified_mpl_style()


def get_timestamp_folder():
    """生成时间戳文件夹名，格式为 val_MMDDHHmm（无年份，24小时制）。"""
    return "val_" + datetime.now().strftime("%m%d%H%M")


def resolve_output_dir(workspace_root, source_dir_name, output_tag=None):
    """
    解析输出目录：results_analyse/{source_dir_name}/val_MMDDHHmm/

    Args:
        workspace_root: 工作区根目录 Path。
        source_dir_name: 实验组名称，如 train_origin 或 train_ex。

    Returns:
        Path: 带时间戳的完整输出目录。
    """
    root = Path(workspace_root)
    base = root / RESULTS_ANALYSE_ROOT / source_dir_name
    stamp = get_timestamp_folder()
    if output_tag:
        stamp = f"{stamp}_{output_tag}"
    out_dir = base / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def apply_mplstyle_if_requested(style_path):
    """按需加载基础 mplstyle，再叠加仓库统一风格。"""
    if not style_path:
        return None
    style_file = Path(style_path)
    if not style_file.is_absolute():
        style_file = Path.cwd() / style_file
    if not style_file.exists():
        raise FileNotFoundError(f"未找到 mplstyle 文件: {style_file}")
    apply_unified_mpl_style(str(style_file))
    return style_file


class TrainingResultAnalyzer:
    """YOLO PCB 训练结果分析器。

    加载 results.csv、args.yaml 等，支持对比折线图、最佳值柱状图、
    单指标训练曲线、详细文本报告及可选推理报告。
    支持 train_origin 与 train_ex 两组实验，并可将分析结果保存到统一目录。
    """

    def __init__(
        self,
        source_dir,
        output_dir=None,
        workspace_root=None,
        selected_experiments=None,
        output_tag=None,
    ):
        """初始化分析器。

        Args:
            source_dir: 实验根目录名或路径，如 "train_origin" 或 "train_ex"。
            output_dir: 输出目录。若为 None，则根据 workspace_root 与 source_dir 自动生成带时间戳的目录。
            workspace_root: 工作区根目录，用于自动生成 output_dir。默认为当前工作目录。
            selected_experiments: 要参与分析的实验名称列表；None 表示分析全部。
        """
        self.workspace_root = Path(workspace_root or os.getcwd())
        self.source_dir_name = Path(source_dir).name
        self.train_ex_dir = self.workspace_root / self.source_dir_name
        self.selected_experiments = selected_experiments  # None = 全部
        self.output_tag = output_tag
        self.theme = get_visual_theme()

        if output_dir is not None:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = resolve_output_dir(
                self.workspace_root, self.source_dir_name, output_tag=self.output_tag
            )

        self.experiments = {}
        self.colors = {}
        self.display_labels = {}
        self.compact_labels = {}
        self.line_styles = {}
        self.markers = {}

        # 创建输出目录（如果不存在）
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 定义颜色与线型方案，优先保证论文图中的统一观感和区分度
        self.color_palette = build_series_palette()
        self.line_style_cycle = ["-", "--", "-.", ":"]
        self.marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]

        self.metrics = {}
        self.task_type = None
        self.metric_aliases = {
            "metrics/mAP50(B)": "mAP50(Box)",
            "metrics/mAP50-95(B)": "mAP50-95(Box)",
            "metrics/precision(B)": "Precision(Box)",
            "metrics/recall(B)": "Recall(Box)",
            "metrics/mAP50(M)": "mAP50(Mask)",
            "metrics/mAP50-95(M)": "mAP50-95(Mask)",
            "metrics/precision(M)": "Precision(Mask)",
            "metrics/recall(M)": "Recall(Mask)",
            "train/box_loss": "Train Box Loss",
            "train/seg_loss": "Train Seg Loss",
            "train/cls_loss": "Train Class Loss",
            "train/dfl_loss": "Train DFL Loss",
            "train/obj_loss": "Train Object Loss",
            "train/loss": "Train Total Loss",
            "val/box_loss": "Val Box Loss",
            "val/seg_loss": "Val Seg Loss",
            "val/cls_loss": "Val Class Loss",
            "val/dfl_loss": "Val DFL Loss",
            "val/obj_loss": "Val Object Loss",
            "val/loss": "Val Total Loss",
        }

    def load_config(self, exp_folder):
        """加载实验配置文件（args.yaml / hparams.yaml / config.yaml）。

        Args:
            exp_folder: 实验目录 Path。

        Returns:
            dict | None: 解析后的配置字典，未找到或解析失败时为 None。
        """
        config_files = [
            exp_folder / "args.yaml",
            exp_folder / "hparams.yaml",
            exp_folder / "config.yaml",
        ]

        for config_file in config_files:
            if config_file.exists():
                try:
                    import yaml

                    with open(config_file, "r") as f:
                        return yaml.safe_load(f)
                except Exception as e:
                    print(f"  [警告] {exp_folder.name}: 读取配置文件失败 - {e}")
                    return None

        return None

    def load_all_results(self):
        """加载实验的 results.csv 及配置，填充 experiments、colors。

        若初始化时指定了 selected_experiments，则仅加载这些实验；否则加载全部。

        Returns:
            bool: 至少加载成功一个实验时返回 True，否则 False。
        """
        if not self.train_ex_dir.exists():
            print(f"错误: 目录 {self.train_ex_dir} 不存在")
            return False

        # 获取所有实验文件夹
        all_folders = [f for f in self.train_ex_dir.iterdir() if f.is_dir()]
        if not all_folders:
            print("错误: 未找到任何实验结果文件夹")
            return False

        # 按 selected_experiments 过滤
        if self.selected_experiments is not None:
            name_set = set(self.selected_experiments)
            experiment_folders = [f for f in all_folders if f.name in name_set]
            missing = name_set - {f.name for f in experiment_folders}
            if missing:
                print(
                    f"警告: 以下实验未找到或缺少 results.csv，已跳过: {sorted(missing)}"
                )
        else:
            experiment_folders = all_folders

        if not experiment_folders:
            print("错误: 没有可加载的实验")
            return False

        print(
            f"找到 {len(experiment_folders)} 个实验结果 (来自 {self.source_dir_name}):"
        )

        for exp_folder in sorted(experiment_folders):
            results_file = exp_folder / "results.csv"
            if results_file.exists():
                try:
                    df = pd.read_csv(results_file)
                    exp_name = exp_folder.name

                    # 提取关键信息
                    self.experiments[exp_name] = {
                        "data": df,
                        "folder": exp_folder,
                        "epochs": len(df),
                        "final_metrics": self._get_final_metrics(df),
                        "config": self.load_config(exp_folder),
                    }

                    print(f"  [成功] {exp_name}: {len(df)} epochs")

                except Exception as e:
                    print(f"  [失败] {exp_folder.name}: 读取失败 - {e}")
            else:
                print(f"  [失败] {exp_folder.name}: 未找到 results.csv")

        # 为每个实验分配稳定样式
        for i, exp_name in enumerate(sorted(self.experiments.keys())):
            self.colors[exp_name] = self.color_palette[i % len(self.color_palette)]
            self.line_styles[exp_name] = self.line_style_cycle[
                (i // len(self.color_palette)) % len(self.line_style_cycle)
            ]
            self.markers[exp_name] = self.marker_cycle[i % len(self.marker_cycle)]

        self._build_display_labels()
        self._finalize_metric_configuration()

        return len(self.experiments) > 0

    def _build_display_labels(self):
        """为实验生成稳定、紧凑且带语义的显示标签。"""
        names = sorted(self.experiments.keys())
        self.display_labels = {}
        self.compact_labels = {}
        used_labels = set()
        for i, name in enumerate(names, 1):
            alias = f"E{i:02d}"
            compact = self._make_compact_experiment_label(name)
            label = f"{alias} {compact}".strip()
            if label in used_labels:
                label = f"{label} #{i}"
            used_labels.add(label)
            self.display_labels[name] = alias
            self.compact_labels[name] = label

    def _make_compact_experiment_label(self, name):
        """将长实验目录名压缩为适合图例展示的短标签。"""
        raw = name
        raw = raw.replace("Ex_", "")
        raw = raw.replace("SegmentData_", "")
        raw = raw.replace("_seg", "")

        model_match = re.search(r"(11s|12s|v8s|rtdetr[_-]l|RTDETR[_-]L)", raw, re.I)
        model = model_match.group(1).replace("_", "-").upper() if model_match else ""
        model = model.replace("RTDETR-L", "RTDETR-L")
        if model in {"11S", "12S", "V8S"}:
            model = model.lower()

        parts = []
        if model:
            parts.append(model)

        if "standardLoss" in raw:
            parts.append("STD")

        loss_match = re.search(r"WassersteinLoss_?([0-9.]+)", raw, re.I)
        if loss_match:
            parts.append(f"NWD{loss_match.group(1)}")

        if "DINOP2" in raw.upper():
            parts.append("+DINO")

        c_match = re.search(r"_C([0-9.]+)", raw, re.I)
        if c_match and "seg" in name.lower():
            parts.append(f"C{c_match.group(1)}")

        if not parts:
            tokens = [t for t in re.split(r"[_-]+", raw) if t]
            parts = tokens[:3]

        return " ".join(parts)

    def _alias_text_lines(self):
        """返回图外展示的实验别名映射。"""
        names = sorted(self.experiments.keys())
        return [self.compact_labels[name] for name in names]

    def _add_alias_panel(self, fig):
        """在图右侧添加统一的实验说明区。"""
        alias_ax = fig.add_axes([0.83, 0.14, 0.15, 0.72])
        alias_ax.axis("off")
        alias_ax.text(
            0.0,
            1.0,
            "Experiments\n" + "\n".join(self._alias_text_lines()),
            ha="left",
            va="top",
            fontsize=9,
            family="monospace",
            linespacing=1.35,
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": self.theme["annotation_bg"],
                "edgecolor": self.theme["annotation_edge"],
                "alpha": 0.96,
            },
            color=self.theme["text"],
        )

    def _apply_shared_figure_style(self, fig, title):
        """统一图表风格，并在图外展示别名映射。"""
        fig.patch.set_facecolor(self.theme["figure_bg"])
        fig.suptitle(
            title,
            fontsize=16,
            fontweight="bold",
            y=0.985,
            color=self.theme["text"],
        )
        self._add_alias_panel(fig)
        fig.subplots_adjust(
            left=0.08,
            right=0.81,
            top=0.90,
            bottom=0.12,
            wspace=0.28,
            hspace=0.36,
        )

    def _apply_single_axis_style(self, fig, ax, title):
        """统一单图样式，并在图外展示别名映射。"""
        fig.patch.set_facecolor(self.theme["figure_bg"])
        ax.set_title(title, fontsize=16, fontweight="bold", pad=16, color=self.theme["text"])
        self._add_alias_panel(fig)
        fig.subplots_adjust(left=0.09, right=0.81, top=0.90, bottom=0.24)

    def _annotate_selected_bars(self, ax, bars, values, epochs, is_loss):
        """仅标注关键柱，避免柱状图文本拥挤。"""
        count = len(values)
        if count <= 8:
            selected_idx = range(count)
        else:
            top_n = 4
            selected_idx = list(range(min(top_n, count))) + list(
                range(max(top_n, count - 2), count)
            )

        value_range = max(values) - min(values) if len(values) > 1 else max(values)
        offset = max(abs(value_range) * 0.04, max(values) * 0.01, 0.005)

        for i in selected_idx:
            bar = bars[i]
            value = values[i]
            epoch = epochs[i]
            height = bar.get_height()
            y = height + offset if height >= 0 else height - offset
            va = "bottom" if height >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y,
                f"{value:.3f}\n@{epoch}",
                ha="center",
                va=va,
                fontsize=8,
                fontweight="bold",
                color=self.theme["muted_text"],
            )

    def _finalize_metric_configuration(self):
        """根据已加载实验自动识别任务类型与可用指标。"""
        all_columns = set()
        task_hints = []
        for exp in self.experiments.values():
            df = exp["data"]
            all_columns.update(df.columns)
            config = exp.get("config") or {}
            task = config.get("task")
            if isinstance(task, str):
                task_hints.append(task.lower())

        if any(task == "segment" for task in task_hints) or "metrics/mAP50(M)" in all_columns:
            self.task_type = "segment"
        else:
            self.task_type = "detect"

        ordered_metrics = self._preferred_metric_order()
        self.metrics = {
            metric: self.metric_aliases.get(metric, metric)
            for metric in ordered_metrics
            if metric in all_columns
        }

    def _preferred_metric_order(self):
        if self.task_type == "segment":
            return [
                "metrics/mAP50(M)",
                "metrics/mAP50-95(M)",
                "metrics/precision(M)",
                "metrics/recall(M)",
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)",
                "metrics/precision(B)",
                "metrics/recall(B)",
                "train/box_loss",
                "train/seg_loss",
                "train/cls_loss",
                "train/dfl_loss",
                "val/box_loss",
                "val/seg_loss",
                "val/cls_loss",
                "val/dfl_loss",
                "train/loss",
                "val/loss",
            ]
        return [
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "metrics/precision(B)",
            "metrics/recall(B)",
            "train/box_loss",
            "train/obj_loss",
            "train/cls_loss",
            "train/dfl_loss",
            "val/box_loss",
            "val/obj_loss",
            "val/cls_loss",
            "val/dfl_loss",
            "train/loss",
            "val/loss",
        ]

    def get_performance_metrics(self):
        """返回当前任务优先使用的性能指标。"""
        preferred = (
            ["metrics/mAP50(M)", "metrics/mAP50-95(M)", "metrics/precision(M)", "metrics/recall(M)"]
            if self.task_type == "segment"
            else ["metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)"]
        )
        return [m for m in preferred if m in self.metrics]

    def get_loss_metrics(self):
        """返回当前任务优先使用的损失指标。"""
        preferred = (
            ["train/seg_loss", "val/seg_loss", "train/box_loss", "val/box_loss", "train/cls_loss", "val/cls_loss"]
            if self.task_type == "segment"
            else ["train/loss", "val/loss", "train/box_loss", "val/box_loss", "train/cls_loss", "val/cls_loss"]
        )
        return [m for m in preferred if m in self.metrics]

    def get_primary_metric(self):
        """返回训练趋势分析中的主指标。"""
        candidates = (
            ["metrics/mAP50(M)", "metrics/mAP50(B)"]
            if self.task_type == "segment"
            else ["metrics/mAP50(B)"]
        )
        for metric in candidates:
            if metric in self.metrics:
                return metric
        return None

    def _get_final_metrics(self, df):
        """获取最后一个 epoch 的指标（仅 self.metrics 中的列）。

        Args:
            df: 实验的 results.csv DataFrame。

        Returns:
            dict: 列名 -> 最后一行的值。
        """
        final_metrics = {}
        for col in df.columns:
            if col in self.metrics:
                final_metrics[col] = df[col].iloc[-1]
        return final_metrics

    def find_best_values(self, metric_name):
        """计算各实验在指定指标上的最佳值及对应 epoch。

        Args:
            metric_name: results.csv 中的列名。

        Returns:
            dict: {实验名: {'value': float, 'epoch': int}}。损失类指标取最小，其余取最大。
        """
        best_values = {}
        for exp_name, exp_data in self.experiments.items():
            df = exp_data["data"]
            if metric_name in df.columns:
                # 判断指标类型：损失指标是越小越好，其他指标是越大越好
                if any(
                    prefix in metric_name
                    for prefix in ["loss", "val/loss", "train/loss"]
                ):
                    best_idx = df[metric_name].idxmin()  # 损失值越小越好
                    best_value = df[metric_name].iloc[best_idx]
                else:
                    best_idx = df[metric_name].idxmax()  # 性能指标越大越好
                    best_value = df[metric_name].iloc[best_idx]
                best_epoch = df.index[best_idx] + 1  # epoch从1开始
                best_values[exp_name] = {"value": best_value, "epoch": best_epoch}
        return best_values

    def calculate_statistics(self, metric_name):
        """计算指定指标的统计信息"""
        values = []
        for exp_name, exp_data in self.experiments.items():
            df = exp_data["data"]
            if metric_name in df.columns:
                # 获取该指标的最佳值
                best_value = self.find_best_values(metric_name)[exp_name]["value"]
                values.append(best_value)

        if not values:
            return None

        return {
            "mean": np.mean(values),
            "median": np.median(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "count": len(values),
        }

    def get_best_model_path(self, exp_name):
        """获取指定实验的最佳模型文件路径"""
        if exp_name not in self.experiments:
            return None

        exp_folder = self.experiments[exp_name]["folder"]
        best_model_path = exp_folder / "weights" / "best.pt"

        if best_model_path.exists():
            return str(best_model_path)
        else:
            print(f"警告: 实验 {exp_name} 的最佳模型文件不存在: {best_model_path}")
            return None

    def run_inference(self, model_path, data_path, split="val"):
        """运行YOLO模型推理"""
        import subprocess
        import tempfile

        # 创建临时结果目录
        with tempfile.TemporaryDirectory() as temp_dir:
            # 构建推理命令
            command = [
                "python",
                "val.py",
                "--weights",
                model_path,
                "--data",
                data_path,
                "--split",
                split,
                "--save-txt",
                "--save-conf",
                "--name",
                f"inference_{split}",
                "--project",
                temp_dir,
            ]

            print(f"\n正在运行推理: {' '.join(command)}")

            try:
                # 运行推理命令
                result = subprocess.run(
                    command, cwd=os.getcwd(), capture_output=True, text=True, check=True
                )

                # 解析推理结果
                return self._parse_inference_result(result.stdout)

            except subprocess.CalledProcessError as e:
                print(f"推理失败: {e.stderr}")
                return None

    def _parse_inference_result(self, output):
        """解析推理结果输出"""
        # 提取关键指标
        metrics = {}

        # 示例输出格式（根据实际YOLO输出调整）
        #                   all        327        0.989        0.994        0.999        0.882
        lines = output.split("\n")
        for line in lines:
            if "all" in line and len(line.split()) >= 6:
                parts = line.strip().split()
                try:
                    metrics["precision"] = float(parts[2])
                    metrics["recall"] = float(parts[3])
                    metrics["mAP50"] = float(parts[4])
                    metrics["mAP50-95"] = float(parts[5])
                except ValueError:
                    continue

        return metrics

    def run_inference_on_all_best_models(self, data_path="data/pcb.yaml"):
        """对所有实验的最佳模型运行推理"""
        inference_results = {}

        for exp_name in sorted(self.experiments.keys()):
            best_model_path = self.get_best_model_path(exp_name)
            if best_model_path:
                print(f"\n{'='*60}")
                print(f"正在处理实验: {exp_name}")
                print(f"最佳模型路径: {best_model_path}")

                # 运行训练集推理
                train_exs = self.run_inference(
                    best_model_path, data_path, split="train"
                )

                # 运行验证集推理
                val_results = self.run_inference(
                    best_model_path, data_path, split="val"
                )

                inference_results[exp_name] = {"train": train_exs, "val": val_results}

        return inference_results

    def generate_inference_report(self, inference_results):
        """生成推理结果报告"""
        print("\n" + "=" * 70)
        print("最佳模型推理结果分析报告")
        print("=" * 70)

        if not inference_results:
            print("\n[警告] 未获取到推理结果")
            return

        # 训练集结果对比
        print("\n" + "-" * 70)
        print("训练集推理结果对比")
        print("-" * 70)

        # 表头
        print(
            "   "
            + " | ".join(
                [
                    "实验名称".ljust(25),
                    "Precision".ljust(10),
                    "Recall".ljust(10),
                    "mAP50".ljust(10),
                    "mAP50-95".ljust(10),
                ]
            )
        )
        print("   " + "-" * 25 + "-" * 12 * 4)

        for exp_name, results in sorted(inference_results.items()):
            if results["train"]:
                print(
                    "   "
                    + " | ".join(
                        [
                            exp_name.ljust(25),
                            f"{results['train']['precision']:.4f}".ljust(10),
                            f"{results['train']['recall']:.4f}".ljust(10),
                            f"{results['train']['mAP50']:.4f}".ljust(10),
                            f"{results['train']['mAP50-95']:.4f}".ljust(10),
                        ]
                    )
                )

        # 验证集结果对比
        print("\n" + "-" * 70)
        print("验证集推理结果对比")
        print("-" * 70)

        # 表头
        print(
            "   "
            + " | ".join(
                [
                    "实验名称".ljust(25),
                    "Precision".ljust(10),
                    "Recall".ljust(10),
                    "mAP50".ljust(10),
                    "mAP50-95".ljust(10),
                ]
            )
        )
        print("   " + "-" * 25 + "-" * 12 * 4)

        for exp_name, results in sorted(inference_results.items()):
            if results["val"]:
                print(
                    "   "
                    + " | ".join(
                        [
                            exp_name.ljust(25),
                            f"{results['val']['precision']:.4f}".ljust(10),
                            f"{results['val']['recall']:.4f}".ljust(10),
                            f"{results['val']['mAP50']:.4f}".ljust(10),
                            f"{results['val']['mAP50-95']:.4f}".ljust(10),
                        ]
                    )
                )

        # 最佳模型总结
        print("\n" + "-" * 70)
        print("最佳模型总结")
        print("-" * 70)

        # 找出各指标最佳模型
        metrics = ["precision", "recall", "mAP50", "mAP50-95"]
        for split in ["train", "val"]:
            print(f"\n{split}集最佳模型：")
            for metric in metrics:
                best_exp = None
                best_val = -1

                for exp_name, results in inference_results.items():
                    if results[split] and results[split][metric] > best_val:
                        best_val = results[split][metric]
                        best_exp = exp_name

                if best_exp:
                    print(f"  {metric}: {best_exp} (值: {best_val:.4f})")

        print("\n" + "=" * 70)
        print("推理结果分析报告生成完成。")
        print("=" * 70)

    def plot_comparison(
        self, metrics_to_plot=None, save_path="training_comparison.png", layout="auto"
    ):
        """生成性能对比折线图"""
        if metrics_to_plot is None:
            metrics_to_plot = self.get_performance_metrics()[:2]

        # 过滤掉不存在的指标
        valid_metrics = []
        for metric in metrics_to_plot:
            has_data = any(
                metric in exp["data"].columns for exp in self.experiments.values()
            )
            if has_data:
                valid_metrics.append(metric)
            else:
                print(f"警告: 指标 {metric} 在所有实验中都不存在，已跳过")

        if not valid_metrics:
            print("错误: 没有有效的指标可以绘制")
            return None

        n_metrics = len(valid_metrics)

        # 自动计算布局，尽量使用2行布局
        if layout == "auto":
            if n_metrics <= 2:
                rows, cols = 1, n_metrics
            elif n_metrics <= 4:
                rows, cols = 2, (n_metrics + 1) // 2
            else:
                rows, cols = 3, (n_metrics + 2) // 3
        else:
            rows, cols = layout

        # 创建子图
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5.2 * rows))
        axes = axes.flatten() if n_metrics > 1 else [axes]

        # 为每个指标生成图表
        for idx, metric in enumerate(valid_metrics):
            ax = axes[idx]
            metric_name = self.metrics.get(metric, metric)

            # 绘制每个实验的曲线
            for exp_name in sorted(self.experiments.keys()):
                df = self.experiments[exp_name]["data"]
                if metric in df.columns:
                    epochs = df.index + 1  # epoch从1开始
                    values = df[metric].values

                    ax.plot(
                        epochs,
                        values,
                        label=self.compact_labels[exp_name],
                        color=self.colors[exp_name],
                        linestyle=self.line_styles[exp_name],
                        linewidth=2.7,
                        alpha=0.96,
                        marker=self.markers[exp_name],
                        markersize=4.8,
                        markerfacecolor="white",
                        markeredgecolor=self.colors[exp_name],
                        markeredgewidth=1.1,
                        markevery=max(1, len(epochs) // 12),
                    )  # 间隔显示标记

            # 设置图表样式
            ax.set_xlabel("训练轮次 (Epochs)", fontsize=12, fontweight="bold", color=self.theme["text"])
            ax.set_ylabel(metric_name, fontsize=12, fontweight="bold", color=self.theme["text"])
            ax.set_title(f"{metric_name}", fontsize=13, fontweight="bold", pad=14, color=self.theme["text"])

            ax.grid(True, axis="y", alpha=0.75, color=self.theme["grid"], linewidth=0.9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color(self.theme["spine"])
            ax.spines["bottom"].set_color(self.theme["spine"])

            # 设置x轴范围
            max_epochs = max([len(exp["data"]) for exp in self.experiments.values()])
            ax.set_xlim(1, max_epochs)

            # 改进刻度标签
            ax.tick_params(axis="both", which="major", labelsize=10, colors=self.theme["muted_text"])
            ax.set_facecolor(self.theme["axes_bg"])

        # 隐藏多余的子图
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)

        handles = []
        labels = []
        for ax in axes[:n_metrics]:
            h, l = ax.get_legend_handles_labels()
            for hh, ll in zip(h, l):
                if ll not in labels:
                    handles.append(hh)
                    labels.append(ll)
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower left",
                bbox_to_anchor=(0.08, 0.01),
                ncol=min(3, max(1, len(labels))),
                fontsize=8.5,
                framealpha=0.95,
                title="Legend",
                title_fontsize=9,
            )
        self._apply_shared_figure_style(fig, f"{self.source_dir_name} 指标对比")
        # 确保保存路径在输出目录中
        save_path = Path(save_path)
        if not save_path.is_absolute():
            save_path = self.output_dir / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor=self.theme["figure_bg"],
            edgecolor="none",
        )
        print(f"图表已保存至: {save_path}")
        plt.close()

        return fig

    def plot_best_values_comparison(
        self, metrics_to_plot=None, save_path="best_values_comparison.png"
    ):
        """生成最佳值对比柱状图"""
        # 默认为所有性能指标
        if metrics_to_plot is None:
            metrics_to_plot = self.get_performance_metrics()

        # 过滤掉不存在的指标
        valid_metrics = []
        for metric in metrics_to_plot:
            has_data = any(
                metric in exp["data"].columns for exp in self.experiments.values()
            )
            if has_data:
                valid_metrics.append(metric)
            else:
                print(f"警告: 指标 {metric} 在所有实验中都不存在，已跳过")

        if not valid_metrics:
            print("错误: 没有有效的指标可以绘制")
            return None

        n_metrics = len(valid_metrics)
        rows = (n_metrics + 1) // 2  # 2列布局
        cols = 2

        # 创建子图
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5.4 * rows))
        axes = axes.flatten() if n_metrics > 1 else [axes]

        for idx, metric in enumerate(valid_metrics):
            if idx >= len(axes):
                break

            ax = axes[idx]
            metric_name = self.metrics.get(metric, metric)
            best_values = self.find_best_values(metric)

            if not best_values:
                continue

            # 准备数据
            exp_names = list(best_values.keys())
            values = [best_values[exp]["value"] for exp in exp_names]
            epochs = [best_values[exp]["epoch"] for exp in exp_names]

            # 按值排序
            sorted_data = sorted(
                zip(exp_names, values, epochs),
                key=lambda x: x[1],
                reverse=not any(
                    prefix in metric for prefix in ["loss", "val/loss", "train/loss"]
                ),
            )
            exp_names, values, epochs = zip(*sorted_data)
            alias_names = [self.display_labels[name] for name in exp_names]

            # 绘制柱状图
            bars = ax.bar(
                range(len(exp_names)),
                values,
                color=[self.colors[exp] for exp in exp_names],
                alpha=0.94,
                edgecolor=self.theme["panel_edge"],
                linewidth=1.1,
            )

            is_loss = any(
                prefix in metric for prefix in ["loss", "val/loss", "train/loss"]
            )
            self._annotate_selected_bars(ax, bars, values, epochs, is_loss)

            # 设置图表样式
            ax.set_xlabel("实验组", fontsize=12, fontweight="bold", color=self.theme["text"])
            ax.set_ylabel(metric_name, fontsize=12, fontweight="bold", color=self.theme["text"])
            ax.set_title(f"{metric_name}", fontsize=13, fontweight="bold", pad=14, color=self.theme["text"])
            ax.set_xticks(range(len(exp_names)))
            ax.set_xticklabels(
                alias_names,
                rotation=35,
                ha="right",
                fontsize=9,
                fontweight="bold",
                color=self.theme["muted_text"],
            )
            ax.grid(True, alpha=0.75, axis="y", color=self.theme["grid"], linewidth=0.9)
            ax.set_facecolor(self.theme["axes_bg"])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color(self.theme["spine"])
            ax.spines["bottom"].set_color(self.theme["spine"])

            # 改进刻度标签
            ax.tick_params(axis="both", which="major", labelsize=10, colors=self.theme["muted_text"])

        # 隐藏多余的子图
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)

        self._apply_shared_figure_style(fig, f"{self.source_dir_name} 最佳值对比")
        # 确保保存路径在输出目录中
        save_path = Path(save_path)
        if not save_path.is_absolute():
            save_path = self.output_dir / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor=self.theme["figure_bg"],
            edgecolor="none",
        )
        print(f"最佳值对比图已保存至: {save_path}")
        plt.close()

        return fig

    def generate_report(self):
        """生成详细分析报告"""
        print("\n" + "=" * 70)
        print("YOLO PCB 训练结果详细分析报告")
        print("=" * 70)

        # 总体统计
        print("\n总体统计：")
        print(f"  实验总数: {len(self.experiments)}")
        print(f"  分析指标数: {len(self.metrics)}")
        print(
            f"  训练轮次范围: {min([exp['epochs'] for exp in self.experiments.values()])} - {max([exp['epochs'] for exp in self.experiments.values()])}"
        )

        # 指标分类
        perf_metrics = [m for m in self.metrics if "metrics/" in m]
        loss_metrics = [m for m in self.metrics if "loss" in m]
        print(f"  性能指标: {len(perf_metrics)} 个")
        print(f"  损失指标: {len(loss_metrics)} 个")

        # 每个实验的详细信息
        print("\n" + "-" * 70)
        print("各实验详细信息：")
        print("-" * 70)

        for exp_name in sorted(self.experiments.keys()):
            exp = self.experiments[exp_name]
            print(f"\n{exp_name}:")
            print(f"  训练轮次: {exp['epochs']}")

            # 最终性能
            print("  最终性能：")
            for metric, metric_name in self.metrics.items():
                if metric in exp["final_metrics"]:
                    print(f"    {metric_name}: {exp['final_metrics'][metric]:.4f}")

            # 最佳性能
            print("  最佳性能：")
            for metric, metric_name in self.metrics.items():
                if metric in exp["data"].columns:
                    best_value = self.find_best_values(metric)[exp_name]["value"]
                    best_epoch = self.find_best_values(metric)[exp_name]["epoch"]
                    print(f"    {metric_name}: {best_value:.4f} (Epoch {best_epoch})")

        # 最佳值总结
        print("\n" + "-" * 70)
        print("各指标最佳值总结：")
        print("-" * 70)

        for metric, metric_name in self.metrics.items():
            best_values = self.find_best_values(metric)
            if best_values:
                # 找到全局最佳
                if any(
                    prefix in metric for prefix in ["loss", "val/loss", "train/loss"]
                ):
                    best_exp = min(
                        best_values.keys(), key=lambda x: best_values[x]["value"]
                    )
                    worst_exp = max(
                        best_values.keys(), key=lambda x: best_values[x]["value"]
                    )
                else:
                    best_exp = max(
                        best_values.keys(), key=lambda x: best_values[x]["value"]
                    )
                    worst_exp = min(
                        best_values.keys(), key=lambda x: best_values[x]["value"]
                    )

                best_val = best_values[best_exp]["value"]
                best_epoch = best_values[best_exp]["epoch"]
                worst_val = best_values[worst_exp]["value"]

                print(f"\n{metric_name}:")
                print(f"  最佳: {best_val:.4f} ({best_exp}, Epoch {best_epoch})")
                print(f"  最差: {worst_val:.4f} ({worst_exp})")

                # 统计信息
                stats = self.calculate_statistics(metric)
                if stats:
                    print(
                        f"  统计信息: 平均值={stats['mean']:.4f}, 中位数={stats['median']:.4f}, 标准差={stats['std']:.4f}"
                    )

                # 显示所有实验的排名
                if any(
                    prefix in metric for prefix in ["loss", "val/loss", "train/loss"]
                ):
                    sorted_exp = sorted(
                        best_values.keys(), key=lambda x: best_values[x]["value"]
                    )
                else:
                    sorted_exp = sorted(
                        best_values.keys(),
                        key=lambda x: best_values[x]["value"],
                        reverse=True,
                    )

                print("  排名：")
                for i, exp in enumerate(sorted_exp, 1):
                    val = best_values[exp]["value"]
                    epoch = best_values[exp]["epoch"]
                    print(f"    {i}. {exp}: {val:.4f} (Epoch {epoch})")

        # 训练趋势分析
        print("\n" + "-" * 70)
        print("训练趋势分析：")
        print("-" * 70)

        # 分析每个实验的收敛情况
        for exp_name in sorted(self.experiments.keys()):
            exp = self.experiments[exp_name]
            df = exp["data"]
            print(f"\n{exp_name} 训练趋势:")

            primary_metric = self.get_primary_metric()
            if primary_metric and primary_metric in df.columns:
                mAP50_data = df[primary_metric]
                # 计算最后10%轮次的平均值和方差
                last_10pct = int(len(mAP50_data) * 0.1)
                if last_10pct >= 3:
                    last_values = mAP50_data.iloc[-last_10pct:]
                    avg_last = np.mean(last_values)
                    std_last = np.std(last_values)
                    if mAP50_data.iloc[0] != 0:
                        improvement = (
                            (mAP50_data.iloc[-1] - mAP50_data.iloc[0])
                            / mAP50_data.iloc[0]
                            * 100
                        )
                        improvement_text = f"{improvement:.2f}%"
                    else:
                        improvement_text = "起始值为 0，跳过百分比计算"
                    metric_label = self.metrics.get(primary_metric, primary_metric)
                    print(
                        f"  {metric_label}: 初始={mAP50_data.iloc[0]:.4f}, 最终={mAP50_data.iloc[-1]:.4f}, 提升={improvement_text}"
                    )
                    print(
                        f"  {metric_label}: 最后{last_10pct}轮平均值={avg_last:.4f}, 标准差={std_last:.4f} (收敛性: {'好' if std_last < 0.01 else '一般' if std_last < 0.02 else '差'})"
                    )

        # 配置比较
        print("\n" + "-" * 70)
        print("实验配置比较：")
        print("-" * 70)

        # 收集所有配置的公共键
        all_configs = []
        for exp in self.experiments.values():
            if exp["config"]:
                all_configs.append(exp["config"])

        if all_configs:
            # 获取所有配置的公共键
            common_keys = set(all_configs[0].keys())
            for config in all_configs[1:]:
                common_keys.intersection_update(config.keys())

            # 只显示重要的配置项
            important_keys = [
                "batch_size",
                "epochs",
                "lr0",
                "lrf",
                "momentum",
                "weight_decay",
                "model",
            ]
            relevant_keys = [key for key in important_keys if key in common_keys]

            if relevant_keys:
                print("\n关键配置项对比：")
                print(
                    "   "
                    + " | ".join(
                        [key.ljust(15) for key in ["实验名称"] + relevant_keys]
                    )
                )
                print("   " + "-" * 15 + "-" * 17 * len(relevant_keys))

                for exp_name in sorted(self.experiments.keys()):
                    exp = self.experiments[exp_name]
                    if exp["config"]:
                        values = [exp_name.ljust(15)]
                        for key in relevant_keys:
                            value = str(exp["config"].get(key, "N/A")).ljust(15)
                            values.append(value)
                        print("   " + " | ".join(values))

        print("\n" + "=" * 70)
        print("分析报告生成完成。")
        print("=" * 70)

    def plot_training_curves(self, metrics_to_plot=None):
        """
        为每个指标单独绘制训练曲线图
        x轴为轮次(epoch)，y轴为指标值

        Args:
            metrics_to_plot: 要绘制的指标列表，如果为None则绘制所有指标
        """
        if metrics_to_plot is None:
            metrics_to_plot = self.get_performance_metrics() + self.get_loss_metrics()

        # 过滤掉不存在的指标
        valid_metrics = []
        for metric in metrics_to_plot:
            has_data = any(
                metric in exp["data"].columns for exp in self.experiments.values()
            )
            if has_data:
                valid_metrics.append(metric)
            else:
                print(f"警告: 指标 {metric} 在所有实验中都不存在，已跳过")

        if not valid_metrics:
            print("错误: 没有有效的指标可以绘制")
            return

        print(f"\n正在为 {len(valid_metrics)} 个指标绘制训练曲线图...")

        # 为每个指标单独绘制一张图
        for metric in valid_metrics:
            metric_name = self.metrics.get(metric, metric)

            # 创建新图
            fig, ax = plt.subplots(figsize=(13, 7.2))

            # 绘制每个实验的曲线
            for exp_name in sorted(self.experiments.keys()):
                df = self.experiments[exp_name]["data"]
                if metric in df.columns:
                    # 获取epoch列，如果存在则使用，否则使用索引+1
                    if "epoch" in df.columns:
                        epochs = df["epoch"].values
                    else:
                        epochs = df.index + 1  # epoch从1开始

                    values = df[metric].values

                    # 过滤掉NaN值
                    valid_mask = ~np.isnan(values)
                    epochs_clean = epochs[valid_mask]
                    values_clean = values[valid_mask]

                    if len(epochs_clean) > 0:
                        ax.plot(
                            epochs_clean,
                            values_clean,
                            label=self.compact_labels[exp_name],
                            color=self.colors[exp_name],
                            linestyle=self.line_styles[exp_name],
                            linewidth=2.8,
                            alpha=0.96,
                            marker=self.markers[exp_name],
                            markersize=4.8,
                            markerfacecolor="white",
                            markeredgecolor=self.colors[exp_name],
                            markeredgewidth=1.1,
                            markevery=max(1, len(epochs_clean) // 18),
                        )  # 间隔显示标记

            # 设置图表样式
            ax.set_xlabel("训练轮次 (Epochs)", fontsize=14, fontweight="bold", color=self.theme["text"])
            ax.set_ylabel(metric_name, fontsize=14, fontweight="bold", color=self.theme["text"])
            ax.set_title(metric_name, fontsize=15, fontweight="bold", pad=14, color=self.theme["text"])

            ax.grid(True, axis="y", alpha=0.78, color=self.theme["grid"], linewidth=0.95)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color(self.theme["spine"])
            ax.spines["bottom"].set_color(self.theme["spine"])

            # 设置x轴范围
            max_epoch = 1
            has_epoch_column = False
            for exp in self.experiments.values():
                df = exp["data"]
                if "epoch" in df.columns:
                    has_epoch_column = True
                    max_epoch = max(max_epoch, df["epoch"].max())
                else:
                    max_epoch = max(max_epoch, len(df))

            if has_epoch_column:
                ax.set_xlim(1, max_epoch)
            else:
                ax.set_xlim(1, max_epoch)

            # 改进刻度标签
            ax.tick_params(axis="both", which="major", labelsize=11, colors=self.theme["muted_text"])
            ax.set_facecolor(self.theme["axes_bg"])

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                fig.legend(
                    handles,
                    labels,
                    loc="lower center",
                bbox_to_anchor=(0.44, 0.04),
                ncol=min(3, max(1, len(labels))),
                fontsize=8.5,
                framealpha=0.95,
                title="Legend",
                    title_fontsize=9,
                )

            # 保存图片
            # 将指标名称转换为文件名安全的格式
            safe_metric_name = metric_name.replace(" ", "_").replace("/", "_")
            save_path = self.output_dir / f"training_curve_{safe_metric_name}.png"

            self._apply_single_axis_style(fig, ax, f"{self.source_dir_name} 训练曲线")
            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
                facecolor=self.theme["figure_bg"],
                edgecolor="none",
            )
            print(f"  [成功] {metric_name} 训练曲线已保存: {save_path}")
            plt.close()

        print(f"\n所有训练曲线图已保存至: {self.output_dir}")

    def write_lists_txt(self):
        """将参与分析的所有实验名称写入 lists.txt。"""
        if not self.experiments:
            return
        lists_path = self.output_dir / "lists.txt"
        aliases_path = self.output_dir / "experiment_aliases.txt"
        names = sorted(self.experiments.keys())
        with open(lists_path, "w", encoding="utf-8") as f:
            f.write("# 参与分析的所有实验组\n")
            f.write("# 分析来源: " + self.source_dir_name + "\n")
            f.write(
                "# 生成时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n"
            )
            f.write("-" * 40 + "\n")
            for name in names:
                compact = self.compact_labels.get(name, self.display_labels.get(name, "--"))
                f.write(f"{compact}  ->  {name}\n")
        with open(aliases_path, "w", encoding="utf-8") as f:
            f.write("# 图表实验别名映射\n")
            f.write("# 分析来源: " + self.source_dir_name + "\n")
            f.write(
                "# 生成时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n"
            )
            f.write("-" * 60 + "\n")
            for name in names:
                compact = self.compact_labels.get(name, self.display_labels.get(name, "--"))
                f.write(f"{compact} = {name}\n")
        print(f"实验列表已保存至: {lists_path}")
        print(f"实验别名映射已保存至: {aliases_path}")


def _interactive_select_source_and_experiments(workspace_root):
    """交互式选择实验组及要分析的实验。

    Returns:
        tuple: (source_dir_name, selected_experiments) 或 (None, None) 表示取消。
    """
    root = Path(workspace_root)
    available = [
        d.name for d in root.iterdir() if d.is_dir() and d.name in SUPPORTED_SOURCE_DIRS
    ]
    if not available:
        print("错误: 工作区中未找到 train_origin、train_ex 或 train_segment 目录")
        return None, None

    print("\n可选实验组:")
    for i, name in enumerate(available, 1):
        count = len(
            [
                f
                for f in (root / name).iterdir()
                if f.is_dir() and (f / "results.csv").exists()
            ]
        )
        print(f"  {i}. {name} ({count} 个实验)")
    print(f"  0. 退出")

    try:
        choice = input("\n请选择实验组编号 (直接回车默认选 1): ").strip() or "1"
        idx = int(choice)
        if idx == 0:
            return None, None
        if idx < 1 or idx > len(available):
            print("无效选择")
            return None, None
        source_name = available[idx - 1]
    except ValueError:
        print("无效输入")
        return None, None

    source_path = root / source_name
    exp_folders = [
        f for f in source_path.iterdir() if f.is_dir() and (f / "results.csv").exists()
    ]
    exp_names = sorted([f.name for f in exp_folders])

    if not exp_names:
        print(f"错误: {source_name} 中无有效实验")
        return None, None

    print(f"\n{source_name} 中的实验:")
    for i, name in enumerate(exp_names, 1):
        print(f"  {i}. {name}")
    print(f"  0. 全部选择")
    print(f"  输入编号或编号区间，如 1,3 或 1-3，多个用逗号分隔")

    sel = input("\n请选择要分析的实验 (直接回车选全部): ").strip()
    if not sel or sel == "0":
        return source_name, None  # 全部

    selected = set()
    for part in sel.replace("，", ",").split(","):
        part = part.strip()
        if "-" in part:
            try:
                a, b = part.split("-", 1)
                lo, hi = int(a.strip()), int(b.strip())
                for i in range(lo, hi + 1):
                    if 1 <= i <= len(exp_names):
                        selected.add(exp_names[i - 1])
            except ValueError:
                pass
        else:
            try:
                i = int(part)
                if 1 <= i <= len(exp_names):
                    selected.add(exp_names[i - 1])
            except ValueError:
                pass

    if not selected:
        print("未选择有效实验，将分析全部")
        return source_name, None
    return source_name, sorted(selected)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="分析 train_origin、train_ex 与 train_segment 目录下的 YOLO 训练结果。"
    )
    parser.add_argument(
        "--source",
        "-s",
        choices=SUPPORTED_SOURCE_DIRS,
        help="实验来源目录，可选 train_origin、train_ex 或 train_segment。",
    )
    parser.add_argument(
        "--experiments",
        "-e",
        type=str,
        help="指定要分析的实验名称，多个实验用逗号分隔；留空则分析全部。",
    )
    parser.add_argument(
        "--workspace",
        "-w",
        type=str,
        default=os.getcwd(),
        help="工作区根目录，默认使用当前目录。",
    )
    parser.add_argument(
        "--run-inference",
        action="store_true",
        help="额外对每个实验的最佳模型执行推理评估。",
    )
    parser.add_argument(
        "--mplstyle",
        type=str,
        help="可选的 Matplotlib mplstyle 文件路径，例如 D:/YOLO_PCB/deeplearning.mplstyle。",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default="",
        help="可选输出目录后缀，仅用于区分不同分析批次，不用于区分图表风格。",
    )
    args = parser.parse_args()

    workspace_root = Path(args.workspace)
    print("YOLO PCB 训练结果分析工具")
    print("=" * 50)

    style_file = None
    if args.mplstyle:
        try:
            style_file = apply_mplstyle_if_requested(args.mplstyle)
            print(f"已加载样式文件: {style_file}")
        except Exception as exc:
            print(f"加载样式文件失败: {exc}")
            return

    # 确定实验组和实验列表
    if args.source is not None:
        source_name = args.source
        selected_experiments = None
        if args.experiments:
            selected_experiments = [
                x.strip() for x in args.experiments.split(",") if x.strip()
            ]
    else:
        result = _interactive_select_source_and_experiments(workspace_root)
        if result == (None, None):
            print("已取消")
            return
        source_name, selected_experiments = result

    # 创建分析器（自动生成 results_analyse/{source}/val_MMDDHHmm/ 输出目录）
    analyzer = TrainingResultAnalyzer(
        source_dir=source_name,
        workspace_root=workspace_root,
        selected_experiments=selected_experiments,
        output_tag=args.output_tag or None,
    )

    # 加载数据
    print("\n正在加载训练结果...")
    if not analyzer.load_all_results():
        print("加载失败，请检查目录路径")
        return

    print(f"\n成功加载 {len(analyzer.experiments)} 个实验数据")

    # 生成分析报告
    analyzer.generate_report()

    # 生成对比图表
    print("\n正在生成对比图表...")
    performance_metrics = analyzer.get_performance_metrics()
    loss_metrics = analyzer.get_loss_metrics()

    # 1. 生成主要性能指标对比图
    analyzer.plot_comparison(
        metrics_to_plot=performance_metrics[:2],
        save_path="training_comparison_main.png",
    )

    # 2. 生成所有性能指标对比图
    analyzer.plot_comparison(
        metrics_to_plot=performance_metrics,
        save_path="training_comparison_perf.png",
    )

    # 3. 生成损失指标对比图
    analyzer.plot_comparison(
        metrics_to_plot=loss_metrics,
        save_path="training_comparison_loss.png",
    )

    # 4. 生成所有指标对比图
    analyzer.plot_comparison(
        metrics_to_plot=list(analyzer.metrics.keys()),
        save_path="training_comparison_all.png",
    )

    # 5. 生成最佳值对比图（性能指标）
    analyzer.plot_best_values_comparison(
        metrics_to_plot=performance_metrics,
        save_path="best_values_comparison_perf.png",
    )

    # 6. 生成最佳值对比图（损失指标）
    analyzer.plot_best_values_comparison(
        metrics_to_plot=loss_metrics,
        save_path="best_values_comparison_loss.png",
    )

    # 7. 生成最佳值对比图（所有指标）
    analyzer.plot_best_values_comparison(
        metrics_to_plot=list(analyzer.metrics.keys()),
        save_path="best_values_comparison_all.png",
    )

    # 8. 生成单个指标的训练曲线图（新增功能）
    print("\n正在生成单个指标训练曲线图...")
    analyzer.plot_training_curves(
        metrics_to_plot=performance_metrics
        + [m for m in loss_metrics if m not in performance_metrics]
    )

    # 写入参与分析的实验列表
    analyzer.write_lists_txt()

    # 询问用户是否运行推理
    run_inference = args.run_inference
    if not run_inference:
        print("\n" + "=" * 50)
        print("最佳模型推理功能")
        print("=" * 50)
        print("是否对所有实验的最佳模型重新运行推理？")
        print("注意：这将花费较长时间，取决于实验数量和数据集大小")
        response = input("输入 'y' 或 'yes' 运行推理，其他键跳过: ")
        run_inference = response.lower() in ["y", "yes"]

    if run_inference:
        print("\n开始运行最佳模型推理...")
        inference_results = analyzer.run_inference_on_all_best_models()
        analyzer.generate_inference_report(inference_results)

    print("\n分析完成！")
    print(f"\n生成的文件保存在: {analyzer.output_dir}")
    print("\n对比图表:")
    print("  - training_comparison_main.png (主要性能指标对比)")
    print("  - training_comparison_perf.png (所有性能指标对比)")
    print("  - training_comparison_loss.png (损失指标对比)")
    print("  - training_comparison_all.png (所有指标对比)")
    print("  - best_values_comparison_perf.png (性能指标最佳值对比)")
    print("  - best_values_comparison_loss.png (损失指标最佳值对比)")
    print("  - best_values_comparison_all.png (所有指标最佳值对比)")
    print("\n训练曲线图:")
    print("  - training_curve_*.png (各指标单独的训练曲线图)")

    if run_inference:
        print("\n推理结果已生成报告显示在控制台。")


if __name__ == "__main__":
    main()
