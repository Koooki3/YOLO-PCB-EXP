#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按实验目录生成模型一图流与 modelAnalyse.txt。"""

from __future__ import annotations

import argparse
from copy import deepcopy
import io
import math
import os
import re
import sys
import time
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
from matplotlib import patches
import torch
import yaml

from chart_style import apply_unified_mpl_style, build_series_palette, get_visual_theme


WORKSPACE_ROOT = Path(__file__).resolve().parent
TRAIN_DIRS = ("train_origin", "train_ex", "train_segment")
RESULT_ROOT = WORKSPACE_ROOT / "model_analyse"
DEFAULT_PYTHON = r"C:\anaconda\envs\kooki\python.exe"

THEME = apply_unified_mpl_style()
SERIES_COLORS = build_series_palette()


@dataclass
class ExperimentRef:
    source_dir: str
    name: str
    path: Path
    args: dict[str, Any]


def _safe_load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _format_num(num: float, digits: int = 2) -> str:
    if num is None or (isinstance(num, float) and not math.isfinite(num)):
        return "N/A"
    return f"{num:.{digits}f}"


def _format_m(num: float) -> str:
    if num is None:
        return "N/A"
    return f"{num / 1e6:.2f} M"


def _format_mb(num_bytes: int | None) -> str:
    if not num_bytes:
        return "N/A"
    return f"{num_bytes / (1024 ** 2):.2f} MB"


def _normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _canonical_experiment_name(name: str) -> str:
    text = name.lower()
    text = text.replace("pcbfinal_", "")
    text = re.sub(r"_dinop\d+", "", text)
    text = re.sub(r"_dinov\d+_[a-z0-9_]+", "", text)
    text = re.sub(r"_dino[a-z0-9_]*", "", text)
    text = re.sub(r"_fixed", "", text)
    text = re.sub(r"_c\d+(?:\.\d+)?", "", text)
    text = text.replace("wassersteinloss_", "wassersteinloss")
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def discover_experiments(workspace: Path) -> list[ExperimentRef]:
    refs: list[ExperimentRef] = []
    for source_dir in TRAIN_DIRS:
        source_path = workspace / source_dir
        if not source_path.exists():
            continue
        for exp_dir in sorted(p for p in source_path.iterdir() if p.is_dir()):
            args_path = exp_dir / "args.yaml"
            if not args_path.exists():
                continue
            try:
                refs.append(
                    ExperimentRef(
                        source_dir=source_dir,
                        name=exp_dir.name,
                        path=exp_dir,
                        args=_safe_load_yaml(args_path),
                    )
                )
            except Exception:
                continue
    return refs


def resolve_experiment(identifier: str | None, workspace: Path) -> ExperimentRef:
    experiments = discover_experiments(workspace)
    if not experiments:
        raise FileNotFoundError("未找到任何带 args.yaml 的训练实验目录。")

    if identifier:
        ident = identifier.strip()
        for ref in experiments:
            if ref.name == ident or str(ref.path) == ident or ref.path.name == Path(ident).name:
                return ref
        raise FileNotFoundError(f"未找到实验: {identifier}")

    print("可分析实验：")
    for idx, ref in enumerate(experiments, 1):
        print(f"  {idx:>2}. {ref.source_dir}/{ref.name}")
    choice = input("\n请选择实验编号: ").strip()
    if not choice:
        raise ValueError("未提供实验编号。")
    index = int(choice)
    if index < 1 or index > len(experiments):
        raise ValueError("实验编号超出范围。")
    return experiments[index - 1]


def find_best_weight(exp_path: Path) -> Path | None:
    weights_dir = exp_path / "weights"
    if not weights_dir.exists():
        return None
    for candidate in ("best.pt", "last.pt"):
        path = weights_dir / candidate
        if path.exists():
            return path
    return None


def build_model(args_dict: dict[str, Any], weight_path: Path | None):
    from ultralytics import RTDETR, YOLO

    model_hint = str(args_dict.get("model", "")).lower()
    is_rtdetr = "rtdetr" in model_hint
    wrapper_cls = RTDETR if is_rtdetr else YOLO
    load_path = str(weight_path) if weight_path else str(args_dict["model"])
    wrapper = wrapper_cls(load_path)
    return wrapper


def get_module_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    if hasattr(model, "model") and isinstance(model.model, torch.nn.Sequential):
        return list(model.model)
    return list(model.children())


def count_leaf_layers(model: torch.nn.Module) -> int:
    return sum(1 for m in model.modules() if len(list(m.children())) == 0)


def estimate_trainable_params(model: torch.nn.Module) -> int:
    total = sum(p.numel() for p in model.parameters())
    frozen_dino = 0
    for module in model.modules():
        if module.__class__.__name__ in {"DINO2Backbone", "DINO3Backbone"} and getattr(module, "freeze_backbone", False):
            dino_model = getattr(module, "dino_model", None)
            if dino_model is not None:
                frozen_dino += sum(p.numel() for p in dino_model.parameters())
    estimate = max(total - frozen_dino, 0)
    return estimate


def collect_module_type_counts(model: torch.nn.Module) -> Counter:
    counts: Counter = Counter()
    for module in get_module_layers(model):
        counts[module.__class__.__name__] += 1
    return counts


def summarize_architecture(model: torch.nn.Module, yaml_dict: dict[str, Any]) -> dict[str, Any]:
    layers = get_module_layers(model)
    backbone_defs = list(yaml_dict.get("backbone", []) or [])
    head_defs = list(yaml_dict.get("head", []) or [])
    combined_defs = backbone_defs + head_defs

    output_head = layers[-1].__class__.__name__ if layers else "Unknown"
    output_scales = 0
    if combined_defs:
        last_def = combined_defs[-1]
        if isinstance(last_def, list) and len(last_def) >= 1 and isinstance(last_def[0], list):
            output_scales = len(last_def[0])

    selected_layers = []
    total_defs = len(combined_defs)
    for idx, layer_def in enumerate(combined_defs):
        if not isinstance(layer_def, list) or len(layer_def) < 4:
            continue
        from_idx, repeats, module_name, args = layer_def
        text = f"#{idx:02d} {module_name} x{repeats}"
        if module_name in {"DINO2Backbone", "DINO3Backbone"}:
            variant = args[0] if args else "unknown"
            frozen = bool(args[1]) if len(args) > 1 else False
            out_ch = args[2] if len(args) > 2 else "match"
            text += f" [{variant}, freeze={frozen}, out={out_ch}]"
            selected_layers.append(text)
        elif module_name in {"Detect", "Segment", "RTDETRDecoder"}:
            text += f" [from={from_idx}]"
            selected_layers.append(text)
        elif idx < 3 or idx >= max(0, total_defs - 4):
            text += f" [from={from_idx}]"
            selected_layers.append(text)

    if not selected_layers:
        for idx, layer in enumerate(layers[: min(8, len(layers))]):
            selected_layers.append(f"#{idx:02d} {layer.__class__.__name__}")

    return {
        "layer_count": len(layers),
        "leaf_layer_count": count_leaf_layers(model),
        "backbone_layer_count": len(backbone_defs),
        "head_layer_count": len(head_defs),
        "output_head": output_head,
        "output_scales": output_scales,
        "selected_layers": selected_layers[:10],
    }


def capture_dino_io_shapes(model: torch.nn.Module, imgsz: int) -> dict[str, tuple[int, int, int]]:
    hooks = []
    captured: dict[str, tuple[int, int, int]] = {}

    def _make_hook(name: str):
        def _hook(module, inputs, outputs):
            x = inputs[0] if inputs else None
            if isinstance(x, (list, tuple)):
                x = x[0]
            if isinstance(x, torch.Tensor):
                _, c, h, w = x.shape
                captured[name] = (int(c), int(h), int(w))
        return _hook

    dino_idx = 0
    for module in model.modules():
        if module.__class__.__name__ in {"DINO2Backbone", "DINO3Backbone"}:
            name = f"dino_{dino_idx}"
            hooks.append(module.register_forward_hook(_make_hook(name)))
            dino_idx += 1

    if not hooks:
        return captured

    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        dummy = torch.randn(1, 3, imgsz, imgsz, device=device)
        model(dummy)

    for hook in hooks:
        hook.remove()
    return captured


def estimate_single_dino_gflops(module: torch.nn.Module, input_shape: tuple[int, int, int] | None) -> float:
    if input_shape is None:
        return 0.0
    in_channels, height, width = input_shape
    model_name = getattr(module, "model_name", "dinov3_vits16")
    spec_map = getattr(module, "dinov3_specs", {})
    spec = spec_map.get(model_name, {})
    model_name_lower = str(model_name).lower()
    patch_size = int(getattr(module, "patch_size", spec.get("patch_size", 16)) or 16)
    embed_dim = int(getattr(module, "embed_dim", spec.get("embed_dim", 384)) or 384)
    out_channels = int(getattr(module, "output_channels", in_channels) or in_channels)
    patch_grid = max(224 // patch_size, 1)

    if "vits" in model_name_lower or "small" in model_name_lower:
        dino_flops = 8.5
    elif "vitb" in model_name_lower or "base" in model_name_lower:
        dino_flops = 28.0
    elif "vitl" in model_name_lower or "large" in model_name_lower:
        dino_flops = 95.0
    elif "vith" in model_name_lower or "huge" in model_name_lower:
        dino_flops = 260.0
    else:
        dino_flops = float(spec.get("params", 21)) * 0.4

    input_proj_flops = 2.0 * (in_channels * 64 * 3 * 3 * height * width + 64 * 3 * height * width) / 1e9
    feature_adapter_flops = 2.0 * (embed_dim * out_channels * patch_grid * patch_grid) / 1e9
    spatial_proj_flops = 2.0 * (out_channels * out_channels * 3 * 3 * patch_grid * patch_grid) / 1e9
    fusion_flops = 2.0 * ((in_channels + out_channels) * out_channels * 3 * 3 * height * width) / 1e9
    return dino_flops + input_proj_flops + feature_adapter_flops + spatial_proj_flops + fusion_flops


def compute_raw_thop_gflops(model: torch.nn.Module, imgsz: int) -> float:
    try:
        import thop
    except ImportError:
        return 0.0

    model = model.to(next(model.parameters()).device)
    p = next(model.parameters())
    imgsz_pair = [imgsz, imgsz]
    has_dino = any(m.__class__.__name__ in {"DINO2Backbone", "DINO3Backbone"} for m in model.modules())
    try:
        if has_dino:
            im = torch.empty((1, p.shape[1], *imgsz_pair), device=p.device)
            return thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9 * 2
        stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32
        im = torch.empty((1, p.shape[1], stride, stride), device=p.device)
        flops = thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9 * 2
        return flops * imgsz_pair[0] / stride * imgsz_pair[1] / stride
    except Exception:
        try:
            im = torch.empty((1, p.shape[1], *imgsz_pair), device=p.device)
            return thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9 * 2
        except Exception:
            return 0.0


def collect_dino_analysis(model: torch.nn.Module, imgsz: int) -> dict[str, Any]:
    dino_modules = []
    io_shapes = capture_dino_io_shapes(model, imgsz)
    dino_layers = [m for m in model.modules() if m.__class__.__name__ in {"DINO2Backbone", "DINO3Backbone"}]
    for dino_idx, module in enumerate(dino_layers):
        if module.__class__.__name__ not in {"DINO2Backbone", "DINO3Backbone"}:
            continue
        key = f"dino_{dino_idx}"
        input_shape = io_shapes.get(key)
        dino_model = getattr(module, "dino_model", None)
        dino_params = sum(p.numel() for p in dino_model.parameters()) if dino_model is not None else None
        module_params = sum(p.numel() for p in module.parameters())
        spec = getattr(module, "model_spec", {}) or getattr(module, "dinov3_specs", {}).get(getattr(module, "model_name", ""), {})
        dino_modules.append(
            {
                "class_name": module.__class__.__name__,
                "variant": getattr(module, "model_name", "unknown"),
                "freeze_backbone": bool(getattr(module, "freeze_backbone", False)),
                "embed_dim": getattr(module, "embed_dim", spec.get("embed_dim")),
                "patch_size": getattr(module, "patch_size", spec.get("patch_size")),
                "output_channels": getattr(module, "output_channels", None),
                "dataset_type": getattr(module, "dataset_type", spec.get("dataset", "unknown")),
                "spec_params_m": spec.get("params"),
                "module_params": module_params,
                "dino_submodule_params": dino_params,
                "input_shape": input_shape,
                "estimated_gflops": estimate_single_dino_gflops(module, input_shape),
            }
        )
    return {
        "has_dino": bool(dino_modules),
        "module_count": len(dino_modules),
        "modules": dino_modules,
        "estimated_total_dino_gflops": sum(item["estimated_gflops"] for item in dino_modules),
    }


def infer_baseline_experiment(current: ExperimentRef, experiments: list[ExperimentRef]) -> ExperimentRef | None:
    current_key = _canonical_experiment_name(current.name)
    current_norm = _normalize_token(current_key)
    candidates = []
    for ref in experiments:
        if ref.source_dir != current.source_dir or ref.name == current.name:
            continue
        if "dino" in ref.name.lower():
            continue
        ref_norm = _normalize_token(_canonical_experiment_name(ref.name))
        if ref_norm == current_norm:
            candidates.append((0, ref))
            continue
        overlap = sum(ch1 == ch2 for ch1, ch2 in zip(current_norm, ref_norm))
        if current_norm in ref_norm or ref_norm in current_norm:
            candidates.append((1, ref))
        elif overlap > 6:
            candidates.append((2, ref))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], len(x[1].name)))
    return candidates[0][1]


def build_baseline_comparison(current_metrics: dict[str, Any], baseline_metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if baseline_metrics is None:
        return None
    return {
        "params_delta": current_metrics["params"] - baseline_metrics["params"],
        "gflops_delta": current_metrics["gflops"] - baseline_metrics["gflops"],
        "cpu_ms_delta": (
            current_metrics["cpu_avg_ms"] - baseline_metrics["cpu_avg_ms"]
            if current_metrics["cpu_avg_ms"] is not None and baseline_metrics["cpu_avg_ms"] is not None
            else None
        ),
        "baseline_name": baseline_metrics["experiment"].name,
    }


def benchmark_cpu_latency(wrapper, imgsz: int, warmup: int, iters: int, repeats: int = 3) -> float:
    model = wrapper.model.to("cpu").eval()
    per_repeat_ms: list[float] = []
    with torch.inference_mode():
        dummy = torch.randn(1, 3, imgsz, imgsz, device="cpu")
        for _ in range(max(warmup, 0)):
            model(dummy)
        for _ in range(max(repeats, 1)):
            start = time.perf_counter()
            for _ in range(max(iters, 1)):
                model(dummy)
            elapsed = (time.perf_counter() - start) * 1000.0 / max(iters, 1)
            per_repeat_ms.append(elapsed)
    per_repeat_ms.sort()
    return per_repeat_ms[len(per_repeat_ms) // 2]


def capture_model_summary(model: torch.nn.Module, imgsz: int) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        model.info(detailed=False, verbose=True, imgsz=imgsz)
    return buffer.getvalue().strip()


def analyze_experiment(
    experiment: ExperimentRef,
    workspace: Path,
    cpu_warmup: int,
    cpu_iters: int,
    skip_cpu: bool = False,
) -> dict[str, Any]:
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    weight_path = find_best_weight(experiment.path)
    wrapper = build_model(experiment.args, weight_path)
    model = wrapper.model
    model.eval()

    imgsz = int(experiment.args.get("imgsz", 640))
    yaml_dict = getattr(model, "yaml", {}) or {}
    task = experiment.args.get("task") or ("segment" if model.model[-1].__class__.__name__ == "Segment" else "detect")
    params = int(get_num_params(model))
    trainable_params = int(estimate_trainable_params(model))
    raw_gflops = float(compute_raw_thop_gflops(model, imgsz))
    gflops = float(get_flops(model, imgsz))
    weight_size = weight_path.stat().st_size if weight_path and weight_path.exists() else None
    arch = summarize_architecture(model, yaml_dict)
    module_counts = collect_module_type_counts(model)
    dino = collect_dino_analysis(model, imgsz)
    model_summary_text = capture_model_summary(model, imgsz)

    cpu_avg_ms = None
    if not skip_cpu:
        cpu_avg_ms = benchmark_cpu_latency(wrapper, imgsz, warmup=cpu_warmup, iters=cpu_iters)

    metrics = {
        "experiment": experiment,
        "task": task,
        "imgsz": imgsz,
        "weight_path": weight_path,
        "weight_size": weight_size,
        "params": params,
        "trainable_params": trainable_params,
        "gflops": gflops,
        "raw_gflops": raw_gflops,
        "cpu_avg_ms": cpu_avg_ms,
        "arch": arch,
        "module_counts": module_counts,
        "dino": dino,
        "yaml_path": experiment.args.get("model"),
        "model_summary_text": model_summary_text,
    }

    experiments = discover_experiments(workspace)
    baseline_ref = infer_baseline_experiment(experiment, experiments) if dino["has_dino"] else None
    metrics["baseline_ref"] = baseline_ref
    if baseline_ref:
        baseline_metrics = analyze_experiment(
            baseline_ref,
            workspace,
            cpu_warmup=cpu_warmup,
            cpu_iters=cpu_iters,
            skip_cpu=skip_cpu,
        )
        metrics["baseline_metrics"] = baseline_metrics
        metrics["baseline_delta"] = build_baseline_comparison(metrics, baseline_metrics)
    else:
        metrics["baseline_metrics"] = None
        metrics["baseline_delta"] = None

    return metrics


def compose_report_text(metrics: dict[str, Any]) -> str:
    experiment = metrics["experiment"]
    arch = metrics["arch"]
    dino = metrics["dino"]
    lines = []
    lines.append("YOLO_PCB Model Analyse Report")
    lines.append("=" * 72)
    lines.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f"Experiment: {experiment.name}")
    lines.append(f"Source Dir: {experiment.source_dir}")
    lines.append(f"Task: {metrics['task']}")
    lines.append(f"Input Size: {metrics['imgsz']} x {metrics['imgsz']}")
    lines.append(f"Model Config: {metrics['yaml_path']}")
    lines.append(f"Weight Path: {metrics['weight_path'] or 'N/A'}")
    lines.append("")
    lines.append("Core Metrics")
    lines.append("-" * 72)
    lines.append(f"Checkpoint Size: {_format_mb(metrics['weight_size'])}")
    lines.append(f"Total Params: {_format_m(metrics['params'])} ({metrics['params']:,})")
    lines.append(f"Estimated Trainable Params: {_format_m(metrics['trainable_params'])} ({metrics['trainable_params']:,})")
    lines.append(f"Corrected GFLOPs: {_format_num(metrics['gflops'], 2)}")
    if metrics["raw_gflops"]:
        lines.append(f"Raw THOP GFLOPs: {_format_num(metrics['raw_gflops'], 2)}")
        if abs(metrics["raw_gflops"] - metrics["gflops"]) > 0.05:
            ratio = metrics["raw_gflops"] / max(metrics["gflops"], 1e-9)
            lines.append(f"Raw/Corrected Ratio: {_format_num(ratio, 2)}x")
    lines.append(f"Top-level Layer Count: {arch['layer_count']}")
    lines.append(f"Leaf Layer Count: {arch['leaf_layer_count']}")
    lines.append(f"Backbone Layers in YAML: {arch['backbone_layer_count']}")
    lines.append(f"Head Layers in YAML: {arch['head_layer_count']}")
    lines.append(f"Output Head: {arch['output_head']}")
    lines.append(f"Output Feature Scales: {arch['output_scales'] or 'N/A'}")
    lines.append(
        "CPU Avg Inference: "
        + (f"{metrics['cpu_avg_ms']:.2f} ms / image" if metrics["cpu_avg_ms"] is not None else "Skipped")
    )
    lines.append("")
    lines.append("Architecture Snapshot")
    lines.append("-" * 72)
    for item in arch["selected_layers"]:
        lines.append(item)

    lines.append("")
    lines.append("Module Type Distribution")
    lines.append("-" * 72)
    for module_name, count in metrics["module_counts"].most_common():
        lines.append(f"{module_name:<20} {count}")

    lines.append("")
    lines.append("DINO-Specific Analysis")
    lines.append("-" * 72)
    if not dino["has_dino"]:
        lines.append("No DINO backbone module detected in this experiment.")
    else:
        lines.append(
            f"DINO Modules: {dino['module_count']} | Estimated DINO-related GFLOPs: {_format_num(dino['estimated_total_dino_gflops'], 2)}"
        )
        for idx, item in enumerate(dino["modules"], 1):
            lines.append(
                f"[DINO-{idx}] {item['class_name']} | variant={item['variant']} | freeze={item['freeze_backbone']} | "
                f"embed_dim={item['embed_dim']} | patch={item['patch_size']} | out={item['output_channels']} | dataset={item['dataset_type']}"
            )
            if item["input_shape"]:
                c, h, w = item["input_shape"]
                lines.append(f"        YOLO->DINO input feature shape: C={c}, H={h}, W={w}")
            lines.append(
                f"        Spec Params={item['spec_params_m']}M | DINO Submodule Params={_format_m(item['dino_submodule_params'])} | "
                f"Whole DINO Block Params={_format_m(item['module_params'])} | Estimated GFLOPs={_format_num(item['estimated_gflops'], 2)}"
            )
        lines.append("        Note: local DINO integration always resizes the injected CNN feature to 224x224 before DINO forward.")
        lines.append("        Note: corrected GFLOPs follow the current local ultralytics DINO counter, not vanilla thop raw output.")
        if abs(metrics["raw_gflops"] - metrics["gflops"]) > 0.05:
            lines.append("        Note: raw THOP is retained only as a reference. The report uses corrected GFLOPs from the local DINO custom counter.")

    if metrics["baseline_delta"]:
        delta = metrics["baseline_delta"]
        lines.append("")
        lines.append("Baseline Delta")
        lines.append("-" * 72)
        lines.append(f"Matched Non-DINO Baseline: {delta['baseline_name']}")
        lines.append(f"Params Delta: {_format_m(delta['params_delta'])} ({delta['params_delta']:+,})")
        lines.append(f"GFLOPs Delta: {_format_num(delta['gflops_delta'], 2)}")
        lines.append(
            "CPU Delta: "
            + (f"{delta['cpu_ms_delta']:+.2f} ms" if delta["cpu_ms_delta"] is not None else "N/A")
        )

    lines.append("")
    lines.append("Ultralytics Summary")
    lines.append("-" * 72)
    lines.append(
        metrics["model_summary_text"]
        or (
            f"{arch['output_head']} summary: {arch['leaf_layer_count']} leaf layers, "
            f"{metrics['params']:,} parameters, corrected {metrics['gflops']:.1f} GFLOPs"
        )
    )
    return "\n".join(lines) + "\n"


def _draw_metric_card(ax, x, y, w, h, title, value, accent):
    card = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.04",
        facecolor=THEME["panel_bg"],
        edgecolor=THEME["panel_edge"],
        linewidth=1.2,
    )
    ax.add_patch(card)
    ax.add_patch(patches.Rectangle((x, y + h - 0.04), w, 0.04, color=accent, alpha=0.98))
    ax.text(x + 0.03, y + h - 0.10, title, fontsize=10.5, color=THEME["muted_text"], ha="left", va="top")
    value_fontsize = 17 if "\n" not in str(value) else 14
    value_y = y + (0.11 if "\n" not in str(value) else 0.07)
    ax.text(
        x + 0.03,
        value_y,
        value,
        fontsize=value_fontsize,
        fontweight="bold",
        color=THEME["text"],
        ha="left",
        va="bottom",
        linespacing=1.18,
    )


def _draw_panel(ax, title: str, accent: str):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel = patches.FancyBboxPatch(
        (0.0, 0.0),
        1.0,
        1.0,
        boxstyle="round,pad=0.014,rounding_size=0.035",
        facecolor=THEME["panel_bg"],
        edgecolor=THEME["panel_edge"],
        linewidth=1.2,
    )
    ax.add_patch(panel)
    ax.add_patch(patches.Rectangle((0.0, 0.92), 1.0, 0.08, color=accent, alpha=0.92))
    ax.text(0.03, 0.955, title, fontsize=13, fontweight="bold", color="white", ha="left", va="center")


def draw_overview_figure(metrics: dict[str, Any], output_path: Path):
    experiment = metrics["experiment"]
    arch = metrics["arch"]
    dino = metrics["dino"]
    baseline_delta = metrics["baseline_delta"]
    baseline_metrics = metrics.get("baseline_metrics")

    fig = plt.figure(figsize=(16, 10), facecolor=THEME["figure_bg"])
    gs = fig.add_gridspec(3, 12, height_ratios=[1.05, 1.25, 1.20], hspace=0.22, wspace=0.18)

    ax_header = fig.add_subplot(gs[0, :])
    ax_flow = fig.add_subplot(gs[1, :6])
    ax_perf = fig.add_subplot(gs[1, 6:])
    ax_layers = fig.add_subplot(gs[2, :6])
    ax_dino = fig.add_subplot(gs[2, 6:])

    ax_header.set_xlim(0, 1)
    ax_header.set_ylim(0, 1)
    ax_header.axis("off")

    banner = patches.FancyBboxPatch(
        (0.0, 0.66),
        1.0,
        0.34,
        boxstyle="round,pad=0.018,rounding_size=0.045",
        facecolor=THEME["header_fill"],
        edgecolor=THEME["header_edge"],
        linewidth=1.4,
    )
    ax_header.add_patch(banner)
    ax_header.text(0.03, 0.86, experiment.name, fontsize=24, fontweight="bold", color=THEME["text"], ha="left", va="center")
    ax_header.text(
        0.03,
        0.73,
        "Single-experiment profiling with corrected local GFLOPs, CPU latency, architecture summary and DINO impact.",
        fontsize=11.2,
        color=THEME["muted_text"],
        ha="left",
        va="center",
    )

    chip_texts = [
        experiment.source_dir,
        f"task={metrics['task']}",
        f"imgsz={metrics['imgsz']}",
        f"head={arch['output_head']}",
        f"scales={arch['output_scales'] or 'N/A'}",
    ]
    chip_x = 0.03
    for idx, text in enumerate(chip_texts):
        chip_w = 0.09 + 0.0062 * len(text)
        chip = patches.FancyBboxPatch(
            (chip_x, 0.60),
            chip_w,
            0.08,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor=THEME["annotation_bg"],
            edgecolor=SERIES_COLORS[(idx + 2) % len(SERIES_COLORS)],
            linewidth=1.1,
        )
        ax_header.add_patch(chip)
        ax_header.text(chip_x + chip_w / 2, 0.64, text, fontsize=10.2, color=THEME["text"], ha="center", va="center")
        chip_x += chip_w + 0.012

    delta_text = (
        f"{baseline_delta['gflops_delta']:+.2f} GFLOPs\n"
        + (f"{baseline_delta['cpu_ms_delta']:+.2f} ms" if baseline_delta and baseline_delta["cpu_ms_delta"] is not None else "baseline linked")
        if baseline_delta
        else f"{arch['leaf_layer_count']} leaf layers"
    )
    cards = [
        ("Checkpoint", _format_mb(metrics["weight_size"]), SERIES_COLORS[0]),
        ("Params", _format_m(metrics["params"]), SERIES_COLORS[1]),
        ("Corrected GFLOPs", _format_num(metrics["gflops"], 2), SERIES_COLORS[2]),
        ("Raw THOP", _format_num(metrics["raw_gflops"], 2) if metrics["raw_gflops"] else "N/A", SERIES_COLORS[4]),
        ("CPU Avg", f"{metrics['cpu_avg_ms']:.2f} ms" if metrics["cpu_avg_ms"] is not None else "Skipped", SERIES_COLORS[3]),
        ("Delta / Depth", delta_text, SERIES_COLORS[5] if baseline_delta else SERIES_COLORS[6]),
    ]
    for idx, (title, value, accent) in enumerate(cards):
        _draw_metric_card(ax_header, 0.01 + idx * 0.164, 0.08, 0.152, 0.42, title, value, accent)

    _draw_panel(ax_flow, "Architecture Ribbon", SERIES_COLORS[2])
    flow_steps = [
        ("Input", f"3 x {metrics['imgsz']}"),
        ("Backbone", f"{arch['backbone_layer_count']} YAML"),
        ("DINO" if dino["has_dino"] else "Enhancer", dino["modules"][0]["variant"] if dino["has_dino"] else "none"),
        ("Head", f"{arch['head_layer_count']} YAML"),
        ("Output", f"{arch['output_head']} x {arch['output_scales'] or 'N/A'}"),
    ]
    step_y = 0.47
    step_w = 0.16
    for idx, (title, desc) in enumerate(flow_steps):
        x = 0.03 + idx * 0.19
        fill = THEME["annotation_bg"] if idx % 2 == 0 else THEME["warm_fill"]
        pill = patches.FancyBboxPatch(
            (x, step_y),
            step_w,
            0.24,
            boxstyle="round,pad=0.018,rounding_size=0.05",
            facecolor=fill,
            edgecolor=SERIES_COLORS[(idx + 1) % len(SERIES_COLORS)],
            linewidth=1.5,
        )
        ax_flow.add_patch(pill)
        ax_flow.text(x + step_w / 2, step_y + 0.16, title, fontsize=11.6, fontweight="bold", color=THEME["text"], ha="center", va="center")
        ax_flow.text(x + step_w / 2, step_y + 0.07, desc, fontsize=9.6, color=THEME["muted_text"], ha="center", va="center")
        if idx < len(flow_steps) - 1:
            ax_flow.annotate(
                "",
                xy=(x + step_w + 0.02, step_y + 0.12),
                xytext=(x + step_w, step_y + 0.12),
                arrowprops=dict(arrowstyle="->", color=THEME["spine"], lw=2.0),
            )
    note_text = (
        "DINO path: YOLO feature -> 224 resize -> DINO tokens -> adapter -> resize back -> fusion"
        if dino["has_dino"]
        else "Pure YOLO path: CNN backbone -> neck/head -> output without DINO injection"
    )
    ax_flow.text(
        0.03,
        0.24,
        note_text,
        fontsize=10.0,
        color=THEME["text"],
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.28", facecolor=THEME["annotation_bg"], edgecolor=THEME["annotation_edge"]),
    )

    _draw_panel(ax_perf, "Performance Snapshot", SERIES_COLORS[1])
    rows = [
        ("Corrected GFLOPs", metrics["gflops"], baseline_metrics["gflops"] if baseline_metrics else None, SERIES_COLORS[2], ""),
        ("Parameters (M)", metrics["params"] / 1e6, baseline_metrics["params"] / 1e6 if baseline_metrics else None, SERIES_COLORS[1], "M"),
        ("CPU Latency (ms)", metrics["cpu_avg_ms"], baseline_metrics["cpu_avg_ms"] if baseline_metrics else None, SERIES_COLORS[3], "ms"),
    ]
    y_positions = [0.73, 0.46, 0.19]
    for (label, current, baseline, accent, unit), y in zip(rows, y_positions):
        values = [v for v in (current, baseline) if v is not None]
        row_max = max(values) if values else 1.0
        row_max = max(row_max, 1e-6)
        ax_perf.text(0.04, y + 0.10, label, fontsize=10.8, fontweight="bold", color=THEME["text"], ha="left", va="center")
        ax_perf.add_patch(patches.Rectangle((0.04, y + 0.01), 0.78, 0.045, color=THEME["grid"], alpha=0.35))
        current_w = 0.78 * (current / row_max) if current is not None else 0.0
        ax_perf.add_patch(patches.Rectangle((0.04, y + 0.01), current_w, 0.045, color=accent, alpha=0.92))
        ax_perf.text(0.04, y - 0.02, "Current", fontsize=9.3, color=THEME["muted_text"], ha="left", va="center")
        ax_perf.text(0.84, y + 0.032, f"{current:.2f}{unit}", fontsize=10.1, color=THEME["text"], ha="left", va="center")
        if baseline is not None:
            ax_perf.add_patch(patches.Rectangle((0.04, y - 0.08), 0.78, 0.026, color=THEME["grid"], alpha=0.22))
            base_w = 0.78 * (baseline / row_max)
            ax_perf.add_patch(patches.Rectangle((0.04, y - 0.08), base_w, 0.026, color=SERIES_COLORS[7], alpha=0.95))
            ax_perf.text(0.04, y - 0.12, "Baseline", fontsize=9.2, color=THEME["muted_text"], ha="left", va="center")
            ax_perf.text(0.84, y - 0.067, f"{baseline:.2f}{unit}", fontsize=9.5, color=THEME["muted_text"], ha="left", va="center")

    _draw_panel(ax_layers, "Selected Layers", SERIES_COLORS[6])
    layer_lines = arch["selected_layers"][:9] if arch["selected_layers"] else ["N/A"]
    for idx, line in enumerate(layer_lines):
        y = 0.84 - idx * 0.085
        ax_layers.text(0.04, y, line, fontsize=10.2, family="monospace", color=THEME["text"], ha="left", va="center")
    ax_layers.text(
        0.04,
        0.08,
        f"Top-level layers: {arch['layer_count']}   |   Leaf layers: {arch['leaf_layer_count']}",
        fontsize=10.0,
        color=THEME["muted_text"],
        ha="left",
        va="center",
    )

    _draw_panel(ax_dino, "DINO / Runtime Notes", SERIES_COLORS[5])
    notes = []
    if dino["has_dino"]:
        for idx, item in enumerate(dino["modules"], 1):
            notes.append(
                f"DINO-{idx}: {item['variant']} | freeze={item['freeze_backbone']} | embed={item['embed_dim']} | patch={item['patch_size']} | out={item['output_channels']}"
            )
            if item["input_shape"]:
                c, h, w = item["input_shape"]
                notes.append(f"input feature: C={c}, H={h}, W={w}")
            notes.append(f"est. DINO GFLOPs={_format_num(item['estimated_gflops'], 2)} | block params={_format_m(item['module_params'])}")
        notes.append(f"Total DINO-related GFLOPs: {_format_num(dino['estimated_total_dino_gflops'], 2)}")
        notes.append("Use corrected local DINO counter for interpretation.")
    else:
        notes.append("No DINO module detected in this experiment.")
        notes.append("Corrected GFLOPs and raw THOP are expected to match closely.")
    if baseline_delta:
        notes.append("")
        notes.append(f"Matched baseline: {baseline_delta['baseline_name']}")
        notes.append(f"Param delta: {_format_m(baseline_delta['params_delta'])}")
        notes.append(f"GFLOPs delta: {_format_num(baseline_delta['gflops_delta'], 2)}")
        if baseline_delta["cpu_ms_delta"] is not None:
            notes.append(f"CPU delta: {baseline_delta['cpu_ms_delta']:+.2f} ms")
    ax_dino.text(0.04, 0.84, "\n".join(notes), fontsize=10.1, color=THEME["text"], ha="left", va="top")

    fig.suptitle("Model Oneflow Analysis", fontsize=17, fontweight="bold", y=0.992, color=THEME["text"])
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=THEME["figure_bg"])
    plt.close(fig)


def ensure_output_dir(experiment: ExperimentRef) -> Path:
    output_dir = RESULT_ROOT / experiment.name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按 train_* 实验生成模型一图流和 modelAnalyse.txt。")
    parser.add_argument("--experiment", "-e", type=str, help="实验名或实验目录路径，例如 Ex_12s_960_WassersteinLoss0.7_DINOP2")
    parser.add_argument("--workspace", "-w", type=str, default=str(WORKSPACE_ROOT), help="工作区根目录，默认当前仓库根目录。")
    parser.add_argument("--cpu-warmup", type=int, default=5, help="CPU 推理预热轮数。")
    parser.add_argument("--cpu-iters", type=int, default=20, help="CPU 平均推理统计轮数。")
    parser.add_argument("--skip-cpu", action="store_true", help="跳过 CPU 平均推理耗时统计。")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    experiment = resolve_experiment(args.experiment, workspace)
    output_dir = ensure_output_dir(experiment)

    metrics = analyze_experiment(
        experiment,
        workspace=workspace,
        cpu_warmup=args.cpu_warmup,
        cpu_iters=args.cpu_iters,
        skip_cpu=args.skip_cpu,
    )

    report_path = output_dir / "modelAnalyse.txt"
    figure_path = output_dir / "model_oneflow.png"

    report_text = compose_report_text(metrics)
    report_path.write_text(report_text, encoding="utf-8")
    draw_overview_figure(metrics, figure_path)

    print(f"分析完成: {experiment.name}")
    print(f"文本报告: {report_path}")
    print(f"一图流: {figure_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("已中断。", file=sys.stderr)
        raise
