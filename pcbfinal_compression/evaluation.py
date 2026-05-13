from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnx
import torch
from onnx import numpy_helper
from ultralytics import YOLO


@dataclass
class ArtifactStats:
    checkpoint_bytes: int
    parameter_count: int | None
    nonzero_count: int | None
    nonzero_ratio: float | None
    effective_nonzero_bytes: int | None


def infer_task_from_model(model_path: Path) -> str:
    return YOLO(str(model_path)).task


def infer_imgsz_from_args(model_path: Path, fallback: int = 640) -> int:
    import yaml

    args_path = model_path.parent.parent / "args.yaml" if model_path.parent.name == "weights" else None
    if args_path and args_path.exists():
        data = yaml.safe_load(args_path.read_text(encoding="utf-8"))
        imgsz = data.get("imgsz")
        if isinstance(imgsz, (list, tuple)) and imgsz:
            return int(imgsz[0])
        if isinstance(imgsz, (int, float)):
            return int(imgsz)
    return fallback


def metric_keys_for_task(task: str) -> tuple[str, str]:
    if task == "segment":
        return "metrics/mAP50(M)", "metrics/mAP50-95(M)"
    return "metrics/mAP50(B)", "metrics/mAP50-95(B)"


def _stored_pt_module(model_path: Path):
    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    module = checkpoint.get("ema") or checkpoint.get("model")
    if module is None:
        raise RuntimeError(f"未在 checkpoint 中找到 model/ema: {model_path}")
    module = module.float()
    module.eval()
    return module


def compute_pt_artifact_stats(model_path: Path) -> ArtifactStats:
    module = _stored_pt_module(model_path)
    parameter_count = 0
    nonzero_count = 0
    effective_nonzero_bytes = 0
    for param in module.parameters():
        parameter_count += int(param.numel())
        current_nonzero = int(torch.count_nonzero(param).item())
        nonzero_count += current_nonzero
        effective_nonzero_bytes += current_nonzero * int(param.element_size())
    nonzero_ratio = (nonzero_count / parameter_count) if parameter_count else None
    return ArtifactStats(
        checkpoint_bytes=model_path.stat().st_size,
        parameter_count=parameter_count,
        nonzero_count=nonzero_count,
        nonzero_ratio=nonzero_ratio,
        effective_nonzero_bytes=effective_nonzero_bytes,
    )


def compute_onnx_artifact_stats(model_path: Path) -> ArtifactStats:
    model = onnx.load(str(model_path))
    parameter_count = 0
    nonzero_count = 0
    effective_nonzero_bytes = 0
    for initializer in model.graph.initializer:
        array = numpy_helper.to_array(initializer)
        parameter_count += int(array.size)
        current_nonzero = int(np.count_nonzero(array))
        nonzero_count += current_nonzero
        effective_nonzero_bytes += current_nonzero * int(array.dtype.itemsize)
    nonzero_ratio = (nonzero_count / parameter_count) if parameter_count else None
    return ArtifactStats(
        checkpoint_bytes=model_path.stat().st_size,
        parameter_count=parameter_count or None,
        nonzero_count=nonzero_count or None,
        nonzero_ratio=nonzero_ratio,
        effective_nonzero_bytes=effective_nonzero_bytes or None,
    )


def compute_artifact_stats(model_path: Path) -> ArtifactStats:
    if model_path.suffix.lower() == ".onnx":
        return compute_onnx_artifact_stats(model_path)
    return compute_pt_artifact_stats(model_path)


def evaluate_artifact(
    artifact_path: Path,
    task: str,
    data_yaml: Path,
    imgsz: int,
    device: str,
    batch: int,
) -> tuple[dict[str, float], dict[str, float]]:
    model = YOLO(str(artifact_path), task=task)
    result = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=0,
        plots=False,
        verbose=False,
    )
    metrics = {key: float(value) for key, value in result.results_dict.items()}
    speed = {key: float(value) for key, value in result.speed.items()}
    return metrics, speed


def compute_retention(
    task: str,
    metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    min_retention: float,
) -> tuple[float | None, float | None, bool]:
    primary_key, secondary_key = metric_keys_for_task(task)
    baseline_primary = baseline_metrics.get(primary_key)
    baseline_secondary = baseline_metrics.get(secondary_key)
    primary_retention = None if not baseline_primary else metrics.get(primary_key, 0.0) / baseline_primary
    secondary_retention = None if not baseline_secondary else metrics.get(secondary_key, 0.0) / baseline_secondary
    is_pass = (
        primary_retention is not None
        and secondary_retention is not None
        and primary_retention >= min_retention
        and secondary_retention >= min_retention
    )
    return primary_retention, secondary_retention, is_pass


def save_prediction_samples(
    artifact_path: Path,
    task: str,
    sample_paths: list[Path],
    output_dir: Path,
    imgsz: int,
    device: str,
) -> list[Path]:
    if not sample_paths:
        return []
    model = YOLO(str(artifact_path), task=task)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for index, sample_path in enumerate(sample_paths, start=1):
        kwargs = {
            "source": str(sample_path),
            "imgsz": imgsz,
            "device": device,
            "conf": 0.25,
            "iou": 0.45,
            "verbose": False,
            "save": False,
        }
        if task == "segment":
            kwargs["retina_masks"] = True
        results = model.predict(**kwargs)
        if not results:
            continue
        result = results[0]
        if task == "segment":
            plotted = result.plot(boxes=False, labels=False, masks=True, conf=False)
        else:
            plotted = result.plot(boxes=True, labels=True, masks=True, conf=True)
        save_path = output_dir / f"{index:02d}_{Path(result.path).stem}.jpg"
        cv2.imwrite(str(save_path), cv2.cvtColor(plotted, cv2.COLOR_RGB2BGR))
        saved_paths.append(save_path)
    return saved_paths
