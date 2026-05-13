#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import gc
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pcbfinal_compression.common import (  # noqa: E402
    ensure_dir,
    format_bytes,
    infer_experiment_name,
    parse_csv_list,
    safe_relpath,
    set_runtime_env,
    setup_logger,
    stage_header,
    timestamp_now,
    unique_sorted,
    write_json,
)

set_runtime_env()

import onnxruntime as ort  # noqa: E402
from pcbfinal_compression.dataset_utils import collect_split_images, sample_evenly  # noqa: E402
from pcbfinal_compression.evaluation import (  # noqa: E402
    compute_artifact_stats,
    compute_retention,
    evaluate_artifact,
    infer_imgsz_from_args,
    infer_task_from_model,
    metric_keys_for_task,
    save_prediction_samples,
)
from pcbfinal_compression.pruning import prune_checkpoint  # noqa: E402
from pcbfinal_compression.quantization import (  # noqa: E402
    QUANT_PROFILE_DESCRIPTIONS,
    create_dynamic_int8_linear_mixed,
    create_fp16_onnx,
    create_static_int8_conv_mixed,
    export_fp32_onnx,
)
from pcbfinal_compression.reporting import (  # noqa: E402
    plot_summary_charts,
    save_records_csv_json,
    write_compression_report,
    write_root_overview,
)


DEFAULT_PRUNE_RATIOS = "0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60"
DEFAULT_QUANT_PROFILES = "dynamic_int8_linear_mixed"
DEFAULT_MIN_RETENTION = 0.95


@dataclass
class OutputLayout:
    root_dir: Path
    output_dir: Path
    models_dir: Path
    baseline_dir: Path
    fp32_dir: Path
    quant_dir: Path
    prune_dir: Path
    final_dir: Path
    charts_dir: Path
    temp_dir: Path
    run_log_path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PCBFINAL mixed-precision quantization and pruning study.")
    parser.add_argument("--model", type=Path, required=True, help="Input .pt checkpoint.")
    parser.add_argument("--data", type=Path, required=True, help="Dataset YAML path.")
    parser.add_argument("--task", type=str, default="auto", choices=["auto", "detect", "segment"], help="Task type.")
    parser.add_argument("--imgsz", type=int, default=None, help="Validation/export image size.")
    parser.add_argument("--device", type=str, default="cpu", help="Preferred evaluation device.")
    parser.add_argument("--val-batch", type=int, default=4, help="Validation batch size for PyTorch artifacts.")
    parser.add_argument("--calib-images", type=int, default=32, help="Calibration image count for static INT8.")
    parser.add_argument("--sample-images", type=int, default=6, help="Sample visualization count.")
    parser.add_argument(
        "--min-retention",
        type=float,
        default=DEFAULT_MIN_RETENTION,
        help="Minimum retention threshold applied to both mAP50 and mAP50-95.",
    )
    parser.add_argument(
        "--quant-profiles",
        type=str,
        default=DEFAULT_QUANT_PROFILES,
        help="Comma-separated quant profiles.",
    )
    parser.add_argument(
        "--prune-ratios",
        type=str,
        default=DEFAULT_PRUNE_RATIOS,
        help="Comma-separated prune ratios.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("PCBFINAL"),
        help="Root directory for study outputs.",
    )
    parser.add_argument(
        "--archive-layout",
        type=str,
        default="pcbfinal",
        choices=["pcbfinal", "timestamped"],
        help="Output layout style.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="When using pcbfinal layout, clear the previous experiment folder and archived models first.",
    )
    return parser


def resolve_task(task_arg: str, model_path: Path) -> str:
    return infer_task_from_model(model_path) if task_arg == "auto" else task_arg


def resolve_common_eval_device(requested_device: str, quant_profiles: list[str], logger) -> str:
    if not quant_profiles:
        logger.info("No quantization profiles enabled, using requested eval device: %s", requested_device)
        return requested_device
    ort_providers = set(ort.get_available_providers())
    if requested_device != "cpu" and "CUDAExecutionProvider" in ort_providers:
        logger.info("ORT CUDAExecutionProvider is available, using requested eval device: %s", requested_device)
        return requested_device
    if requested_device != "cpu":
        logger.info("ORT only exposes CPUExecutionProvider in this environment, fallback eval device: cpu")
    return "cpu"


def sanitize_token(text: str) -> str:
    cleaned = []
    for char in text:
        if char.isalnum() or char in {"_", "-"}:
            cleaned.append(char)
        elif char == ".":
            cleaned.append("p")
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_")


def model_archive_name(task: str, experiment_name: str, stage_name: str, suffix: str) -> str:
    return f"{sanitize_token(task)}__{sanitize_token(experiment_name)}__{sanitize_token(stage_name)}{suffix}"


def replace_record_artifact_path(record: dict, artifact_path: Path) -> None:
    record["artifact_path"] = str(artifact_path.resolve())
    record["artifact_suffix"] = artifact_path.suffix.lower()


def copy_model_artifact(source_path: Path, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() != target_path.resolve():
        shutil.copy2(source_path, target_path)
    return target_path


def make_model_target(layout: OutputLayout, task: str, experiment_name: str, stage_name: str, suffix: str) -> Path:
    return layout.models_dir / model_archive_name(task, experiment_name, stage_name, suffix)


def clear_matching_archived_models(models_dir: Path, task: str, experiment_name: str) -> None:
    pattern = f"{sanitize_token(task)}__{sanitize_token(experiment_name)}__*"
    for path in models_dir.glob(pattern):
        if path.is_file():
            path.unlink()


def build_output_layout(
    output_root: Path,
    archive_layout: str,
    overwrite_output: bool,
    task: str,
    experiment_name: str,
) -> OutputLayout:
    root_dir = ensure_dir(output_root.resolve())
    if archive_layout == "pcbfinal":
        models_dir = ensure_dir(root_dir / "models")
        output_dir = root_dir / f"{sanitize_token(task)}__{sanitize_token(experiment_name)}"
        if overwrite_output and output_dir.exists():
            shutil.rmtree(output_dir)
        if overwrite_output:
            clear_matching_archived_models(models_dir, task, experiment_name)
        output_dir = ensure_dir(output_dir)
    else:
        output_dir = ensure_dir(root_dir / f"{sanitize_token(experiment_name)}_{timestamp_now()}")
        models_dir = ensure_dir(output_dir / "artifacts")

    baseline_dir = ensure_dir(output_dir / "baseline")
    fp32_dir = ensure_dir(output_dir / "fp32_onnx")
    quant_dir = ensure_dir(output_dir / "quantization")
    prune_dir = ensure_dir(output_dir / "pruning")
    final_dir = ensure_dir(output_dir / "final")
    charts_dir = ensure_dir(final_dir / "charts")
    temp_dir = ensure_dir(output_dir / "_temp")
    run_log_path = output_dir / "run.log"
    return OutputLayout(
        root_dir=root_dir,
        output_dir=output_dir,
        models_dir=models_dir,
        baseline_dir=baseline_dir,
        fp32_dir=fp32_dir,
        quant_dir=quant_dir,
        prune_dir=prune_dir,
        final_dir=final_dir,
        charts_dir=charts_dir,
        temp_dir=temp_dir,
        run_log_path=run_log_path,
    )


def build_record(
    family: str,
    candidate_name: str,
    artifact_path: Path,
    task: str,
    status: str,
    stats,
    metrics: dict[str, float] | None,
    speed: dict[str, float] | None,
    baseline_metrics: dict[str, float] | None,
    min_retention: float,
    notes: str = "",
    prune_ratio: float | None = None,
) -> dict:
    metric_primary_key, metric_secondary_key = metric_keys_for_task(task)
    metrics = metrics or {}
    speed = speed or {}
    retention_primary = None
    retention_secondary = None
    if baseline_metrics and metrics and family != "baseline":
        retention_primary, retention_secondary, passed = compute_retention(task, metrics, baseline_metrics, min_retention)
        status = "PASS" if passed else "FAIL"
    primary_threshold = baseline_metrics.get(metric_primary_key, 0.0) * min_retention if baseline_metrics else None
    secondary_threshold = baseline_metrics.get(metric_secondary_key, 0.0) * min_retention if baseline_metrics else None
    record = {
        "family": family,
        "candidate_name": candidate_name,
        "artifact_path": str(artifact_path.resolve()),
        "artifact_suffix": artifact_path.suffix.lower(),
        "task": task,
        "status": status,
        "notes": notes,
        "checkpoint_bytes": stats.checkpoint_bytes,
        "parameter_count": stats.parameter_count,
        "nonzero_count": stats.nonzero_count,
        "nonzero_ratio": stats.nonzero_ratio,
        "effective_nonzero_bytes": stats.effective_nonzero_bytes,
        "retention_primary": retention_primary,
        "retention_secondary": retention_secondary,
        "metric_primary_key": metric_primary_key,
        "metric_secondary_key": metric_secondary_key,
        "primary_threshold": primary_threshold,
        "secondary_threshold": secondary_threshold,
        "speed_preprocess_ms": speed.get("preprocess"),
        "speed_inference_ms": speed.get("inference"),
        "speed_postprocess_ms": speed.get("postprocess"),
        "prune_ratio": prune_ratio,
    }
    record.update(metrics)
    return record


def evaluate_candidate(
    family: str,
    candidate_name: str,
    artifact_path: Path,
    task: str,
    data_yaml: Path,
    imgsz: int,
    device: str,
    batch: int,
    baseline_metrics: dict[str, float] | None,
    min_retention: float,
    logger,
    notes: str = "",
    prune_ratio: float | None = None,
) -> dict:
    stats = compute_artifact_stats(artifact_path)
    metrics, speed = evaluate_artifact(
        artifact_path=artifact_path,
        task=task,
        data_yaml=data_yaml,
        imgsz=imgsz,
        device=device,
        batch=batch,
    )
    record = build_record(
        family=family,
        candidate_name=candidate_name,
        artifact_path=artifact_path,
        task=task,
        status="baseline" if family == "baseline" else "evaluated",
        stats=stats,
        metrics=metrics,
        speed=speed,
        baseline_metrics=baseline_metrics,
        min_retention=min_retention,
        notes=notes,
        prune_ratio=prune_ratio,
    )
    logger.info(
        "%s | status=%s | size=%s | %s=%.6f | %s=%.6f | inference=%.3f ms",
        candidate_name,
        record["status"],
        format_bytes(record["checkpoint_bytes"]),
        record["metric_primary_key"],
        record.get(record["metric_primary_key"], 0.0),
        record["metric_secondary_key"],
        record.get(record["metric_secondary_key"], 0.0),
        record.get("speed_inference_ms") or 0.0,
    )
    return record


def choose_best_quant(quant_records: list[dict]) -> dict | None:
    passed = [row for row in quant_records if row["status"] == "PASS"]
    if not passed:
        return None
    return sorted(
        passed,
        key=lambda row: (
            row["checkpoint_bytes"],
            row.get("speed_inference_ms") or float("inf"),
            -((row.get("retention_primary") or 0.0) + (row.get("retention_secondary") or 0.0)) / 2.0,
        ),
    )[0]


def choose_best_prune(prune_records: list[dict]) -> dict | None:
    passed = [row for row in prune_records if row["status"] == "PASS"]
    if not passed:
        return None
    return sorted(
        passed,
        key=lambda row: (
            -(row.get("prune_ratio") or 0.0),
            row.get("effective_nonzero_bytes") or float("inf"),
            row.get("speed_inference_ms") or float("inf"),
        ),
    )[0]


def choose_final_candidate(fp32_record: dict | None, quant_records: list[dict], prune_records: list[dict]) -> dict | None:
    passed_candidates = [
        row
        for row in [*quant_records, *prune_records]
        if row["status"] == "PASS" and row.get("artifact_path") and Path(str(row["artifact_path"])).exists()
    ]
    if passed_candidates:
        return sorted(
            passed_candidates,
            key=lambda row: (
                row["checkpoint_bytes"],
                row.get("speed_inference_ms") or float("inf"),
                -((row.get("retention_primary") or 0.0) + (row.get("retention_secondary") or 0.0)) / 2.0,
            ),
        )[0]
    if fp32_record and fp32_record["status"] == "PASS" and fp32_record.get("artifact_path"):
        return fp32_record
    return None


def build_fine_prune_ratios(prune_records: list[dict]) -> list[float]:
    passed = [row for row in prune_records if row["status"] == "PASS" and row.get("prune_ratio") is not None]
    if not passed:
        return []
    highest_pass = max(float(row["prune_ratio"]) for row in passed)
    nearby = [highest_pass + offset for offset in (-0.04, -0.02, 0.02, 0.04, 0.06)]
    existing = {round(float(row["prune_ratio"]), 4) for row in prune_records if row.get("prune_ratio") is not None}
    fine = [round(value, 4) for value in nearby if 0.01 <= value <= 0.95 and round(value, 4) not in existing]
    return unique_sorted(fine)


def save_stage_validation(stage_dir: Path, record: dict) -> None:
    save_records_csv_json([record], stage_dir / "validation.csv", stage_dir / "validation.json")


def cleanup_prune_temps(prune_records: list[dict], best_prune_record: dict | None, temp_dir: Path) -> None:
    best_path = Path(best_prune_record["artifact_path"]).resolve() if best_prune_record else None
    for row in prune_records:
        artifact_value = row.get("artifact_path")
        if not artifact_value:
            continue
        artifact_path = Path(artifact_value)
        if best_path is not None and artifact_path.resolve() == best_path:
            continue
        if artifact_path.exists():
            try:
                artifact_path.resolve().relative_to(temp_dir.resolve())
            except ValueError:
                continue
            artifact_path.unlink()
            row["artifact_path"] = ""
            row["artifact_suffix"] = artifact_path.suffix.lower()
            row["notes"] = (row.get("notes") or "") + "; temporary sweep artifact cleaned"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def main() -> None:
    args = build_parser().parse_args()
    model_path = args.model.resolve()
    data_yaml = args.data.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model does not exist: {model_path}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML does not exist: {data_yaml}")

    task = resolve_task(args.task, model_path)
    imgsz = args.imgsz or infer_imgsz_from_args(model_path)
    experiment_name = infer_experiment_name(model_path)
    layout = build_output_layout(
        output_root=args.output_root,
        archive_layout=args.archive_layout,
        overwrite_output=args.overwrite_output,
        task=task,
        experiment_name=experiment_name,
    )
    logger = setup_logger(layout.run_log_path)

    quant_profiles = [value for value in parse_csv_list(args.quant_profiles, cast=str) if value]
    prune_ratios = unique_sorted(parse_csv_list(args.prune_ratios, cast=float))
    eval_device = resolve_common_eval_device(args.device, quant_profiles, logger)

    stage_header(logger, "PCBFINAL Compression Study")
    logger.info("workspace        : %s", REPO_ROOT)
    logger.info("model            : %s", model_path)
    logger.info("data             : %s", data_yaml)
    logger.info("task             : %s", task)
    logger.info("imgsz            : %s", imgsz)
    logger.info("requested device : %s", args.device)
    logger.info("eval device      : %s", eval_device)
    logger.info("quant profiles   : %s", quant_profiles)
    logger.info("prune ratios     : %s", prune_ratios)
    logger.info("retention gate   : %.2f", args.min_retention)
    logger.info("archive layout   : %s", args.archive_layout)
    logger.info("output root      : %s", layout.root_dir)
    logger.info("experiment dir   : %s", layout.output_dir)
    logger.info("models dir       : %s", layout.models_dir)

    calibration_images = sample_evenly(collect_split_images(data_yaml, "train"), args.calib_images)
    sample_images = sample_evenly(collect_split_images(data_yaml, "val"), args.sample_images)
    logger.info("calibration images: %d", len(calibration_images))
    logger.info("sample images     : %d", len(sample_images))

    stage_header(logger, "[1/6] Baseline Evaluation")
    baseline_record = evaluate_candidate(
        family="baseline",
        candidate_name="baseline_pt",
        artifact_path=model_path,
        task=task,
        data_yaml=data_yaml,
        imgsz=imgsz,
        device=eval_device,
        batch=args.val_batch,
        baseline_metrics=None,
        min_retention=args.min_retention,
        logger=logger,
        notes="Original PyTorch checkpoint",
    )
    baseline_model_path = copy_model_artifact(
        model_path,
        make_model_target(layout, task, experiment_name, "baseline_pt", model_path.suffix.lower()),
    )
    replace_record_artifact_path(baseline_record, baseline_model_path)
    baseline_samples = save_prediction_samples(
        artifact_path=baseline_model_path,
        task=task,
        sample_paths=sample_images,
        output_dir=layout.baseline_dir / "samples",
        imgsz=imgsz,
        device=eval_device,
    )
    save_stage_validation(layout.baseline_dir, baseline_record)

    stage_header(logger, "[2/6] FP32 ONNX Export And Validation")
    fp32_onnx_path = export_fp32_onnx(
        model_path=model_path,
        task=task,
        imgsz=imgsz,
        target_path=make_model_target(layout, task, experiment_name, "fp32_onnx", ".onnx"),
        logger=logger,
    )
    fp32_record = evaluate_candidate(
        family="export",
        candidate_name="fp32_onnx",
        artifact_path=fp32_onnx_path,
        task=task,
        data_yaml=data_yaml,
        imgsz=imgsz,
        device=eval_device,
        batch=1,
        baseline_metrics=baseline_record,
        min_retention=args.min_retention,
        logger=logger,
        notes="Exported FP32 ONNX baseline",
    )
    fp32_samples = save_prediction_samples(
        artifact_path=fp32_onnx_path,
        task=task,
        sample_paths=sample_images[: min(3, len(sample_images))],
        output_dir=layout.fp32_dir / "samples",
        imgsz=imgsz,
        device=eval_device,
    )
    save_stage_validation(layout.fp32_dir, fp32_record)

    stage_header(logger, "[3/6] Quantization Profiles")
    quant_records: list[dict] = []
    quant_sample_sections: dict[str, list[Path]] = {}
    if quant_profiles:
        quant_builders = {
            "fp16_onnx": lambda target: create_fp16_onnx(fp32_onnx_path, target, logger),
            "dynamic_int8_linear_mixed": lambda target: create_dynamic_int8_linear_mixed(fp32_onnx_path, target, logger),
            "static_int8_conv_mixed": lambda target: create_static_int8_conv_mixed(
                fp32_onnx_path,
                target,
                calibration_images,
                imgsz,
                logger,
            ),
        }
        for profile in quant_profiles:
            candidate_stage_dir = ensure_dir(layout.quant_dir / profile)
            candidate_path = make_model_target(layout, task, experiment_name, profile, ".onnx")
            try:
                if profile not in quant_builders:
                    raise RuntimeError(f"Unknown quantization profile: {profile}")
                quant_builders[profile](candidate_path)
                record = evaluate_candidate(
                    family="quant",
                    candidate_name=profile,
                    artifact_path=candidate_path,
                    task=task,
                    data_yaml=data_yaml,
                    imgsz=imgsz,
                    device=eval_device,
                    batch=1,
                    baseline_metrics=baseline_record,
                    min_retention=args.min_retention,
                    logger=logger,
                    notes=QUANT_PROFILE_DESCRIPTIONS.get(profile, ""),
                )
                quant_records.append(record)
                save_stage_validation(candidate_stage_dir, record)
                try:
                    quant_sample_sections[profile] = save_prediction_samples(
                        artifact_path=candidate_path,
                        task=task,
                        sample_paths=sample_images[: min(3, len(sample_images))],
                        output_dir=candidate_stage_dir / "samples",
                        imgsz=imgsz,
                        device=eval_device,
                    )
                except Exception as sample_exc:
                    logger.warning("Quantization samples export failed for %s | %s", profile, sample_exc)
            except Exception as exc:
                logger.exception("Quantization profile failed: %s", profile)
                failed_record = build_record(
                    family="quant",
                    candidate_name=profile,
                    artifact_path=candidate_path,
                    task=task,
                    status="unsupported",
                    stats=type(
                        "Stats",
                        (),
                        {
                            "checkpoint_bytes": candidate_path.stat().st_size if candidate_path.exists() else 0,
                            "parameter_count": None,
                            "nonzero_count": None,
                            "nonzero_ratio": None,
                            "effective_nonzero_bytes": None,
                        },
                    )(),
                    metrics=None,
                    speed=None,
                    baseline_metrics=baseline_record,
                    min_retention=args.min_retention,
                    notes=str(exc),
                )
                quant_records.append(failed_record)
                save_stage_validation(candidate_stage_dir, failed_record)
            finally:
                gc.collect()

    stage_header(logger, "[4/6] Pruning Sweep")
    prune_records: list[dict] = []
    prune_temp_dir = ensure_dir(layout.temp_dir / "prune")
    for ratio in prune_ratios:
        candidate_path = prune_temp_dir / f"{sanitize_token(model_path.stem)}_prune_{ratio:.2f}.pt"
        prune_info = prune_checkpoint(
            source_model_path=model_path,
            target_model_path=candidate_path,
            amount=ratio,
            task=task,
            logger=logger,
        )
        record = evaluate_candidate(
            family="prune",
            candidate_name=f"prune_{ratio:.2f}",
            artifact_path=candidate_path,
            task=task,
            data_yaml=data_yaml,
            imgsz=imgsz,
            device=eval_device,
            batch=args.val_batch,
            baseline_metrics=baseline_record,
            min_retention=args.min_retention,
            logger=logger,
            notes=f"global L1 mask pruning on {prune_info['prunable_module_count']} modules",
            prune_ratio=ratio,
        )
        prune_records.append(record)
        gc.collect()

    fine_prune_ratios = build_fine_prune_ratios(prune_records)
    if fine_prune_ratios:
        logger.info("Fine prune sweep ratios: %s", fine_prune_ratios)
        for ratio in fine_prune_ratios:
            candidate_path = prune_temp_dir / f"{sanitize_token(model_path.stem)}_prune_{ratio:.2f}.pt"
            prune_info = prune_checkpoint(
                source_model_path=model_path,
                target_model_path=candidate_path,
                amount=ratio,
                task=task,
                logger=logger,
            )
            record = evaluate_candidate(
                family="prune",
                candidate_name=f"prune_{ratio:.2f}",
                artifact_path=candidate_path,
                task=task,
                data_yaml=data_yaml,
                imgsz=imgsz,
                device=eval_device,
                batch=args.val_batch,
                baseline_metrics=baseline_record,
                min_retention=args.min_retention,
                logger=logger,
                notes=f"fine sweep around best pass, modules={prune_info['prunable_module_count']}",
                prune_ratio=ratio,
            )
            prune_records.append(record)
            gc.collect()

    best_quant_record = choose_best_quant(quant_records)
    best_prune_record = choose_best_prune(prune_records)
    best_prune_samples: list[Path] = []
    if best_prune_record:
        best_prune_source = Path(best_prune_record["artifact_path"])
        best_prune_archived = copy_model_artifact(
            best_prune_source,
            make_model_target(
                layout,
                task,
                experiment_name,
                f"best_prune_{best_prune_record['prune_ratio']:.2f}",
                best_prune_source.suffix.lower(),
            ),
        )
        replace_record_artifact_path(best_prune_record, best_prune_archived)
        best_prune_stage = ensure_dir(layout.prune_dir / "best_prune")
        save_stage_validation(best_prune_stage, best_prune_record)
        best_prune_samples = save_prediction_samples(
            artifact_path=best_prune_archived,
            task=task,
            sample_paths=sample_images,
            output_dir=best_prune_stage / "samples",
            imgsz=imgsz,
            device=eval_device,
        )
    cleanup_prune_temps(prune_records, best_prune_record, layout.temp_dir)

    final_best_record = choose_final_candidate(fp32_record, quant_records, prune_records)
    final_record: dict | None = None
    final_samples: list[Path] = []
    if final_best_record and final_best_record.get("artifact_path"):
        final_source = Path(final_best_record["artifact_path"])
        if final_source.exists():
            final_target = copy_model_artifact(
                final_source,
                make_model_target(layout, task, experiment_name, "final_compressed", final_source.suffix.lower()),
            )
            final_record = dict(final_best_record)
            final_record["candidate_name"] = "final_compressed"
            replace_record_artifact_path(final_record, final_target)
            save_stage_validation(layout.final_dir, final_record)
            final_samples = save_prediction_samples(
                artifact_path=final_target,
                task=task,
                sample_paths=sample_images[: min(3, len(sample_images))],
                output_dir=layout.final_dir / "samples",
                imgsz=imgsz,
                device=eval_device,
            )

    stage_header(logger, "[5/6] Reports And Charts")
    summary_records = [baseline_record, fp32_record, *quant_records, *prune_records]
    save_records_csv_json(summary_records, layout.output_dir / "summary.csv", layout.output_dir / "summary.json")
    save_records_csv_json(
        quant_records,
        layout.quant_dir / "quantization_candidates.csv",
        layout.quant_dir / "quantization_candidates.json",
    )
    save_records_csv_json(
        prune_records,
        layout.prune_dir / "pruning_sweep.csv",
        layout.prune_dir / "pruning_sweep.json",
    )

    comparison_records = [baseline_record, fp32_record]
    if best_quant_record:
        comparison_records.append(best_quant_record)
    if best_prune_record:
        comparison_records.append(best_prune_record)
    if final_record:
        comparison_records.append(final_record)
    save_records_csv_json(
        comparison_records,
        layout.final_dir / "source_vs_compressed.csv",
        layout.final_dir / "source_vs_compressed.json",
    )

    best_choice_payload = {
        "baseline": baseline_record,
        "fp32_onnx": fp32_record,
        "best_quantization": best_quant_record,
        "best_pruning": best_prune_record,
        "final_compressed": final_record,
    }
    write_json(layout.final_dir / "best_choice.json", best_choice_payload)

    chart_paths = plot_summary_charts(summary_records, task, layout.charts_dir)
    sample_sections = {
        "baseline": baseline_samples,
        "fp32_onnx": fp32_samples,
        "best_prune": best_prune_samples,
        "final": final_samples,
    }
    sample_sections.update({name: paths for name, paths in quant_sample_sections.items() if paths})
    write_compression_report(
        report_path=layout.final_dir / "compression_report.md",
        workspace=REPO_ROOT,
        task=task,
        baseline_record=baseline_record,
        fp32_record=fp32_record,
        quant_records=quant_records,
        prune_records=prune_records,
        best_quant_record=best_quant_record,
        best_prune_record=best_prune_record,
        final_best_record=final_record,
        min_retention=args.min_retention,
        chart_paths=chart_paths,
        sample_sections=sample_sections,
    )
    overview_paths = write_root_overview(layout.root_dir, REPO_ROOT)

    stage_header(logger, "[6/6] Final Summary")
    logger.info("baseline validation    : %s", safe_relpath(layout.baseline_dir / "validation.csv", REPO_ROOT))
    logger.info("fp32 onnx validation   : %s", safe_relpath(layout.fp32_dir / "validation.csv", REPO_ROOT))
    logger.info("summary.csv            : %s", safe_relpath(layout.output_dir / "summary.csv", REPO_ROOT))
    logger.info(
        "quantization candidates: %s",
        safe_relpath(layout.quant_dir / "quantization_candidates.csv", REPO_ROOT),
    )
    logger.info("pruning sweep          : %s", safe_relpath(layout.prune_dir / "pruning_sweep.csv", REPO_ROOT))
    logger.info("source vs compressed   : %s", safe_relpath(layout.final_dir / "source_vs_compressed.csv", REPO_ROOT))
    logger.info("compression report     : %s", safe_relpath(layout.final_dir / "compression_report.md", REPO_ROOT))
    for overview_path in overview_paths:
        logger.info("root overview          : %s", safe_relpath(overview_path, REPO_ROOT))
    if best_quant_record:
        logger.info("best quantization      : %s (%s)", best_quant_record["candidate_name"], best_quant_record["status"])
    else:
        logger.info("best quantization      : none")
    if best_prune_record:
        logger.info("best pruning           : %s (%s)", best_prune_record["candidate_name"], best_prune_record["status"])
    else:
        logger.info("best pruning           : none")
    if final_record:
        logger.info("final compressed       : %s", safe_relpath(Path(final_record["artifact_path"]), REPO_ROOT))
    else:
        logger.info("final compressed       : none")


if __name__ == "__main__":
    main()
