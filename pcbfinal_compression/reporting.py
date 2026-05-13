from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from chart_style import apply_unified_mpl_style
from pcbfinal_compression.common import format_bytes, safe_relpath
from pcbfinal_compression.evaluation import metric_keys_for_task


def dataframe_from_records(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    return frame.sort_values(by=["family", "candidate_name"], kind="stable").reset_index(drop=True)


def save_records_csv_json(records: list[dict], csv_path: Path, json_path: Path) -> None:
    frame = dataframe_from_records(records)
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(frame.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")


def plot_summary_charts(summary_records: list[dict], task: str, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not summary_records:
        return []
    theme = apply_unified_mpl_style()
    frame = dataframe_from_records(summary_records)
    metric_primary_key, metric_secondary_key = metric_keys_for_task(task)
    chart_paths: list[Path] = []

    candidate_frame = frame[frame["family"] != "baseline"].copy()
    candidate_frame["checkpoint_mb"] = candidate_frame["checkpoint_bytes"] / (1024 * 1024)
    candidate_frame["inference_ms"] = candidate_frame["speed_inference_ms"]
    candidate_frame["primary_metric"] = candidate_frame[metric_primary_key]
    candidate_frame["secondary_metric"] = candidate_frame[metric_secondary_key]
    candidate_frame["mean_retention"] = (
        candidate_frame["retention_primary"].fillna(0.0) + candidate_frame["retention_secondary"].fillna(0.0)
    ) / 2.0

    def _scatter(x_col: str, y_col: str, title: str, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(8.6, 5.4))
        for family, family_frame in candidate_frame.groupby("family"):
            ax.scatter(family_frame[x_col], family_frame[y_col], label=family, s=90, alpha=0.95)
            for _, row in family_frame.iterrows():
                ax.annotate(
                    row["candidate_name"],
                    (row[x_col], row[y_col]),
                    fontsize=9,
                    xytext=(5, 5),
                    textcoords="offset points",
                    color=theme["text"],
                )
        ax.set_title(title)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.35)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)

    size_vs_primary = output_dir / "size_vs_primary_metric.png"
    _scatter("checkpoint_mb", "primary_metric", f"Checkpoint Size vs {metric_primary_key}", size_vs_primary)
    chart_paths.append(size_vs_primary)

    size_vs_secondary = output_dir / "size_vs_secondary_metric.png"
    _scatter("checkpoint_mb", "secondary_metric", f"Checkpoint Size vs {metric_secondary_key}", size_vs_secondary)
    chart_paths.append(size_vs_secondary)

    speed_vs_retention = output_dir / "speed_vs_retention.png"
    _scatter("inference_ms", "mean_retention", "Inference Speed vs Mean Retention", speed_vs_retention)
    chart_paths.append(speed_vs_retention)

    prune_frame = candidate_frame[candidate_frame["family"] == "prune"].copy()
    if not prune_frame.empty:
        fig, ax = plt.subplots(figsize=(8.6, 5.4))
        prune_frame = prune_frame.sort_values("prune_ratio")
        ax.plot(prune_frame["prune_ratio"], prune_frame["primary_metric"], marker="o", label=metric_primary_key)
        ax.plot(prune_frame["prune_ratio"], prune_frame["secondary_metric"], marker="s", label=metric_secondary_key)
        ax.axhline(prune_frame["primary_threshold"].iloc[0], linestyle="--", linewidth=1.4, label="primary threshold")
        ax.axhline(prune_frame["secondary_threshold"].iloc[0], linestyle=":", linewidth=1.4, label="secondary threshold")
        ax.set_title("Pruning Ratio vs Metrics")
        ax.set_xlabel("prune_ratio")
        ax.set_ylabel("metric")
        ax.grid(True, alpha=0.35)
        ax.legend()
        fig.tight_layout()
        prune_chart = output_dir / "prune_ratio_vs_metrics.png"
        fig.savefig(prune_chart, dpi=220, bbox_inches="tight")
        plt.close(fig)
        chart_paths.append(prune_chart)

    return chart_paths


def write_compression_report(
    report_path: Path,
    workspace: Path,
    task: str,
    baseline_record: dict,
    fp32_record: dict | None,
    quant_records: list[dict],
    prune_records: list[dict],
    best_quant_record: dict | None,
    best_prune_record: dict | None,
    final_best_record: dict | None,
    min_retention: float,
    chart_paths: list[Path],
    sample_sections: dict[str, list[Path]],
) -> None:
    metric_primary_key, metric_secondary_key = metric_keys_for_task(task)
    gate_pct = min_retention * 100.0
    lines = [
        "# PCBFINAL Compression Study Report",
        "",
        "## 1. Summary",
        f"- Task: `{task}`",
        f"- Baseline model: `{safe_relpath(Path(baseline_record['artifact_path']), workspace)}`",
        f"- Primary retention gate: `{metric_primary_key}` >= `{gate_pct:.0f}%` of baseline",
        f"- Secondary retention gate: `{metric_secondary_key}` >= `{gate_pct:.0f}%` of baseline",
        "",
        "## 2. Baseline",
        f"- Checkpoint size: `{format_bytes(baseline_record['checkpoint_bytes'])}`",
        f"- Effective nonzero size: `{format_bytes(baseline_record['effective_nonzero_bytes'])}`",
        f"- {metric_primary_key}: `{baseline_record.get(metric_primary_key, 0.0):.6f}`",
        f"- {metric_secondary_key}: `{baseline_record.get(metric_secondary_key, 0.0):.6f}`",
        f"- Inference speed: `{baseline_record.get('speed_inference_ms', 0.0):.3f} ms/image`",
        "",
        "## 3. FP32 ONNX Validation",
    ]
    if fp32_record:
        lines.extend(
            [
                f"- Model: `{safe_relpath(Path(fp32_record['artifact_path']), workspace)}`",
                f"- Status: `{fp32_record['status']}`",
                f"- Checkpoint size: `{format_bytes(fp32_record['checkpoint_bytes'])}`",
                f"- {metric_primary_key}: `{fp32_record.get(metric_primary_key, 0.0):.6f}`",
                f"- {metric_secondary_key}: `{fp32_record.get(metric_secondary_key, 0.0):.6f}`",
                f"- Primary retention: `{(fp32_record.get('retention_primary') or 0.0):.4f}`",
                f"- Secondary retention: `{(fp32_record.get('retention_secondary') or 0.0):.4f}`",
                f"- Inference speed: `{fp32_record.get('speed_inference_ms', 0.0):.3f} ms/image`",
                "",
            ]
        )
    else:
        lines.extend(["- no FP32 ONNX validation recorded", ""])
    lines.extend(
        [
            "## 4. Quantization Candidates",
            "",
            "| candidate | status | checkpoint size | primary retention | secondary retention | inference ms |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in quant_records:
        lines.append(
            f"| {row['candidate_name']} | {row['status']} | {format_bytes(row['checkpoint_bytes'])} | "
            f"{row.get('retention_primary') or 0.0:.4f} | {row.get('retention_secondary') or 0.0:.4f} | "
            f"{row.get('speed_inference_ms') or 0.0:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 5. Pruning Sweep",
            "",
            "| candidate | prune ratio | status | checkpoint size | effective nonzero size | primary retention | secondary retention |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in prune_records:
        lines.append(
            f"| {row['candidate_name']} | {row.get('prune_ratio') or 0.0:.2f} | {row['status']} | "
            f"{format_bytes(row['checkpoint_bytes'])} | {format_bytes(row['effective_nonzero_bytes'])} | "
            f"{row.get('retention_primary') or 0.0:.4f} | {row.get('retention_secondary') or 0.0:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 6. Recommendations",
            f"- Best quantization: `{best_quant_record['candidate_name']}`" if best_quant_record else "- Best quantization: none passed retention gate",
            f"- Best pruning: `{best_prune_record['candidate_name']}`" if best_prune_record else "- Best pruning: none passed retention gate",
            f"- Final compressed artifact: `{final_best_record['candidate_name']}`" if final_best_record else "- Final compressed artifact: none",
            "",
            "## 7. Charts",
        ]
    )
    for chart_path in chart_paths:
        rel = safe_relpath(chart_path, workspace)
        lines.append(f"- [{chart_path.name}]({rel})")
    lines.append("")
    lines.append("## 8. Sample Predictions")
    for section_name, paths in sample_sections.items():
        lines.append(f"### {section_name}")
        if not paths:
            lines.append("- no samples exported")
        else:
            for sample_path in paths:
                rel = safe_relpath(sample_path, workspace)
                lines.append(f"- [{sample_path.name}]({rel})")
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_root_overview(root_dir: Path, workspace: Path) -> list[Path]:
    rows: list[dict] = []
    markdown_lines = [
        "# PCBFINAL Overview",
        "",
        "| task | experiment | slot | candidate | status | size | effective nonzero | primary metric | secondary metric | inference ms | artifact | report |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]

    for best_choice_path in sorted(root_dir.glob("*/final/best_choice.json")):
        experiment_dir = best_choice_path.parent.parent
        report_path = experiment_dir / "final" / "compression_report.md"
        payload = json.loads(best_choice_path.read_text(encoding="utf-8"))
        for slot in ("baseline", "fp32_onnx", "best_quantization", "best_pruning", "final_compressed"):
            record = payload.get(slot)
            if not record:
                continue
            primary_key = record.get("metric_primary_key")
            secondary_key = record.get("metric_secondary_key")
            artifact_path = Path(record["artifact_path"]) if record.get("artifact_path") else None
            row = {
                "task": record.get("task"),
                "experiment": experiment_dir.name,
                "slot": slot,
                "candidate_name": record.get("candidate_name"),
                "status": record.get("status"),
                "checkpoint_bytes": record.get("checkpoint_bytes"),
                "effective_nonzero_bytes": record.get("effective_nonzero_bytes"),
                "metric_primary_key": primary_key,
                "metric_secondary_key": secondary_key,
                "primary_metric": record.get(primary_key) if primary_key else None,
                "secondary_metric": record.get(secondary_key) if secondary_key else None,
                "speed_inference_ms": record.get("speed_inference_ms"),
                "artifact_path": str(artifact_path.resolve()) if artifact_path else "",
                "artifact_relpath": safe_relpath(artifact_path, workspace) if artifact_path else "",
                "report_relpath": safe_relpath(report_path, workspace),
            }
            rows.append(row)
            markdown_lines.append(
                "| {task} | {experiment} | {slot} | {candidate_name} | {status} | {checkpoint_size} | "
                "{effective_nonzero_size} | {primary_metric:.6f} | {secondary_metric:.6f} | {speed:.3f} | "
                "[artifact]({artifact_relpath}) | [report]({report_relpath}) |".format(
                    task=row["task"],
                    experiment=row["experiment"],
                    slot=row["slot"],
                    candidate_name=row["candidate_name"],
                    status=row["status"],
                    checkpoint_size=format_bytes(row["checkpoint_bytes"]),
                    effective_nonzero_size=format_bytes(row["effective_nonzero_bytes"]),
                    primary_metric=float(row["primary_metric"] or 0.0),
                    secondary_metric=float(row["secondary_metric"] or 0.0),
                    speed=float(row["speed_inference_ms"] or 0.0),
                    artifact_relpath=row["artifact_relpath"] or "#",
                    report_relpath=row["report_relpath"],
                )
            )

    overview_paths: list[Path] = []
    if rows:
        frame = pd.DataFrame(rows)
        csv_path = root_dir / "overview.csv"
        json_path = root_dir / "overview.json"
        md_path = root_dir / "overview.md"
        frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
        json_path.write_text(frame.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
        overview_paths.extend([csv_path, json_path, md_path])
    return overview_paths
