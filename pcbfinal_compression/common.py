from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable


def set_runtime_env() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_now() -> str:
    return datetime.now().strftime("%m%d%H%M%S")


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"pcbfinal_compression.{log_path.stem}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def stage_header(logger: logging.Logger, title: str) -> None:
    line = "=" * 88
    logger.info(line)
    logger.info(title)
    logger.info(line)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_csv_list(text: str | None, cast=float) -> list:
    if not text:
        return []
    values = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        values.append(cast(raw))
    return values


def format_bytes(num_bytes: int | float | None) -> str:
    if num_bytes is None:
        return "n/a"
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def safe_relpath(path: Path, start: Path) -> str:
    try:
        return str(path.resolve().relative_to(start.resolve()))
    except ValueError:
        return str(path.resolve())


def infer_experiment_name(model_path: Path) -> str:
    if model_path.parent.name == "weights" and model_path.parent.parent.name:
        return model_path.parent.parent.name
    return model_path.stem


def unique_sorted(values: Iterable[float], digits: int = 4) -> list[float]:
    return sorted({round(float(value), digits) for value in values})
