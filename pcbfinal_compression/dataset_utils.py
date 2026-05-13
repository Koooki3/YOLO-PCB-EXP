from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_data_yaml(data_yaml: Path) -> dict:
    return yaml.safe_load(data_yaml.read_text(encoding="utf-8"))


def resolve_data_path(data_yaml: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (data_yaml.parent / path).resolve()


def collect_split_images(data_yaml: Path, split: str) -> list[Path]:
    data = load_data_yaml(data_yaml)
    split_dir = resolve_data_path(data_yaml, data.get(split))
    if split_dir is None or not split_dir.exists():
        return []
    images = [path for path in split_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTS]
    images.sort()
    return images


def sample_evenly(paths: list[Path], count: int) -> list[Path]:
    if count <= 0 or not paths:
        return []
    if count >= len(paths):
        return list(paths)
    if count == 1:
        return [paths[len(paths) // 2]]
    indexes = np.linspace(0, len(paths) - 1, count, dtype=int)
    return [paths[idx] for idx in indexes]


def letterbox(
    image: np.ndarray,
    new_shape: tuple[int, int] | int,
    color: tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    height, width = image.shape[:2]
    ratio = min(new_shape[0] / height, new_shape[1] / width)
    new_unpad = (int(round(width * ratio)), int(round(height * ratio)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if (width, height) != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def preprocess_onnx_image(image_path: Path, imgsz: int) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"读取图片失败: {image_path}")
    image = letterbox(image, (imgsz, imgsz))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
    return image[None, ...]


def build_calibration_reader(image_paths: Iterable[Path], input_name: str, imgsz: int):
    from onnxruntime.quantization import CalibrationDataReader

    class _CalibrationReader(CalibrationDataReader):
        def __init__(self, paths: list[Path], ort_input_name: str, image_size: int):
            self._paths = iter(paths)
            self._input_name = ort_input_name
            self._imgsz = image_size

        def get_next(self):
            try:
                path = next(self._paths)
            except StopIteration:
                return None
            return {self._input_name: preprocess_onnx_image(path, self._imgsz)}

    return _CalibrationReader(list(image_paths), input_name, imgsz)
