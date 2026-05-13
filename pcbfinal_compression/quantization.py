from __future__ import annotations

import logging
import shutil
from pathlib import Path

import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantFormat, QuantType, quantize_dynamic, quantize_static
from onnxruntime.transformers.float16 import convert_float_to_float16
from ultralytics import YOLO

from pcbfinal_compression.dataset_utils import build_calibration_reader


QUANT_PROFILE_DESCRIPTIONS = {
    "fp16_onnx": "FP32 ONNX -> FP16 mixed precision",
    "dynamic_int8_linear_mixed": "Dynamic INT8 for MatMul/Gemm only",
    "static_int8_conv_mixed": "Static INT8 for Conv only with representative calibration",
}


def export_fp32_onnx(
    model_path: Path,
    task: str,
    imgsz: int,
    target_path: Path,
    logger: logging.Logger,
) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        logger.info("复用已存在 FP32 ONNX: %s", target_path)
        return target_path

    logger.info("导出 FP32 ONNX -> %s", target_path)
    source_onnx_path = model_path.with_suffix(".onnx")
    source_existed_before = source_onnx_path.exists()
    model = YOLO(str(model_path), task=task)
    source_onnx = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            opset=14,
            simplify=True,
            half=False,
            dynamic=False,
            batch=1,
            device="cpu",
        )
    )
    shutil.copy2(source_onnx, target_path)
    if source_onnx != target_path and source_onnx.exists() and not source_existed_before:
        source_onnx.unlink()
    return target_path


def create_fp16_onnx(fp32_path: Path, target_path: Path, logger: logging.Logger) -> Path:
    logger.info("生成 FP16 ONNX -> %s", target_path)
    model = onnx.load(str(fp32_path))
    converted = convert_float_to_float16(model, keep_io_types=True)
    onnx.save(converted, str(target_path))
    return target_path


def create_dynamic_int8_linear_mixed(fp32_path: Path, target_path: Path, logger: logging.Logger) -> Path:
    logger.info("生成 Dynamic INT8 ONNX -> %s", target_path)
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(target_path),
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    return target_path


def create_static_int8_conv_mixed(
    fp32_path: Path,
    target_path: Path,
    calibration_images: list[Path],
    imgsz: int,
    logger: logging.Logger,
) -> Path:
    logger.info("生成 Static INT8 ONNX -> %s", target_path)
    if not calibration_images:
        raise RuntimeError("Static INT8 量化缺少校准图像。")
    session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    reader = build_calibration_reader(calibration_images, input_name, imgsz)
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(target_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["Conv"],
    )
    return target_path
