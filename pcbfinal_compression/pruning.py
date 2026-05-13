from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from ultralytics import YOLO


def _dino_prefixes(model: nn.Module) -> list[str]:
    prefixes = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == "DINO3Backbone":
            prefixes.append(name)
    return prefixes


def collect_prunable_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    dino_prefixes = _dino_prefixes(model)
    modules: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        if any(name == prefix or name.startswith(prefix + ".") for prefix in dino_prefixes):
            continue
        if not hasattr(module, "weight") or module.weight is None:
            continue
        modules.append((name, module))
    return modules


def prune_checkpoint(
    source_model_path: Path,
    target_model_path: Path,
    amount: float,
    task: str,
    logger: logging.Logger,
) -> dict[str, int | float]:
    logger.info("执行全局 L1 掩码剪枝 amount=%.2f -> %s", amount, target_model_path)
    wrapper = YOLO(str(source_model_path), task=task)
    prunable_modules = collect_prunable_modules(wrapper.model)
    if not prunable_modules:
        raise RuntimeError("未找到可剪枝的非 DINO Conv/Linear 模块。")
    prune.global_unstructured(
        [(module, "weight") for _, module in prunable_modules],
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    for _, module in prunable_modules:
        prune.remove(module, "weight")
    target_model_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper.save(str(target_model_path))
    parameter_count = sum(int(param.numel()) for param in wrapper.model.parameters())
    nonzero_count = sum(int(torch.count_nonzero(param).item()) for param in wrapper.model.parameters())
    return {
        "prunable_module_count": len(prunable_modules),
        "parameter_count": parameter_count,
        "nonzero_count": nonzero_count,
        "nonzero_ratio": (nonzero_count / parameter_count) if parameter_count else 0.0,
    }
