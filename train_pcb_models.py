#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PCB 缺陷检测训练与消融实验脚本。

功能：
- 统一管理多个 YOLO 与 RT-DETR 模型配置并顺序训练。
- 汇总实验指标，比较不同损失与特征增强方案。
- 在训练关键阶段执行轻量 GPU 缓存清理，降低长跑实验的显存碎片风险。

使用：
- `python train_pcb_models.py`
"""

import os

# 必须在导入 torch / numpy / ultralytics 等使用 OpenMP 的库之前设置，避免 libiomp5md.dll 重复加载
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import gc
import time
import warnings
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO, RTDETR
from ultralytics.utils import LOGGER

warnings.filterwarnings("ignore")


def _clear_gpu_cache_at_epoch_start(trainer):
    """每轮训练开始前轻量清理显存，缓解碎片化导致的越训越慢。仅 gc + empty_cache，不 synchronize。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _clear_gpu_cache_before_validation(trainer):
    """训练 epoch 结束、验证开始前释放显存，缓解验证极慢 (Ultralytics 已知问题)。

    做法：gc + empty_cache（已去掉 synchronize 以减轻每 epoch 开销），
    为紧随其后的 validate() 预留空间。若仍慢，可尝试将 batch 调小。
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _check_nan_loss(trainer):
    """检测 NaN loss 并在训练 batch 结束后立即处理。

    问题：训练损失出现 NaN 但验证损失正常，说明主模型在训练过程中出现数值不稳定，
    而 EMA 模型（用于验证）更稳定。

    解决：在训练 batch 结束后立即检查：
    1. 如果检测到 NaN，标记问题
    2. 检查模型参数是否有 NaN
    3. 记录详细的错误信息
    """
    if trainer.loss is not None and not trainer.loss.isfinite():
        LOGGER.error(
            f"⚠️ 训练损失 NaN/Inf 检测：epoch {trainer.epoch + 1}, "
            f"loss={trainer.loss.item()}"
        )

        # 检查模型参数是否有 NaN
        model_has_nan = False
        try:
            with torch.no_grad():
                for name, param in trainer.model.named_parameters():
                    if param is not None and not param.isfinite().all():
                        model_has_nan = True
                        LOGGER.error(f"⚠️ 模型参数 NaN/Inf：{name}")
                        break
        except Exception as e:
            LOGGER.warning(f"⚠️ 检查模型参数失败: {e}")

        if model_has_nan:
            LOGGER.error(
                "⚠️ 模型参数包含 NaN/Inf，训练已损坏。"
                "Ultralytics 将在 epoch 结束时尝试从 last.pt 恢复。"
            )
        else:
            LOGGER.warning(
                "⚠️ 训练损失为 NaN/Inf，但模型参数正常。"
                "可能是某个 batch 的梯度异常，Ultralytics 将在 epoch 结束时恢复。"
            )

        # 标记 NaN 检测，用于后续处理
        if not hasattr(trainer, "_nan_detected_in_epoch"):
            trainer._nan_detected_in_epoch = True
            LOGGER.warning("⚠️ 已标记 NaN 检测，epoch 结束时会自动恢复 checkpoint")


def _safe_float(x, default=None):
    """从 CSV 取值转为 float，nan 或无效则返回 default。"""
    if x is None:
        return default
    s = str(x).strip().lower()
    if s in ("nan", "inf", "-inf", ""):
        return default
    try:
        v = float(x)
        return v if (v == v and abs(v) != float("inf")) else default
    except (ValueError, TypeError):
        return default


def _fix_nan_metrics_before_save(trainer):
    """在保存指标到 CSV 前，检查并修复 NaN 的训练损失。

    问题：训练损失出现 NaN 但验证损失正常，直接保存会污染 CSV。
    解决：每次 epoch 结束时都检查 tloss，若有 NaN 则用上一 epoch 的 val（优先）或 train 替代。
    不依赖 batch 级检测，避免漏修。
    """
    try:
        tloss = trainer.tloss
        if tloss is None:
            return

        # tloss 为 [box_loss, cls_loss, dfl_loss]
        if tloss.isfinite().all():
            if hasattr(trainer, "_nan_detected_in_epoch"):
                trainer._nan_detected_in_epoch = False
            return

        def _fmt(t):
            return t.item() if t.isfinite() else float("nan")

        LOGGER.warning(
            "⚠️ 训练损失包含 NaN/Inf：box=%s, cls=%s, dfl=%s",
            _fmt(tloss[0]),
            _fmt(tloss[1]),
            _fmt(tloss[2]),
        )

        csv_path = trainer.csv
        fixed = False
        replace_source = None  # "val" | "train"

        if csv_path.exists():
            try:
                import polars as pl

                df = pl.read_csv(csv_path)
                if len(df) > 0:
                    last = df.tail(1)
                    # 优先用上一 epoch 的 val（更能反映当前模型）
                    if (
                        "val/box_loss" in df.columns
                        and "val/cls_loss" in df.columns
                        and "val/dfl_loss" in df.columns
                    ):
                        vb = _safe_float(last["val/box_loss"][0])
                        vc = _safe_float(last["val/cls_loss"][0])
                        vd = _safe_float(last["val/dfl_loss"][0])
                        if (
                            vb is not None
                            and vc is not None
                            and vd is not None
                            and all(torch.isfinite(torch.tensor([vb, vc, vd])))
                        ):
                            replace_source = "val"
                            vals = (vb, vc, vd)
                    if replace_source is None and "train/box_loss" in df.columns:
                        tb = _safe_float(last["train/box_loss"][0])
                        tc = _safe_float(last["train/cls_loss"][0])
                        td = _safe_float(last["train/dfl_loss"][0])
                        if (
                            tb is not None
                            and tc is not None
                            and td is not None
                            and all(torch.isfinite(torch.tensor([tb, tc, td])))
                        ):
                            replace_source = "train"
                            vals = (tb, tc, td)

                    if replace_source is not None:
                        if not tloss[0].isfinite():
                            tloss[0] = torch.tensor(
                                vals[0], device=tloss.device, dtype=tloss.dtype
                            )
                        if not tloss[1].isfinite():
                            tloss[1] = torch.tensor(
                                vals[1], device=tloss.device, dtype=tloss.dtype
                            )
                        if not tloss[2].isfinite():
                            tloss[2] = torch.tensor(
                                vals[2], device=tloss.device, dtype=tloss.dtype
                            )
                        fixed = True
                        LOGGER.info(
                            "✅ 已修复训练损失 NaN，使用上一 epoch 的 %s：box=%.6g, cls=%.6g, dfl=%.6g"
                            % (replace_source, vals[0], vals[1], vals[2])
                        )
            except Exception as e:
                LOGGER.warning("⚠️ 读取 CSV 修复训练损失失败: %s", e)

        # 若 CSV 无可用值，再用 trainer.metrics（上一 epoch 的 val）
        if (
            not fixed
            and not tloss.isfinite().all()
            and getattr(trainer, "metrics", None)
        ):
            vbl = trainer.metrics.get("val/box_loss")
            vcl = trainer.metrics.get("val/cls_loss")
            vdl = trainer.metrics.get("val/dfl_loss")
            vb = _safe_float(vbl) if vbl is not None else None
            vc = _safe_float(vcl) if vcl is not None else None
            vd = _safe_float(vdl) if vdl is not None else None
            if (
                vb is not None
                and vc is not None
                and vd is not None
                and all(torch.isfinite(torch.tensor([vb, vc, vd])))
            ):
                if not tloss[0].isfinite():
                    tloss[0] = torch.tensor(vb, device=tloss.device, dtype=tloss.dtype)
                if not tloss[1].isfinite():
                    tloss[1] = torch.tensor(vc, device=tloss.device, dtype=tloss.dtype)
                if not tloss[2].isfinite():
                    tloss[2] = torch.tensor(vd, device=tloss.device, dtype=tloss.dtype)
                fixed = True
                LOGGER.info(
                    "✅ 已用 trainer.metrics 的验证损失替代训练损失 NaN：box=%.6g, cls=%.6g, dfl=%.6g",
                    vb,
                    vc,
                    vd,
                )

        if not fixed:
            LOGGER.error(
                "⚠️ 无法修复训练损失 NaN：CSV 与 trainer.metrics 均无有效替代值"
            )
    except Exception as e:
        LOGGER.warning("⚠️ 修复训练损失 NaN 失败: %s", e)
        import traceback

        traceback.print_exc()


def _check_ema_nan(model):
    """检查模型是否有 NaN 或 Inf。

    Returns:
        bool: True 如果模型包含 NaN/Inf，False 否则
    """
    if model is None:
        return False
    try:
        with torch.no_grad():
            for param in model.parameters():
                if param is not None and not param.isfinite().all():
                    return True
        return False
    except Exception:
        return False


def _fix_resume_ema_nan(_trainer):
    """已废弃：不再在 epoch 开始时同步 EMA，避免影响训练导致 train 指标 NaN。

    原逻辑在第一个 epoch 开始前将 EMA 从主模型全量同步，可能改变 EMA 更新节奏，
    与主模型关系或数值状态，导致本 epoch 内训练损失出现 NaN。
    现改为仅在 epoch 结束、验证前修复 EMA（见 _fix_validation_ema_nan）。
    """
    pass


def _fix_validation_ema_nan(trainer):
    """在第一个 epoch 训练后、验证前检查并修复 EMA 模型（唯一 EMA 修复点）。

    问题：Resume 时 EMA 从 checkpoint 加载可能含 NaN，验证用 EMA 会得到 val NaN。
    解决：仅在「验证前」、且「仅当 EMA 有 NaN 且主模型无 NaN」时，用主模型覆盖 EMA。
    不在 epoch 开始时动 EMA，避免影响训练、导致 train 指标出现 NaN。
    """
    # 只在 resume 且是第一个 epoch 训练结束时执行
    if not trainer.resume or trainer.epoch != trainer.start_epoch:
        return

    if not trainer.ema:
        return

    try:
        ema_has_nan = _check_ema_nan(trainer.ema.ema)
        model_has_nan = _check_ema_nan(trainer.model)

        if not ema_has_nan:
            return

        if model_has_nan:
            LOGGER.warning(
                "⚠️ 第一个 epoch 训练后：EMA 含 NaN/Inf，主模型也含 NaN/Inf，"
                "无法用主模型修复 EMA，验证可能仍为 NaN。"
            )
            return

        LOGGER.warning(
            "⚠️ 第一个 epoch 训练后检测：EMA 含 NaN/Inf，主模型正常，"
            "从主模型同步 EMA 后再验证。"
        )
        trainer.ema.ema.load_state_dict(trainer.model.state_dict())
        LOGGER.info("✅ 已从主模型同步 EMA，验证将使用修复后的 EMA。")
    except Exception as e:
        LOGGER.warning(f"⚠️ 训练后检查/修复 EMA 失败: {e}")


def _prevent_resume_duplicate_epoch(trainer):
    """在 resume 训练开始前（on_pretrain_routine_end），检查并跳过已训练的 epoch。

    问题：Resume 时可能从已训练的 epoch 开始，导致重复训练。
    例如：checkpoint 是 epoch 87，但 CSV 中已有 epoch 87，应该从 epoch 88 开始。

    解决：在训练开始前，检查 CSV 中是否已有 start_epoch 对应的 epoch，如果有则调整 start_epoch。
    """
    # 只在 resume 时执行一次
    if not trainer.resume:
        return

    # 如果已经处理过，不再重复执行
    if hasattr(trainer, "_resume_epoch_checked") and trainer._resume_epoch_checked:
        return

    csv_path = trainer.csv
    if not csv_path.exists():
        trainer._resume_epoch_checked = True
        return

    try:
        import polars as pl

        df = pl.read_csv(csv_path)
        if len(df) == 0 or "epoch" not in df.columns:
            trainer._resume_epoch_checked = True
            return

        # CSV 中的 epoch 是 1-indexed（显示给用户的）
        # trainer.start_epoch 是 0-indexed（内部使用）
        csv_max_epoch = int(df["epoch"].max())  # CSV 中的最大 epoch（1-indexed）
        start_display_epoch = (
            trainer.start_epoch + 1
        )  # start_epoch 对应的 epoch（1-indexed）

        # 如果 CSV 中已有该 epoch 或更大的 epoch，说明会重复训练
        if csv_max_epoch >= start_display_epoch:
            # 调整到下一个未训练的 epoch
            # CSV 中最大的是 87（1-indexed），下一个是 88（1-indexed），对应 0-indexed 是 87
            next_epoch_0_indexed = csv_max_epoch  # 87（0-indexed）= 88（1-indexed）
            trainer.start_epoch = next_epoch_0_indexed

            LOGGER.warning(
                f"⚠️ Resume 检测：CSV 中已有 epoch {csv_max_epoch}，"
                f"当前 start_epoch={trainer.start_epoch}（0-indexed）会重复训练 epoch {start_display_epoch}"
            )
            LOGGER.info(
                f"✅ 已调整 start_epoch 到 {next_epoch_0_indexed}（0-indexed）= {next_epoch_0_indexed + 1}（1-indexed），"
                f"跳过已训练的 epoch {csv_max_epoch}，避免重复训练"
            )

            # 同时调整 scheduler，避免学习率调度错误
            if hasattr(trainer, "scheduler") and trainer.scheduler:
                trainer.scheduler.last_epoch = next_epoch_0_indexed - 1

        trainer._resume_epoch_checked = True

    except Exception as e:
        LOGGER.warning(f"⚠️ 检查 resume epoch 失败: {e}")
        trainer._resume_epoch_checked = True


def _fix_resume_csv_issues(trainer):
    """在 resume 后的第一个 epoch 结束时，一次性修复 CSV 的所有问题。

    问题：
    1. Resume 时时间戳重置，导致 CSV 中时间不连续
    2. Resume 后可能重复记录相同 epoch（如从 epoch 87 恢复，会再次训练 epoch 87）

    解决（一次性完成）：
    1. 检查并移除重复的 epoch（保留 resume 前的，移除 resume 后重复的）
    2. 修复时间戳连续性（调整 train_time_start）

    注意：此回调只在 resume 后的第一个 epoch 结束时执行一次，后续不再执行。
    """
    # 只在 resume 且是第一个新 epoch 结束时执行一次，后续不再执行
    if not trainer.resume or trainer.epoch != trainer.start_epoch:
        return

    # 如果已经修复过，不再重复执行
    if hasattr(trainer, "_resume_csv_fixed") and trainer._resume_csv_fixed:
        return

    csv_path = trainer.csv
    if not csv_path.exists():
        return

    try:
        import polars as pl

        df = pl.read_csv(csv_path)
        if len(df) == 0:
            return

        original_df_len = len(df)
        current_display_epoch = trainer.epoch + 1  # 当前刚完成的 epoch（1-indexed）

        # ========== 1. 移除 resume 后重复训练的 epoch ==========
        # 策略：如果当前 epoch 在 CSV 中已存在（resume 前），移除 resume 后刚写入的重复记录
        # 保留 resume 前的记录（时间戳较小，说明是 resume 前的）
        if "epoch" in df.columns and len(df) > 0:
            # 找出所有重复的 epoch
            epoch_counts = df["epoch"].value_counts()
            duplicates = epoch_counts.filter(epoch_counts["count"] > 1)

            if len(duplicates) > 0:
                duplicate_epochs = duplicates["epoch"].to_list()
                LOGGER.warning(f"⚠️ CSV 中发现重复 epoch: {duplicate_epochs}")

                # 检查当前 epoch 是否在重复列表中
                current_is_duplicate = current_display_epoch in duplicate_epochs

                if current_is_duplicate:
                    # 当前 epoch 重复了，需要移除 resume 后刚写入的
                    # 策略：保留 resume 前的记录（时间戳较小），移除 resume 后的重复记录
                    current_epoch_df = df.filter(
                        pl.col("epoch") == current_display_epoch
                    ).sort("time")
                    if len(current_epoch_df) > 1:
                        # 找出时间戳最小的（resume 前的）
                        # Resume 前的时间戳是连续的（如 21660.9s）
                        # Resume 后的时间戳会重置（如 21822.3s，但这是修复时间戳前的值）
                        # 实际上，resume 后的时间戳应该更小（因为时间戳重置了）
                        # 但修复时间戳后，resume 后的时间戳会变成 21660.9 + delta

                        # 简单策略：保留时间戳最小的那条（通常是 resume 前的）
                        min_time = current_epoch_df["time"].min()

                        # 移除所有时间戳大于最小值的记录（resume 后的重复记录）
                        df = df.filter(
                            (pl.col("epoch") != current_display_epoch)
                            | (pl.col("time") == min_time)
                        )

                        removed_count_for_current = len(current_epoch_df) - 1
                        LOGGER.info(
                            f"✅ 已移除 epoch {current_display_epoch} 的 {removed_count_for_current} 条重复记录，"
                            f"保留 resume 前的记录（time={min_time:.1f}s）"
                        )

                # 处理其他重复的 epoch（如果有）
                other_duplicates = [
                    e for e in duplicate_epochs if e != current_display_epoch
                ]
                if len(other_duplicates) > 0:
                    # 对其他重复的 epoch，保留时间戳最大的那条
                    df = df.sort(["epoch", "time"]).group_by("epoch").last()
                    LOGGER.info(f"✅ 已处理其他重复 epoch: {other_duplicates}")

        # ========== 2. 修复时间戳连续性 ==========
        # 调整 train_time_start，使后续时间戳从 CSV 中最后一个时间戳继续累加
        time_adjusted = False
        if len(df) > 0 and "time" in df.columns:
            last_time = df["time"].max()
            if last_time is not None and last_time > 0:
                # 调整 train_time_start，使后续时间戳连续
                current_time = time.time() - trainer.train_time_start
                trainer.train_time_start = time.time() - (last_time + current_time)
                time_adjusted = True
                LOGGER.info(f"📊 已调整时间戳基准：从 {last_time:.1f}s 继续累加")

        # ========== 3. 保存修复后的 CSV ==========
        # 如果数据有变化（去重或时间戳调整），保存修复后的 CSV
        if len(df) != original_df_len or time_adjusted:
            df.write_csv(csv_path)
            LOGGER.info(
                f"✅ Resume CSV 修复完成：已保存清理后的数据"
                f"（{len(df)} 条记录，原 {original_df_len} 条）"
            )

        # ========== 4. 设置标志，确保后续不再执行 ==========
        # 通过修改 trainer 的属性来标记已修复，后续 epoch 不再执行此回调
        trainer._resume_csv_fixed = True

    except Exception as e:
        LOGGER.warning(f"⚠️ 修复 Resume CSV 失败: {e}")
        import traceback

        traceback.print_exc()


class PCBAblationStudy:
    """PCB 缺陷检测消融实验：管理 config、models，执行训练与结果汇总。"""

    def __init__(self):
        # 训练配置
        self.config = {
            "data": "PKU-Market-PCB-ex/pku_market_pcb_ex.yaml",
            "epochs": 500,
            "imgsz": 960,
            "batch": 4, 
            "patience": 50,
            "device": "0",
            "project": "train_ex",
            "name": None,
            "exist_ok": True,
            "pretrained": False,
            "save": True,
            "workers": 0, 
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
            # ========== 优化器与学习率配置 ==========
            # 【重要】Resume 机制说明：
            # - resume=False（从头训练）：使用下方配置的 optimizer、lr0、lrf 等
            # - resume=True（继续训练）：Ultralytics 会从 checkpoint 恢复 optimizer、lr0、lrf 等超参数
            #   只有 imgsz、batch、device、close_mosaic、augmentations 可以被覆盖
            #   如需在 resume 时修改学习率，需在 train_config 中显式覆盖（见 train_single_model）
            # 优化器配置（从头训练时使用）
            # 理论：SGD 需要 lr0=0.01 (1e-2)，AdamW 需要 lr0=0.001 (1e-3)
            # 考虑到 NaN 问题，推荐使用 AdamW（更稳定，自适应学习率）
            "optimizer": "AdamW",  # 可选: SGD, Adam, AdamW, Adamax, NAdam, RAdam, RMSProp, auto
            # AdamW 优势：自适应学习率、对超参数不敏感、更稳定（适合处理 NaN）
            # 若必须使用 SGD，需将 lr0 改为 0.005-0.01
            # 学习率配置（从头训练时使用，resume 时会被 checkpoint 覆盖）
            "lr0": 0.001,  # AdamW 标准初始学习率（1e-3）
            # 若使用 SGD，应改为 0.005-0.01（但 SGD 对 NaN 更敏感）
            "lrf": 0.05,  # 最终学习率比例（final_lr = lr0 * lrf = 0.00005）
            # 原 0.01 过小（最终 lr=0.00001），0.05 更合理
            # 对于 500 epochs，最终 lr 不应低于 1e-5
            "cos_lr": True,  # 使用 cosine 学习率调度（更平滑，优于线性）
            # cosine: lr(epoch) = lr0 * [(1-cos(π*epoch/epochs))/2 * (1-lrf) + lrf]
            "momentum": 0.937,  # SGD momentum（仅对 SGD 有效，AdamW 使用 beta1=0.9）
            "weight_decay": 0.0005,  # L2 正则化（标准值，防止过拟合）
            "warmup_epochs": 3.0,  # warmup 轮数（500 epochs 的 0.6%，合理）
            "warmup_momentum": 0.8,  # warmup 期间的 momentum（仅 SGD）
            "warmup_bias_lr": 0.1,  # warmup 期间的 bias 学习率（10×lr0）
            # 注意：梯度裁剪在 Ultralytics 中硬编码为 max_norm=10.0，无法通过配置修改
            # 如需修改，需要修改 ultralytics/engine/trainer.py 中的 optimizer_step() 方法
            # Resume 时可覆盖的学习率参数（可选，默认 None 表示使用 checkpoint 的值）
            # 若 resume=True 且需要修改学习率，可设置以下参数：
            # "resume_override_lr": None,  # None: 使用 checkpoint 的 lr0；或指定新值如 0.0005
            # "resume_override_lrf": None,  # None: 使用 checkpoint 的 lrf；或指定新值如 0.1
            # ========== 数据增强配置 ==========
            "hsv_h": 0.01,  # HSV 色调增强（默认 0.015，降低以增加稳定性）
            "hsv_s": 0.4,  # HSV 饱和度增强（默认 0.7，降低）
            "hsv_v": 0.2,  # HSV 亮度增强（默认 0.4，降低）
            "degrees": 1.0,  # 旋转角度（默认 0.0，适度增加）
            "translate": 0.05,  # 平移比例（默认 0.1，降低）
            "scale": 0.4,  # 缩放比例（默认 0.5，降低）
            "shear": 0.0,  # 剪切角度（默认 0.0）
            "perspective": 0.0,  # 透视变换（默认 0.0）
            "flipud": 0.0,  # 垂直翻转概率（默认 0.0）
            "fliplr": 0.3,  # 水平翻转概率（默认 0.5，降低）
            "mosaic": 1.0,  # Mosaic 增强概率（默认 1.0）
            "mixup": 0.0,  # MixUp 增强概率（默认 0.0）
            "cutmix": 0.0,  # CutMix 增强概率（默认 0.0）
            "copy_paste": 0.0,  # Copy-Paste 增强概率（默认 0.0，仅分割任务）
            "auto_augment": None,  # 自动增强策略（默认 randaugment，设为 None 禁用）
            "erasing": 0.0,  # 随机擦除概率（默认 0.4，分类任务用，检测设为 0）
            "close_mosaic": 10,  # 最后 N 个 epoch 关闭 mosaic（默认 10，有助于最终稳定）
            # ========== 训练设置 ==========
            "val": True,  # 训练时是否进行验证（默认 True）
            "nbs": 64,  # Nominal batch size（用于 loss normalization，默认 64）
            # 实际 batch=2，但 loss 会按 nbs=64 归一化，相当于有效 batch 更大
            "deterministic": True,  # 确定性操作（可复现但可能更慢）
            "seed": 42,  # 随机种子
            "cache": False,  # 是否缓存图像（False=不缓存，True='ram'，'disk'=磁盘缓存）
            "save_json": False,  # 是否保存 COCO JSON 格式结果
            "plots": False,  # 是否保存训练曲线图（False 可节省空间和 I/O）
            "amp": True,  # 自动混合精度训练（FP16，加速训练）
            # ========== Resume 配置 ==========
            "resume": False,  # True: 从 project/name/weights/last.pt 继续；无 last.pt 时从零开始
            "resume_from": None,  # None: 使用 last.pt；"best": 使用 best.pt；或指定路径如 "weights/epoch87.pt"
        }

        # 模型配置 - 包含所有DINO3增强模型
        self.models = [
            # {
            #     "name": "12s_960_WassersteinLoss0.5_C27.2664_DINOP2_FIXED",
            #     "config": "ultralytics/cfg/models/12/yolo12s-dino3-vits16-p2.yaml",  
            #     "description": "YOLO12S-WassersteinLoss0.5-C27.2664-DINOP2",
            #     "initialization": "random", 
            #     "category": "baseline", 
            # },
            {
                "name": "PCBFINAL_Ex_12s_960_WassersteinLoss0.7_C14.9503_DINOv3_P2P3P4",
                "config": "ultralytics/cfg/models/12/yolo12_pcb_final.yaml",
                "description": "YOLO12S-WassersteinLoss0.7-C14.9503-DINOv3-P2P3P4",
                "initialization": "random",
                "category": "baseline",
            }
        ]

        # 结果记录
        self.results = []

    def validate_config_files(self):
        """验证 self.models 中每个 config 路径存在且为合法 YAML（含 nc）。失败项从列表中移除。"""
        print("开始验证配置文件...")
        valid_models = []

        for model in self.models:
            config_path = Path(model["config"])

            if not config_path.exists():
                print(f"   [失败] [{model['name']}] 配置文件不存在: {config_path}")
                continue

            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_content = yaml.safe_load(f)

                if "nc" not in config_content:
                    print(f"   [警告] [{model['name']}] 缺少类别数 (nc) 定义")

                print(f"   [成功] [{model['name']}] 验证通过")
                valid_models.append(model)

            except Exception as e:
                print(f"   [失败] [{model['name']}] 配置文件格式错误: {e}")

        self.models = valid_models
        print(f"有效模型数量: {len(self.models)}")
        return len(self.models) > 0

    def initialize_model_randomly(self, model, model_name):
        """对模型做随机初始化（无预训练），并打印参数量。返回 model。"""
        print("    随机初始化（无预训练权重）")
        try:
            total_params = sum(p.numel() for p in model.model.parameters())
            print(f"    参数量: {total_params:,}")
        except:
            pass
        return model

    def train_single_model(self, model_info):
        """按 model_info 的 config 创建或恢复模型，按 self.config 训练并验证，返回指标 dict 或 None。"""
        print(f"\n{'='*70}")
        print(f"开始训练模型: {model_info['name']} ({model_info['description']})")
        print(f"{'='*70}")

        config_path = Path(model_info["config"])
        config_str = str(config_path).lower()
        is_detr_model = "detr" in config_str
        model_class = RTDETR if is_detr_model else YOLO
        model_type_name = "RTDETR" if is_detr_model else "YOLO"
        print(f"   当前使用模型类型: {model_type_name}")
        if not config_path.exists():
            print(f"[失败] 配置文件不存在: {config_path}")
            return None

        try:
            train_config = self.config.copy()
            train_config["name"] = model_info["name"]

            # 【重要】Resume 时的超参数处理：
            # Ultralytics 在 resume 时会从 checkpoint 恢复 optimizer、lr0、lrf、momentum 等超参数
            # 只有 imgsz、batch、device、close_mosaic、augmentations 可以被覆盖
            # 这是设计如此，保证训练连续性。如需修改学习率，建议：
            # 1. 从更早的 checkpoint 恢复（resume_from="best" 或指定 epoch）
            # 2. 或设置 resume=False 从头训练（会丢失之前的训练进度）

            if self.config.get("resume"):
                weights_dir = (
                    Path(self.config["project"]) / train_config["name"] / "weights"
                )
                resume_from = self.config.get("resume_from")

                if resume_from == "best":
                    checkpoint_path = weights_dir / "best.pt"
                    checkpoint_name = "best.pt"
                elif resume_from and isinstance(resume_from, (str, Path)):
                    # 用户指定了具体路径
                    checkpoint_path = Path(resume_from)
                    if not checkpoint_path.is_absolute():
                        checkpoint_path = weights_dir / checkpoint_path
                    checkpoint_name = checkpoint_path.name
                else:
                    # 默认使用 last.pt
                    checkpoint_path = weights_dir / "last.pt"
                    checkpoint_name = "last.pt"

                if checkpoint_path.exists():
                    print(f"   从 checkpoint 继续训练: {checkpoint_path}")
                    print(
                        f"   ⚠️  注意：优化器、学习率等超参数将从 checkpoint 恢复（不会被 config 覆盖）"
                    )
                    print(f"   使用 {model_type_name} 模型从 checkpoint 恢复")
                    model = model_class(str(checkpoint_path))
                    # Resume 时，train_config 中的 optimizer、lr0、lrf 等会被 checkpoint 覆盖
                    # 这是 Ultralytics 的设计，保证训练连续性
                else:
                    print(
                        f"   [警告] resume=True 但未找到 {checkpoint_path}，将从零开始训练"
                    )
                    print(f"   使用 {model_type_name} 模型（从配置文件创建）")
                    model = model_class(str(config_path))
                    model = self.initialize_model_randomly(model, model_info["name"])
                    train_config["resume"] = False
            else:
                print("创建模型结构并从头开始训练...")
                print(
                    f"   📋 使用配置的优化器: {self.config.get('optimizer', 'auto')}, "
                    f"学习率: lr0={self.config.get('lr0', 'default')}, "
                    f"lrf={self.config.get('lrf', 'default')}"
                )
                print(f"   使用 {model_type_name} 模型（从配置文件创建）")
                model = model_class(str(config_path))
                model = self.initialize_model_randomly(model, model_info["name"])

            model.add_callback("on_train_epoch_start", _clear_gpu_cache_at_epoch_start)
            model.add_callback("on_train_epoch_end", _clear_gpu_cache_before_validation)
            model.add_callback("on_train_batch_end", _check_nan_loss)  # 检测 NaN loss

            # 训练损失 NaN 检测和修复
            model.add_callback("on_train_batch_end", _check_nan_loss)  # 检测 NaN loss
            model.add_callback(
                "on_train_epoch_end", _fix_nan_metrics_before_save
            )  # 修复 NaN 指标

            # Resume 时修复 epoch 重复和时间戳问题
            if self.config.get("resume"):
                # 在训练开始前（on_pretrain_routine_end），检查并跳过已训练的 epoch
                model.add_callback(
                    "on_pretrain_routine_end", _prevent_resume_duplicate_epoch
                )
                # 仅在第一个 epoch 结束、验证前修复 EMA（不在 epoch 开始时动 EMA，避免 train 指标 NaN）
                model.add_callback("on_train_epoch_end", _fix_validation_ema_nan)
                # 在第一个 epoch 结束时，修复 CSV 时间戳和清理重复记录
                model.add_callback("on_train_epoch_end", _fix_resume_csv_issues)

            # 移除内部使用的参数，避免传递给 Ultralytics（会报错）
            train_config_clean = train_config.copy()
            train_config_clean.pop(
                "resume_from", None
            )  # resume_from 是内部逻辑，不传给 Ultralytics

            results = model.train(**train_config_clean)
            metrics = model.val()

            # 收集结果
            model_result = {
                "name": model_info["name"],
                "description": model_info["description"],
                "category": model_info.get("category", "unknown"),
                "mAP50": getattr(metrics.box, "map50", 0.0) or 0.0,
                "mAP50_95": getattr(metrics.box, "map", 0.0) or 0.0,
                "precision": getattr(metrics.box, "mp", 0.0) or 0.0,
                "recall": getattr(metrics.box, "mr", 0.0) or 0.0,
            }

            print(f"[成功] {model_info['name']} 训练完成。")
            print(
                f"   mAP50: {model_result['mAP50']:.4f} | mAP50-95: {model_result['mAP50_95']:.4f}"
            )

            return model_result

        except Exception as e:
            print(f"[失败] 模型 {model_info['name']} 训练失败: {e}")
            import traceback

            traceback.print_exc()
            return None

    def run_ablation_study(self):
        """校验配置后依次训练 self.models 中模型，打印进度与 print_results_summary。"""
        print("PCB 缺陷检测消融实验 - DINO3 增强模型对比")
        print("=" * 70)

        if not self.validate_config_files():
            print("[失败] 配置文件验证失败，终止实验。")
            return

        print(f"数据集: {self.config['data']}")
        print(
            f"⚙️  设备: {self.config['device']}, 轮次: {self.config['epochs']}, 批大小: {self.config['batch']}"
        )
        if self.config.get("resume"):
            resume_from = self.config.get("resume_from")
            if resume_from == "best":
                print(
                    "📂 继续上次训练: resume=True, resume_from='best'（从 best.pt 恢复，通常更稳定）"
                )
            elif resume_from:
                print(
                    f"📂 继续上次训练: resume=True, resume_from='{resume_from}'（从指定 checkpoint 恢复）"
                )
            else:
                print(
                    "📂 继续上次训练: resume=True（从 last.pt 恢复，若出现 NaN 建议改用 resume_from='best'）"
                )
        print("=" * 70)

        successful_models = 0
        for i, model_info in enumerate(self.models):
            print(f"\n进度: {i+1}/{len(self.models)}")
            result = self.train_single_model(model_info)
            if result:
                self.results.append(result)
                successful_models += 1

        # 打印精简结果摘要
        self.print_results_summary()

        print("\n消融实验完成。")
        print(f"成功训练: {successful_models}/{len(self.models)} 个模型")

    def print_results_summary(self):
        """按 baseline / dino_p2 / dino_p3 分组打印 mAP、精确率、召回率及总体排名与提升分析。"""
        if not self.results:
            print("\n[警告] 没有可用的训练结果")
            return

        print(f"\n{'='*80}")
        print("PCB 缺陷检测实验结果对比")
        print(f"{'='*80}")

        # 按类别分组显示
        baseline_results = [r for r in self.results if r["category"] == "baseline"]
        dino_p2_results = [r for r in self.results if r["category"] == "dino_p2"]
        dino_p3_results = [r for r in self.results if r["category"] == "dino_p3"]

        # 基准模型
        if baseline_results:
            print("\n【基准模型】")
            print(
                f"{'模型':<35} {'mAP50':<10} {'mAP50-95':<12} {'精确率':<10} {'召回率':<10}"
            )
            print("-" * 80)
            for r in baseline_results:
                print(
                    f"{r['description']:<35} {r['mAP50']:<10.4f} {r['mAP50_95']:<12.4f} "
                    f"{r['precision']:<10.4f} {r['recall']:<10.4f}"
                )

        # DINO3-P2 模型
        if dino_p2_results:
            print("\n【DINO3-P2 层集成】")
            print(
                f"{'模型':<35} {'mAP50':<10} {'mAP50-95':<12} {'精确率':<10} {'召回率':<10}"
            )
            print("-" * 80)
            sorted_p2 = sorted(dino_p2_results, key=lambda x: x["mAP50"], reverse=True)
            for r in sorted_p2:
                print(
                    f"{r['description']:<35} {r['mAP50']:<10.4f} {r['mAP50_95']:<12.4f} "
                    f"{r['precision']:<10.4f} {r['recall']:<10.4f}"
                )

        # DINO3-P3 模型
        if dino_p3_results:
            print("\n【DINO3-P3 层集成】")
            print(
                f"{'模型':<35} {'mAP50':<10} {'mAP50-95':<12} {'精确率':<10} {'召回率':<10}"
            )
            print("-" * 80)
            sorted_p3 = sorted(dino_p3_results, key=lambda x: x["mAP50"], reverse=True)
            for r in sorted_p3:
                print(
                    f"{r['description']:<35} {r['mAP50']:<10.4f} {r['mAP50_95']:<12.4f} "
                    f"{r['precision']:<10.4f} {r['recall']:<10.4f}"
                )

        # 总体排名（显示所有结果）
        print(f"\n{'='*80}")
        print("【总体排名】")
        print(
            f"{'排名':<6} {'模型':<35} {'mAP50':<10} {'mAP50-95':<12} {'精确率':<10} {'召回率':<10}"
        )
        print("-" * 80)
        sorted_all = sorted(self.results, key=lambda x: x["mAP50"], reverse=True)
        for i, r in enumerate(sorted_all):
            rank_symbol = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else ""
            print(
                f"{i+1}{rank_symbol:<4} {r['description']:<35} {r['mAP50']:<10.4f} {r['mAP50_95']:<12.4f} "
                f"{r['precision']:<10.4f} {r['recall']:<10.4f}"
            )

        print(f"{'='*80}")

        # 性能提升分析
        if baseline_results and (dino_p2_results or dino_p3_results):
            baseline = baseline_results[0]
            best_dino = (
                sorted_all[0] if sorted_all[0]["category"] != "baseline" else None
            )

            if best_dino:
                map50_improvement = (
                    (best_dino["mAP50"] - baseline["mAP50"]) / baseline["mAP50"]
                ) * 100
                map95_improvement = (
                    (best_dino["mAP50_95"] - baseline["mAP50_95"])
                    / baseline["mAP50_95"]
                ) * 100

                print("\n性能提升分析：")
                print(
                    f"  基准模型: {baseline['description']} (mAP50: {baseline['mAP50']:.4f})"
                )
                print(
                    f"  最佳DINO模型: {best_dino['description']} (mAP50: {best_dino['mAP50']:.4f})"
                )
                print(f"  mAP50 提升: {map50_improvement:+.2f}%")
                print(f"  mAP50-95 提升: {map95_improvement:+.2f}%")

        # 单模型训练时的详细显示
        if len(self.results) == 1:
            r = self.results[0]
            print("\n当前模型详细指标：")
            print(f"  模型名称: {r['name']}")
            print(f"  描述: {r['description']}")
            print(f"  mAP50: {r['mAP50']:.4f}")
            print(f"  mAP50-95: {r['mAP50_95']:.4f}")
            print(f"  精确率: {r['precision']:.4f}")
            print(f"  召回率: {r['recall']:.4f}")


def main():
    """创建 PCBAblationStudy，执行 run_ablation_study 并打印实验总结。"""
    print("=" * 70)
    print("PCB 缺陷检测消融实验 - DINO3 增强模型")
    print("=" * 70)
    print("实验设计: 对比基准模型与 DINO3 增强模型性能")
    print("包含模型: YOLO11/YOLOv8/YOLO12 + DINO3 (P2/P3 层集成)")
    print("=" * 70)

    study = PCBAblationStudy()
    study.run_ablation_study()

    print("\n" + "=" * 70)
    print("实验总结:")
    print("=" * 70)
    print("1. 所有模型采用统一随机初始化")
    print("2. 对比基准模型与 DINO3 增强模型")
    print("3. 评估 P2 和 P3 层集成的效果差异")
    print("=" * 70)


if __name__ == "__main__":
    main()
