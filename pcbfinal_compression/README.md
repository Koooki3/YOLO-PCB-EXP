# PCBFINAL Compression Toolkit

`pcbfinal_compression/` 提供一套面向 `PCBFINAL` 检测/分割模型的压缩实验入口，统一做：

- 先导出并验证 FP32 ONNX
- 混合精度量化候选自动生成与判定
- 掩码稀疏化剪枝比例自动扫比
- 压缩前后模型大小、非零参数规模、`mAP50` / `mAP50-95` 与推理速度对比
- 可视化图表、样例预测图和 Markdown 报告导出

## Environment

- Python: `C:\anaconda\envs\kooki\python.exe`
- 脚本默认设置：
  - `KMP_DUPLICATE_LIB_OK=TRUE`
  - `YOLO_AUTOINSTALL=False`
- 当前实现基于：
  - PyTorch / Ultralytics
  - ONNX Runtime
  - `onnxruntime.quantization`
  - `onnxruntime.transformers.float16`

## Entry Point

唯一入口是 [run_compression_study.py](/D:/YOLO_PCB/pcbfinal_compression/run_compression_study.py)。

```bash
C:\anaconda\envs\kooki\python.exe pcbfinal_compression/run_compression_study.py ^
  --model D:\YOLO_PCB\train_ex\PCBFINAL_Ex_12s_960_WassersteinLoss0.7_C14.9503_DINOv3_P2P3P4\weights\best.pt ^
  --data D:\YOLO_PCB\PKU-Market-PCB-ex\pku_market_pcb_ex.yaml ^
  --task detect ^
  --imgsz 960 ^
  --device cpu ^
  --min-retention 0.95 ^
  --quant-profiles dynamic_int8_linear_mixed ^
  --output-root D:\YOLO_PCB\PCBFINAL ^
  --archive-layout pcbfinal
```

## Detect Example

```bash
C:\anaconda\envs\kooki\python.exe pcbfinal_compression/run_compression_study.py ^
  --model D:\YOLO_PCB\train_ex\PCBFINAL_Ex_12s_960_WassersteinLoss0.7_C14.9503_DINOv3_P2P3P4\weights\best.pt ^
  --data D:\YOLO_PCB\PKU-Market-PCB-ex\pku_market_pcb_ex.yaml ^
  --task detect ^
  --imgsz 960 ^
  --device cpu ^
  --val-batch 4 ^
  --calib-images 32 ^
  --sample-images 6 ^
  --min-retention 0.95 ^
  --quant-profiles dynamic_int8_linear_mixed ^
  --output-root D:\YOLO_PCB\PCBFINAL ^
  --archive-layout pcbfinal ^
  --overwrite-output
```

## Segment Example

```bash
C:\anaconda\envs\kooki\python.exe pcbfinal_compression/run_compression_study.py ^
  --model D:\YOLO_PCB\train_segment\PCBFINAL_SegmentData_12s_standardLoss_DINOv3_P2P3P4_seg\weights\best.pt ^
  --data D:\YOLO_PCB\SegmentData\SegmentData.yaml ^
  --task segment ^
  --imgsz 288 ^
  --device cpu ^
  --val-batch 4 ^
  --sample-images 4 ^
  --min-retention 0.95 ^
  --quant-profiles dynamic_int8_linear_mixed ^
  --output-root D:\YOLO_PCB\PCBFINAL ^
  --archive-layout pcbfinal ^
  --overwrite-output
```

## CLI

- `--model`
- `--data`
- `--task {auto,detect,segment}`
- `--imgsz`
- `--device`
- `--val-batch`
- `--calib-images`
- `--sample-images`
- `--min-retention`
- `--quant-profiles`
- `--prune-ratios`
- `--output-root`
- `--archive-layout {pcbfinal,timestamped}`
- `--overwrite-output`

## Quantization Profiles

- `fp16_onnx`
  - 导出 FP32 ONNX，再转换为 FP16 mixed precision。
  - 当前环境只有 ORT CPU provider，通常只作为参考候选。
- `dynamic_int8_linear_mixed`
  - 仅对 `MatMul/Gemm` 执行动态 INT8，优先覆盖 DINO 线性路径。
  - 这是当前默认且优先的混合精度量化路线。
- `static_int8_conv_mixed`
  - 仅对 `Conv` 执行静态 INT8，使用代表性校准图像。

## Pruning Policy

- 只对非 `DINO3Backbone` 的 `Conv2d/Linear` 做全局 L1 掩码剪枝。
- 默认粗扫：
  - `0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60`
- 如果粗扫存在通过项，会自动围绕最高通过比率增加 `0.02` 步长的细扫。

## Pass / Fail Rule

- `detect`
  - `metrics/mAP50(B)` 与 `metrics/mAP50-95(B)` 都必须保留到 baseline 的 `95%` 及以上。
- `segment`
  - `metrics/mAP50(M)` 与 `metrics/mAP50-95(M)` 都必须保留到 baseline 的 `95%` 及以上。
  - Box 指标会记录，但不用于最终通过门槛。

## Console Stages

脚本运行时会按阶段输出状态，控制台与 `run.log` 保持一致：

- `[1/6] Baseline Evaluation`
- `[2/6] FP32 ONNX Export And Validation`
- `[3/6] Quantization Profiles`
- `[4/6] Pruning Sweep`
- `[5/6] Reports and Charts`
- `[6/6] Final Summary`

每个候选都会打印当前状态、模型体积、主/次指标和推理耗时，最终汇总会明确给出最佳量化方案、最佳剪枝方案、最终压缩产物以及 `PCBFINAL` 根目录总览文件路径。

## Outputs

默认输出到：

`D:\YOLO_PCB\PCBFINAL\`

其中 `pcbfinal` 归档布局固定为：

- `PCBFINAL/<task>__<experiment>/baseline/`
- `PCBFINAL/<task>__<experiment>/fp32_onnx/`
- `PCBFINAL/<task>__<experiment>/quantization/`
- `PCBFINAL/<task>__<experiment>/pruning/`
- `PCBFINAL/<task>__<experiment>/final/`
- `PCBFINAL/models/`
- `PCBFINAL/overview.csv`
- `PCBFINAL/overview.json`
- `PCBFINAL/overview.md`

固定包含：

- `summary.csv`
- `summary.json`
- `quantization_candidates.csv`
- `pruning_sweep.csv`
- `best_choice.json`
- `compression_report.md`
- `run.log`
- `charts/*.png`
- `samples/*/*.jpg`
- `baseline/validation.csv`
- `fp32_onnx/validation.csv`
- `final/validation.csv`
- `final/source_vs_compressed.csv`
- `final/source_vs_compressed.json`
- `PCBFINAL/models/*.pt|*.onnx`

## Notes

- 当前环境没有 `tensorrt`，且 ORT 只有 CPU provider；因此默认研究路线是 ORT mixed precision + mask pruning。
- 当前默认压缩顺序是“baseline -> FP32 ONNX -> dynamic_int8_linear_mixed -> pruning sweep -> final archive”。
- 掩码剪枝通常不会显著缩小 `.pt` 物理文件体积，所以报告会同时展示：
  - `checkpoint_bytes`
  - `effective_nonzero_bytes`
  - `nonzero_ratio`
