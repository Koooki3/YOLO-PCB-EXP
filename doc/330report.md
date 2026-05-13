PCBFINAL 两周的结构调整、轻量化尝试和压缩测试

## 更新

1. PCBFINAL 的 detect 结构轻量化  
2. segment 分支测试
3. 量化/剪枝压缩实验流程

## PCBFINAL 结构

### detect：`P2P3P4` 三尺度检测头指标优于其他组合

| 实验                                                    | 结构                            | 参数量(M) | best mAP50(B) | best mAP50-95(B) | 相对 4-scale 参数变化 | 备注           |
| ----------------------------------------------------- | ----------------------------- | -----: | ------------: | ---------------: | --------------: | ------------ |
| `Ex_12s_960_WassersteinLoss0.7_C14.9503_DINOP2_FIXED` | 4-scale 参考分支，保留 `P2/P3/P4/P5` |  33.24 |       0.79061 |          0.45550 |              基准 | 精度最好，但也最重    |
| `PCBFINAL_..._P3P4`                                   | 2-scale，轻量化比较激进               |  26.36 |       0.73322 |          0.37548 |         -20.68% | 精度掉得比较明显     |
| `PCBFINAL_..._P2P3P4`                                 | 3-scale，保留浅层 `P2` 和中层 `P3/P4` |  26.54 |       0.78318 |          0.44164 |         -20.15% | 最均衡          |
| `PCBFINAL_..._P3P4P5`                                 | 3-scale，但保留的是中深层 `P3/P4/P5`   |  33.11 |       0.66270 |          0.34403 |          -0.39% | 几乎没减重，精度反而最差 |

- 相比 4-scale 参考分支减参 `20.15%`，但 `mAP50(B)` 还保留 `99.06%`，`mAP50-95(B)` 还保留 `96.96%`
- `P3P4` 组合说明P2和P5同时丢失是不可靠的，尤其在小缺陷和复杂纹理场景
- `P3P4P5` 说明深层语义不能直接替代浅层细节

结合已有的 [results_analyse/feature_compare_dino_nwd/analysis_report.md](results_analyse/feature_compare_dino_nwd/analysis_report.md)：

- detect 检测收益主层还是在 `P3`
- 浅层 `P2` 对小缺陷定位有帮助
- 所以 `P2P3P4` 这个组合是合理的

![PCBFINAL 检测结构对比](results_analyse/train_ex/val_03241634/training_comparison_perf.png)

### segment

| 实验 | 参数量(M) | best mAP50(M) | best mAP50-95(M) | 现阶段怎么判断 |
|---|---:|---:|---:|---|
| `PCBFINAL_SegmentData_12s_standardLoss_DINOv3_P2P3P4_seg` | 26.99 | 0.95333 | 0.56188 | 当前可用的分割主线 |
| `PCBFINAL_SegmentData_12s_standardLoss_DINOv3_P2P3P4_seg_ehanced` | 27.53 | 0.16331 | 0.03719 | 收敛较差 |

- 分割以标准版 `P2P3P4_seg` 为准

再结合 [results_analyse/feature_compare_segmentdata/analysis_report.md](results_analyse/feature_compare_segmentdata/analysis_report.md) 的已有分析：

- DINO 对分割是有效的
- 但 `DINO+NWD` 在 SegmentData 上更像是把模型往 recall 和浅层前景增强方向推
- 它并没有像 detect 那样，继续把最终 mask mAP 再往上抬

![SegmentData 定性对比示例](results_analyse/feature_compare_segmentdata/class_oneflows/paper_qualitative_oneflow_mouse_bite_01_mouse_bite_01_2.jpg)

## 压缩实验

### 新代码

这次已经把压缩实验工具单独整理到了：

- 入口脚本：[pcbfinal_compression/run_compression_study.py](pcbfinal_compression/run_compression_study.py)
- 使用说明：[pcbfinal_compression/README.md](pcbfinal_compression/README.md)

1. 自动判断混合精度/量化候选过线  
2. 自动扫描剪枝比率  
3. 自动对比压缩前后的模型大小、`mAP50`、`mAP50-95` 和推理耗时  
4. detect / segment 共用一套流程  
5. 输出完整日志、图表、样例图和 Markdown 报告  

### detect：以 `PCBFINAL_..._P2P3P4` 为对象的实测结果

量化/剪枝指标阈值

- `mAP50(B)` ≥ `0.70774`
- `mAP50-95(B)` ≥ `0.39956`

#### 量化结果

| 实验                          | 状态          |           大小 | mAP50(B) | mAP50-95(B) | 相对 baseline 保留率 | 推理耗时(ms/img) | 备注           |
| --------------------------- | ----------- | -----------: | -------: | ----------: | --------------: | -----------: | ------------ |
| baseline PT                 | baseline    | 53,866,602 B |  0.78638 |     0.44395 |     100% / 100% |       615.03 | 基线           |
| `dynamic_int8_linear_mixed` | PASS        | 42,735,465 B |  0.76510 |     0.42955 | 97.29% / 96.76% |       425.15 | 参数搜索下最优量化方案  |
| `static_int8_conv_mixed`    | FAIL        | 93,000,253 B |  0.74764 |     0.39380 | 95.07% / 88.70% |       446.29 | 超出指标阈值       |
| `fp16_onnx`                 | unsupported | 53,455,870 B |        - |           - |               - |            - | 本机 ORT 初始化失败 |

- 文件体积下降了 `20.66%`
- `mAP50(B)` 和 `mAP50-95(B)` 保留了 `97.29% / 96.76%`
- CPU 推理耗时下降了 `30.87%`

#### 剪枝结果

| 实验           | 状态       |       物理文件大小 |        有效非零字节 |    非零比例 | mAP50(B) | mAP50-95(B) | 相对 baseline 保留率 | 推理耗时(ms/img) |
| ------------ | -------- | -----------: | ------------: | ------: | -------: | ----------: | --------------: | -----------: |
| baseline PT  | baseline | 53,866,602 B | 106,161,888 B | 99.998% |  0.78638 |     0.44395 |     100% / 100% |       615.03 |
| `prune_0.52` | PASS     | 53,917,081 B | 100,303,992 B |  94.48% |  0.71920 |     0.40777 | 91.46% / 91.85% |       606.06 |
| `prune_0.54` | FAIL     | 53,917,081 B | 100,078,688 B |  94.15% |  0.69853 |     0.39900 | 88.83% / 89.87% |       670.22 |

- 最大安全剪枝率 `0.52`
- `.pt` 物理大小基本不变

## detect 和 segment 适配

detect 和 segment 模型能共用：

- baseline、量化候选和剪枝候选统一用 `Ultralytics YOLO.val()`
- detect 用 `metrics/mAP50(B)` 和 `metrics/mAP50-95(B)` 判定
- segment 用 `metrics/mAP50(M)` 和 `metrics/mAP50-95(M)` 判定
- ONNX 加载时会显式指定 `task`，避免 segment ONNX 被当成 detect
- detect 导出 bbox overlay，segment 导出 mask overlay

### segment 的兼容性验证结果

- `mAP50(M)` ≥ `0.85775`
- `mAP50-95(M)` ≥ `0.49874`

| 候选 | 状态 | 大小 | mAP50(M) | mAP50-95(M) | 相对 baseline 保留率 | 推理耗时(ms/img) | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| baseline PT | baseline | 54,492,660 B | 0.95305 | 0.55415 | 100% / 100% | 134.15 | 基线 |
| `dynamic_int8_linear_mixed` | PASS | 43,174,477 B | 0.90092 | 0.50957 | 94.53% / 91.95% | 67.56 | 当前最优量化方案 |
| `prune_0.21` | PASS | 54,547,149 B | 0.94876 | 0.53645 | 99.55% / 96.81% | 189.29 | 当前最大安全剪枝率 |

- 文件体积下降了 `20.77%`
- mask 主指标都能过 90% 
- CPU 推理速度也有明显提升，降幅接近 `49.64%`

- 能找到安全比率 `0.21`
- segment 出现延迟上升

## PCBFINAL 的混合精度量化和剪枝

### DINOv3 

1. DINOv3 官方仓库本身要求 `PyTorch >= 2.7.1`，而且官方说明里也明确说训练/评测代码主要在 Linux 上测试过。当前仓库虽然能在 Windows 跑，但这不是一个和官方完全一致的部署环境 
2. ONNX Runtime 官方 mixed precision 文档也写得很明确：CPU 版 ORT 不支持 float16 ops，自动 mixed precision 工具需要 GPU。可现在 `kooki` 环境里的 ORT 只有 `CPUExecutionProvider` 
3.  detect 实测中，`fp16_onnx` 已经在 DINO 相关 `SimplifiedLayerNormFusion` 节点初始化时报错了
4. 从现有特征对比报告来看，DINOv3 的收益主要集中在浅层/中层表征增强

### 合理的模型压缩路线

1. 先导出并验证 FP32 ONNX  
2. 优先对 `MatMul/Gemm` 做 `dynamic_int8_linear_mixed`  
3. 统一重验 detect / segment 的 `mAP50` 和 `mAP50-95`  
4. 只保留双指标都不低于 baseline 95%（或更高）的候选  
5. 对非 `DINO3Backbone` 的 `Conv2d/Linear` 做全局 L1 掩码剪枝扫比 （可选）
6. 同时记录文件体积、有效非零字节、速度和精度

这条路线的好处是：

- 风险可控下做到尽可能保证压缩后的模型精度
- detect 和 segment 均适配
- 结果是可量化、可复验、可交付的

### 官方资料

- ONNX Runtime Quantization: [https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- ONNX Runtime Float16 / Mixed Precision: [https://onnxruntime.ai/docs/performance/model-optimizations/float16.html](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html)
- PyTorch Pruning Tutorial: [https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html)
- PyTorch Semi-Structured Sparse Tutorial: [https://docs.pytorch.org/tutorials/advanced/semi_structured_sparse.html](https://docs.pytorch.org/tutorials/advanced/semi_structured_sparse.html)
- NVIDIA Model Optimizer: [https://nvidia.github.io/Model-Optimizer/](https://nvidia.github.io/Model-Optimizer/)
- DINOv3 Official Repository: [https://github.com/facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)
