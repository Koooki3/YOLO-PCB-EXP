# NWD（Normalized Wasserstein Distance）理论与集成说明文档

基于论文 *A Normalized Gaussian Wasserstein Distance for Tiny Object Detection*（Wang et al., arXiv:2110.13389）及本仓库中的 YOLO 集成实现整理。

> **公式预览**：文内数学公式使用 `$...$`（行内）与 `$$...$$`（块级）书写。若在 VS Code/Cursor 中公式未渲染，可安装支持 KaTeX/MathJax 的 Markdown 扩展（如 Markdown All in One、Markdown+Math）后重新打开预览。

---

## 一、理论算法

### 1.1 动机与问题

- **小目标检测难点**：小目标（如 AI-TOD 中 &lt; 16×16 像素，平均绝对尺寸约 12.8 像素，见 Wasserstein.pdf）仅含少量像素，外观信息少，IoU 等基于交并比的度量对**位置偏差非常敏感**。
- **IoU 的局限**（论文 Fig.1–2）：
  - 对微小目标，几个像素的偏移就会导致 IoU 骤降：**6×6 像素目标仅 1 像素对角线偏移，IoU 从 0.53 降到 0.06**；同量偏移下 36×36 目标 IoU 仅从 0.90 降到 0.65，说明 IoU 对尺度敏感、对 tiny 目标过于苛刻。
  - 当预测框与真值**无重叠**（|P∩G|=0）或**完全包含**（|P∩G|=P 或 G）时，IoU-Loss **无法提供有效梯度**；这两种情况在 tiny 目标中很常见。
  - 基于 IoU 的分配易导致小目标正样本过少（论文：AI-TOD 上平均每 gt 的正样本数 &lt; 1）。
- **NWD 的目标**：用**分布距离**度量框的相似度，使其对尺度更不敏感、对位置偏差更平滑（论文 Fig.2 中 NWD–偏差曲线在不同尺度下重合），并在无重叠/完全包含时仍可求导。

### 1.2 边界框的高斯分布建模

将水平边界框视为 2D 高斯分布，中心权重高、向边界递减，与“前景在中心、背景在边界”的直觉一致。

- **水平框表示**：$R = (c_x, c_y, w, h)$（中心、宽、高）。
- **内接椭圆**（等高线）：

  $$
  \frac{(x-\mu_x)^2}{\sigma_x^2} + \frac{(y-\mu_y)^2}{\sigma_y^2} = 1
  $$

  其中 $\mu_x = c_x,\ \mu_y = c_y,\ \sigma_x = w/2,\ \sigma_y = h/2$。
- **2D 高斯** $\mathcal{N}(\mu, \Sigma)$ 与协方差矩阵：

  $$
  \mu = \begin{bmatrix} c_x \\ c_y \end{bmatrix}, \quad
  \Sigma = \begin{bmatrix} w^2/4 & 0 \\ 0 & h^2/4 \end{bmatrix}
  $$

  即水平框 $R = (c_x, c_y, w, h)$ 与高斯 $\mathcal{N}(\mu, \Sigma)$ 一一对应，两框的相似度转化为两高斯的分布距离。

### 1.3 二阶 Wasserstein 距离

对两个 2D 高斯 $\mathcal{N}(m_1, \Sigma_1)$ 与 $\mathcal{N}(m_2, \Sigma_2)$，**二阶 Wasserstein 距离的平方**为（论文 Wasserstein.pdf Eq.5–6）：

$$
W_2^2(\mu_1, \mu_2) = \|m_1 - m_2\|_2^2 + \left\| \Sigma_1^{1/2} - \Sigma_2^{1/2} \right\|_F^2
$$

其中 $\|\cdot\|_F$ 为 Frobenius 范数。对**对角协方差**（水平框对应的高斯），上式可简化为（论文 Eq.7）：

$$
W_2^2(N_a, N_b) = \left\| \left[ c_{xa},\, c_{ya},\, \frac{w_a}{2},\, \frac{h_a}{2} \right]^\top - \left[ c_{xb},\, c_{yb},\, \frac{w_b}{2},\, \frac{h_b}{2} \right]^\top \right\|_2^2
$$

即：

$$
W_2^2 = (c_{xa}-c_{xb})^2 + (c_{ya}-c_{yb})^2 + \left( \frac{w_a-w_b}{2} \right)^2 + \left( \frac{h_a-h_b}{2} \right)^2
$$

### 1.4 归一化 NWD 度量

$W_2^2$ 是距离，为得到与 IoU 类似的**相似度**（0–1，越大越相似），论文定义 **Normalized Wasserstein Distance (NWD)**（Eq.8）：

$$
\mathrm{NWD}(N_a, N_b) = \exp\left( -\frac{\sqrt{W_2^2(N_a, N_b)}}{C} \right)
$$

- $C$：与数据集相关的常数。论文（Wasserstein.pdf Sec.3.2）将 C 取为数据集的**平均绝对尺寸**（AI-TOD 约 12.8 像素），并称在一定范围内鲁棒（实验见 supplementary）。
- 值域：$(0, 1]$，越大表示两框越相似；无重叠或完全包含时仍为正值，可求导。

### 1.5 NWD 作为损失函数

论文将 NWD 用作回归损失（Wasserstein.pdf Eq.9）：

$$
\mathcal{L}_{\mathrm{NWD}} = 1 - \mathrm{NWD}(N_p, N_g)
$$

其中 $N_p$ 为预测框对应的高斯，$N_g$ 为真值框对应的高斯。论文明确指出：在上述两种情形——(1) 预测框与真值无重叠 $|P\cap G|=0$，(2) 一方完全包含另一方 $|P\cap G|=P$ 或 $G$——NWD 损失仍可提供有效梯度，而 IoU-Loss 无法；这两种情况在 tiny 目标中非常常见。

### 1.6 NWD 相对 IoU 的优势（论文总结）

1. **尺度更不敏感**：不同尺度下 NWD–偏差曲线更一致（论文 Fig.2）。
2. **对位置偏差更平滑**：相同像素偏差下，NWD 变化比 IoU 平缓，有利于区分正负样本。
3. **可度量无重叠或包含关系**：不依赖交并比，无重叠时仍有有意义的相似度与梯度。

---

## 二、本仓库中的集成实现

### 2.1 涉及文件与位置

- **文件**：`ultralytics/utils/loss.py`
- **内容**：
  - `wasserstein_distance_sq()`：计算 $W_2^2$（支持 xywh / xyxy）
  - `nwd_metric()`：由 $W_2^2$ 得到 NWD
  - `BboxLoss.forward()`：IoU + NWD 联合边界框损失

### 2.2 核心函数与公式对应

#### 2.2.1 $W_2^2$：`wasserstein_distance_sq(box1, box2, xywh)`

- **含义**：与论文 Eq.7 一致，计算两框对应高斯之间的 $W_2^2$。
- **输入**：
  - `box1`、`box2`：可为 `(N, 4)` 或 `(4, N)`；若 `box2` 为 `(N, 4)` 会在内部转置为 `(4, N)` 以与 `box1` 逐元素对齐。
  - `xywh`：`True` 表示 (cx, cy, w, h)；`False` 表示 (x1, y1, x2, y2)（xyxy）。
- **xyxy → (cx, cy, w, h)**：
  - $c_x = (x_1+x_2)/2,\ c_y = (y_1+y_2)/2,\ w = x_2-x_1,\ h = y_2-y_1$
- **计算**：
  - $p_1 = (c_{x1}-c_{x2})^2 + (c_{y1}-c_{y2})^2$
  - $p_2 = ((w_1-w_2)/2)^2 + ((h_1-h_2)/2)^2$
  - 返回 $W_2^2 = p_1 + p_2$（与论文一致）。

#### 2.2.2 NWD：`nwd_metric(pred_boxes, target_boxes, xywh=False, C=20.0)`

- **含义**：对应论文 Eq.8，$\mathrm{NWD} = \exp(-\sqrt{W_2^2}/C)$。
- **实现要点**：
  - 内部调用 `wasserstein_distance_sq(pred_boxes.T, target_boxes, xywh=xywh)` 得到 $W_2^2$（向量，逐样本）。
  - 使用 $\sqrt{W_2^2 + \epsilon}$（$\epsilon=10^{-6}$）避免零梯度与数值问题。
- **默认 C=20.0**：与论文“与数据集平均尺寸相关”一致；可根据数据集（如 PCB 小目标平均尺寸）调整。

### 2.3 与 YOLO 训练的集成：`BboxLoss.forward`

- **输入**（与原生 YOLO 一致）：
  - `pred_bboxes`、`target_bboxes`：xyxy，形状含 batch 与 anchor 维度；在 loss 内按 `fg_mask` 取前景。
  - `target_scores`、`target_scores_sum`、`fg_mask`：来自 TAL 分配，用于加权与归一化。
- **流程简述**：
  1. 用 `fg_mask` 取前景预测与真值：`pred_fg`、`target_fg`（xyxy）。
  2. **IoU**：`bbox_iou(pred_fg, target_fg, xywh=False, CIoU=True)`，与原有一致。
  3. **NWD**：`nwd_metric(pred_fg, target_fg, xywh=False, C=20.0)`，得到逐样本 NWD。
  4. **加权联合损失**（与论文 Eq.9 的 $1-\mathrm{NWD}$ 一致，并保留 YOLO 的 score 加权；可调参数 $\mathrm{rate}\in[0,1]$ 为 NWD 权重）：
     $$
     \mathcal{L}_{\mathrm{iou}} = (1-\mathrm{rate})\, \frac{\sum (1 - \mathrm{IoU}) \cdot w}{\sum w}
     + \mathrm{rate}\, \frac{\sum (1 - \mathrm{NWD}) \cdot w}{\sum w}
     $$
     $\mathrm{rate}=0.5$ 即 IoU:NWD=1:1；本仓库在 PCB 缺陷上推荐 **rate=0.7**（见第四节）。其中 $w$ 为 `target_scores` 在前景上的和（`weight`），与原生 IoU 损失使用同一加权方式。
  5. DFL 损失不变；最终返回 `(loss_iou, loss_dfl)`。

### 2.4 数据格式与常数

| 项目         | 说明                                                                                                                                                                               |
| :--------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 框格式        | YOLO 内部使用 xyxy；在 `xywh=False` 下会先转为 (cx, cy, w, h) 再算 $W_2^2$ 与 NWD。                                                                                                             |
| 常数 C       | 默认 20.0；论文（Wasserstein.pdf）将 C 取为数据集平均绝对尺寸（AI-TOD 约 12.8 像素），并称在一定范围内鲁棒；PCB 小目标可尝试 10–25 间调参。                                                                                    |
| IoU/NWD 比例 | 可调参数 `rate`：$\mathcal{L}_{\mathrm{iou}} = (1-\mathrm{rate})\cdot(1-\mathrm{IoU}) + \mathrm{rate}\cdot(1-\mathrm{NWD})$。本仓库在 PCB 缺陷数据上实验表明 **rate=0.7** 时 mAP50-95 与综合评分最优，详见第四节。 |

#### 2.4.1 基于 YOLO txt 标注统计数据集 C（理论 + 脚本 + 实验）

**理论口径（与论文一致）**：论文将 $C$ 设置为数据集的**平均绝对尺寸**（average absolute size）。在仅有 YOLO txt 标注（归一化 $w,h$）且训练输入统一为 `imgsz×imgsz` 的情况下，可用以下方式近似计算绝对尺寸（单位：像素）：

- **像素宽高**：$w_{px}=w\cdot \mathrm{imgsz},\ h_{px}=h\cdot \mathrm{imgsz}$
- **绝对尺寸标量**（默认推荐，鲁棒且常用）：$s=\sqrt{w_{px}h_{px}}$
- **论文式 C**：$C=\mathrm{mean}(s)$

> 说明：若训练/推理采用 letterbox 且原图长宽比差异很大，更精确的做法是用每张原图的 $W,H$ 将归一化框还原到原图像素后再统计；但在你当前“训练统一 960×960 输入”的设定下，上述近似足以作为 C 的起点，并可在小范围内微调。

**脚本**：`compute_nwd_c.py`（工程根目录）  
功能：递归扫描 `--labels` 指定路径下所有 `*.txt`（不依赖 train/val 划分），解析 YOLO 行格式 `<cls> <cx> <cy> <w> <h>`，将 `w,h` 按 `--imgsz` 转为像素，并输出 C 与分布统计（min/分位数/median/trimmed mean/mean）。

**使用示例**（你的环境）：

```bash
& C:\anaconda\envs\kooki\python.exe d:/YOLO_PCB/compute_nwd_c.py --labels "D:\YOLO_PCB\PKU-Market-PCB-ex\labels" --imgsz 960 --size-mode sqrt_area
```

其中 `--size-mode sqrt_area` 即 $s=\sqrt{w_{px}h_{px}}$（推荐）。

**实验 1：PKU-Market-PCB-ex**

- **labels_root**：`D:\YOLO_PCB\PKU-Market-PCB-ex\labels`
- **imgsz 假设**：960×960
- **size_mode**：sqrt_area（$s=\sqrt{w_{px}h_{px}}$）
- **统计结果**（来自脚本输出）：

| 指标 | 值（像素） |
| :--- | :--- |
| recursive txt | 236 files |
| total boxes | 601 |
| min | 3.85763 |
| p10 | 6.78725 |
| p30 | 9.12298 |
| median (p50) | 12.5672 |
| p70 | 16.0031 |
| p90 | 28.4262 |
| max | 48.626 |
| trimmed_mean_10% | 13.461 |
| **mean（论文式 C）** | **14.9503** |

按论文“平均绝对尺寸”定义，该数据集的 **$C\approx 14.95$**（像素）。

**实验 2：PKU-Market-PCB**

- **labels_root**：`D:\YOLO_PCB\PKU-Market-PCB\labels`
- **imgsz 假设**：960×960
- **size_mode**：sqrt_area（$s=\sqrt{w_{px}h_{px}}$）
- **统计结果**（来自脚本输出）：

| 指标 | 值（像素） |
| :--- | :--- |
| recursive txt | 693 files |
| total boxes | 2953 |
| min | 11.3787 |
| p10 | 18.4174 |
| p30 | 22.9582 |
| median (p50) | 26.1046 |
| p70 | 29.9708 |
| p90 | 37.0391 |
| max | 87.674 |
| trimmed_mean_10% | 26.6525 |
| **mean（论文式 C）** | **27.2664** |

按论文“平均绝对尺寸”定义，该数据集的 **$C\approx 27.27$**（像素）。

**如何使用这些统计量**：

- **论文复现口径**：直接使用 `mean` 作为 $C$。
- **长尾/极端大框较多时**：可对比 `median` 或 `p30`（更稳健），再在一个小范围内试验（例如 \(\{0.8C,\,1.0C,\,1.2C\}\)）观察 mAP50-95。

### 2.5 调用链（训练时）

1. `v8DetectionLoss`（或等价检测 Loss 类）在每步前向中得到 `pred_bboxes`、`target_bboxes` 等。
2. 调用 `self.bbox_loss(pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask)`。
3. `BboxLoss.forward` 内计算 CIoU + NWD，并得到 `loss_iou` 与 `loss_dfl`，与分类损失等一起反传。

### 2.6 与论文用法的对应关系

- **论文（Wasserstein.pdf）**：NWD 可替代 IoU 用于 (1) 标签分配、(2) NMS、(3) 回归损失；三者同时替换时在 AI-TOD 上提升最大（如 Faster R-CNN 表 3：RPN 三模块全用 NWD 时 12 epoch AP 17.8%），但若六模块全用 NWD 需更长训练（24 epoch 才接近收敛）。
- **本实现**：仅在 **(3) 回归损失** 中集成 NWD，与 CIoU 以可调 `rate` 加权联合；标签分配与 NMS 仍使用原有 IoU/置信度逻辑。这样在保持与 Ultralytics 评测（mAP 基于 IoU）一致的前提下，用 NWD 改善小目标回归梯度；若在分配或 NMS 中引入 NWD，最优 `rate` 可能略有变化。

---

## 三、小结

- **理论**：NWD 将框建模为 2D 高斯，用 $W_2^2$ 的指数归一化得到相似度，并用作 $\mathcal{L}_{\mathrm{NWD}} = 1 - \mathrm{NWD}$，适合小目标与无重叠/包含情况。
- **实现**：`loss.py` 中 $W_2^2$ 与 NWD 的公式与论文一致，支持 xyxy；在 `BboxLoss` 中以加权和形式与 CIoU 联合，保留 YOLO 的 score 加权与 DFL，便于复现与调参。
- **调参**：联合损失采用可调权重 $\mathcal{L}_{\mathrm{iou}} = (1-\mathrm{rate})(1-\mathrm{IoU}) + \mathrm{rate}(1-\mathrm{NWD})$；在 PCB 缺陷检测上 **rate=0.7** 取得最佳 mAP50-95 与综合评分，详见第四节。

---

## 四、IoU–NWD 混合权重（rate）实验分析与结论

本节基于同一 YOLO 架构与同一 PCB 数据集，在不同 **rate**（NWD 在 box 回归损失中的权重）下的训练结果，从理论、Ultralytics 训练逻辑与实验数据三方面说明：**为何 rate=0.7 时训练最稳、最佳模型（mAP50-95）最好**。实验命名格式：`Ex_12s_960_WassersteinLoss_0.x`，对应 `rate=x`；best 模型由 Ultralytics 按验证集 **mAP50-95** 选出的 epoch 保存。

### 4.1 论文（Wasserstein.pdf）要点回顾

- **IoU 敏感性**（论文 Fig.1）：6×6 像素目标仅 1 像素对角线偏移，IoU 从 0.53 骤降至 0.06；36×36 目标同量偏移时 IoU 仅从 0.90 降至 0.65。说明 IoU 对 tiny 目标**尺度敏感、离散位置偏差惩罚过重**。
- **NWD 特性**（论文 Fig.2）：(1) 尺度不变性——不同尺度下 NWD–偏差曲线重合；(2) 对位置偏差更平滑；(3) 在 $|P\cap G|=0$ 或 $|P\cap G|=P/G$ 时仍可度量相似度并给出梯度。
- **损失形式**（论文 Eq.9）：$\mathcal{L}_{\mathrm{NWD}} = 1 - \mathrm{NWD}(N_p, N_g)$，在无重叠或完全包含时仍可优化。
- **常数 C**：论文将 C 设为数据集平均绝对尺寸（AI-TOD 约 12.8），并称在一定范围内鲁棒；本实现默认 C=20.0，可按 PCB 目标尺度微调。对本文两套 PCB 数据的 YOLO 标注，按论文口径统计得到：PKU-Market-PCB-ex **$C\approx 14.95$**，PKU-Market-PCB **$C\approx 27.27$**（见 §2.4.1）。
- **应用范围**：论文在标签分配、NMS、回归损失三处替换 IoU；本仓库仅在回归损失中与 CIoU 加权联合，分配与 NMS 仍用 IoU，以保持与 mAP 评测一致。

### 4.2 rate sweep 实验结果摘要

在 **loss** 中采用 $\mathcal{L}_{\mathrm{iou}} = (1-\mathrm{rate})(1-\mathrm{IoU}) + \mathrm{rate}(1-\mathrm{NWD})$，其它超参一致，得到各 rate 下 best 模型的指标（best 由验证集 mAP50-95 最大 epoch 确定）：

| rate | boxloss | clsloss | dflloss | mAP50 | mAP50-95 | best epoch | score (0.1×mAP50+0.9×mAP50-95) |
|------|---------|---------|---------|-------|----------|------------|----------------------------------|
| 0.1  | 0.946   | 0.872   | 0.680   | 0.704 | 0.332    | 308        | 0.369                            |
| 0.2  | 1.733   | 1.161   | 0.912   | 0.606 | 0.304    | 256        | 0.334                            |
| 0.3  | 1.388   | 0.986   | 0.855   | 0.643 | 0.329    | 307        | 0.361                            |
| 0.4  | 1.029   | 0.845   | 0.858   | 0.688 | 0.333    | 431        | 0.368                            |
| 0.5  | 0.870   | 0.772   | 0.857   | 0.723 | 0.369    | 416        | 0.404                            |
| 0.6  | 1.116   | 1.320   | 0.905   | 0.550 | 0.278    | 214        | 0.305                            |
| **0.7** | **0.517** | **0.720** | **0.837** | **0.710** | **0.379** | **443** | **0.412** |
| 0.8  | 0.389   | 0.756   | 0.838   | 0.656 | 0.358    | 443        | 0.388                            |
| 0.9  | 0.278   | 0.882   | 0.863   | 0.682 | 0.370    | 365        | 0.402                            |
| 1.0  | 0.169   | 1.691   | 0.985   | 0.439 | 0.183    | 138        | 0.208                            |

主要现象：

- **boxloss** 随 rate 增大整体下降，rate=1 时最小，说明纯 NWD 在“训练意义上的几何距离”上极易优化。
- **mAP50-95** 非单调：从 0.1→0.5→**0.7** 上升并在 **0.7 达到最高 0.379**；0.8/0.9 略降，**rate=1 大幅跌至 0.183**。
- **rate=1** 时 best epoch 仅 138，早早就达到“最优”（实为在 NWD 空间最优，与 IoU 评测错位）；**rate=0.7** 的 best 出现在 443 epoch，训练周期长、收敛稳。

### 4.3 为何 rate 过小或过大都不利

**IoU 与 NWD 的度量差异**

- **IoU（CIoU）**：与 COCO 式 mAP 完全对齐，但对小目标与轻微偏移**过于敏感**；无重叠时梯度消失。PCB 缺陷多为细小目标，IoU 易导致正样本过少、梯度不稳定（与论文在 AI-TOD 上“每 gt 正样本数&lt;1”一致）。
- **NWD**：对尺度与位置偏差更平滑，无重叠时仍有梯度；但**评测与 NMS 仍基于 IoU**，纯优化 NWD 会与“IoU 空间的最优”产生**度量偏差**。

**rate 过小（如 0.1–0.4）**

- NWD 仅弱辅助，监督仍以 IoU 为主；小目标上 IoU 的敏感性与梯度问题改善有限，mAP50-95 提升有限（约 0.33–0.36）。

**rate 过大（尤其 1.0）**

- 训练目标近似“NWD 检测器”：boxloss 数值很小、收敛很快，但优化的是高斯分布距离，与 IoU 阈值边界不对齐。
- 论文指出 NWD 可同时用于分配、NMS、损失；本实现**仅损失用 NWD**，若损失中完全去掉 IoU，会与分配/NMS/评测的 IoU 逻辑不一致，导致 mAP50-95 崩塌、best 过早出现。

**rate≈0.7 的折中**

- **NWD 权重足够大**：小目标与无重叠/包含样本获得稳定梯度，缓解 IoU 对微小偏差的“过敏”（与论文 Fig.2 的平滑性一致）。
- **保留足够 IoU 权重**：回归目标仍与 IoU 评测、分配、NMS 对齐，避免度量错位。
- 在“小目标多、评测仍为 IoU”的 PCB 场景下，0.7 在**梯度质量**与**评测一致性**之间取得最佳平衡，故 mAP50-95 与综合评分最高。

### 4.4 与 Ultralytics 训练逻辑的对应

- **损失结构**：总损失为 `box_loss×box_gain + cls_loss×cls_gain + dfl_loss×dfl_gain`（如 box=7.5, cls=0.5, dfl=1.5）；`box_loss` 即上述 IoU–NWD 加权和。`rate` 只改变 box 回归项的组成，不改变 gain。
- **best.pt 选择**：按验证集 **mAP50-95** 最大 epoch 保存；因此“训练目标”与“选 best 的指标”一致时更有利。rate=0.7 时两者一致性好；rate=1 时优化目标与 mAP 脱节，best 出现早且指标差。
- **训练时长与收敛**：rate=0.7 的 best 出现在 443 epoch（cosine lr + patience=50），说明中后期仍在稳定提升；rate=1 在 138 epoch 即达“最优”，符合“在 NWD 空间早收敛、在 IoU 空间却偏离”的现象。

### 4.5 PCB 缺陷数据与 rate 选择

- PCB 缺陷含大量细小目标（划痕、残铜等），与论文关注的 tiny object（AI-TOD 平均约 12.8 像素）类似；IoU 对 1–2 像素偏差惩罚过重，易导致正样本不足。
- NWD 对中心与尺寸的平滑度量更适合这类目标；但最终质检仍依赖 IoU 阈值判定，故需保留 IoU 分量。
- 经验上：**NWD 权重大于 0.5 以强化小目标梯度，但小于 1 以保持与 IoU 评测对齐**；在本批实验与超参下 **rate=0.7** 为最优折中。

### 4.6 结论归纳

- **rate 太小**：NWD 仅弱辅助，IoU 主导，小目标梯度与正样本问题改善有限 → mAP50-95 一般。
- **rate 太大（尤其 1）**：训练目标变成 NWD 空间最优，与 IoU 分配/NMS/评测错位 → boxloss 虽小，mAP50-95 显著下降，best 过早出现。
- **rate=0.7**：NWD 足够强以稳定 tiny 缺陷训练、平滑位置偏差，又保留足够 IoU 以对齐评测与 pipeline → **训练最稳、best epoch 最靠后、mAP50-95 与 score 最高**。

若后续在**标签分配或 NMS** 中引入 NWD（对齐论文全流程），最优 rate 可能略有变化，可再扫一遍 0.5–0.8。

---

**参考文献**
Wang J., Xu C., Yang W., Yu L., *A Normalized Gaussian Wasserstein Distance for Tiny Object Detection*, arXiv:2110.13389. 本仓库同时参考本地 `Wasserstein.pdf` 以核对公式与实验设置。
代码：<https://github.com/jwwangchn/NWD>
