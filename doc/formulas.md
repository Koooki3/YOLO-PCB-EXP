# 公式与原理整理（YOLO / NWD / DINO）

## 0. 来源顺序（按正文使用顺序）

1. 本地文献：`D:\桌面\毕设\参考文献\included\ALH.pdf`（用于 YOLOv8/YOLO11 在 PCB 场景下的结构与实验背景）[1]。  
2. 本地文档：`D:\YOLO_PCB\NWD理论与集成说明文档.md`（用于 NWD 推导与工程口径核对）[2]。  
3. 本地代码：`D:\YOLO_PCB\ultralytics\utils\loss.py`、`D:\YOLO_PCB\ultralytics\utils\tal.py`（用于 YOLOv8/11 训练目标与 TAL 公式口径核对）[3-4]。  
4. 官方/原始公开来源：Ultralytics YOLO11 文档、YOLOv12 arXiv、NWD arXiv、DINO/DINOv2/DINOv3 arXiv[5-10]。  

> 说明：你要求“官方口径优先”。因此“YOLO12”采用可公开核验的学术论文 `YOLOv12: Attention-Centric Real-Time Object Detectors` 口径[6]，并与 Ultralytics 的 YOLO11 体系明确区分。

---

## 1. YOLOv8 / YOLO11 / YOLO12 关键公式与原理

## 1.1 YOLOv8（及 YOLO11 继承口径）训练目标

结合本地 `ultralytics` 实现，检测主损失可写为：

$$
\mathcal{L}_{det}=\lambda_{box}\mathcal{L}_{box}+\lambda_{cls}\mathcal{L}_{cls}+\lambda_{dfl}\mathcal{L}_{dfl}
$$

其中三项分别为：

1) 分类 BCE 项（与目标分配后的 soft target）[3]：

$$
\mathcal{L}_{cls}=\frac{1}{S}\sum_{i}\mathrm{BCE}\left(\hat{\mathbf{p}}_i,\mathbf{y}_i\right)
$$

2) 边框回归项（CIoU 形式）[3]：

$$
\mathcal{L}_{box}=\frac{1}{S}\sum_{i}w_i\left(1-\mathrm{CIoU}\left(\hat{b}_i,b_i\right)\right)
$$

3) DFL（Distribution Focal Loss）离散回归项[3]：

$$
\mathcal{L}_{dfl}=\frac{1}{S}\sum_i w_i\!\left[-\!\!\sum_{k}\!\left(q_{ik}^{(l)}\log \hat{p}_{ik}^{(l)}+q_{ik}^{(r)}\log \hat{p}_{ik}^{(r)}\right)\right]
$$

其中 $S=\sum_i y_i$ 为归一化项，$w_i$ 为前景样本权重。  

### Task-Aligned Assigner（TAL）核心度量

YOLOv8/11 的正样本匹配常用任务对齐度量[4]：

$$
\mathrm{align}_{ij}=s_{ij}^{\alpha}\cdot u_{ij}^{\beta}
$$

其中 $s_{ij}$ 为分类得分、$u_{ij}$ 为 IoU（实现中可取 CIoU 裁剪到非负），$\alpha,\beta$ 为超参数。该式体现“分类置信 + 定位质量”联合选正样本。

## 1.2 YOLO11 的迭代要点（官方文档可核验部分）

Ultralytics 官方说明 YOLO11 相比 YOLOv8 在 backbone/neck 与训练流程上做效率-精度改进，但**未单独发布完整学术论文**[5]。因此数学目标函数层面可视作延续上式（检测仍是 box/cls/dfl 主线，任务分配仍基于 TAL 机制）[3-5]。

## 1.3 YOLO12（attention-centric 论文口径）

YOLOv12 公开论文强调“以注意力为中心，同时保持实时性”[6]。在可迁移实现中，关键数学部件是多头自注意力：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

$$
\mathrm{MHA}(X)=\mathrm{Concat}\left(\mathrm{head}_1,\dots,\mathrm{head}_h\right)W^O
$$

并与残差/前馈构成块：

$$
X' = X + \mathrm{MHA}(X),\qquad
Y = X' + \mathrm{FFN}(X')
$$

原理上，YOLO12 的“迭代核心”是把 CNN 主导的特征交互替换为更强的全局建模机制，同时通过结构设计保持延迟可控[6]。

---

## 2. 归一化 Wasserstein 距离（NWD）损失：关键公式与原理

以下以论文与本地实现一致口径给出[2,7]。

## 2.1 边框高斯建模

对边框 $R=(c_x,c_y,w,h)$，定义二维高斯：

$$
\mu=\begin{bmatrix}c_x\\c_y\end{bmatrix},\qquad
\Sigma=\begin{bmatrix}w^2/4&0\\0&h^2/4\end{bmatrix}
$$

## 2.2 二阶 Wasserstein 距离（闭式）

对两个高斯 $N_a(\mu_a,\Sigma_a),N_b(\mu_b,\Sigma_b)$：

$$
W_2^2(N_a,N_b)=\|\mu_a-\mu_b\|_2^2+\left\|\Sigma_a^{1/2}-\Sigma_b^{1/2}\right\|_F^2
$$

对水平框对应对角协方差可化简为：

$$
W_2^2=(c_{xa}-c_{xb})^2+(c_{ya}-c_{yb})^2+\left(\frac{w_a-w_b}{2}\right)^2+\left(\frac{h_a-h_b}{2}\right)^2
$$

## 2.3 归一化相似度与损失

$$
\mathrm{NWD}(N_a,N_b)=\exp\!\left(-\frac{\sqrt{W_2^2(N_a,N_b)}}{C}\right)
$$

$$
\mathcal{L}_{NWD}=1-\mathrm{NWD}(N_p,N_g)
$$

其中 $C$ 与数据集目标绝对尺度相关（论文采用数据集平均绝对尺寸）[7]。  
相较 IoU 度量，NWD 对 tiny object 的微小像素偏移更平滑，且在“无重叠/包含”情形仍保留可学习信号[7]。

## 2.4 与 YOLO 回归联合（工程常见形式）

在本地工程说明中，常见混合写法为[2-3]：

$$
\mathcal{L}_{box}^{mix}=(1-r)\,\mathcal{L}_{IoU}+r\,\mathcal{L}_{NWD},\quad r\in[0,1]
$$

该式用于在“IoU 对齐评测标准”与“NWD 对小目标友好梯度”之间做折中。

---

## 3. DINO 迭代（DINO -> DINOv2 -> DINOv3）与特征增强公式

## 3.1 DINO（Self-Distillation with No Labels）基础公式

学生-教师分支的核心是跨视角蒸馏交叉熵[8]：

$$
\mathcal{L}_{DINO}
=-\sum_{x\in\mathcal{V}_s}\sum_{x'\in\mathcal{V}_t}
p_t(x')^\top\log p_s(x)
$$

其中

$$
p_s=\mathrm{softmax}\!\left(\frac{z_s}{\tau_s}\right),\qquad
p_t=\mathrm{softmax}\!\left(\frac{z_t-c}{\tau_t}\right)
$$

教师由 EMA 更新：

$$
\theta_t \leftarrow m\theta_t+(1-m)\theta_s
$$

中心向量更新（用于抑制塌缩）：

$$
c\leftarrow m_c c+(1-m_c)\,\frac{1}{B}\sum_{b=1}^{B}z_t^{(b)}
$$

## 3.2 DINOv2：图像级 + patch 级目标协同

DINOv2 延续教师-学生框架，并结合图像级与局部 patch 级监督（常见写法）[9]：

$$
\mathcal{L}_{DINOv2}=\lambda_g\mathcal{L}_{global}+\lambda_p\mathcal{L}_{patch}+\lambda_r\mathcal{L}_{reg}
$$

原理：通过全局一致性 + 局部细粒度一致性，提升表征在分类与密集任务上的兼容性（即“同一特征同时服务 image-level 与 pixel/patch-level”）。

## 3.3 DINOv3：特征增强重点——Gram Anchoring

DINOv3 公开摘要明确提出 Gram anchoring 来解决长训练下 dense feature 退化问题[10]。  
可迁移实现中，核心思想可写为“对特征相关性结构进行锚定正则”：

$$
G(F)=\frac{1}{HW}F^\top F
$$

$$
\mathcal{L}_{gram}=\left\|G(F_{cur})-G(F_{anchor})\right\|_F^2
$$

并加入总目标：

$$
\mathcal{L}_{DINOv3}=\mathcal{L}_{ssl}+\lambda_{gram}\mathcal{L}_{gram}
$$

其中 $F_{anchor}$ 可理解为训练中用于稳定 dense 几何关系的“锚定统计”。  
原理上，它约束特征通道间二阶统计不发生灾难性漂移，从而增强密集特征稳定性（分割/检测等下游更受益）[10]。

---

## 4. 二次正确性检查（已执行）

1. 公式维度：  
   - NWD 的 $W_2^2$、指数归一化形式与本地文档及论文口径一致。  
   - DINO 蒸馏项保持“概率分布交叉熵 + EMA 教师 + 中心化”三件套。  
2. 符号一致性：  
   - YOLO 损失统一用 $\mathcal{L}_{box},\mathcal{L}_{cls},\mathcal{L}_{dfl}$；  
   - DINO 系列统一区分学生 $s$、教师 $t$。  
3. 版本边界：  
   - 明确区分 Ultralytics YOLO11（无正式论文）与学术 YOLOv12（arXiv 论文）[5-6]。  
4. 可移植性：  
   - 所有公式均为标准 LaTeX，未使用平台私有语法。  

---

## 5. 参考文献（GBT 7714-2025，按正文首次出现顺序）

[1] HOU Y Q, ZHANG X D. A lightweight and high-accuracy framework for printed circuit board defect detection[J]. Engineering Applications of Artificial Intelligence, 2025, 148: 110375. DOI:10.1016/j.engappai.2025.110375.  

[2] YOLO_PCB 项目组. NWD理论与集成说明文档[EB/OL]. (2026-03-10)[2026-04-05]. `D:\YOLO_PCB\NWD理论与集成说明文档.md`.  

[3] JOCHER G, QIU J, CHAURASIA A, et al. Ultralytics loss implementation[EB/OL]. (2024)[2026-04-05]. `D:\YOLO_PCB\ultralytics\utils\loss.py`.  

[4] JOCHER G, QIU J, CHAURASIA A, et al. TaskAlignedAssigner implementation[EB/OL]. (2024)[2026-04-05]. `D:\YOLO_PCB\ultralytics\utils\tal.py`.  

[5] JOCHER G, QIU J. Ultralytics YOLO11[EB/OL]. (2024-09-10)[2026-04-05]. https://docs.ultralytics.com/models/yolo11/.  

[6] TIAN Y J, YE Q X, DOERMANN D. YOLOv12: Attention-Centric Real-Time Object Detectors[EB/OL]. arXiv:2502.12524, 2025[2026-04-05]. https://arxiv.org/abs/2502.12524. DOI:10.48550/arXiv.2502.12524.  

[7] WANG J W, XU C, YANG W, et al. A Normalized Gaussian Wasserstein Distance for Tiny Object Detection[EB/OL]. arXiv:2110.13389, 2022[2026-04-05]. https://arxiv.org/abs/2110.13389. DOI:10.48550/arXiv.2110.13389.  

[8] CARON M, TOUVRON H, MISRA I, et al. Emerging Properties in Self-Supervised Vision Transformers[EB/OL]. arXiv:2104.14294, 2021[2026-04-05]. https://arxiv.org/abs/2104.14294. DOI:10.48550/arXiv.2104.14294.  

[9] OQUAB M, DARCET T, MOUTAKANNI T, et al. DINOv2: Learning Robust Visual Features without Supervision[EB/OL]. arXiv:2304.07193, 2024[2026-04-05]. https://arxiv.org/abs/2304.07193. DOI:10.48550/arXiv.2304.07193.  

[10] SIMEONI O, VO H V, SEITZER M, et al. DINOv3[EB/OL]. arXiv:2508.10104, 2025[2026-04-05]. https://arxiv.org/abs/2508.10104. DOI:10.48550/arXiv.2508.10104.  
