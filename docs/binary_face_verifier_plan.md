# RetinaFace 误检抑制（二分类复核）训练与检测流程方案

## 1. 目标与总思路
- 在 RetinaFace 检测头后增加一个 **Face / Non-Face 二分类复核器（Verifier）**。
- 检测阶段采用两级决策：
  1) RetinaFace 先召回候选框；
  2) Verifier 对候选框裁剪图进行真假人脸判定并重打分。
- 目标：在可控召回下降下显著降低 FP（False Positive）。

---

## 2. 数据构建方案（决定上限）

### 2.1 正样本（Face）
- 来源 A：WIDER FACE 标注框（高质量正样本）。
- 来源 B：RetinaFace 在训练集/近域数据上的高 IoU 预测框（带轻微偏移，增强鲁棒性）。
- 采样策略：
  - 框抖动（中心、尺度、长宽比随机扰动）；
  - 多尺寸分桶（小脸/中脸/大脸）均衡采样；
  - 低光、IR 灰度、模糊、遮挡样本加权。

### 2.2 负样本（Non-Face）
- 来源 A：纯背景图 + 无标注区域裁剪。
- 来源 B：**难负样本**（Hard Negatives）：RetinaFace 高分但与任意 GT IoU < 阈值的预测框。
- 来源 C：容易混淆目标（手、耳朵、衣服纹理、广告人像、玩偶、雕像、屏幕里的人脸等）。

### 2.3 标签规则建议
- IoU >= 0.5：正样本。
- IoU < 0.3：负样本。
- 0.3~0.5：忽略（不参与训练）以减少噪声。
- 边界框太小（如 min(w,h)<10）可单独分桶，避免劣质监督污染主干训练。

### 2.4 数据比例与课程学习
- 初期建议 Pos:Neg = 1:1 或 1:2。
- 随训练推进逐步提高难负样本占比（例如从 20% 提到 60%）。
- 每轮迭代加入最新线上/验证集误检回流样本（闭环 hard mining）。

---

## 3. 二分类机模型设计

### 3.1 输入与预处理
- 输入：RetinaFace 输出框对应 ROI。
- 裁剪策略：
  - 扩边 `margin`（如 0.1~0.25）保留上下文；
  - 对越界框做 padding；
  - 统一到固定尺寸（112/128/160）。
- 归一化：与主模型一致或按分类模型标准均值方差。

### 3.2 模型结构建议（按部署预算）
- 轻量实时：MobileNetV3-small / EfficientNet-Lite0。
- 平衡精度：MobileNetV3-large / ConvNeXt-Tiny（裁剪版）。
- 极致精度：Swin-T / ViT-Tiny（对吞吐要求高时不推荐）。

### 3.3 损失函数与不均衡处理
- 主损失：BCEWithLogits 或 CE。
- 推荐：Focal Loss（抑制易分类样本，聚焦难负样本）。
- 类别不平衡：class weight、重采样、或 logit-adjustment。

### 3.4 训练技巧
- 强增强：ColorJitter、Blur、JPEG、Cutout、随机灰度、仿 IR 变换。
- 但避免破坏人脸关键结构（几何增强幅度要小）。
- Label smoothing 可轻量启用（如 0.05）。
- EMA、Cosine LR、Warmup、Early Stopping 提升稳定性。

---

## 4. 训练流程（推荐）

1. **阶段A：离线初训**
   - 用通用正负样本库训练 verifier 基线模型。
2. **阶段B：联合数据再训练**
   - 用当前 RetinaFace 的误检结果做 hard negative 迭代训练。
3. **阶段C：阈值与校准**
   - 在验证集做温度缩放/Platt 校准，让概率更可解释。
4. **阶段D：面向业务指标调参**
   - 按目标 Recall 下最小化 FP/image 或 FPR@TPR。

---

## 5. 检测/推理流程（两级级联）

1. RetinaFace 前向得到 `(box, score_det, landm)`。
2. 先做一次宽松阈值筛选（提高召回）。
3. 对候选框裁剪 ROI，送入 verifier 得到 `score_cls`。
4. 融合打分（建议之一）：
   - `score_final = (score_det^a) * (score_cls^b)`，其中 a,b 通过验证集网格搜索。
5. 用 `score_final` 做最终阈值过滤 + NMS。
6. 可选：按框尺寸使用分段阈值（小脸阈值更严格）。

> 关键点：不要只“串联 hard threshold”，应做分数融合 + 分桶阈值，通常可显著降低误杀。

---

## 6. 指标与验收

### 6.1 离线指标
- 检测端：mAP、Recall、Precision、FP/image。
- 级联系统：
  - 固定 Recall 比较 FP/image 降幅；
  - 或固定 FP/image 比较 Recall 损失；
  - ROC / PR 曲线与 AUC。

### 6.2 线上指标
- 场景分桶（白天/夜间、可见光/IR、远距小脸）。
- 统计误检 Top-K case 并每周回流。

### 6.3 验收建议
- 目标示例：在 Recall 下降不超过 0.5~1.0 个百分点的前提下，FP/image 下降 30%+。

---

## 7. 与现有仓库流程对齐建议
- 沿用当前 RetinaFace 训练与推理参数体系（置信度阈值、NMS、top-k）。
- 在 negative mining 管道中新增 verifier 数据导出模式（保存 ROI 与标签）。
- 形成固定“挖掘→训练→评估→回流”的周期化流程。

---

## 8. 常见坑与规避
- **坑1：负样本太“容易”** → 线上降 FP 不明显。
  - 规避：提高 hard negative 占比，持续回流误检。
- **坑2：只看总体精度** → 小脸/夜间场景崩。
  - 规避：按场景分桶评估和阈值分桶。
- **坑3：阈值未校准** → 模型分数不可比较。
  - 规避：做概率校准与版本间统一评估协议。
- **坑4：ROI 预处理不一致** → 训练推理分布偏移。
  - 规避：严格复用同一裁剪/归一化配置。

---

## 9. 最小可落地版本（MVP）
1. 先固定一个轻量 verifier（MobileNetV3-small）。
2. 使用当前模型跑一轮全量 hard negative 挖掘。
3. 训练二分类 20~30 epoch，Focal Loss + class weight。
4. 推理端引入 score 融合和单阈值。
5. 达成初步降 FP 后，再做分桶阈值和概率校准。

该方案可以在不改动 RetinaFace 主干结构的情况下，快速获得“误检显著下降”的收益，同时保留后续向端到端联合训练演进的空间。
