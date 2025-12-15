# PCB瑕疵检测高级技术指南

## 概述

本文档详细介绍了可以提升PCB瑕疵检测性能的各种高级技术，包括注意力机制、损失函数优化、数据增强策略等。

## 1. 注意力机制

### 1.1 CBAM (Convolutional Block Attention Module)

**优势：**
- 结合通道注意力和空间注意力
- 计算开销相对较小
- 对小目标检测效果显著
- 即插即用，易于集成

**适用场景：**
- PCB瑕疵通常较小，CBAM能有效聚焦关键区域
- 多种瑕疵类型，通道注意力有助于特征选择

**使用示例：**
```python
from pcb_detection.models import CBAM

# 在YOLO backbone中添加CBAM
cbam = CBAM(in_channels=512, reduction_ratio=16)
enhanced_features = cbam(backbone_features)
```

### 1.2 其他注意力机制对比

| 注意力机制 | 计算复杂度 | 小目标效果 | 内存占用 | 推荐指数 |
|-----------|-----------|-----------|----------|----------|
| CBAM      | 低        | 优秀      | 低       | ⭐⭐⭐⭐⭐ |
| SE Block  | 很低      | 良好      | 很低     | ⭐⭐⭐⭐   |
| ECA       | 很低      | 良好      | 很低     | ⭐⭐⭐⭐   |
| Coord Attention | 中   | 优秀      | 中       | ⭐⭐⭐⭐   |

## 2. 损失函数优化

### 2.1 Focal Loss

**解决问题：**
- 类别不平衡（某些瑕疵类型样本较少）
- 难易样本不平衡

**配置建议：**
```python
from pcb_detection.models import FocalLoss

# 针对PCB瑕疵检测的配置
focal_loss = FocalLoss(
    alpha=1.0,      # 可根据类别分布调整
    gamma=2.0,      # 聚焦参数，越大越关注难样本
    reduction='mean'
)
```

### 2.2 IoU Loss系列

**CIoU Loss推荐配置：**
```python
from pcb_detection.models import IoULoss

ciou_loss = IoULoss(loss_type='ciou')  # 最适合小目标检测
```

**各IoU Loss对比：**
- **IoU**: 基础版本
- **GIoU**: 解决无重叠问题
- **DIoU**: 考虑中心点距离
- **CIoU**: 综合考虑重叠、距离、长宽比（推荐）

### 2.3 组合损失函数

```python
from pcb_detection.models import ComboLoss

combo_loss = ComboLoss(
    cls_loss_weight=1.0,    # 分类损失权重
    bbox_loss_weight=2.0,   # 边界框损失权重（小目标可适当增加）
    obj_loss_weight=1.0,    # 目标性损失权重
    use_focal=True,         # 使用Focal Loss
    use_iou=True,          # 使用IoU Loss
    iou_type='ciou'        # 使用CIoU
)
```

## 3. 数据增强策略

### 3.1 基础增强

**推荐配置：**
```python
from pcb_detection.data import DataAugmentation

basic_aug = DataAugmentation({
    'rotation_range': (-15, 15),        # 适度旋转
    'scale_range': (0.8, 1.2),         # 缩放范围
    'brightness_range': (-0.2, 0.2),   # 亮度调整
    'contrast_range': (0.8, 1.2),      # 对比度调整
    'flip_horizontal': True,            # 水平翻转
    'flip_vertical': True,              # 垂直翻转
    'augmentation_prob': 0.5            # 应用概率
})
```

### 3.2 高级增强

**Mosaic增强：**
```python
from pcb_detection.data import MosaicAugmentation

mosaic = MosaicAugmentation(
    image_size=640,
    prob=0.5  # 建议训练后期降低概率
)
```

**Copy-Paste增强：**
```python
from pcb_detection.data import CopyPasteAugmentation

copy_paste = CopyPasteAugmentation(
    prob=0.3,      # 适中概率，避免过度增强
    max_paste=3    # 最多粘贴3个瑕疵
)
```

### 3.3 Albumentations集成

```python
from pcb_detection.data import AlbumentationsAugmentation

albu_aug = AlbumentationsAugmentation(
    image_size=640,
    train=True  # 训练模式使用更激进的增强
)
```

## 4. 训练策略优化

### 4.1 多尺度训练

**配置示例：**
```python
# 训练配置
training_config = {
    'multi_scale': True,
    'scale_range': (0.5, 1.5),  # 尺度范围
    'scale_step': 32,           # 尺度步长
    'scale_freq': 10            # 每10个epoch改变一次尺度
}
```

### 4.2 渐进式训练

**阶段性训练策略：**
1. **第一阶段（0-100 epoch）**：基础增强 + 大学习率
2. **第二阶段（100-200 epoch）**：添加高级增强 + 中等学习率
3. **第三阶段（200-300 epoch）**：精细调优 + 小学习率

### 4.3 学习率调度

```python
# 推荐的学习率调度策略
lr_scheduler = {
    'type': 'cosine',
    'initial_lr': 0.01,
    'min_lr': 0.0001,
    'warmup_epochs': 5,
    'warmup_lr': 0.001
}
```

## 5. 模型架构优化

### 5.1 FPN + PANet

**特征金字塔网络配置：**
```python
fpn_config = {
    'in_channels': [256, 512, 1024],  # 不同层级的通道数
    'out_channels': 256,              # 输出通道数
    'num_outs': 3,                    # 输出层数
    'start_level': 0,                 # 起始层级
    'add_extra_convs': True           # 添加额外卷积层
}
```

### 5.2 检测头优化

**针对小目标的检测头配置：**
```python
detection_head_config = {
    'num_classes': 5,
    'in_channels': 256,
    'feat_channels': 256,
    'stacked_convs': 4,               # 堆叠卷积层数
    'use_depthwise': False,           # 是否使用深度可分离卷积
    'norm_cfg': {'type': 'BN'},       # 归一化配置
    'act_cfg': {'type': 'SiLU'}       # 激活函数配置
}
```

## 6. 后处理优化

### 6.1 Soft-NMS

**配置示例：**
```python
soft_nms_config = {
    'type': 'soft_nms',
    'iou_threshold': 0.5,
    'sigma': 0.5,                     # Soft-NMS参数
    'min_score': 0.05,                # 最小置信度
    'method': 'gaussian'              # 'gaussian' 或 'linear'
}
```

### 6.2 测试时增强 (TTA)

```python
tta_config = {
    'scales': [0.8, 1.0, 1.2],        # 多尺度测试
    'flips': [False, True],           # 翻转测试
    'merge_method': 'nms'             # 结果融合方法
}
```

## 7. 推荐的技术组合

### 7.1 基础版本（快速原型）

```python
# 适合快速验证和基线建立
recommended_basic = {
    'attention': 'se',                # SE Block
    'loss': 'focal',                  # Focal Loss
    'augmentation': 'basic',          # 基础增强
    'backbone': 'yolov8n'             # 轻量级模型
}
```

### 7.2 性能版本（追求精度）

```python
# 适合追求最高检测精度
recommended_performance = {
    'attention': 'cbam',              # CBAM注意力
    'loss': 'combo',                  # 组合损失
    'augmentation': 'advanced',       # 高级增强
    'backbone': 'yolov8l',            # 大模型
    'fpn': True,                      # 特征金字塔
    'multi_scale': True,              # 多尺度训练
    'tta': True                       # 测试时增强
}
```

### 7.3 平衡版本（精度与速度兼顾）

```python
# 适合实际部署应用
recommended_balanced = {
    'attention': 'cbam',              # CBAM注意力
    'loss': 'combo',                  # 组合损失
    'augmentation': 'moderate',       # 适度增强
    'backbone': 'yolov8m',            # 中等模型
    'fpn': True,                      # 特征金字塔
    'multi_scale': False,             # 单尺度训练
    'tta': False                      # 不使用TTA
}
```

## 8. 实施建议

### 8.1 渐进式实施

1. **第一步**：建立基线模型（基础YOLO + 基础增强）
2. **第二步**：添加注意力机制（推荐CBAM）
3. **第三步**：优化损失函数（添加Focal Loss和CIoU Loss）
4. **第四步**：增强数据增强（添加Mosaic和Copy-Paste）
5. **第五步**：多尺度训练和TTA

### 8.2 超参数调优顺序

1. **学习率**：最重要，优先调优
2. **损失函数权重**：根据验证集表现调整
3. **增强强度**：避免过度增强导致性能下降
4. **注意力机制参数**：通常使用默认值即可

### 8.3 性能监控

**关键指标：**
- mAP@0.5：主要评估指标
- 各类别AP：关注小样本类别
- 推理速度：实际部署考虑
- 模型大小：存储和传输考虑

**监控工具：**
```python
# 使用TensorBoard或Weights & Biases进行实验跟踪
import wandb

wandb.init(project="pcb-defect-detection")
wandb.config.update({
    "attention": "cbam",
    "loss": "combo",
    "augmentation": "advanced"
})
```

## 9. 常见问题和解决方案

### 9.1 过拟合问题

**症状**：训练集性能好，验证集性能差
**解决方案**：
- 增加数据增强强度
- 使用Dropout和正则化
- 减小模型复杂度
- 早停策略

### 9.2 小目标检测困难

**症状**：大瑕疵检测好，小瑕疵检测差
**解决方案**：
- 使用多尺度训练
- 增加FPN层数
- 调整anchor尺寸
- 使用Copy-Paste增强

### 9.3 类别不平衡

**症状**：常见瑕疵检测好，稀有瑕疵检测差
**解决方案**：
- 使用Focal Loss
- 调整类别权重
- 数据重采样
- 生成合成样本

### 9.4 推理速度慢

**症状**：模型精度高但推理速度不满足要求
**解决方案**：
- 使用轻量级backbone
- 模型剪枝和量化
- 减少检测头复杂度
- 优化后处理流程

## 10. 总结

对于PCB瑕疵检测任务，推荐的技术栈：

**核心技术：**
1. **CBAM注意力机制** - 提升小目标检测能力
2. **Focal Loss + CIoU Loss** - 处理类别不平衡和小目标回归
3. **高级数据增强** - Mosaic + Copy-Paste + Albumentations
4. **多尺度训练** - 提升不同尺寸瑕疵的检测能力

**实施策略：**
- 渐进式添加技术，每次验证效果
- 重点关注小目标和稀有类别的检测性能
- 平衡精度和速度，根据实际需求选择配置
- 充分利用数据增强，但避免过度增强

通过合理组合这些技术，预期可以将mAP从基线的0.005-0.01提升到0.3-0.5以上，显著改善PCB瑕疵检测的实用性。