# PCB瑕疵检测高级技术快速开始指南

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 验证安装
python verify_setup.py
```

### 2. 使用预定义配置

我们提供了三种预定义配置，适合不同的使用场景：

#### 基础配置 - 快速原型
```python
from pcb_detection.utils import load_config
from pcb_detection.models import YOLODetector

# 加载基础配置
config = load_config("config/advanced_config.yaml", "basic_config")

# 创建模型
model = YOLODetector(config)
```

#### 平衡配置 - 推荐使用
```python
# 精度与速度兼顾的配置
config = load_config("config/advanced_config.yaml", "balanced_config")
model = YOLODetector(config)
```

#### 性能配置 - 追求最高精度
```python
# 最高精度配置
config = load_config("config/advanced_config.yaml", "performance_config")
model = YOLODetector(config)
```

### 3. 使用高级技术组件

#### 3.1 注意力机制

```python
from pcb_detection.models import CBAM, AttentionBlock

# 直接使用CBAM
cbam = CBAM(in_channels=512, reduction_ratio=16)
enhanced_features = cbam(backbone_features)

# 使用可配置的注意力块
attention = AttentionBlock(in_channels=512, attention_type="cbam")
```

#### 3.2 高级损失函数

```python
from pcb_detection.models import FocalLoss, IoULoss, ComboLoss

# Focal Loss处理类别不平衡
focal_loss = FocalLoss(alpha=1.0, gamma=2.0)

# CIoU Loss优化小目标
ciou_loss = IoULoss(loss_type='ciou')

# 组合损失函数
combo_loss = ComboLoss(
    cls_loss_weight=1.0,
    bbox_loss_weight=2.0,  # 增加bbox权重关注小目标
    obj_loss_weight=1.0,
    use_focal=True,
    use_iou=True,
    iou_type='ciou'
)
```

#### 3.3 高级数据增强

```python
from pcb_detection.data import PCBAdvancedAugmentation

# 创建高级增强管道
advanced_aug = PCBAdvancedAugmentation(
    image_size=640,
    mosaic_prob=0.5,
    copy_paste_prob=0.3,
    mixup_prob=0.2,
    use_albumentations=True
)

# 应用增强
augmented_image, augmented_annotations = advanced_aug(images, annotations_list)
```

### 4. 训练流程示例

#### 4.1 基础训练流程

```python
from pcb_detection.data import PCBDataset, DatasetSplitter
from pcb_detection.models import YOLODetector
from pcb_detection.training import Trainer

# 1. 准备数据
dataset = PCBDataset("训练集-PCB_DATASET", mode="train")
splitter = DatasetSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
train_data, val_data, test_data = splitter.split_dataset(dataset.annotations)

# 2. 创建模型
config = load_config("config/advanced_config.yaml", "balanced_config")
model = YOLODetector(config)

# 3. 训练
trainer = Trainer(model, config)
trainer.train(train_data, val_data)
```

#### 4.2 渐进式训练策略

```python
# 阶段1: 基础训练
basic_config = load_config("config/advanced_config.yaml", "basic_config")
model = YOLODetector(basic_config)
trainer = Trainer(model, basic_config)
trainer.train(train_data, val_data, epochs=100)

# 阶段2: 添加高级技术
balanced_config = load_config("config/advanced_config.yaml", "balanced_config")
model.upgrade_config(balanced_config)  # 升级配置
trainer.train(train_data, val_data, epochs=100, resume=True)

# 阶段3: 最终优化
performance_config = load_config("config/advanced_config.yaml", "performance_config")
model.upgrade_config(performance_config)
trainer.train(train_data, val_data, epochs=100, resume=True)
```

### 5. 消融实验

```python
from pcb_detection.evaluation import AblationStudy

# 创建消融实验
ablation = AblationStudy(base_config="basic_config")

# 测试不同技术的贡献
results = ablation.run_experiments([
    "cbam_only",           # 仅CBAM注意力
    "advanced_loss_only",  # 仅高级损失
    "advanced_aug_only",   # 仅高级增强
    "multi_scale_only",    # 仅多尺度训练
    "performance_config"   # 全部技术
])

# 生成对比报告
ablation.generate_report(results)
```

### 6. 模型评估和可视化

```python
from pcb_detection.evaluation import Evaluator
from pcb_detection.visualization import Visualizer

# 评估模型
evaluator = Evaluator()
metrics = evaluator.evaluate(model, test_data)

print(f"mAP@0.5: {metrics.map_50:.3f}")
print(f"各类别AP: {metrics.ap_per_class}")

# 可视化结果
visualizer = Visualizer()
for image, gt_annotations in test_data:
    predictions = model.predict(image)
    comparison = visualizer.create_comparison_image(image, gt_annotations, predictions)
    visualizer.save_visualization(comparison, f"results/comparison_{i}.jpg")
```

### 7. 性能优化和部署

#### 7.1 模型优化

```python
from pcb_detection.optimization import ModelOptimizer

optimizer = ModelOptimizer()

# 模型剪枝
pruned_model = optimizer.prune_model(model, sparsity=0.3)

# 模型量化
quantized_model = optimizer.quantize_model(model, calibration_data=val_data)

# 导出ONNX
optimizer.export_onnx(model, "pcb_detector.onnx")
```

#### 7.2 部署配置

```python
# 边缘设备部署
edge_config = load_config("config/advanced_config.yaml", "deployment_configs.edge_device")

# 服务器部署
server_config = load_config("config/advanced_config.yaml", "deployment_configs.server_deployment")

# 离线分析
offline_config = load_config("config/advanced_config.yaml", "deployment_configs.offline_analysis")
```

## 🎯 性能基准

### 预期性能提升

| 配置 | mAP@0.5 | 推理速度(FPS) | 模型大小(MB) | 适用场景 |
|------|---------|---------------|--------------|----------|
| 基础配置 | 0.15-0.25 | 60+ | 6 | 快速原型 |
| 平衡配置 | 0.35-0.45 | 30-40 | 25 | 生产部署 |
| 性能配置 | 0.45-0.55 | 15-25 | 87 | 离线分析 |

### 技术贡献分析

| 技术 | mAP提升 | 主要优势 |
|------|---------|----------|
| CBAM注意力 | +0.08-0.12 | 小目标检测 |
| Focal Loss | +0.05-0.08 | 类别不平衡 |
| CIoU Loss | +0.03-0.05 | 边界框精度 |
| Mosaic增强 | +0.06-0.10 | 上下文理解 |
| Copy-Paste增强 | +0.04-0.07 | 样本多样性 |
| 多尺度训练 | +0.05-0.08 | 尺度鲁棒性 |

## 🔧 故障排除

### 常见问题

1. **内存不足**
   ```python
   # 减小batch size
   config['training']['batch_size'] = 8
   
   # 使用梯度累积
   config['training']['gradient_accumulation_steps'] = 2
   ```

2. **训练不收敛**
   ```python
   # 降低学习率
   config['training']['learning_rate'] = 0.005
   
   # 增加warmup
   config['training']['warmup_epochs'] = 10
   ```

3. **过拟合**
   ```python
   # 增加数据增强
   config['augmentation']['basic']['prob'] = 0.8
   
   # 添加正则化
   config['training']['weight_decay'] = 0.0005
   ```

### 性能调优建议

1. **小目标检测差**：增加Copy-Paste和Mosaic增强概率
2. **类别不平衡**：使用Class Balanced Loss
3. **推理速度慢**：选择轻量级backbone和注意力机制
4. **精度不够**：使用性能配置和模型集成

## 📚 更多资源

- [详细技术指南](advanced_techniques_guide.md)
- [API文档](api_reference.md)
- [配置参考](config_reference.md)
- [最佳实践](best_practices.md)

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！