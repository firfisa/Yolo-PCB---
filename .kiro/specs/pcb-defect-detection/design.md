# PCB瑕疵检测系统设计文档

## 概述

本设计文档描述了一个基于深度学习的PCB瑕疵检测系统，该系统使用YOLO架构进行目标检测，能够识别和定位五种PCB瑕疵类型。系统设计重点关注高精度检测、实时性能和可视化展示，通过多种优化技术提升检测效果。

## 架构

### 系统架构图

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   数据预处理     │───▶│   YOLO模型      │───▶│   后处理模块     │
│   - 图像加载     │    │   - 特征提取     │    │   - NMS        │
│   - 数据增强     │    │   - 检测头      │    │   - 置信度过滤   │
│   - 标准化      │    │   - 损失计算     │    │   - 坐标转换    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   数据管理      │    │   训练管理      │    │   评估模块      │
│   - 数据集加载   │    │   - 训练循环     │    │   - mAP计算     │
│   - 标注解析     │    │   - 验证       │    │   - AP分类计算   │
│   - 批次生成     │    │   - 检查点保存   │    │   - 可视化生成   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 核心组件

1. **数据处理层**: 负责数据加载、预处理和增强
2. **模型层**: YOLO检测网络，包含骨干网络、颈部网络和检测头
3. **训练层**: 训练管理、损失计算和优化
4. **评估层**: 性能评估和指标计算
5. **可视化层**: 结果展示和对比图生成

## 组件和接口

### 数据处理组件

```python
class PCBDataset:
    """PCB数据集处理类"""
    def __init__(self, data_path: str, mode: str)
    def load_annotations(self) -> List[Dict]
    def apply_augmentation(self, image: np.ndarray) -> np.ndarray
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]

class DataAugmentation:
    """数据增强类"""
    def __init__(self, config: Dict)
    def random_rotation(self, image: np.ndarray, angle_range: Tuple[int, int])
    def random_scaling(self, image: np.ndarray, scale_range: Tuple[float, float])
    def color_jittering(self, image: np.ndarray, brightness: float, contrast: float)
```

### 模型组件

```python
class YOLODetector:
    """YOLO检测器主类"""
    def __init__(self, model_config: Dict, num_classes: int = 5)
    def forward(self, x: torch.Tensor) -> torch.Tensor
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor
    def predict(self, image: np.ndarray) -> List[Detection]

class DetectionHead:
    """检测头组件"""
    def __init__(self, in_channels: int, num_classes: int, num_anchors: int)
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor
```

### 评估组件

```python
class Evaluator:
    """评估器类"""
    def __init__(self, iou_threshold: float = 0.5)
    def calculate_map(self, predictions: List, ground_truths: List) -> float
    def calculate_ap_per_class(self, predictions: List, ground_truths: List) -> Dict[str, float]
    def generate_metrics_report(self) -> Dict

class Visualizer:
    """可视化类"""
    def __init__(self, class_names: List[str], colors: List[Tuple])
    def draw_detections(self, image: np.ndarray, detections: List) -> np.ndarray
    def create_comparison_grid(self, images: List, gt_detections: List, pred_detections: List) -> np.ndarray
```

## 数据模型

### 检测结果数据结构

```python
@dataclass
class Detection:
    """单个检测结果"""
    bbox: Tuple[float, float, float, float]  # (x, y, w, h) 归一化坐标
    confidence: float                        # 置信度分数
    class_id: int                           # 类别ID
    class_name: str                         # 类别名称

@dataclass
class EvaluationMetrics:
    """评估指标"""
    map_50: float                           # mAP@IoU=0.5
    ap_per_class: Dict[str, float]          # 每类AP
    precision: float                        # 精确率
    recall: float                          # 召回率
    f1_score: float                        # F1分数

@dataclass
class TrainingConfig:
    """训练配置"""
    model_name: str = "yolov8n"            # 模型名称
    epochs: int = 300                      # 训练轮数
    batch_size: int = 16                   # 批次大小
    learning_rate: float = 0.01            # 学习率
    image_size: int = 640                  # 图像尺寸
    augmentation: bool = True              # 是否使用数据增强
```

### 数据集格式

```python
# YOLO格式标注
# 每行格式: class_id x_center y_center width height (归一化坐标)
# 示例: 0 0.521000 0.235075 0.362000 0.450249

class_mapping = {
    0: "Mouse_bite",      # 鼠标咬痕
    1: "Open_circuit",    # 开路
    2: "Short",           # 短路  
    3: "Spur",            # 毛刺
    4: "Spurious_copper"  # 杂散铜
}
```

## 正确性属性

*属性是应该在系统所有有效执行中保持为真的特征或行为——本质上是关于系统应该做什么的正式陈述。属性作为人类可读规范和机器可验证正确性保证之间的桥梁。*

### 属性1: mAP计算一致性
*对于任何*预测结果集合和真实标注集合，在IoU阈值0.5下计算的mAP应该在0到1之间且计算结果一致
**验证需求: 需求 1.1**

### 属性2: 类别AP完整性  
*对于任何*评估过程，系统应该为所有五个瑕疵类别(Mouse_bite, Open_circuit, Short, Spur, Spurious_copper)都计算并报告AP值
**验证需求: 需求 1.2**

### 属性3: 结构化结果存储
*对于任何*评估指标计算，结果应该以包含所有必需字段的有效JSON或CSV格式存储
**验证需求: 需求 1.4**

### 属性4: 多IoU阈值评估
*对于任何*IoU阈值集合，系统应该为每个阈值提供相应的评估指标
**验证需求: 需求 1.5**

### 属性5: 数据增强保持有效性
*对于任何*有效的PCB图像，应用旋转、缩放或颜色变换后仍应产生有效的图像数据
**验证需求: 需求 2.1**

### 属性6: 集成预测一致性
*对于任何*多个模型预测集合，集成方法应该产生单一的、格式一致的预测结果
**验证需求: 需求 2.5**

### 属性7: 可视化格式正确性
*对于任何*输入图像和检测结果，可视化系统应该生成左侧为真实标注、右侧为预测结果的并排对比图像
**验证需求: 需求 3.1**

### 属性8: 标签映射正确性
*对于任何*瑕疵检测结果，显示的标签应该正确映射到对应的单字母代码(M, O, S, P, C)
**验证需求: 需求 3.2**

### 属性9: 颜色映射唯一性
*对于任何*瑕疵类型，边界框应该使用唯一且一致的颜色进行绘制
**验证需求: 需求 3.3**

### 属性10: 批量可视化网格生成
*对于任何*图像集合，批量可视化应该生成包含所有输入图像的正确网格布局
**验证需求: 需求 3.5**

### 属性11: 标注格式转换往返一致性
*对于任何*有效的标注数据，转换为YOLO格式后再转换回原格式应该保持数据等价性
**验证需求: 需求 4.1**

### 属性12: 数据验证完整性
*对于任何*数据集，验证过程应该能够识别所有图像-标签不匹配和标注不一致问题
**验证需求: 需求 4.2**

### 属性13: 数据分割类别平衡
*对于任何*数据集分割操作，各分割中的类别分布比例应该与原始数据集保持一致
**验证需求: 需求 4.3**

### 属性14: 图像预处理标准化
*对于任何*输入图像，预处理后的图像应该具有指定的尺寸和标准化的像素值范围
**验证需求: 需求 4.5**

### 属性15: 训练日志记录
*对于任何*训练会话，系统应该记录训练进度指标和验证指标到日志文件
**验证需求: 需求 5.1**

### 属性16: 置信度阈值应用
*对于任何*推理操作和置信度阈值，输出的检测结果应该只包含置信度高于阈值的检测
**验证需求: 需求 5.2**

### 属性17: 结果导出往返一致性
*对于任何*预测结果，导出为JSON或CSV格式后再导入应该保持数据完整性
**验证需求: 需求 5.3**

### 属性18: 最佳模型保存
*对于任何*训练过程，系统应该自动保存验证性能最佳的模型检查点
**验证需求: 需求 5.4**

### 属性19: 性能指标跟踪
*对于任何*推理操作，性能监控应该记录推理时间和资源使用指标
**验证需求: 需求 5.5**

### 属性20: 多尺度训练覆盖
*对于任何*多尺度训练配置，训练过程应该使用配置中指定的所有图像分辨率
**验证需求: 需求 6.1**

### 属性21: 类别不平衡处理
*对于任何*类别不平衡的数据集，损失函数应该对少数类别给予更高权重
**验证需求: 需求 6.3**

### 属性22: NMS重复检测去除
*对于任何*包含重叠检测的预测结果，非极大值抑制应该去除冗余的重复检测
**验证需求: 需求 6.4**

## 错误处理

### 数据错误处理
- **无效图像格式**: 系统应该检测并报告不支持的图像格式，提供清晰的错误信息
- **标注文件缺失**: 当标注文件不存在时，系统应该优雅地处理并继续处理其他文件
- **标注格式错误**: 检测到格式不正确的标注时，系统应该记录错误并跳过该标注
- **类别ID超出范围**: 当遇到未定义的类别ID时，系统应该报告错误并提供有效范围

### 模型错误处理  
- **模型加载失败**: 当模型文件损坏或不兼容时，提供详细的错误诊断信息
- **内存不足**: 在GPU内存不足时，自动调整批次大小或切换到CPU模式
- **训练中断**: 实现训练检查点机制，支持从中断点恢复训练
- **推理失败**: 当单个图像推理失败时，记录错误但继续处理其他图像

### 可视化错误处理
- **图像尺寸不匹配**: 自动调整图像尺寸以适应可视化布局
- **颜色空间转换错误**: 处理不同颜色空间的图像转换问题
- **字体渲染失败**: 提供备用字体选项确保标签正常显示

## 测试策略

### 双重测试方法

本系统采用单元测试和基于属性的测试相结合的方法：

**单元测试**覆盖：
- 具体示例验证正确行为
- 组件间集成点测试  
- 单元测试有助于发现具体错误，但避免编写过多测试
- 属性测试负责处理大量输入覆盖

**基于属性的测试**要求：
- 使用Hypothesis（Python）作为属性测试库
- 每个属性测试配置最少运行100次迭代，因为属性测试过程是随机的
- 每个属性测试必须用注释明确引用设计文档中的正确性属性
- 使用格式：'**特性: pcb-defect-detection, 属性 {编号}: {属性文本}**'
- 每个正确性属性必须由单个属性测试实现
- 在设计文档的测试策略部分明确这些要求

**属性测试标注示例**：
```python
def test_map_calculation_consistency():
    """
    **特性: pcb-defect-detection, 属性 1: mAP计算一致性**
    """
    # 测试实现
```

### 测试覆盖范围

1. **数据处理测试**
   - 数据加载和解析正确性
   - 数据增强变换有效性
   - 标注格式转换准确性

2. **模型功能测试**  
   - 模型前向传播输出格式
   - 损失函数计算正确性
   - 预测结果后处理

3. **评估指标测试**
   - mAP和AP计算准确性
   - 不同IoU阈值下的评估
   - 指标报告格式验证

4. **可视化测试**
   - 图像绘制和标注正确性
   - 颜色映射和标签显示
   - 批量可视化布局

5. **集成测试**
   - 端到端训练流程
   - 完整推理管道
   - 结果保存和加载