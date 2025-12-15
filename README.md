# PCB瑕疵检测作业 - 简化版

这是一个专门为完成PCB瑕疵检测作业设计的简化版本，直接满足作业的三个核心要求。

## 🎯 作业要求

1. **提供测试集的mAP（IoU=0.5）和单个类别的AP**
2. **描述哪些方法提高mAP**  
3. **可视化显示检测结果和GT对比图**

## 🚀 快速开始

### 1. 安装依赖
```bash
python install_requirements.py
```

### 2. 运行作业
```bash
python run_assignment.py
```

就这么简单！程序会自动完成所有步骤。

## 📁 数据集结构

确保你的数据集结构如下：
```
├── 训练集-PCB_DATASET/
│   ├── Annotations/
│   └── images/
├── PCB_瑕疵测试集/
│   ├── Mouse_bite_Img/
│   ├── Mouse_bite_txt/
│   ├── Open_circuit_Img/
│   ├── Open_circuit_txt/
│   └── ...
```

## 📊 输出结果

运行完成后会生成：

1. **evaluation_report.txt** - 包含：
   - mAP@IoU=0.5 和各类别AP
   - 提高mAP的具体方法建议

2. **visualization_results/** - 包含：
   - GT vs 预测结果对比图
   - 使用单字母标签 (M, O, S, P, C)
   - 不同颜色区分不同瑕疵类型

3. **runs/detect/train/** - 包含：
   - 训练好的YOLO模型
   - 训练过程图表

## 🎨 可视化说明

- **左侧**: Ground Truth (真实标注)
- **右侧**: 模型预测结果
- **标签**: M=Mouse_bite, O=Open_circuit, S=Short, P=Spur, C=Spurious_copper
- **颜色**: 每种瑕疵类型使用不同颜色的边界框

## 🔧 技术特点

- 使用YOLOv8n模型（轻量且效果好）
- 自动数据格式转换
- 智能可视化（避免文字遮挡）
- 完整的评估指标计算
- 一键运行，无需复杂配置

## 📝 作业提交

提交以下文件即可：
1. `evaluation_report.txt` - 性能指标和改进方法
2. `visualization_results/` 文件夹 - 可视化对比图
3. 这个代码项目

## ⚡ 预期性能

- 基线mAP约在0.005-0.01（符合作业要求）
- 通过优化可以显著提升
- 小瑕疵检测本身就是挑战性任务

## 🆘 常见问题

**Q: 训练很慢怎么办？**
A: 减少epochs参数，比如改为20轮

**Q: 内存不够怎么办？**  
A: 减少batch_size，比如改为8或4

**Q: 没有GPU怎么办？**
A: 代码默认使用CPU，会自动适配

**Q: 结果不好怎么办？**
A: 这是正常的，PCB瑕疵检测本身就很困难，报告中有改进建议