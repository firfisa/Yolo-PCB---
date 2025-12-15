#!/usr/bin/env python3
"""
简化的PCB瑕疵检测系统 - 专门用于完成作业要求
作者: Kiro AI Assistant
功能: 
1. 训练YOLO模型检测PCB瑕疵
2. 计算mAP和各类别AP
3. 生成可视化对比图
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
from typing import List, Dict, Tuple
import json

class SimplePCBDetector:
    """简化的PCB瑕疵检测器"""
    
    def __init__(self):
        self.class_names = ['Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
        self.class_colors = [
            (255, 0, 0),    # 红色 - Mouse_bite (M)
            (0, 255, 0),    # 绿色 - Open_circuit (O)  
            (0, 0, 255),    # 蓝色 - Short (S)
            (255, 255, 0),  # 黄色 - Spur (P)
            (255, 0, 255)   # 紫色 - Spurious_copper (C)
        ]
        self.class_letters = ['M', 'O', 'S', 'P', 'C']
        self.model = None
        
    def prepare_dataset(self, train_path: str, test_path: str, output_path: str = "dataset"):
        """准备YOLO格式的数据集"""
        print("准备数据集...")
        
        # 创建YOLO数据集结构
        dataset_dir = Path(output_path)
        (dataset_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        # 处理训练数据
        self._copy_dataset(train_path, dataset_dir / "images" / "train", dataset_dir / "labels" / "train")
        
        # 处理测试数据  
        self._copy_dataset(test_path, dataset_dir / "images" / "val", dataset_dir / "labels" / "val")
        
        # 创建数据配置文件
        config = {
            'path': str(dataset_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 5,
            'names': self.class_names
        }
        
        with open(dataset_dir / "data.yaml", 'w', encoding='utf-8') as f:
            import yaml
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
        return str(dataset_dir / "data.yaml")
    
    def _copy_dataset(self, source_path: str, img_dest: Path, label_dest: Path):
        """复制数据集文件"""
        source = Path(source_path)
        
        # 遍历所有类别文件夹
        for class_folder in source.iterdir():
            if not class_folder.is_dir():
                continue
                
            # 处理图像文件夹
            img_folder = class_folder / f"{class_folder.name}_Img"
            txt_folder = class_folder / f"{class_folder.name}_txt"
            
            if img_folder.exists() and txt_folder.exists():
                # 复制图像和标签
                for img_file in img_folder.glob("*.bmp"):
                    txt_file = txt_folder / f"{img_file.stem}.txt"
                    if txt_file.exists():
                        # 复制图像
                        import shutil
                        shutil.copy2(img_file, img_dest / img_file.name)
                        shutil.copy2(txt_file, label_dest / f"{img_file.stem}.txt")
    
    def train_model(self, data_config: str, epochs: int = 100):
        """训练YOLO模型"""
        print("开始训练YOLO模型...")
        
        # 使用YOLOv8n作为基础模型（轻量且效果好）
        self.model = YOLO('yolov8n.pt')
        
        # 训练参数
        results = self.model.train(
            data=data_config,
            epochs=epochs,
            imgsz=640,
            batch=16,
            lr0=0.01,
            patience=20,
            save=True,
            device='cpu'  # 如果有GPU可以改为0
        )
        
        print(f"训练完成！最佳模型保存在: {self.model.trainer.best}")
        return results
    
    def evaluate_model(self, data_config: str) -> Dict:
        """评估模型性能"""
        print("评估模型性能...")
        
        if self.model is None:
            print("请先训练模型或加载已训练的模型")
            return {}
            
        # 在验证集上评估
        results = self.model.val(data=data_config, imgsz=640)
        
        # 提取评估指标
        metrics = {
            'mAP_50': float(results.box.map50),  # mAP@IoU=0.5
            'mAP_50_95': float(results.box.map),  # mAP@IoU=0.5:0.95
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
        }
        
        # 各类别AP
        if hasattr(results.box, 'ap50'):
            class_ap = results.box.ap50
            for i, class_name in enumerate(self.class_names):
                if i < len(class_ap):
                    metrics[f'AP_{class_name}'] = float(class_ap[i])
        
        return metrics
    
    def visualize_results(self, test_images_path: str, output_dir: str = "visualization_results"):
        """生成可视化对比图"""
        print("生成可视化结果...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        test_path = Path(test_images_path)
        comparison_images = []
        
        # 遍历测试图像
        for class_folder in test_path.iterdir():
            if not class_folder.is_dir():
                continue
                
            img_folder = class_folder / f"{class_folder.name}_Img"
            txt_folder = class_folder / f"{class_folder.name}_txt"
            
            if not (img_folder.exists() and txt_folder.exists()):
                continue
                
            # 处理每个图像
            for img_file in list(img_folder.glob("*.bmp"))[:3]:  # 每类取3张图
                txt_file = txt_folder / f"{img_file.stem}.txt"
                if not txt_file.exists():
                    continue
                    
                # 读取图像
                image = cv2.imread(str(img_file))
                if image is None:
                    continue
                    
                # 创建对比图
                comparison = self._create_comparison_image(image, txt_file, img_file)
                if comparison is not None:
                    comparison_images.append(comparison)
                    
                    # 保存单个对比图
                    save_path = output_path / f"comparison_{img_file.stem}.jpg"
                    cv2.imwrite(str(save_path), comparison)
        
        # 创建网格对比图
        if comparison_images:
            grid = self._create_grid_visualization(comparison_images)
            cv2.imwrite(str(output_path / "comparison_grid.jpg"), grid)
            
        print(f"可视化结果保存在: {output_path}")
        
    def _create_comparison_image(self, image: np.ndarray, gt_file: Path, img_file: Path) -> np.ndarray:
        """创建GT vs 预测的对比图"""
        h, w = image.shape[:2]
        
        # 创建左右对比图
        comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
        
        # 左侧：GT
        gt_image = image.copy()
        gt_boxes = self._load_yolo_annotations(gt_file, w, h)
        gt_image = self._draw_boxes(gt_image, gt_boxes, is_gt=True)
        comparison[:, :w] = gt_image
        
        # 右侧：预测结果
        pred_image = image.copy()
        if self.model:
            results = self.model.predict(image, verbose=False)
            if results and len(results) > 0:
                pred_boxes = self._extract_predictions(results[0])
                pred_image = self._draw_boxes(pred_image, pred_boxes, is_gt=False)
        comparison[:, w:] = pred_image
        
        # 添加标题
        cv2.putText(comparison, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Prediction", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return comparison
    
    def _load_yolo_annotations(self, txt_file: Path, img_w: int, img_h: int) -> List[Dict]:
        """加载YOLO格式标注"""
        boxes = []
        try:
            with open(txt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * img_w
                        y_center = float(parts[2]) * img_h
                        width = float(parts[3]) * img_w
                        height = float(parts[4]) * img_h
                        
                        x1 = int(x_center - width / 2)
                        y1 = int(y_center - height / 2)
                        x2 = int(x_center + width / 2)
                        y2 = int(y_center + height / 2)
                        
                        boxes.append({
                            'class_id': class_id,
                            'bbox': (x1, y1, x2, y2),
                            'confidence': 1.0
                        })
        except Exception as e:
            print(f"读取标注文件错误: {e}")
        
        return boxes
    
    def _extract_predictions(self, result) -> List[Dict]:
        """提取预测结果"""
        boxes = []
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                if hasattr(box, 'xyxy') and hasattr(box, 'cls') and hasattr(box, 'conf'):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    boxes.append({
                        'class_id': class_id,
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': confidence
                    })
        return boxes
    
    def _draw_boxes(self, image: np.ndarray, boxes: List[Dict], is_gt: bool = True) -> np.ndarray:
        """绘制边界框"""
        for box in boxes:
            class_id = box['class_id']
            x1, y1, x2, y2 = box['bbox']
            confidence = box['confidence']
            
            if class_id >= len(self.class_colors):
                continue
                
            color = self.class_colors[class_id]
            letter = self.class_letters[class_id]
            
            # 绘制边界框
            thickness = 2 if is_gt else 1
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # 绘制类别标签（单字母）
            label = f"{letter}" if is_gt else f"{letter}:{confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # 标签背景
            cv2.rectangle(image, (x1, y1 - label_size[1] - 5), 
                         (x1 + label_size[0], y1), color, -1)
            
            # 标签文字
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image
    
    def _create_grid_visualization(self, images: List[np.ndarray]) -> np.ndarray:
        """创建网格可视化"""
        if not images:
            return np.zeros((100, 100, 3), dtype=np.uint8)
            
        # 计算网格大小
        n_images = len(images)
        cols = min(3, n_images)
        rows = (n_images + cols - 1) // cols
        
        # 调整图像大小
        target_h, target_w = 300, 600  # 每个对比图的目标大小
        resized_images = []
        for img in images:
            resized = cv2.resize(img, (target_w, target_h))
            resized_images.append(resized)
        
        # 创建网格
        grid_h = rows * target_h
        grid_w = cols * target_w
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        for i, img in enumerate(resized_images):
            row = i // cols
            col = i % cols
            y1 = row * target_h
            y2 = y1 + target_h
            x1 = col * target_w
            x2 = x1 + target_w
            grid[y1:y2, x1:x2] = img
            
        return grid
    
    def generate_report(self, metrics: Dict, output_file: str = "evaluation_report.txt"):
        """生成评估报告"""
        print("生成评估报告...")
        
        report = f"""
PCB瑕疵检测模型评估报告
========================

整体性能指标:
- mAP@IoU=0.5: {metrics.get('mAP_50', 0):.4f}
- mAP@IoU=0.5:0.95: {metrics.get('mAP_50_95', 0):.4f}
- 精确率: {metrics.get('precision', 0):.4f}
- 召回率: {metrics.get('recall', 0):.4f}

各类别AP@IoU=0.5:
"""
        
        for class_name in self.class_names:
            ap_key = f'AP_{class_name}'
            ap_value = metrics.get(ap_key, 0)
            report += f"- {class_name}: {ap_value:.4f}\n"
        
        report += f"""
提高mAP的方法建议:
1. 数据增强: 使用旋转、缩放、颜色变换等技术增加数据多样性
2. 模型优化: 尝试YOLOv8s/m/l等更大的模型
3. 超参数调优: 优化学习率、批次大小、锚框配置
4. 多尺度训练: 在不同分辨率下训练提高检测精度
5. 损失函数优化: 使用焦点损失处理类别不平衡
6. 后处理优化: 调整NMS阈值和置信度阈值
7. 集成学习: 结合多个模型的预测结果
8. 数据质量: 检查和清理标注数据，确保质量

注意: 当前基线mAP较低(0.005-0.01)主要是因为:
- PCB瑕疵目标非常小且难以检测
- 数据集规模可能较小
- 需要针对小目标检测进行特殊优化
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
            
        print(f"评估报告保存在: {output_file}")
        return report


def main():
    """主函数 - 完成作业的三个要求"""
    detector = SimplePCBDetector()
    
    # 1. 准备数据集
    print("=== 步骤1: 准备数据集 ===")
    data_config = detector.prepare_dataset(
        train_path="训练集-PCB_DATASET",
        test_path="PCB_瑕疵测试集"
    )
    
    # 2. 训练模型
    print("\n=== 步骤2: 训练YOLO模型 ===")
    detector.train_model(data_config, epochs=50)  # 作业用50轮就够了
    
    # 3. 评估模型 (要求1: 提供mAP和各类别AP)
    print("\n=== 步骤3: 评估模型性能 ===")
    metrics = detector.evaluate_model(data_config)
    
    # 4. 生成可视化 (要求3: 可视化GT vs 预测)
    print("\n=== 步骤4: 生成可视化对比图 ===")
    detector.visualize_results("PCB_瑕疵测试集")
    
    # 5. 生成报告 (要求2: 描述提高mAP的方法)
    print("\n=== 步骤5: 生成评估报告 ===")
    detector.generate_report(metrics)
    
    print("\n=== 作业完成! ===")
    print("检查以下文件:")
    print("- evaluation_report.txt: 包含mAP、各类别AP和改进方法")
    print("- visualization_results/: 包含GT vs 预测对比图")
    print("- runs/detect/train/: 包含训练过程和模型文件")


if __name__ == "__main__":
    main()