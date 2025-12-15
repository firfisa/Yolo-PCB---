#!/usr/bin/env python3
"""
PCB瑕疵检测作业 - 一键运行脚本
直接运行这个文件就能完成所有作业要求！
"""

from simple_pcb_detection import SimplePCBDetector
import os

def main():
    print("🚀 开始PCB瑕疵检测作业...")
    print("📋 作业要求:")
    print("   1. 提供测试集的mAP（IoU=0.5）和单个类别的AP")
    print("   2. 描述哪些方法提高mAP") 
    print("   3. 可视化显示检测结果和GT对比")
    print()
    
    # 检查数据集是否存在
    if not os.path.exists("训练集-PCB_DATASET"):
        print("❌ 找不到训练集文件夹: 训练集-PCB_DATASET")
        return
    
    if not os.path.exists("PCB_瑕疵测试集"):
        print("❌ 找不到测试集文件夹: PCB_瑕疵测试集")
        return
    
    print("✅ 数据集检查通过")
    
    try:
        detector = SimplePCBDetector()
        
        # 步骤1: 准备数据
        print("\n📁 准备YOLO格式数据集...")
        data_config = detector.prepare_dataset(
            train_path="训练集-PCB_DATASET",
            test_path="PCB_瑕疵测试集"
        )
        
        # 步骤2: 训练模型
        print("\n🎯 训练YOLO模型 (这可能需要几分钟)...")
        detector.train_model(data_config, epochs=30)  # 30轮够用了
        
        # 步骤3: 评估性能
        print("\n📊 评估模型性能...")
        metrics = detector.evaluate_model(data_config)
        
        # 步骤4: 生成可视化
        print("\n🎨 生成可视化对比图...")
        detector.visualize_results("PCB_瑕疵测试集")
        
        # 步骤5: 生成报告
        print("\n📝 生成评估报告...")
        detector.generate_report(metrics)
        
        print("\n🎉 作业完成！")
        print("\n📄 结果文件:")
        print("   📊 evaluation_report.txt - 包含mAP、AP和改进方法")
        print("   🖼️  visualization_results/ - GT vs 预测对比图")
        print("   🏆 runs/detect/train/ - 训练结果和最佳模型")
        
        # 显示关键结果
        print(f"\n🔍 关键结果预览:")
        print(f"   mAP@0.5: {metrics.get('mAP_50', 0):.4f}")
        for i, class_name in enumerate(detector.class_names):
            ap_key = f'AP_{class_name}'
            ap_value = metrics.get(ap_key, 0)
            print(f"   {class_name} AP: {ap_value:.4f}")
            
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        print("💡 请检查:")
        print("   - 是否安装了所需依赖: pip install ultralytics opencv-python matplotlib")
        print("   - 数据集文件夹是否存在且格式正确")

if __name__ == "__main__":
    main()