#!/usr/bin/env python3
"""
快速安装PCB检测作业所需的依赖
"""

import subprocess
import sys

def install_package(package):
    """安装单个包"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} 安装成功")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ {package} 安装失败")
        return False

def main():
    print("🔧 安装PCB瑕疵检测作业所需依赖...")
    
    # 核心依赖包
    packages = [
        "ultralytics",      # YOLO模型
        "opencv-python",    # 图像处理
        "matplotlib",       # 可视化
        "numpy",           # 数值计算
        "Pillow",          # 图像处理
        "PyYAML",          # 配置文件
        "pandas",          # 数据处理
        "tqdm"             # 进度条
    ]
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n📦 安装完成: {success_count}/{len(packages)} 个包安装成功")
    
    if success_count == len(packages):
        print("🎉 所有依赖安装成功！现在可以运行: python run_assignment.py")
    else:
        print("⚠️  部分依赖安装失败，请手动安装或检查网络连接")

if __name__ == "__main__":
    main()