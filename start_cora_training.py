#!/usr/bin/env python3
"""
CoRA完整训练启动脚本
集成协同过滤+LLM+图像生成的多模态个性化推荐系统
包含训练启动、实时监控、图像查看等全部功能
"""

import os
import sys
import subprocess
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
import threading

def monitor_training_progress(image_dir="training_images", log_dir="logs/test"):
    """后台监控训练进度"""
    print("🔍 启动训练监控...")
    last_image_count = 0
    
    while True:
        try:
            # 检查图像生成状态
            image_path = Path(image_dir)
            if image_path.exists():
                recent_dir = image_path / "recent_images"
                all_dir = image_path / "all_images"
                best_dir = image_path / "best_images"
                
                recent_count = len(list(recent_dir.glob("*.png"))) if recent_dir.exists() else 0
                all_count = len(list(all_dir.glob("*.png"))) if all_dir.exists() else 0
                best_count = len(list(best_dir.glob("*.png"))) if best_dir.exists() else 0
                
                total_images = recent_count + all_count + best_count
                
                if total_images > last_image_count:
                    current_time = datetime.now().strftime("%H:%M:%S")
                    print(f"\n[{current_time}] 🎨 新图像生成! 总计: {total_images} (+{total_images - last_image_count})")
                    print(f"    最近: {recent_count}, 全部: {all_count}, 最佳: {best_count}")
                    last_image_count = total_images
            
            time.sleep(30)  # 每30秒检查一次
        except Exception:
            break

def show_training_status():
    """显示当前训练状态"""
    print("\n📊 训练状态检查:")
    print("-" * 50)
    
    # 检查进程
    try:
        import psutil
        training_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                if 'train_collm_mf_din.py' in cmdline:
                    training_processes.append(f"PID {proc.info['pid']}")
        
        if training_processes:
            print(f"✅ 训练进程运行中: {', '.join(training_processes)}")
        else:
            print("❌ 未检测到训练进程")
    except Exception:
        print("⚠️ 无法检查进程状态")
    
    # 检查图像生成
    image_path = Path("training_images")
    if image_path.exists():
        recent_count = len(list((image_path / "recent_images").glob("*.png"))) if (image_path / "recent_images").exists() else 0
        all_count = len(list((image_path / "all_images").glob("*.png"))) if (image_path / "all_images").exists() else 0
        best_count = len(list((image_path / "best_images").glob("*.png"))) if (image_path / "best_images").exists() else 0
        
        print(f"🖼️ 已生成图像: 最近{recent_count}张, 全部{all_count}张, 最佳{best_count}张")
        
        # 显示最新图像
        recent_dir = image_path / "recent_images"
        if recent_dir.exists():
            recent_images = list(recent_dir.glob("*.png"))
            if recent_images:
                latest = max(recent_images, key=lambda x: x.stat().st_mtime)
                mtime = datetime.fromtimestamp(latest.stat().st_mtime)
                print(f"    最新图像: {latest.name} ({mtime.strftime('%H:%M:%S')})")
    else:
        print("📁 图像目录尚未创建")
    
    # 检查GPU状态
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split('\n')[0].split(', ')
            gpu_util = gpu_info[0].strip()
            gpu_mem = gpu_info[1].strip()
            print(f"🔧 GPU状态: 利用率{gpu_util}%, 显存{gpu_mem}MB")
    except Exception:
        pass

def view_recent_images(n=5):
    """查看最近生成的图像"""
    print(f"\n🖼️ 最近 {n} 张生成图像:")
    print("-" * 50)
    
    image_path = Path("training_images")
    recent_dir = image_path / "recent_images"
    metadata_dir = image_path / "metadata"
    
    if not recent_dir.exists():
        print("❌ 图像目录不存在，可能训练还未开始生成图像")
        return
    
    image_files = list(recent_dir.glob("*.png"))
    if not image_files:
        print("📭 暂无图像文件")
        return
    
    image_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    for i, img_file in enumerate(image_files[:n]):
        print(f"\n{i+1}. {img_file.name}")
        print(f"   路径: {img_file}")
        
        # 读取元数据
        metadata_file = metadata_dir / f"{img_file.name}.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                step = metadata.get('step', 'unknown')
                epoch = metadata.get('epoch', 'unknown')
                loss = metadata.get('loss', 0.0)
                avg_score = metadata.get('avg_score', 0.0)
                quality = metadata.get('quality_label', 'unknown')
                
                sample_info = metadata.get('sample_info', {})
                instruction = sample_info.get('instruction', '')[:50]
                weight = sample_info.get('adaptive_weight', 0.0)
                
                print(f"   [步骤{step}|轮次{epoch}] 损失:{loss:.3f} 质量:{quality}({avg_score:.2f}) 权重:{weight:.2f}")
                print(f"   指令: {instruction}{'...' if len(instruction) >= 50 else ''}")
            except Exception:
                size_mb = img_file.stat().st_size / (1024 * 1024)
                mtime = datetime.fromtimestamp(img_file.stat().st_mtime)
                print(f"   大小: {size_mb:.1f}MB, 时间: {mtime.strftime('%H:%M:%S')}")

def start_training_with_monitor():
    """启动训练并开始监控"""
    # 启动监控线程
    monitor_thread = threading.Thread(target=monitor_training_progress, daemon=True)
    monitor_thread.start()
    
    # 启动训练
    config_file = "train_configs/plora_pretrain_mf_ood.yaml"
    cmd = [sys.executable, "train_collm_mf_din.py", "--cfg-path", config_file]
    
    print(f"执行训练命令: {' '.join(cmd)}")
    print("📊 监控已启动，将实时显示图像生成进度...")
    print()
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True)
        
        end_time = time.time()
        duration = end_time - start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        
        print("\n" + "🎉"*20)
        print("✅ CoRA训练完成!")
        print(f"⏱️  训练时长: {hours}h {minutes}m {seconds}s")
        print("🎉"*20)
        
        # 显示最终结果
        view_recent_images(3)
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练失败，退出码: {e.returncode}")
        print("💡 请检查错误信息并重新运行")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  训练被用户中断")
        
    except Exception as e:
        print(f"\n❌ 训练出现未知错误: {e}")

def main():
    print("🚀 启动CoRA多模态个性化推荐训练")
    print("="*80)
    
    # 训练架构总览
    print("📋 训练架构总览:")
    print("  🔧 协同过滤模块: 用户ID+物品ID → 低秩权重增量 → 注入LLM QKVO层")
    print("  🧠 LLM模块: 冻结主体，训练协同特征投影MLP + 评分预测头")
    print("  🎨 扩散模块: instruction+title+adaptive_weight*历史偏好 → LoRA UNet")
    print("  👁  监督模块: 兼容版Qwen2.5-VL评估器（4维度评分）")
    print()
    
    # 可训练模块检查
    print("✅ 可训练模块确认:")
    print("  • 协同特征到LLM的投影MLP (freeze_proj: False)")
    print("  • LLM输出评分的投影层 (enable_score_head: True)")
    print("  • 分层LoRA适配器 (UNet交叉注意力层的QKV矩阵)")
    print("  • LLM的LoRA层 (协同权重注入QKVO，freeze_lora: False)")
    print()
    
    # 冻结模块检查
    print("❄️  冻结模块确认:")
    print("  • LLM主体参数 (保持通用文本理解能力)")
    print("  • 扩散模型UNet主体 (保留基础生成能力)")
    print("  • 协同过滤主体 (freeze_rec: True)")
    print()
    
    # 监督信号说明
    print("🎯 监督信号:")
    print("  1. 指示一致性: 生成图像与instruction+title的匹配度")
    print("  2. 语义准确性: 与item_features(TopK商品)的匹配度")
    print("  3. 图像完整性: 构图完整性和视觉质量")
    print("  4. 质量: 技术表现与美观度")
    print("  5. 评分预测: MSE损失 (pred_rating vs target_rating)")
    print()
    
    # SwanLab记录说明
    print("📊 SwanLab记录指标:")
    print("  • train/loss, train/rating_loss, train/image_loss")
    print("  • image/consistency, image/accuracy, image/integrity, image/quality")
    print("  • image/adaptive_weight")
    print("  • train/lr, train/epoch, train/iter")
    print("  • epoch_mae, epoch_rmse")
    print()
    
    # 检查环境
    print("🔍 环境检查:")
    
    # 检查配置文件
    config_file = "train_configs/plora_pretrain_mf_ood.yaml"
    if os.path.exists(config_file):
        print(f"  ✅ 配置文件存在: {config_file}")
    else:
        print(f"  ❌ 配置文件缺失: {config_file}")
        return
    
    # 检查模型路径
    model_paths = {
        "LLM": "/root/autodl-tmp/vicuna/weight",
        "扩散模型": "/root/autodl-tmp/Stable_Diffusion", 
        "Qwen2.5-VL": "/root/autodl-tmp/Qwen2.5-VL-3B-Instruct",
        "数据集": "/root/autodl-tmp/dataset/amazon/"
    }
    
    for name, path in model_paths.items():
        if os.path.exists(path):
            print(f"  ✅ {name}: {path}")
        else:
            print(f"  ⚠️  {name}: {path} (不存在，但可能在训练中自动处理)")
    
    print()
    
    # 启动训练确认
    print("\n💡 选择启动模式:")
    print("  1. 普通训练 (无监控)")
    print("  2. 训练+实时监控 (推荐)")
    
    choice = input("🚀 请选择模式 (1/2), 或直接回车启动监控模式: ").strip()
    
    if choice == "1":
        # 普通训练模式
        confirm = input("🚀 确认启动普通训练? (y/N): ").lower().strip()
        if confirm not in ['y', 'yes']:
            print("❌ 训练已取消")
            return
            
        print("\n" + "🚀"*20)
        print("开始CoRA训练（普通模式）...")
        print("🚀"*20)
        
        cmd = [sys.executable, "train_collm_mf_din.py", "--cfg-path", config_file]
        print(f"执行命令: {' '.join(cmd)}")
        
        start_time = time.time()
        try:
            result = subprocess.run(cmd, check=True)
            
            end_time = time.time()
            duration = end_time - start_time
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            
            print("\n" + "🎉"*20)
            print("✅ CoRA训练完成!")
            print(f"⏱️  训练时长: {hours}h {minutes}m {seconds}s")
            print("🎉"*20)
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ 训练失败，退出码: {e.returncode}")
            print("💡 请检查错误信息并重新运行")
            
        except KeyboardInterrupt:
            print(f"\n⚠️  训练被用户中断")
            
        except Exception as e:
            print(f"\n❌ 训练出现未知错误: {e}")
            
    else:
        # 监控模式（默认）
        print("\n" + "🚀"*20)
        print("开始CoRA训练（监控模式）...")
        print("🚀"*20)
        
        start_training_with_monitor()
    
    print("\n📊 查看训练结果:")
    print("  • SwanLab面板: 查看实时训练指标")
    print("  • logs/test/: 检查训练日志和检查点")
    print("  • training_images/: 查看生成的个性化图像")
    print("\n💡 使用其他功能:")
    print("  python start_cora_training.py --status  # 检查训练状态")
    print("  python start_cora_training.py --images 5  # 查看最近5张图像")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoRA训练启动和监控工具")
    parser.add_argument("--status", action="store_true", help="检查当前训练状态")
    parser.add_argument("--images", type=int, default=0, help="查看最近生成的N张图像")
    parser.add_argument("--monitor", action="store_true", help="启动训练并开启实时监控")
    
    args = parser.parse_args()
    
    if args.status:
        show_training_status()
    elif args.images > 0:
        view_recent_images(args.images)
    elif args.monitor:
        # 跳过交互式确认，直接启动带监控的训练
        print("🚀 直接启动训练+监控模式...")
        start_training_with_monitor()
    else:
        # 默认交互式模式
        main()
