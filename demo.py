#!/usr/bin/env python3
"""
图像生成模块独立测试脚本
专门测试修复后的图像生成功能，不依赖完整训练流程
"""
import os
import sys
import time
import torch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_image_generation_module():
    """测试图像生成模块"""
    print("🎨 图像生成模块独立测试")
    print("="*60)
    
    # 1. 导入模块
    try:
        from image_personalization.qwen_image_trainer import QwenImageTrainer, QwenImageConfig
        print("✅ 成功导入QwenImageTrainer模块")
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False
    
    # 2. 创建配置
    print("\n🔧 创建测试配置...")
    config = QwenImageConfig(
        base_dir="/root/autodl-tmp/stable-diffusion-3.5-medium",
        num_inference_steps=15,  # 快速测试用较少步数
        true_cfg_scale=2.5,      # 保守CFG
        width=512,               # 较小尺寸加快测试
        height=512,
        use_4bit=True,
        enable_cpu_offload=True,
    )
    print(f"   📐 图像尺寸: {config.width}x{config.height}")
    print(f"   🎯 生成步数: {config.num_inference_steps}")
    print(f"   ⚖️ CFG Scale: {config.true_cfg_scale}")
    
    # 3. 初始化训练器
    print("\n🚀 初始化图像生成器...")
    try:
        trainer = QwenImageTrainer(config)
        print("✅ 图像生成器初始化成功")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 准备测试用例
    test_cases = [
        {
            "name": "简单游戏产品",
            "prompt": "a high-quality product image of gaming headset",
            "expected": "正常生成"
        },
        {
            "name": "电子设备",
            "prompt": "professional product photography of wireless controller",
            "expected": "正常生成"
        },
        {
            "name": "通用产品",
            "prompt": "detailed image showing gaming product",
            "expected": "正常生成"
        },
        {
            "name": "简短描述",
            "prompt": "PlayStation controller product image",
            "expected": "正常生成"
        },
        {
            "name": "Xbox产品",
            "prompt": "a high-quality product image of Xbox 360 Wireless Headset",
            "expected": "测试之前会黑图的case"
        }
    ]
    
    print(f"\n🧪 开始测试 {len(test_cases)} 个用例...")
    
    # 创建保存目录
    save_dir = "test_generated_images"
    os.makedirs(save_dir, exist_ok=True)
    print(f"📁 图像保存目录: {save_dir}/")
    
    # 5. 运行测试
    results = []
    for i, test_case in enumerate(test_cases):
        print(f"\n--- 测试 {i+1}/{len(test_cases)}: {test_case['name']} ---")
        print(f"📝 Prompt: {test_case['prompt']}")
        print(f"🎯 期望: {test_case['expected']}")
        
        start_time = time.time()
        
        try:
            # 生成图像
            images = trainer.generate_image(
                prompt=test_case['prompt'],
                seed=1000 + i  # 使用不同种子
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # 分析结果
            if images and len(images) > 0:
                # 保存图像
                image_filename = f"test_{i+1:02d}_{test_case['name'].replace(' ', '_')}.png"
                image_path = os.path.join(save_dir, image_filename)
                images[0].save(image_path)
                print(f"   💾 图像已保存: {image_path}")
                
                # 检查图像质量
                try:
                    import numpy as np
                    img_array = np.array(images[0])
                    mean_brightness = img_array.mean()
                    std_brightness = img_array.std()
                    h, w, c = img_array.shape
                    
                    # 计算唯一颜色数（采样）
                    sample_size = min(500, h * w)
                    indices = np.random.choice(h * w, sample_size, replace=False)
                    sampled_pixels = img_array.reshape(-1, c)[indices]
                    unique_colors = len(np.unique(sampled_pixels, axis=0))
                    
                    # 判断图像质量
                    is_black = mean_brightness < 10 and std_brightness < 5
                    is_white = mean_brightness > 245 and std_brightness < 5
                    is_monotone = unique_colors < 20
                    
                    if is_black:
                        status = "❌ 纯黑色图像"
                        quality = "black"
                    elif is_white:
                        status = "⚪ 纯白色图像"
                        quality = "white"
                    elif is_monotone:
                        status = "⚠️ 颜色单调"
                        quality = "monotone"
                    else:
                        status = "✅ 正常图像"
                        quality = "normal"
                    
                    print(f"   {status}")
                    print(f"   📊 统计: 亮度={mean_brightness:.1f}, 标准差={std_brightness:.1f}, 颜色={unique_colors}")
                    print(f"   ⏱️ 用时: {duration:.1f}秒")
                    
                    results.append({
                        'name': test_case['name'],
                        'success': True,
                        'quality': quality,
                        'duration': duration,
                        'brightness': mean_brightness,
                        'std': std_brightness,
                        'colors': unique_colors,
                        'saved_path': image_path
                    })
                    
                except Exception as analysis_error:
                    print(f"   ✅ 生成成功（分析失败: {analysis_error}）")
                    print(f"   ⏱️ 用时: {duration:.1f}秒")
                    results.append({
                        'name': test_case['name'],
                        'success': True,
                        'quality': 'unknown',
                        'duration': duration,
                        'saved_path': image_path
                    })
            else:
                print(f"   ❌ 生成失败：无图像输出")
                results.append({
                    'name': test_case['name'],
                    'success': False,
                    'quality': 'failed',
                    'duration': duration
                })
                
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"   ❌ 生成异常: {e}")
            results.append({
                'name': test_case['name'],
                'success': False,
                'quality': 'error',
                'duration': duration,
                'error': str(e)
            })
    
    # 6. 分析总体结果
    print(f"\n📊 测试结果总结")
    print("="*60)
    
    total_tests = len(results)
    successful_tests = [r for r in results if r['success']]
    failed_tests = [r for r in results if not r['success']]
    normal_images = [r for r in results if r.get('quality') == 'normal']
    black_images = [r for r in results if r.get('quality') == 'black']
    
    print(f"🎯 总测试数: {total_tests}")
    print(f"✅ 成功生成: {len(successful_tests)} ({len(successful_tests)/total_tests*100:.1f}%)")
    print(f"❌ 生成失败: {len(failed_tests)} ({len(failed_tests)/total_tests*100:.1f}%)")
    print(f"🎨 正常图像: {len(normal_images)} ({len(normal_images)/total_tests*100:.1f}%)")
    print(f"🖤 黑色图像: {len(black_images)} ({len(black_images)/total_tests*100:.1f}%)")
    
    if len(successful_tests) > 0:
        avg_duration = sum(r['duration'] for r in successful_tests) / len(successful_tests)
        print(f"⏱️ 平均用时: {avg_duration:.1f}秒")
    
    # 详细结果表格
    print(f"\n📋 详细结果:")
    print(f"{'序号':<4} {'名称':<15} {'状态':<8} {'质量':<10} {'用时':<6}")
    print("-" * 50)
    
    for i, result in enumerate(results):
        status_icon = "✅" if result['success'] else "❌"
        quality_desc = {
            'normal': '正常',
            'black': '纯黑',
            'white': '纯白', 
            'monotone': '单调',
            'failed': '失败',
            'error': '异常',
            'unknown': '未知'
        }.get(result.get('quality'), '未知')
        
        print(f"{i+1:<4} {result['name']:<15} {status_icon:<8} {quality_desc:<10} {result['duration']:.1f}s")
    
    # 7. 修复效果评估
    print(f"\n🎯 修复效果评估:")
    
    if len(black_images) == 0 and len(normal_images) >= 3:
        print("   🎉 修复成功！无黑图问题，生成稳定")
        grade = "A"
    elif len(black_images) <= 1 and len(normal_images) >= 2:
        print("   ✅ 修复良好，偶有问题但大幅改善")
        grade = "B"
    elif len(black_images) <= 2:
        print("   🔄 部分修复，仍需优化")
        grade = "C"
    else:
        print("   ⚠️ 修复效果有限，需要进一步调试")
        grade = "D"
    
    print(f"   📈 修复等级: {grade}")
    
    # 8. 建议
    print(f"\n💡 建议:")
    if len(black_images) > 0:
        print("   🔧 仍有黑图问题，建议:")
        print("      - 降低CFG scale (当前: {:.1f})".format(config.true_cfg_scale))
        print("      - 减少生成步数 (当前: {})".format(config.num_inference_steps))
        print("      - 检查模型文件完整性")
        print("      - 重启程序清理GPU缓存")
    
    if len(failed_tests) > 0:
        print("   ⚠️ 有生成失败的用例，检查:")
        print("      - GPU内存是否充足")
        print("      - 模型路径是否正确")
        print("      - 依赖库是否完整")
    
    if len(normal_images) >= 4:
        print("   🎊 图像生成模块工作良好！")
        print("   📈 可以尝试适当提高参数获得更好质量")
    
    # 9. 显示保存的图像列表
    saved_images = [r for r in results if r['success'] and 'saved_path' in r]
    if saved_images:
        print(f"\n📁 生成的图像已保存到: {save_dir}/")
        print("   图像列表:")
        for result in saved_images:
            quality_icon = {
                'normal': '✅',
                'black': '🖤', 
                'white': '⚪',
                'monotone': '🟨',
                'unknown': '❓'
            }.get(result.get('quality'), '❓')
            
            filename = os.path.basename(result['saved_path'])
            print(f"   {quality_icon} {filename}")
        
        print(f"\n💡 打开图像查看:")
        print(f"   cd {save_dir} && ls -la")
        print(f"   或直接查看: {os.path.abspath(save_dir)}")
    
    print("="*60)
    
    return len(normal_images) >= len(test_cases) // 2

def main():
    """主函数"""
    print("🎨 QwenImageTrainer 图像生成模块测试")
    print(f"⏰ 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 检查GPU状态
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("🖥️ 使用CPU模式")
    
    success = test_image_generation_module()
    
    print(f"\n⏰ 结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 测试结果: {'✅ 通过' if success else '❌ 需要优化'}")
    
    # 提供完整训练时的图像保存位置信息
    print(f"\n📌 注意事项:")
    print(f"   🧪 本测试的图像保存在: test_generated_images/")
    print(f"   🚂 正式训练时图像保存在: training_images/")
    print(f"      ├── recent_images/    (最近生成的图像)")
    print(f"      ├── all_images/       (所有保存的图像)")
    print(f"      ├── best_images/      (高质量图像)")
    print(f"      └── metadata/         (元数据信息)")
    print(f"   💡 可以使用 ls -la training_images/ 查看正式训练的图像")

if __name__ == "__main__":
    main()
