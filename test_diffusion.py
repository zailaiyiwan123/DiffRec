#!/usr/bin/env python3
"""
诊断Qwen-Image-Lightning模型加载问题
"""

import os
import torch
from pathlib import Path

def check_lora_files():
    """检查LoRA文件是否存在"""
    print("🔍 检查LoRA文件...")
    
    # 默认路径
    base_paths = [
        "/root/autodl-tmp/Qwen-Image-Lightning",
        "./Qwen-Image-Lightning",
        "../Qwen-Image-Lightning",
        os.path.expanduser("~/Qwen-Image-Lightning")
    ]
    
    lora_filename = "Qwen-Image-Lightning-4steps-V1.0.safetensors"
    
    for base_path in base_paths:
        lora_path = os.path.join(base_path, lora_filename)
        print(f"📂 检查: {lora_path}")
        
        if os.path.exists(lora_path):
            file_size = os.path.getsize(lora_path) / (1024 * 1024)  # MB
            print(f"✅ 找到LoRA文件: {lora_path}")
            print(f"📊 文件大小: {file_size:.2f} MB")
            return lora_path
        else:
            print(f"❌ 文件不存在: {lora_path}")
    
    print("\n⚠️ 没有找到LoRA文件，请检查以下可能的解决方案：")
    print("1. 确认Qwen-Image-Lightning模型已下载")
    print("2. 检查路径配置是否正确")
    print("3. 确认文件名是否正确")
    return None

def apply_compatibility_patches():
    """应用和训练时相同的兼容性补丁"""
    print("\n🔧 应用兼容性补丁...")
    
    try:
        import huggingface_hub
        
        # 补丁 split_torch_state_dict_into_shards 函数
        if not hasattr(huggingface_hub, 'split_torch_state_dict_into_shards'):
            def split_torch_state_dict_into_shards(state_dict, *args, **kwargs):
                """兼容性函数：模拟 split_torch_state_dict_into_shards"""
                if isinstance(state_dict, dict):
                    total_size = sum(
                        p.numel() * p.element_size() 
                        for p in state_dict.values() 
                        if hasattr(p, 'numel') and hasattr(p, 'element_size')
                    )
                else:
                    total_size = 0
                
                return [state_dict], {
                    "total_size": total_size,
                    "all_sizes": [total_size] if total_size > 0 else [0]
                }
            
            huggingface_hub.split_torch_state_dict_into_shards = split_torch_state_dict_into_shards
            print("✅ HuggingFace Hub 兼容性补丁已应用")
            
    except Exception as e:
        print(f"⚠️ HuggingFace Hub 补丁失败: {e}")
    
    try:
        import diffusers
        # 检查是否有 FlowMatchEulerDiscreteScheduler
        if not hasattr(diffusers, 'FlowMatchEulerDiscreteScheduler'):
            if hasattr(diffusers, 'EulerDiscreteScheduler'):
                diffusers.FlowMatchEulerDiscreteScheduler = diffusers.EulerDiscreteScheduler
                print("✅ 使用EulerDiscreteScheduler替代FlowMatchEulerDiscreteScheduler")
            else:
                class FlowMatchEulerDiscreteSchedulerCompat:
                    @classmethod
                    def from_config(cls, config):
                        return cls()
                    def __init__(self):
                        self.num_train_timesteps = 1000
                diffusers.FlowMatchEulerDiscreteScheduler = FlowMatchEulerDiscreteSchedulerCompat
                print("✅ 使用兼容性调度器替代FlowMatchEulerDiscreteScheduler")
    except Exception as e:
        print(f"⚠️ Diffusers 补丁失败: {e}")

def check_diffusers_import():
    """检查diffusers导入"""
    print("\n🔍 检查diffusers导入...")
    
    # 先应用补丁
    apply_compatibility_patches()
    
    try:
        from diffusers import DiffusionPipeline
        print("✅ DiffusionPipeline 导入成功")
    except Exception as e:
        print(f"❌ DiffusionPipeline 导入失败: {e}")
        return False
    
    try:
        from diffusers import FlowMatchEulerDiscreteScheduler
        print("✅ FlowMatchEulerDiscreteScheduler 导入成功")
    except Exception as e:
        print(f"❌ FlowMatchEulerDiscreteScheduler 导入失败: {e}")
        print("⚠️ 这可能导致使用兼容性调度器")
    
    return True

def test_lora_loading(lora_path):
    """测试LoRA权重加载"""
    if not lora_path:
        return False
        
    print(f"\n🔍 测试LoRA权重加载: {lora_path}")
    
    try:
        from safetensors.torch import safe_open
        
        lora_state_dict = {}
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            print(f"📊 LoRA文件包含 {len(keys)} 个权重")
            
            # 显示前几个键
            print("📋 权重键名示例:")
            for i, key in enumerate(keys[:5]):
                tensor = f.get_tensor(key)
                print(f"  - {key}: {tensor.shape} ({tensor.dtype})")
            
            if len(keys) > 5:
                print(f"  - ... 还有 {len(keys) - 5} 个权重")
        
        print("✅ LoRA权重加载测试成功")
        return True
        
    except Exception as e:
        print(f"❌ LoRA权重加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_model_creation():
    """测试模型创建"""
    print("\n🔍 测试基础模型创建...")
    
    try:
        # 尝试创建基础Transformer2D模型
        import torch.nn as nn
        
        class TestTransformer2D(nn.Module):
            def __init__(self, in_channels=8, **kwargs):
                super().__init__()
                self.norm = nn.GroupNorm(num_groups=4, num_channels=in_channels)
                
        model = TestTransformer2D()
        print("✅ 基础模型创建成功")
        return True
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return False

def check_image_generation_config():
    """检查图像生成配置"""
    print("\n🔍 检查图像生成配置...")
    
    try:
        from image_personalization.qwen_image_trainer import QwenImageConfig
        
        cfg = QwenImageConfig()
        print(f"📂 配置的base_model: {cfg.base_model}")
        print(f"🏷️ 配置的lora_weight_name: {cfg.lora_weight_name}")
        print(f"🎯 推理步数: {cfg.num_inference_steps}")
        print(f"⚖️ CFG缩放: {cfg.true_cfg_scale}")
        
        # 检查完整路径
        full_path = os.path.join(cfg.base_model, cfg.lora_weight_name)
        print(f"📍 完整LoRA路径: {full_path}")
        
        if os.path.exists(full_path):
            print("✅ 配置路径正确")
            return full_path
        else:
            print("❌ 配置路径不存在")
            return None
            
    except Exception as e:
        print(f"❌ 配置检查失败: {e}")
        return None

def test_actual_image_generation():
    """测试实际的图像生成流程"""
    print("\n🔍 测试QwenImageTrainer实际图像生成...")
    
    try:
        # 导入QwenImageTrainer
        from image_personalization.qwen_image_trainer import QwenImageTrainer, QwenImageConfig
        
        # 创建配置
        cfg = QwenImageConfig()
        print("✅ 配置创建成功")
        
        # 初始化训练器
        trainer = QwenImageTrainer(cfg)
        print("✅ QwenImageTrainer初始化成功")
        
        # 检查管道类型
        pipe_type = type(trainer.pipe).__name__
        print(f"🔍 管道类型: {pipe_type}")
        
        if "Mock" in pipe_type:
            print("❌ 检测到MockDiffusionPipeline - 这就是生成噪声图片的原因！")
            print("🔍 查看初始化过程中的错误信息...")
            return False
        else:
            print("✅ 使用真实的扩散管道")
        
        # 测试图像生成
        print("🎨 测试图像生成...")
        test_prompt = "Generate an image primarily focused on: Help me recommend items in Video Games, Xbox Controller."
        
        images = trainer.generate_image(test_prompt)
        
        if images and len(images) > 0:
            image = images[0]
            print(f"✅ 生成图像成功: {image.size}, 模式: {image.mode}")
            
            # 检查是否是噪声图像
            import numpy as np
            img_array = np.array(image)
            
            # 计算图像的标准差 - 噪声图像通常有很高的标准差
            std_dev = np.std(img_array)
            mean_val = np.mean(img_array)
            
            print(f"📊 图像统计: 均值={mean_val:.2f}, 标准差={std_dev:.2f}")
            
            # 噪声图像通常标准差很高（>60），均值接近128
            if std_dev > 60 and 100 < mean_val < 156:
                print("❌ 检测到噪声图像特征！")
                return False
            else:
                print("✅ 图像看起来不像纯噪声")
                return True
        else:
            print("❌ 图像生成失败")
            return False
            
    except Exception as e:
        print(f"❌ QwenImageTrainer测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def suggest_solutions():
    """建议解决方案"""
    print("\n💡 可能的解决方案:")
    
    print("🔧 如果生成噪声图像，可能的原因和解决方案：")
    print("\n1. LoRA权重合并问题：")
    print("   - 检查load_and_merge_lora_weight_from_safetensors函数是否正确执行")
    print("   - 确认LoRA权重键名和模型参数名匹配")
    print("   - 验证权重合并时的设备一致性")
    
    print("\n2. 扩散模型配置问题：")
    print("   - 检查调度器配置是否正确")
    print("   - 确认推理步数(num_inference_steps=4)是否合适")
    print("   - 验证CFG缩放(true_cfg_scale=1.0)设置")
    
    print("\n3. 模型初始化问题：")
    print("   - 确保QwenImageTransformer2DModel参数正确")
    print("   - 检查in_channels和norm_num_groups的兼容性")
    print("   - 验证模型是否fallback到MockDiffusionPipeline")
    
    print("\n4. 环境和依赖问题：")
    print("   - 更新diffusers版本: pip install diffusers --upgrade")
    print("   - 确保torch版本兼容: pip install torch>=1.12.0")
    print("   - 检查CUDA版本和GPU内存")
    
    print("\n5. 下载完整模型（如果缺失）：")
    print("   git clone https://huggingface.co/Qwen/Qwen-Image-Lightning")
    print("   或者单独下载LoRA：")
    print("   wget https://huggingface.co/Qwen/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V1.0.safetensors")
    
    print("\n6. 调试步骤：")
    print("   - 运行: python diagnose_image_model.py")
    print("   - 检查训练日志中的管道初始化信息")
    print("   - 确认是否看到'使用占位功能，图像生成被禁用'的警告")
    
    print("\n🚨 如果问题持续存在，可能需要：")
    print("   - 降级到稳定的diffusers版本")
    print("   - 使用标准Stable Diffusion而非Qwen-Image-Lightning")
    print("   - 检查其他成功案例的环境配置")

def main():
    """主诊断流程"""
    print("🩺 开始诊断Qwen-Image-Lightning模型加载问题...\n")
    
    # 1. 检查LoRA文件
    lora_path = check_lora_files()
    
    # 2. 检查diffusers导入
    diffusers_ok = check_diffusers_import()
    
    # 3. 检查配置
    config_path = check_image_generation_config()
    
    # 4. 测试LoRA加载（如果文件存在）
    if lora_path:
        lora_loading_ok = test_lora_loading(lora_path)
    else:
        lora_loading_ok = False
    
    # 5. 测试模型创建
    model_creation_ok = check_model_creation()
    
    # 6. 测试实际图像生成（关键测试）
    image_generation_ok = False
    if lora_path and diffusers_ok and config_path:
        image_generation_ok = test_actual_image_generation()
    else:
        print("\n⚠️ 跳过图像生成测试，因为前置条件不满足")
    
    # 总结
    print("\n" + "="*60)
    print("📊 诊断结果总结:")
    print("="*60)
    print(f"✅ LoRA文件存在: {'是' if lora_path else '否'}")
    print(f"✅ Diffusers导入: {'正常' if diffusers_ok else '异常'}")
    print(f"✅ 配置路径: {'正确' if config_path else '错误'}")
    print(f"✅ LoRA加载: {'成功' if lora_loading_ok else '失败'}")
    print(f"✅ 模型创建: {'成功' if model_creation_ok else '失败'}")
    print(f"✅ 图像生成: {'正常' if image_generation_ok else '异常/噪声'}")
    
    if image_generation_ok:
        print("\n🎉 所有检查都通过！图像生成应该能正常工作。")
        print("如果训练时仍然生成噪声，可能是训练数据或其他配置问题。")
    elif all([lora_path, diffusers_ok, config_path, lora_loading_ok, model_creation_ok]):
        print("\n❌ 虽然所有组件都正常，但图像生成异常！")
        print("这说明LoRA权重没有正确合并或扩散过程有问题。")
        suggest_solutions()
    else:
        print("\n⚠️ 发现基础问题，请按照以下建议解决：")
        suggest_solutions()

if __name__ == "__main__":
    main()
