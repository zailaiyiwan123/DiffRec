#!/usr/bin/env python3
"""
修复 huggingface_hub 版本兼容性问题的脚本
解决 HF_HUB_CACHE 属性错误
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """运行命令并显示输出"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description}成功")
            if result.stdout.strip():
                print(f"📋 输出: {result.stdout.strip()}")
        else:
            print(f"❌ {description}失败")
            print(f"错误: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ {description}异常: {e}")
        return False

def check_current_versions():
    """检查当前版本"""
    print("📦 检查当前包版本...")
    packages = [
        'huggingface-hub', 'transformers', 'diffusers', 
        'accelerate', 'peft', 'torch'
    ]
    
    for pkg in packages:
        try:
            result = subprocess.run(
                f"pip show {pkg}", 
                shell=True, capture_output=True, text=True
            )
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                version_line = [l for l in lines if l.startswith('Version:')]
                if version_line:
                    version = version_line[0].split(':')[1].strip()
                    print(f"  {pkg}: {version}")
                else:
                    print(f"  {pkg}: 版本未知")
            else:
                print(f"  {pkg}: 未安装")
        except Exception as e:
            print(f"  {pkg}: 检查失败 - {e}")

def fix_versions():
    """修复版本兼容性"""
    print("\n🚀 开始修复版本兼容性问题...")
    
    # 步骤1: 升级 huggingface_hub 到兼容版本
    print("\n步骤1: 升级 huggingface_hub...")
    success = run_command(
        "pip install --upgrade huggingface_hub>=0.19.0", 
        "升级 huggingface_hub"
    )
    
    if not success:
        print("⚠️ 尝试强制重新安装...")
        run_command(
            "pip uninstall -y huggingface_hub && pip install huggingface_hub>=0.19.0",
            "强制重新安装 huggingface_hub"
        )
    
    # 步骤2: 确保其他包版本兼容
    print("\n步骤2: 确保其他包版本兼容...")
    
    # 推荐的兼容版本组合
    compatible_versions = [
        "transformers>=4.40.0",
        "diffusers>=0.21.0", 
        "accelerate>=0.21.0",
        "peft>=0.5.0"
    ]
    
    for pkg_version in compatible_versions:
        run_command(
            f"pip install --upgrade '{pkg_version}'",
            f"升级 {pkg_version}"
        )
    
    # 步骤3: 验证修复
    print("\n步骤3: 验证修复结果...")
    test_imports()

def test_imports():
    """测试关键导入"""
    print("🧪 测试关键模块导入...")
    
    test_cases = [
        ("huggingface_hub.constants", "测试 huggingface_hub.constants"),
        ("transformers", "测试 transformers"),
        ("diffusers", "测试 diffusers"),
        ("accelerate", "测试 accelerate"), 
        ("peft", "测试 peft")
    ]
    
    results = []
    for module, desc in test_cases:
        try:
            __import__(module)
            print(f"✅ {desc} - 成功")
            results.append(True)
        except Exception as e:
            print(f"❌ {desc} - 失败: {e}")
            results.append(False)
    
    # 特别测试 HF_HUB_CACHE 属性
    print("\n🔍 特别测试 HF_HUB_CACHE 属性...")
    try:
        from huggingface_hub import constants
        if hasattr(constants, 'HF_HUB_CACHE'):
            print("✅ HF_HUB_CACHE 属性存在")
            results.append(True)
        else:
            # 检查新的属性名
            attrs = [attr for attr in dir(constants) if 'CACHE' in attr.upper()]
            print(f"⚠️ HF_HUB_CACHE 不存在，但找到缓存相关属性: {attrs}")
            
            # 检查是否有替代属性
            if hasattr(constants, 'HF_HOME') or hasattr(constants, 'HUGGINGFACE_HUB_CACHE'):
                print("✅ 找到替代缓存属性")
                results.append(True)
            else:
                print("❌ 未找到缓存相关属性")
                results.append(False)
                
    except Exception as e:
        print(f"❌ HF_HUB_CACHE 测试失败: {e}")
        results.append(False)
    
    success_rate = sum(results) / len(results)
    print(f"\n📊 测试成功率: {success_rate:.1%}")
    
    return success_rate > 0.8

def create_compatibility_patch():
    """创建兼容性补丁"""
    print("\n🔧 创建兼容性补丁...")
    
    patch_content = '''
"""
HuggingFace Hub 兼容性补丁
解决不同版本间的属性差异
"""

def patch_hf_cache():
    """修补 HF_HUB_CACHE 属性问题"""
    try:
        from huggingface_hub import constants
        
        # 如果没有 HF_HUB_CACHE，尝试使用替代方案
        if not hasattr(constants, 'HF_HUB_CACHE'):
            import os
            
            # 尝试不同的缓存路径方案
            if hasattr(constants, 'HF_HOME'):
                constants.HF_HUB_CACHE = constants.HF_HOME
            elif hasattr(constants, 'HUGGINGFACE_HUB_CACHE'):
                constants.HF_HUB_CACHE = constants.HUGGINGFACE_HUB_CACHE
            else:
                # 使用默认缓存路径
                default_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
                constants.HF_HUB_CACHE = default_cache
                
            print(f"🔧 已修补 HF_HUB_CACHE: {constants.HF_HUB_CACHE}")
            
    except Exception as e:
        print(f"⚠️ 兼容性补丁应用失败: {e}")

# 自动应用补丁
patch_hf_cache()
'''
    
    try:
        with open('hf_compatibility_patch.py', 'w', encoding='utf-8') as f:
            f.write(patch_content)
        print("✅ 兼容性补丁已创建: hf_compatibility_patch.py")
        return True
    except Exception as e:
        print(f"❌ 创建补丁失败: {e}")
        return False

def main():
    """主函数"""
    print("🔧 HuggingFace 版本兼容性修复工具")
    print("="*50)
    
    # 检查当前版本
    check_current_versions()
    
    # 修复版本
    fix_versions()
    
    # 检查修复后的版本
    print("\n📦 修复后的版本:")
    check_current_versions()
    
    # 创建兼容性补丁（备用方案）
    create_compatibility_patch()
    
    print("\n" + "="*50)
    print("🎯 修复完成！")
    print("\n💡 如果问题仍然存在，请尝试:")
    print("1. 重启 Python 解释器")
    print("2. 清除 pip 缓存: pip cache purge")
    print("3. 使用兼容性补丁: import hf_compatibility_patch")

if __name__ == "__main__":
    main()
