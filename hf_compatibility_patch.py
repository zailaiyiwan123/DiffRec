
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
