from typing import Optional, Dict, Any
from pathlib import Path
import torch

from .qwen_image_trainer import QwenImageConfig, QwenImageTrainer
from .conditioning import extract_titles_from_his_interaction, build_preference_text, fuse_instruction_and_preference, build_enhanced_prompt

# Global switch: image generation enable/disable
ENABLE_IMAGE_GENERATION = True  # Set to False to disable, True to enable

_qwen_trainer: Optional[QwenImageTrainer] = None


def init_qwen_trainer(lora_path: str = "/root/autodl-tmp/stable-diffusion-3.5-medium"):
    """Initialize Stable Diffusion 3.5 trainer"""
    global _qwen_trainer
    
    if not ENABLE_IMAGE_GENERATION:
        print("[Hook] Image generation disabled, skipping trainer initialization")
        return
    
    if _qwen_trainer is not None:
        return
    
    print("[Hook] Initializing SD3.5 image trainer...")
    cfg = QwenImageConfig(
        base_model="",
        lora_weight_name=None,
        base_dir=lora_path,
    )
    
    try:
        expert = None
        
        _qwen_trainer = QwenImageTrainer(cfg, expert=expert)
        print(f"[Hook] SD3.5 trainer initialized successfully: {type(_qwen_trainer).__name__}")
    except Exception as e:
        print(f"[Hook] Image trainer initialization failed: {e}")
        _qwen_trainer = None

    print("[Hook] SD3.5 trainer initialization completed")


def update_image_module(sample: Dict[str, Any] = None, adaptive_weight: float = 1.0, save_dir: Optional[str] = None, epoch: int = None, qwen_cfg: QwenImageConfig = None, lora_rank: int = 16, lora_alpha: int = 32, expert=None):
    """
    Update image module - supports two calling modes:
    1. Traditional: pass sample for training
    2. New: pass qwen_cfg for initialization
    """
    global _qwen_trainer
    
    if not ENABLE_IMAGE_GENERATION:
        if sample is not None:
            print("[Hook] Image generation disabled, returning zero loss")
            return {"loss": torch.tensor(0.0), "diffusion_loss": torch.tensor(0.0), "expert_scores": {}, "images": []}
        elif qwen_cfg is not None:
            print("[Hook] Image generation disabled, skipping trainer initialization")
            return False
        else:
            print("[Hook] Image generation disabled, no operation")
            return None
    
    if qwen_cfg is not None:
        if _qwen_trainer is not None:
            print("[Hook] Existing trainer instance found, reinitializing")
            _qwen_trainer = None
            
        print(f"[Hook] Initializing SD3.5 trainer with new config: {qwen_cfg.base_dir}")
        try:
            if expert is not None:
                print(f"[Hook] Using provided CLIP evaluator: {type(expert).__name__}")
            else:
                print("[Hook] No evaluator provided, using default config")

            _qwen_trainer = QwenImageTrainer(qwen_cfg, expert=expert)
            print(f"[Hook] SD3.5 trainer initialized successfully: {type(_qwen_trainer).__name__}")
            
            print(f"[Hook] LoRA config: rank={lora_rank}, alpha={lora_alpha}")
            return True
        except Exception as e:
            print(f"[Hook] SD3.5 trainer initialization failed: {e}")
            import traceback
            traceback.print_exc()
            _qwen_trainer = None
            return False
    
    if sample is None:
        print("[Hook] No sample or qwen_cfg provided, cannot process")
        return None
    
    if _qwen_trainer is None:
        print("[Hook] Trainer is None, initializing...")
        init_qwen_trainer(lora_path="/root/autodl-tmp/stable-diffusion-3.5-medium")
        print(f"[Hook] Trainer initialization completed: {_qwen_trainer is not None}")

    instruction = sample.get("instruction", "")
    title = sample.get("title", "")
    his_interaction = sample.get("his_interaction", "")
    item_features = sample.get("item_features", "")

    if _qwen_trainer is None:
        return {"loss": torch.tensor(0.0), "diffusion_loss": 0.0, "expert_scores": {}}
    
    try:
        result = _qwen_trainer.training_step({
            "instruction": instruction,
            "title": title,
            "his_interaction": his_interaction,
            "topk_desc": item_features,
            "adaptive_weight": adaptive_weight,
        }, save_dir=save_dir, epoch=epoch)
        
        if isinstance(result, dict) and 'images' in result:
            print(f"[Hook] Generated images: {len(result['images'])}")
        
        return result
    except Exception as e:
        print(f"[Hook] Image generation call failed: {e}")
        import traceback
        traceback.print_exc()
        import torch
        return {"loss": torch.tensor(0.0), "diffusion_loss": 0.0, "expert_scores": {}}

