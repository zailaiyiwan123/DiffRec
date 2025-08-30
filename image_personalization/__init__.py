# namespace init for image_personalization

# Export main modules
from .hook import init_qwen_trainer, update_image_module

from .smolvlm_evaluator import SmolVLMEvaluator
from .sd_lora_trainer import SDPersonalizationTrainer, SDPersonalizationConfig
from .qwen_image_trainer import QwenImageConfig, QwenImageTrainer

# Compatibility: keep old names
init_sd_trainer = init_qwen_trainer

__all__ = [
    'init_sd_trainer',      # Compatibility
    'init_qwen_trainer',    # New name
    'update_image_module', 
    'QwenVLEvaluator',
    'SmolVLMEvaluator',
    'SDPersonalizationTrainer',
    'SDPersonalizationConfig',
    'QwenImageConfig',
    'QwenImageTrainer'
]
