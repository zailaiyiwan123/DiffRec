"""
PEFT compatibility layer - provides interface compatible with minigpt4.models.ppeft
"""

# Try importing original PEFT
try:
    from peft import LoraConfig as PLoraConfig, get_peft_model
    print("Using original PEFT library")
except ImportError:
    # Try importing from minigpt4.models.ppeft
    try:
        from minigpt4.models.ppeft import PLoraConfig, get_peft_model
        print("Using minigpt4.models.ppeft")
    except ImportError:
        # Create compatibility classes
        class PLoraConfig:
            def __init__(self, r=8, lora_alpha=16, target_modules=None, lora_dropout=0.0, **kwargs):
                self.r = r
                self.lora_alpha = lora_alpha
                self.target_modules = target_modules or []
                self.lora_dropout = lora_dropout
                
        def get_peft_model(model, config):
            print("Warning: Using PEFT compatibility layer")
            return model
            
        print("Warning: PEFT import failed, using compatibility layer")
