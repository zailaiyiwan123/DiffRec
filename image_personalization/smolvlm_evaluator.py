from typing import Dict
import json

import torch
from PIL import Image


class SmolVLMEvaluator:
    """
    Use local SmolVLM-256M-Instruct for multi-dimensional scoring.
    Need to provide model directory path, e.g., "SmolVLM-256M-Instruct".
    """
    def __init__(self, model_dir: str, device: torch.device):
        print(f"🔍 Initializing SmolVLM evaluator: {model_dir}")
        
        # Check transformers import
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            print(f"✅ Transformers module imported successfully")
        except ImportError as e:
            raise ImportError(f"transformers library import failed, please check version: {e}")
            
        # Check from_pretrained methods
        if not hasattr(AutoProcessor, 'from_pretrained'):
            raise RuntimeError("AutoProcessor does not have from_pretrained method")
        if not hasattr(AutoModelForCausalLM, 'from_pretrained'):
            raise RuntimeError("AutoModelForCausalLM does not have from_pretrained method")
            
        self.device = device
        
        # Load processor
        try:
            print("🔄 Loading AutoProcessor...")
            try:
                self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
                print("✅ AutoProcessor loaded successfully")
            except Exception as e1:
                print(f"⚠️ AutoProcessor failed: {e1}")
                print("🔄 Trying to auto-detect Tokenizer type...")
                from transformers import AutoTokenizer, AutoImageProcessor, CLIPImageProcessor
                try:
                    # Auto-detect correct tokenizer type
                    self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
                    print(f"✅ Auto-detected tokenizer: {type(self.tokenizer).__name__}")
                except Exception as e2:
                    print(f"⚠️ AutoTokenizer also failed: {e2}")
                    # Finally try GPT2Tokenizer (based on error message)
                    from transformers import GPT2Tokenizer
                    self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir, trust_remote_code=True)
                    print("✅ Successfully using GPT2Tokenizer")
                    
                try:
                    # Try to load image processor
                    self.image_processor = AutoImageProcessor.from_pretrained(model_dir, trust_remote_code=True)
                    print("✅ AutoImageProcessor loaded successfully")
                except:
                    try:
                        self.image_processor = CLIPImageProcessor.from_pretrained(model_dir, trust_remote_code=True)
                        print("✅ CLIPImageProcessor loaded successfully")
                    except:
                        self.image_processor = None
                        print("⚠️ Image processor loading failed, will only use text")
                        
                self.processor = None  # Manual processing
                print("✅ Manual Tokenizer configuration completed")
        except Exception as e:
            raise RuntimeError(f"Processor loading failed: {e}")
            
        # Load model
        try:
            print("🔄 Loading AutoModelForCausalLM...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir, 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, 
                trust_remote_code=True
            )
            print("✅ AutoModelForCausalLM loaded successfully")
            
            self.model = self.model.to(self.device).eval()
            print(f"✅ Model moved to device {self.device} and set to eval mode")
            
        except Exception as e:
            raise RuntimeError(f"AutoModelForCausalLM loading failed: {e}")

        self.prompt_template = (
            "You are a professional multimodal evaluation expert. Please strictly score the [User Input] and [Reference Image] according to the following dimensions on a 0-5 integer scale (5 being the best). Scoring should be independent and objective, no need to explain reasons.\n\n"
            "### Input Data:\n[Text] \"{instruction}\"\n[Image] \n\n"
            "### Scoring Dimensions and Standards:\n"
            "1. **Instruction Consistency** → Degree of matching between text description and image content:\n   - 5 points: Perfect match of core elements\n   - 3 points: Partial match but with deviations\n   - 0 points: Completely unrelated\n"
            "2. **Semantic Accuracy** → Accuracy of key objects/scenes in the image:\n   - 5 points: All object positions and attributes correct\n   - 3 points: Main objects correct, detail errors\n   - 0 points: Core objects missing or incorrect\n"
            "3. **Image Integrity** → Composition rationality and visual integrity:\n   - 5 points: No defects/occlusion, prominent subject\n   - 3 points: Local occlusion but doesn't affect understanding\n   - 0 points: Serious defects or information loss\n"
            "4. **Quality** → Technical performance and aesthetics:\n   - 5 points: High definition, natural lighting, no noise\n   - 3 points: Medium quality, minor flaws\n   - 0 points: Blurry, distorted or severely distorted\n\n"
            "### Output Format (JSON):\n{\n  \"consistency_score\": [0-5],\n  \"accuracy_score\": [0-5],\n  \"integrity_score\": [0-5],\n  \"quality_score\": [0-5]\n}"
        )

    @torch.no_grad()
    def score_image(self, instruction: str, image: Image.Image) -> Dict[str, float]:
        text = self.prompt_template.format(instruction=instruction)
        
        if self.processor is not None:
            # Use AutoProcessor
            inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=256)
            resp = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        else:
            # Manual processing - can only handle text (image processor may be incompatible)
            print("⚠️ SmolVLM cannot process images, skipping evaluation")
            return {
                "consistency": None,  # Cannot verify
                "accuracy": None,     # Cannot verify  
                "integrity": None,    # Cannot verify
                "quality": None,      # Cannot verify
                "status": "evaluation_failed",
                "reason": "image_processing_unavailable"
            }
        # Parse JSON from response
        try:
            json_start = resp.find("{")
            json_str = resp[json_start:]
            data = json.loads(json_str)
            c = float(data.get("consistency_score", 0))
            a = float(data.get("accuracy_score", 0))
            i = float(data.get("integrity_score", 0))
            q = float(data.get("quality_score", 0))
        except Exception:
            c = a = i = q = 0.0
        return {
            "consistency": c,
            "accuracy": a,
            "integrity": i,
            "quality": q,
        }


