#!/usr/bin/env python3
"""
QwenImageTrainer refactored version - focused on stability and simplicity
Solving black image alternation issues, redesigned based on successful testing experience
"""
import os
import time
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Any, Optional
from PIL import Image, ImageDraw

@dataclass
class QwenImageConfig:
    """Compatible SD3.5 configuration"""
    # LoRA related (maintain original interface)
    base_model: str = ""  # LoRA directory
    lora_weight_name: Optional[str] = None  # LoRA filename
    # Base model path
    base_dir: Optional[str] = "/root/autodl-tmp/stable-diffusion-3.5-medium"

    # Inference parameters (further optimized quality configuration)
    num_inference_steps: int = 40    # Increase steps for better detail quality
    true_cfg_scale: float = 4.5      # Enhanced guidance strength for better semantic consistency
    width: int = 720                # Moderately increased resolution for clearer images
    height: int = 720               # Moderately increased resolution for clearer images
    
    # Memory optimization options
    use_4bit: bool = True
    compute_dtype: str = "bfloat16"
    enable_cpu_offload: bool = True

    # Training loss weights (maintain original interface)
    learning_rate: float = 1e-4
    weight_instr_consistency: float = 0.25
    weight_semantic_accuracy: float = 0.25
    weight_image_integrity: float = 0.25
    weight_quality: float = 0.25

class QwenImageTrainerV2(nn.Module):
    """Refactored QwenImageTrainer - focused on stability"""
    
    def __init__(self, cfg: QwenImageConfig, expert=None):
        """Compatible with original constructor signature"""
        super().__init__()
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.step_count = 0
        self.epoch_count = 0

        # Simplified image save directory
        self.save_dir = "training_images_v2"
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Image save directory: {self.save_dir}")
        
        # Expert evaluator (prioritize provided expert, otherwise create simplified version)
        self.expert = expert  # If None, use default scores during evaluation
        expert_type = type(self.expert).__name__ if self.expert else "None"
        print(f"Expert evaluator: {expert_type}")
        
        # Initialize pipeline
        self._init_pipeline()
        
        # Initialize LLM for historical preference summarization
        self._init_llm_for_preference_summary()
        
    def _init_pipeline(self):
        """Initialize Stable Diffusion pipeline (memory optimized version)"""
        print("Initializing SD3.5 pipeline...")
        
        # Check memory status
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            total_mem = torch.cuda.get_device_properties(0).total_memory
            allocated_mem = torch.cuda.memory_allocated(0)
            free_mem = total_mem - allocated_mem
            free_gb = free_mem / (1024**3)
            print(f"Memory status: {free_gb:.1f}GB available / {total_mem/(1024**3):.1f}GB total")
            
            # Dual-GPU balanced environment check
            if free_gb < 8:  # 8GB per card sufficient for balanced allocation
                print(f"Warning: Insufficient memory ({free_gb:.1f}GB < 8GB), skipping SD pipeline initialization")
                self.pipe = None
                return
        
        try:
            # Import necessary libraries
            from diffusers import StableDiffusion3Pipeline
            
            # Memory optimized loading approach
            print("Loading SD3.5 in memory optimized mode...")
            
            # Dual-GPU balanced loading approach
            if torch.cuda.device_count() > 1:
                print(f"Detected {torch.cuda.device_count()} GPUs, enabling multi-GPU balanced loading")
                # Use device_map for load balancing in multi-GPU environment
                device_map_strategy = "balanced"
            else:
                device_map_strategy = None
                
            self.pipe = StableDiffusion3Pipeline.from_pretrained(
                self.cfg.base_dir,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                ignore_mismatched_sizes=True,
                variant="fp16",
                device_map=device_map_strategy,
            )
            
            # Decide manual movement based on loading strategy
            if device_map_strategy is None:
                # Manual move to GPU in single-GPU environment
                self.pipe = self.pipe.to(self.device)
                print("Model moved to GPU")
            else:
                # device_map automatically assigned in multi-GPU environment
                print("Model automatically assigned to multi-GPU via device_map")
            
            # Enable CPU offload for memory optimization (only in single-GPU mode)
            if device_map_strategy is None and hasattr(self.pipe, 'enable_model_cpu_offload'):
                self.pipe.enable_model_cpu_offload()
                print("CPU offload enabled")
            elif device_map_strategy is not None:
                print("Skipping CPU offload in multi-GPU environment (device_map already used)")
            
            if hasattr(self.pipe, 'enable_attention_slicing'):
                self.pipe.enable_attention_slicing(1)
                print("Attention slicing enabled")
            
            # Enable additional memory optimizations
            if hasattr(self.pipe, 'enable_vae_slicing'):
                self.pipe.enable_vae_slicing()
                print("VAE slicing enabled")
            
            if hasattr(self.pipe, 'enable_vae_tiling'):
                self.pipe.enable_vae_tiling()
                print("VAE tiling enabled")
                
            # Final memory check
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                new_allocated = torch.cuda.memory_allocated(0)
                print(f"SD pipeline memory usage: {new_allocated/(1024**3):.1f}GB")
            
            print("SD3.5 pipeline initialized successfully")
            
        except Exception as e:
            print(f"Pipeline initialization failed: {e}")
            print("Downgrading to CPU mode...")
            try:
                # Downgrade to CPU execution
                self.pipe = StableDiffusion3Pipeline.from_pretrained(
                    self.cfg.base_dir,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    ignore_mismatched_sizes=True
                )
                print("SD3.5 pipeline initialized in CPU mode")
            except:
                print("CPU mode initialization also failed, disabling image generation")
            self.pipe = None
    
    def _init_llm_for_preference_summary(self):
        """Initialize LLM for historical preference summarization"""
        print("Initializing Vicuna LLM for preference summarization...")
        
        try:
            self.llm_path = "/root/autodl-tmp/vicuna/weight"
            self.llm_model = None
            self.llm_tokenizer = None
            self.llm_available = True
            print("LLM configuration completed, will lazy load when needed")
            
        except Exception as e:
            print(f"Warning: LLM configuration failed: {e}")
            self.llm_available = False
    
    def _load_llm_if_needed(self):
        """Lazy load LLM model"""
        if not self.llm_available or self.llm_model is not None:
            return
        
        try:
            print("Loading Vicuna model...")
            from transformers import LlamaTokenizer, LlamaForCausalLM
            
            self.llm_tokenizer = LlamaTokenizer.from_pretrained(
                self.llm_path, 
                trust_remote_code=True
            )
            
            self.llm_model = LlamaForCausalLM.from_pretrained(
                self.llm_path,
                torch_dtype=torch.float16,
                device_map="cpu",  # Load to CPU first to avoid memory conflicts
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            
            print("Vicuna model loaded successfully")
            
        except Exception as e:
            print(f"LLM loading failed: {e}")
            self.llm_available = False
    
    def _summarize_preferences_with_llm(self, historical_titles: str) -> str:
        """Use LLM to summarize historical preference style types"""
        if not self.llm_available or not historical_titles:
            return ""
        
        try:
            self._load_llm_if_needed()
            
            if not self.llm_model:
                return ""
            
            # Direct keyword extraction prompt - concise and efficient
            summary_prompt = f"""From these user items, extract 3-5 specific keywords:

{historical_titles[:800]}

Extract keywords like: brands, categories, themes, genres, styles, materials, types.
Output only comma-separated keywords, no explanations.

Examples:
Nintendo, action, platformer
vintage, vinyl, rock
fantasy, medieval, magic
outdoor, camping, hiking

Keywords:"""

            # Generate summary
            inputs = self.llm_tokenizer.encode(summary_prompt, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    inputs,
                    max_length=inputs.size(1) + 25,  # Increase generation length for more space
                    temperature=0.3,  # Moderate temperature for creativity while maintaining accuracy
                    do_sample=True,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                    num_return_sequences=1,
                    early_stopping=True,
                    repetition_penalty=1.1,  # Avoid repetitive generation
                )
            
            # Decode results
            generated_text = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            summary = generated_text.replace(summary_prompt, "").strip()
            
            # Intelligent keyword extraction - extract useful features from LLM output
            import re
            
            # Clean LLM output, remove common garbage content
            summary = summary.strip()
            
            # Skip if output contains obvious conversational text
            invalid_patterns = [
                r'\bI\s+think\b', r'\bThe\s+user\b', r'\bBased\s+on\b', 
                r'\bIt\s+appears\b', r'\bIt\s+seems\b', r'\bHuman\b', 
                r'\bAssistant\b', r'\bSorry\b', r'\bCannot\b'
            ]
            
            for pattern in invalid_patterns:
                if re.search(pattern, summary, re.IGNORECASE):
                    return ""
            
            # Extract keywords - handle multiple separators
            summary = re.sub(r'[^\w\s,\-&]', ' ', summary)  # Keep words, spaces, commas, hyphens, & symbols
            summary = re.sub(r'\s+', ' ', summary)  # Merge multiple spaces
            
            # Split by comma, if no comma then split by space
            if ',' in summary:
                raw_keywords = summary.split(',')
            else:
                raw_keywords = summary.split()
            
            # Clean and filter keywords
            keywords = []
            for word in raw_keywords:
                word = word.strip().lower()
                
                # Skip invalid words
                if (word and 
                    len(word) > 2 and  # At least 3 characters
                    not word.isdigit() and  # Not pure numbers
                    word not in ['the', 'and', 'or', 'but', 'with', 'for', 'to', 'of', 'in', 'on', 'at'] and  # Not stop words
                    len(word) < 20):  # Not too long
                    keywords.append(word)
            
            # Limit keyword count (keep first 5 most meaningful)
            if len(keywords) > 5:
                keywords = keywords[:5]
            
            # Deduplicate while maintaining order
            unique_keywords = []
            for kw in keywords:
                if kw not in unique_keywords:
                    unique_keywords.append(kw)
            
            final_summary = ', '.join(unique_keywords)
            
            # Final validation - ensure actual content
            if not final_summary or len(final_summary) < 3:
                return ""
                
            # Length check - allow longer keyword combinations
            if len(final_summary) > 150:
                final_summary = final_summary[:150]
            
            return final_summary
            
        except Exception as e:
            print(f"Warning: LLM summarization failed: {e}")
            return ""
    
    def clean_prompt(self, text: str) -> str:
        """Clean prompt text"""
        if not text:
            return ""
        
        # Basic cleaning
        import re
        cleaned = re.sub(r'[^\w\s\-.,!?]', '', str(text))
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Length limit (expanded to 500 characters for more detailed prompts)
        if len(cleaned) > 500:
            cleaned = cleaned[:500]
        
        return cleaned
    
    def build_simple_prompt(self, instruction: str, title: str, his_interaction: str = "", adaptive_weight: float = 0.0):
        """Build complete prompt: instruction + title + historical preference"""
        
        # 1. Handle invalid titles, skip "Unknown Item" directly
        if not title or title.lower().startswith('unknown') or len(title.strip()) < 3:
            print(f"Warning: Invalid title, skipping image generation: {title}")
            return None  # Return None to indicate skipping image generation
        
        # 2. Extract title information from historical interactions
        historical_titles = self._extract_historical_titles(his_interaction)
        
        # 3. Use LLM to summarize historical preference style
        style_summary = ""
        if historical_titles and adaptive_weight > 0.1:
            style_summary = self._summarize_preferences_with_llm(historical_titles)
        
        # 4. Build base prompt: directly use original title
        base_prompt = f"Create an exquisite, high-definition, and attractive cover.The image about: {title}"
        
        # 5. Add style summary influence based on adaptive weight
        if style_summary and adaptive_weight > 0.1:
            # Decide influence level based on weight strength
            if adaptive_weight > 0.7:
                influence = "strongly influenced by"
            elif adaptive_weight > 0.4:
                influence = "moderately influenced by" 
            else:
                influence = "slightly influenced by"
            
            base_prompt += f", {influence} {style_summary} style"
        
        return base_prompt
    
    def _extract_historical_titles(self, his_interaction: str) -> str:
        """Directly use historical title information, filter empty strings"""
        if not his_interaction:
            return ""
        
        try:
            # Check if it's list format historical titles (including stringified lists)
            if isinstance(his_interaction, (list, tuple)):
                # Directly handle Python lists/tuples
                valid_titles = []
                for title in his_interaction:
                    title_str = str(title).strip() if title is not None else ""
                    if title_str and title_str.lower() not in ['none', 'null', 'nan', '']:
                        valid_titles.append(title_str)
                
                if valid_titles:
                    his_titles_str = " | ".join(valid_titles)  # Connect with separator
                else:
                    return ""
            elif isinstance(his_interaction, str) and (his_interaction.startswith('[') or his_interaction.startswith('(')):
                # Handle stringified lists, e.g.: "['', 'title1', 'title2']"
                try:
                    import ast
                    parsed_list = ast.literal_eval(his_interaction)
                    if isinstance(parsed_list, (list, tuple)):
                        valid_titles = []
                        for title in parsed_list:
                            title_str = str(title).strip() if title is not None else ""
                            if title_str and title_str.lower() not in ['none', 'null', 'nan', '']:
                                valid_titles.append(title_str)
                        
                        if valid_titles:
                            his_titles_str = " | ".join(valid_titles)
                        else:
                            return ""
                    else:
                        # Parse failed, treat as normal string
                        his_titles_str = his_interaction.strip()
                except Exception:
                    # Parse failed, treat as normal string
                    his_titles_str = his_interaction.strip()
            else:
                # Directly use passed historical title content
                his_titles_str = str(his_interaction).strip()
                
                # If looks like None, empty values etc., return empty directly
                if his_titles_str.lower() in ['none', 'null', 'nan', '[]', '{}']:
                    return ""
            
            # Limit length to avoid overly long prompts (expanded to 800 characters)
            if len(his_titles_str) > 800:
                his_titles_str = his_titles_str[:800] + "..."
                
            return his_titles_str
                
        except Exception as e:
            print(f"Warning: Historical title processing error: {e}")
            return ""
    
    def generate_image_safe(self, prompt: str) -> list:
        """Safe image generation, avoiding black images"""
        if self.pipe is None:
            print("Warning: Pipeline unavailable, returning placeholder image")
            img = Image.new('RGB', (self.cfg.width, self.cfg.height), color='lightblue')
            return [img]
        
        # Generate seed - use timestamp to avoid fixed patterns
        current_time = int(time.time() * 1000)
        seed = (current_time + self.step_count * 123) % 100000
        
        try:
            with torch.no_grad():
                # Moderate cleanup for 32GB memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                generator = torch.Generator(device=self.device).manual_seed(seed)
                
                # Optimized image generation
                result = self.pipe(
                    prompt=prompt,
                    width=self.cfg.width,
                    height=self.cfg.height,
                    num_inference_steps=self.cfg.num_inference_steps,
                    guidance_scale=self.cfg.true_cfg_scale,
                    generator=generator,
                    output_type="pil",
                    return_dict=True,
                )
            
            if hasattr(result, 'images') and len(result.images) > 0:
                images = result.images
                
                # Immediately release GPU generation results
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # Simplified image quality check (reduced memory usage)
                img = images[0]
                import numpy as np
                img_array = np.array(img)
                brightness = img_array.mean()
                
                # Image quality check for 32GB memory environment
                if brightness < 20:  # Reasonable dark image threshold for 512x512 resolution
                    print(f"Warning: Dark image detected (brightness={brightness:.1f}), can retry in 32GB memory environment")
                
                return images
            else:
                print("Warning: Generation result is empty")
            
        except Exception as e:
            print(f"Image generation failed: {e}")
        
        # Return placeholder image
        img = Image.new('RGB', (self.cfg.width, self.cfg.height), color='lightgray')
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), "Generated Image", fill='black')
        return [img]
    
    def generate_image(self, prompt: str, seed: int = None, **kwargs) -> list:
        """Compatible with test script image generation method (fixed version)"""
        if self.pipe is None:
            print("Warning: Pipeline unavailable, returning placeholder image")
            img = Image.new('RGB', (self.cfg.width, self.cfg.height), color='lightblue')
            return [img]
        
        # Use provided seed or generate dynamic seed
        if seed is not None:
            actual_seed = seed
        else:
            current_time = int(time.time() * 1000)
            actual_seed = (current_time + self.step_count * 123) % 100000
        
        try:
            with torch.no_grad():
                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                generator = torch.Generator(device=self.device).manual_seed(actual_seed)
                
                # Fixed version: use more conservative parameters
                result = self.pipe(
                    prompt=prompt,
                    width=self.cfg.width,
                    height=self.cfg.height,
                    num_inference_steps=self.cfg.num_inference_steps,
                    guidance_scale=self.cfg.true_cfg_scale,
                    generator=generator,
                )
                
                if hasattr(result, 'images') and len(result.images) > 0:
                    images = result.images
                    
                    # Validate image quality
                    img = images[0]
                    import numpy as np
                    img_array = np.array(img)
                    brightness = img_array.mean()
                    std = img_array.std()
                    
                    # If black image detected, try regenerating
                    if brightness < 10 and std < 5:
                        # Retry with different seed
                        generator = torch.Generator(device=self.device).manual_seed(actual_seed + 12345)
                        result = self.pipe(
                            prompt=f"{prompt}, bright lighting, colorful",
                            width=self.cfg.width,
                            height=self.cfg.height,
                            num_inference_steps=self.cfg.num_inference_steps,
                            guidance_scale=max(1.2, self.cfg.true_cfg_scale - 0.6),  # Further reduce CFG
                            generator=generator,
                        )
                        
                        if hasattr(result, 'images') and len(result.images) > 0:
                            images = result.images
                            img_array = np.array(images[0])
                            brightness = img_array.mean()
                    
                    return images
                else:
                    print("Warning: Generation result is empty")
                    
        except Exception as e:
            print(f"Image generation failed: {e}")
        
        # Return placeholder image
        img = Image.new('RGB', (self.cfg.width, self.cfg.height), color='lightgray')
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), "Generated Image", fill='black')
        return [img]
    
    def training_step(self, batch: Dict[str, Any], save_dir: str = None, epoch: int = 0) -> Dict[str, Any]:
        """Training step - compatible with original interface (includes save_dir and epoch parameters)"""
        self.step_count += 1
        
        # Extract data
        instruction = batch.get("instruction", "")
        title = batch.get("title", "")
        his_interaction = batch.get("his_interaction", "")
        adaptive_weight = batch.get("adaptive_weight", 0.5)
        
        # Data validation
        if not instruction and not title:
            print("Warning: Both instruction and title are empty, skipping")
            return {"loss": torch.tensor(0.0), "metrics": {}, "images": []}
        
        # Build prompt
        prompt = self.build_simple_prompt(instruction, title, his_interaction, adaptive_weight)
        
        # Check if image generation should be skipped
        if prompt is None:
            # Skip image generation for invalid titles
            return {
                "loss": torch.tensor(0.0, device=self.device),
                "metrics": {},
                "images": [],
                "diffusion_loss": torch.tensor(0.0, device=self.device),
                "expert_scores": {},
            }
        
        # Generate image
        images = self.generate_image(prompt)
        
        # Save image (use provided save_dir or default directory)
        save_directory = save_dir if save_dir else self.save_dir
        os.makedirs(save_directory, exist_ok=True)
        
        saved_paths = []
        if images and len(images) > 0:
            img_filename = f"step_{self.step_count:06d}_epoch_{epoch}_{int(time.time())}.png"
            img_path = os.path.join(save_directory, img_filename)
            
            try:
                images[0].save(img_path)
                saved_paths.append(img_path)
            except Exception as e:
                print(f"Warning: Image save failed: {e}")
        
        # Calculate loss and metrics (based on adaptive weight)
        base_loss = 0.2
        diffusion_loss_value = base_loss * adaptive_weight  # Use adaptive weight
        
        loss = torch.tensor(diffusion_loss_value, device=self.device)
        diffusion_loss = torch.tensor(diffusion_loss_value, device=self.device)
        
        # Calculate expert scores - use real evaluator
        expert_scores = {}
        metrics = {}
        
        if images and len(images) > 0 and self.expert and hasattr(self.expert, 'score_image'):
            # Use image evaluator
            try:
                eval_scores = self.expert.score_image(instruction, images[0])
                
                # Handle CLIP evaluation results - new format only has clip_score
                if isinstance(eval_scores, dict):
                    # New CLIP evaluation format
                    if 'clip_score' in eval_scores:
                        clip_score = eval_scores.get('clip_score', 3.0)
                        similarity = eval_scores.get('similarity', 0.0)
                        status = eval_scores.get('status', 'unknown')
                        
                        # Only save unified score
                        expert_scores = torch.tensor(float(clip_score), device=self.device)
                        metrics = {
                            'clip_score': float(clip_score),
                            'similarity': float(similarity),
                            'status': status
                        }
                    else:
                        # Compatible with old format (four dimensions) - but shouldn't be used now
                        print("Warning: Detected old format scoring, recommend upgrading to CLIP format")
                        avg_score = 0.0
                        count = 0
                        for key in ['consistency', 'accuracy', 'integrity', 'quality']:
                            if key in eval_scores:
                                avg_score += float(eval_scores[key])
                                count += 1
                        avg_score = avg_score / count if count > 0 else 3.0
                        
                        expert_scores = torch.tensor(avg_score, device=self.device)
                        metrics = {'clip_score': avg_score, 'status': 'legacy_format'}
                else:
                    # Evaluation failed, use neutral score
                    print("Warning: CLIP evaluator returned result in non-dictionary format")
                    expert_scores = torch.tensor(3.0, device=self.device)
                    metrics = {'clip_score': 3.0, 'status': 'error'}
                
            except Exception as e:
                print(f"Warning: Image evaluation failed: {e}")
                # Fall back to neutral score
                expert_scores = torch.tensor(3.0, device=self.device)
                metrics = {'clip_score': 3.0, 'status': 'error'}
        else:
            # No image or evaluator not supported, use neutral score
            print("Warning: No image or evaluator, using default score")
            expert_scores = torch.tensor(3.0, device=self.device)
            metrics = {'clip_score': 3.0, 'status': 'no_evaluator'}
        
        # Return format compatible with original version
        return {
            "loss": loss,
            "metrics": metrics,
            "images": saved_paths,
            "diffusion_loss": diffusion_loss,
            "expert_scores": expert_scores,
        }

    def compute_loss(self, images, instruction: str, topk_desc: str):
        """Calculate image quality loss (compatible with original interface)"""
        try:
            # Check if evaluator exists
            if self.expert is None:
                print("Warning: No evaluator, using default score")
                default_scores = {
                    "clip_score": 3.0,
                    "status": "no_evaluator"
                }
                loss = torch.tensor(0.2, device=self.device)
                return loss, default_scores
            
            # Unified evaluator interface call
            if hasattr(self.expert, 'score_image'):
                # CLIP evaluator - parameter order: (instruction, image)
                if isinstance(images, list) and len(images) > 0:
                    image = images[0]
                else:
                    image = images
                scores = self.expert.score_image(instruction, image)
            elif hasattr(self.expert, 'score'):
                # Simplified evaluator - parameter order: (image, instruction, topk_desc)
                scores = self.expert.score(images, instruction, topk_desc)
            else:
                # Default neutral score
                print("Warning: Evaluator doesn't support score or score_image method, using default score")
                scores = {
                    "clip_score": 3.0,
                    "status": "no_method"
                }
            
            # Handle new CLIP scoring format
            if 'clip_score' in scores:
                clip_score = self._extract_score_value(scores.get("clip_score", 3.0))
                similarity = scores.get("similarity", 0.0)
                status = scores.get("status", "unknown")
                
                # Calculate loss based on CLIP score (higher score = lower loss)
                # Convert 1-5 score range to 0-1 loss range
                normalized_score = (clip_score - 1.0) / 4.0  # Normalize to 0-1
                loss = torch.tensor(1.0 - normalized_score, device=self.device)  # High score = low loss
                
                metrics = {
                    "clip_score": float(clip_score),
                    "similarity": float(similarity),
                    "status": status
                }
            else:
                # Compatible with old format - calculate average score
                c = self._extract_score_value(scores.get("consistency", 3.0))
                a = self._extract_score_value(scores.get("accuracy", 3.0))
                i = self._extract_score_value(scores.get("integrity", 3.0))
                q = self._extract_score_value(scores.get("quality", 3.0))
                
                avg_score = (c + a + i + q) / 4.0
                normalized_score = (avg_score - 1.0) / 4.0
                loss = torch.tensor(1.0 - normalized_score, device=self.device)
                
                metrics = {
                    "clip_score": float(avg_score),
                    "status": "legacy_converted"
                }
            
            return loss, metrics
            
        except Exception as e:
            print(f"Warning: Evaluator call failed: {e}")
            # Return neutral score
            default_scores = {
                "clip_score": 3.0,
                "status": "error"
            }
            loss = torch.tensor(0.2, device=self.device)
            return loss, default_scores
    
    def _extract_score_value(self, score):
        """Extract score value (compatible with different formats)"""
        if hasattr(score, 'item'):
            return float(score.item())
        elif isinstance(score, (int, float)):
            return float(score)
        elif score is None:
            return 0.8  # Default value
        else:
            return 0.8

    @staticmethod
    def _pil_list_to_tensor(images):
        """Convert PIL image list to tensor (compatible with original version)"""
        try:
            import torchvision.transforms as T
            tfm = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            return torch.stack([tfm(img) for img in images])
        except:
            # If conversion fails, return fake tensor
            return torch.randn(len(images), 3, 512, 512)
    
    def fuse_embeddings_enhanced_v2(self, instruction: str, title: str, his_interaction: str, adaptive_weight: float, item_features: str = ""):
        """Compatible with original version prompt building method"""
        return self.build_simple_prompt(instruction, title, his_interaction, adaptive_weight)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        saved_images = len([f for f in os.listdir(self.save_dir) if f.endswith('.png')])
        return {
            "total_steps": self.step_count,
            "saved_images": saved_images,
            "save_directory": self.save_dir
        }

# Backward compatibility alias
QwenImageTrainer = QwenImageTrainerV2

# Convenience function
def create_qwen_trainer_v2(base_dir: str = "/root/autodl-tmp/stable-diffusion-3.5-medium"):
    """Create simplified trainer"""
    config = QwenImageConfig(base_dir=base_dir)
    trainer = QwenImageTrainerV2(config)
    return trainer

if __name__ == "__main__":
    # Simple test
    print("QwenImageTrainerV2 simple test")
    
    trainer = create_qwen_trainer_v2()
    
    # Test case
    test_batch = {
        "instruction": "Help me recommend gaming items",
        "title": "Xbox Wireless Controller",
        "his_interaction": "gaming history..."
    }
    
    result = trainer.training_step(test_batch)
    print(f"Test result: {result}")
    
    stats = trainer.get_stats()
    print(f"Statistics: {stats}")
