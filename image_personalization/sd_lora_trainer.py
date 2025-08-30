import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn

from .image_manager import TrainingImageManager


@dataclass
class SDPersonalizationConfig:
    model_root: str = "/root/autodl-tmp/Stable_Diffusion"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_target_modules: str = "to_q,to_k,to_v"  # Only inject cross-attention Q/K/V
    learning_rate: float = 1e-4
    weight_instr_consistency: float = 0.25
    weight_semantic_accuracy: float = 0.25
    weight_image_integrity: float = 0.25
    weight_quality: float = 0.25
    gradient_checkpointing: bool = True


class ExpertAPI(nn.Module):
    """
    Expert model interface placeholder:
    - compute_instruction_consistency(image, instruction)
    - compute_semantic_accuracy(image, topk_item_desc)
    - compute_image_integrity(image)
    - compute_quality(image)
    Returns scores between 0-1.
    """
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):  # Placeholder
        raise NotImplementedError

    def score(self, image, instruction: str, topk_desc: str) -> Dict[str, torch.Tensor]:
        # Use constants as placeholder here, replace when integrating external multimodal models
        device = image.device
        return {
            "instr_consistency": torch.tensor(0.8, device=device),
            "semantic_accuracy": torch.tensor(0.8, device=device),
            "image_integrity": torch.tensor(0.8, device=device),
            "quality": torch.tensor(0.8, device=device),
        }


class SDPersonalizationTrainer(nn.Module):
    def __init__(self, cfg: SDPersonalizationConfig, expert: Optional[ExpertAPI] = None, smolvlm_dir: Optional[str] = None, qwen_vl_dir: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize image manager
        self.image_manager = TrainingImageManager(
            base_dir="training_images",
            max_images=50,  # Maximum 50 images
            max_recent_images=10,  # Recent 10 images
            cleanup_interval=50,  # Clean up after every 50 saves
        )
        
        # Training step counter
        self.step_count = 0
        self.epoch_count = 0

        # Lazy load Stable Diffusion components (via diffusers or local minimal SD weights)
        from diffusers import StableDiffusionPipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            cfg.model_root, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=False,  # Avoid accelerate detection issues
            device_map=None           # Avoid accelerate detection issues
        ).to(self.device)

        # Freeze UNet/VAE/TextEncoder, only train LoRA weights
        for p in self.pipe.unet.parameters():
            p.requires_grad = False
        for p in self.pipe.vae.parameters():
            p.requires_grad = False
        for p in self.pipe.text_encoder.parameters():
            p.requires_grad = False

        # Enable UNet layered LoRA, inject into cross-attention Q/K/V matrices
        self._inject_lora_into_unet(
            rank=cfg.lora_rank, alpha=cfg.lora_alpha, target_modules=cfg.lora_target_modules
        )
        
        # Ensure entire pipeline is on correct device
        self.pipe.to(self.device)

        # Optimizer only contains LoRA weights
        trainable = [p for p in self.pipe.unet.parameters() if p.requires_grad]
        print(f"📊 Found trainable LoRA parameters: {len(trainable)}")
        if len(trainable) == 0:
            print("⚠️ Warning: No trainable parameters found, checking LoRA injection...")
            # Backup plan: force enable all LoRA parameters
            for n, p in self.pipe.unet.named_parameters():
                if any(target in n for target in ["to_q", "to_k", "to_v"]):
                    p.requires_grad = True
                    trainable.append(p)
                    print(f"🔧 Force enabling parameter: {n}")
        
        if len(trainable) > 0:
            self.optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate)
        else:
            print("❌ Still no trainable parameters, using placeholder optimizer")
            # Create a placeholder parameter to avoid errors
            dummy_param = torch.nn.Parameter(torch.zeros(1, device=self.device))
            self.optimizer = torch.optim.AdamW([dummy_param], lr=cfg.learning_rate)

        # Expert scorer - prioritize using Qwen2.5-VL-3B-Instruct
        if expert is not None:
            self.expert = expert
        elif qwen_vl_dir is not None:
            print("🎯 Using Qwen2.5-VL-3B-Instruct evaluator")
            from .qwen_vl_evaluator import QwenVLEvaluator
            self.expert = QwenVLEvaluator(qwen_vl_dir, self.device)
        elif smolvlm_dir is not None:
            print("⚠️ Using legacy SmolVLM evaluator (recommend upgrading to Qwen2.5-VL)")
            from .smolvlm_evaluator import SmolVLMEvaluator
            self.expert = SmolVLMEvaluator(smolvlm_dir, self.device)
        else:
            self.expert = ExpertAPI()

    def _inject_lora_into_unet(self, rank: int, alpha: int, target_modules: str):
        # Completely skip PEFT, directly use stable manual LoRA implementation
        print("🔧 Using stable manual LoRA implementation (avoiding PEFT compatibility issues)...")
        self._inject_manual_lora(rank, alpha, target_modules)
    
    def _inject_manual_lora(self, rank: int, alpha: int, target_modules: str):
        """Manual LoRA injection, avoiding PEFT version compatibility issues"""
        import torch
        import torch.nn as nn
        
        class SimpleLoRALinear(nn.Module):
            def __init__(self, original_layer, rank, alpha):
                super().__init__()
                self.original_layer = original_layer
                self.rank = rank
                self.alpha = alpha
                
                # Ensure LoRA weights are on the same device as original layer
                device = next(original_layer.parameters()).device
                dtype = next(original_layer.parameters()).dtype
                
                # LoRA weights - use more stable initialization and ensure on correct device
                self.lora_A = nn.Parameter(
                    torch.randn(rank, original_layer.in_features, device=device, dtype=dtype) / (rank ** 0.5)
                )
                self.lora_B = nn.Parameter(
                    torch.zeros(original_layer.out_features, rank, device=device, dtype=dtype)
                )
                
                # Freeze original weights
                for param in original_layer.parameters():
                    param.requires_grad = False
                    
            def forward(self, x, *args, **kwargs):
                # Ensure compatibility with various diffusers calling methods
                original_output = self.original_layer(x, *args, **kwargs)
                
                # Ensure all tensors are on the same device
                device = x.device
                lora_A = self.lora_A.to(device)
                lora_B = self.lora_B.to(device)
                
                # LoRA calculation
                if x.dim() == 3:  # [batch, seq, hidden]
                    lora_x = x.view(-1, x.size(-1))  # [batch*seq, hidden]
                    lora_out = lora_x @ lora_A.T @ lora_B.T  # [batch*seq, out]
                    lora_out = lora_out.view(x.size(0), x.size(1), -1)  # [batch, seq, out]
                else:
                    lora_out = x @ lora_A.T @ lora_B.T
                
                scaling = self.alpha / self.rank
                return original_output + lora_out * scaling
        
        target_list = [mod.strip() for mod in target_modules.split(",")]
        lora_count = 0
        replaced_modules = []
        
        # More precise layer replacement logic
        def replace_layers(module, path=""):
            nonlocal lora_count
            
            children_to_replace = []
            for name, child in module.named_children():
                full_path = f"{path}.{name}" if path else name
                
                # Check if this is a target layer
                is_target = False
                if isinstance(child, nn.Linear):
                    for target in target_list:
                        if target in name or target in full_path:
                            is_target = True
                            break
                
                if is_target:
                    children_to_replace.append((name, child, full_path))
                else:
                    replace_layers(child, full_path)
            
            # Replace layers
            for name, child, full_path in children_to_replace:
                new_layer = SimpleLoRALinear(child, rank, alpha)
                setattr(module, name, new_layer)
                lora_count += 1
                replaced_modules.append(full_path)
                print(f"🔧 LoRA injection: {full_path} ({child.in_features} -> {child.out_features})")
        
        print(f"🎯 Target modules: {target_list}")
        replace_layers(self.pipe.unet)
        
        if lora_count > 0:
            print(f"✅ Manual LoRA injection completed, {lora_count} layers in total")
            print(f"📋 Replaced layers: {replaced_modules[:5]}{'...' if len(replaced_modules) > 5 else ''}")
            
            # Check trainable parameters
            trainable_params = 0
            for name, param in self.pipe.unet.named_parameters():
                if param.requires_grad:
                    trainable_params += param.numel()
            print(f"📊 Trainable LoRA parameters: {trainable_params:,}")
        else:
            print("⚠️ No matching layers found, may need to adjust target_modules")
            print("💡 Linear layer examples in UNet:")
            # Show some actual layer names as reference
            count = 0
            for name, module in self.pipe.unet.named_modules():
                if isinstance(module, nn.Linear) and count < 5:
                    print(f"   {name}")
                    count += 1

    def compute_loss(self, images, instruction: str, topk_desc: str):
        # Use multimodal model scores (0-5), map to 0-1 and convert to loss
        if hasattr(self.expert, "score_image"):
            # Take first image for evaluation
            from torchvision.transforms.functional import to_pil_image
            img0 = to_pil_image(images[0].cpu())
            res = self.expert.score_image(instruction, img0)
            c = torch.tensor(res.get("consistency", 0.0), device=self.device) / 5.0
            a = torch.tensor(res.get("accuracy", 0.0), device=self.device) / 5.0
            i = torch.tensor(res.get("integrity", 0.0), device=self.device) / 5.0
            q = torch.tensor(res.get("quality", 0.0), device=self.device) / 5.0
        else:
            scores = self.expert.score(images, instruction, topk_desc)
            c = scores["instr_consistency"]
            a = scores["semantic_accuracy"]
            i = scores["image_integrity"]
            q = scores["quality"]
        loss = (
            self.cfg.weight_instr_consistency * (1.0 - c) +
            self.cfg.weight_semantic_accuracy * (1.0 - a) +
            self.cfg.weight_image_integrity * (1.0 - i) +
            self.cfg.weight_quality * (1.0 - q)
        )
        metrics = {"consistency": float(c.detach().item()), "accuracy": float(a.detach().item()), "integrity": float(i.detach().item()), "quality": float(q.detach().item())}
        return loss, metrics

    @torch.no_grad()
    def _build_prompt(self, instruction: str, fused_preference: str) -> str:
        if fused_preference:
            return f"{instruction}\nPreferences: {fused_preference}"
        return instruction

    def fuse_embeddings(self, instruction: str, preference_text: str, adaptive_weight: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get instruction and preference embeddings through text encoder, and perform vector-level fusion: instr_emb + w * pref_emb.
        Returns (prompt_embeds, pooled_embeds) for direct feeding to pipeline.
        """
        text_encoder = self.pipe.text_encoder
        tokenizer = self.pipe.tokenizer
        device = self.device
        # Encode two text segments
        instr_ids = tokenizer(
            instruction, return_tensors="pt", padding=True, truncation=True
        ).input_ids.to(device)
        pref_ids = tokenizer(
            preference_text or "", return_tensors="pt", padding=True, truncation=True
        ).input_ids.to(device)
        with torch.no_grad():
            instr_out = text_encoder(instr_ids)[0]  # [B, L1, D]
            pref_out = text_encoder(pref_ids)[0]    # [B, L2, D]
        # Align to same length: use minimum length alignment and apply numerically stable scaling
        min_len = min(instr_out.shape[1], pref_out.shape[1]) if pref_out.shape[1] > 0 else instr_out.shape[1]
        if min_len == 0:
            fused = instr_out
        else:
            # Clip adaptive_weight to 0~1
            aw = max(0.0, min(1.0, float(adaptive_weight)))
            fused = instr_out[:, :min_len, :] + aw * pref_out[:, :min_len, :]
        print(f"[DEBUG] fuse_embeddings: len={min_len}, aw={adaptive_weight}")
        # pooled uses CLS/mean pooling approximation
        pooled = fused.mean(dim=1)
        return fused, pooled

    def fuse_embeddings_enhanced(self, instruction: str, title: str, historical_preference: str, adaptive_weight: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enhanced embedding-level fusion: instruction + title + adaptive_weight * historical_preference
        Args:
            instruction: Instruction text
            title: Current item title 
            historical_preference: Historical preference text
            adaptive_weight: Adaptive weight
        Returns:
            fused_embeds: [B, seq_len, hidden_dim]
            pooled: [B, hidden_dim] for backward compatibility
        """
        tokenizer = self.pipe.tokenizer
        text_encoder = self.pipe.text_encoder
        device = self.device
        
        # 使用改进的文本级融合（更稳定且语义清晰）
        from image_personalization.conditioning import build_enhanced_prompt
        
        aw = max(0.0, min(1.0, float(adaptive_weight)))
        combined_text = build_enhanced_prompt(
            instruction or "",
            title or "",
            historical_preference or "",
            aw
        )
        
        print(f"[DEBUG] Text-level fusion (aw={aw}): {combined_text[:200]}...")
        
        # 编码融合后的文本
        combined_ids = tokenizer(
            combined_text,
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=77  # CLIP标准长度
        ).input_ids.to(device)
        
        with torch.no_grad():
            fused = text_encoder(combined_ids)[0]  # [B, L, D]
        
        print(f"[DEBUG] Fused embedding shape: {fused.shape}")
        
        # pooled 使用 mean pooling
        pooled = fused.mean(dim=1)
        return fused, pooled
    
    def fuse_embeddings_enhanced_v2(self, instruction: str, title: str, historical_preference: str, adaptive_weight: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """备选：真正的embedding级相加融合（如果v1效果不好可以试试）"""
        tokenizer = self.pipe.tokenizer
        text_encoder = self.pipe.text_encoder
        device = self.device
        
        # 统一长度编码
        max_len = 77
        instr_ids = tokenizer(instruction or "", return_tensors="pt", padding='max_length', truncation=True, max_length=max_len).input_ids.to(device)
        title_ids = tokenizer(title or "", return_tensors="pt", padding='max_length', truncation=True, max_length=max_len).input_ids.to(device)  
        hist_ids = tokenizer(historical_preference or "", return_tensors="pt", padding='max_length', truncation=True, max_length=max_len).input_ids.to(device)
        
        with torch.no_grad():
            instr_out = text_encoder(instr_ids)[0]  # [B, 77, D]
            title_out = text_encoder(title_ids)[0]  # [B, 77, D]
            hist_out = text_encoder(hist_ids)[0]    # [B, 77, D]
        
        # 保存原始幅度信息
        instr_magnitude = torch.norm(instr_out, dim=-1, keepdim=True)
        title_magnitude = torch.norm(title_out, dim=-1, keepdim=True) 
        hist_magnitude = torch.norm(hist_out, dim=-1, keepdim=True)
        
        # 归一化到单位向量（保持方向）
        instr_norm = torch.nn.functional.normalize(instr_out, dim=-1)
        title_norm = torch.nn.functional.normalize(title_out, dim=-1)
        hist_norm = torch.nn.functional.normalize(hist_out, dim=-1)
        
        aw = max(0.0, min(1.0, float(adaptive_weight)))
        
        # 方向融合：加权平均方向向量
        base_direction = (instr_norm + title_norm) / 2
        fused_direction = (1 - aw * 0.3) * base_direction + (aw * 0.3) * hist_norm
        
        # 幅度融合：保持合理的embedding幅度
        base_magnitude = (instr_magnitude + title_magnitude) / 2
        fused_magnitude = base_magnitude  # 使用基础幅度，不受历史影响
        
        # Reconstruct embedding: direction × magnitude
        fused = torch.nn.functional.normalize(fused_direction, dim=-1) * fused_magnitude
        
        # 调试信息：检查融合结果的数值范围
        fused_mean = fused.mean().item()
        fused_std = fused.std().item()
        fused_min = fused.min().item()
        fused_max = fused.max().item()
        
        print(f"[DEBUG] Embedding-level fusion: shapes={fused.shape}, aw={aw}")
        print(f"[DEBUG] Fused stats: mean={fused_mean:.4f}, std={fused_std:.4f}, min={fused_min:.4f}, max={fused_max:.4f}")
        
        # 如果数值范围异常，进行修正
        if abs(fused_mean) > 1.0 or fused_std > 2.0 or abs(fused_min) > 5.0 or abs(fused_max) > 5.0:
            print(f"[WARNING] Embedding range abnormal, applying correction...")
            # 重新缩放到合理范围
            fused = torch.clamp(fused, -3.0, 3.0)
            # 重新归一化到稳定范围
            fused = fused * 0.8  # 稍微缩小避免极值
            print(f"[DEBUG] After correction: mean={fused.mean().item():.4f}, std={fused.std().item():.4f}")
        
        pooled = fused.mean(dim=1)
        return fused, pooled

    def training_step(self, batch: Dict[str, Any], save_dir: Optional[str] = None, save_limit: int = 5, epoch: int = None):
        """
        batch fields:
        - instruction: Text instruction
        - title: Current item title (new)
        - fused_preference_text: Textualized preferences (or do embedding-level fusion upstream)
        - topk_desc: Description of Ground Truth TopK items
        New logic: instruction + title + adaptive_weight * historical_preference
        """
        self.step_count += 1
        if epoch is not None:
            self.epoch_count = epoch
            
        instruction = batch.get("instruction", "")
        title = batch.get("title", "")  # New: current item title
        fused_pref_text = batch.get("fused_preference_text", "")
        topk_desc = batch.get("topk_desc", "")

        # Enhanced vector-level fusion: instruction + title + adaptive_weight * historical_preference
        prompt_embeds = None
        
        # Try to use enhanced embedding fusion
        if instruction or title:
            try:
                # Read adaptive weight and historical preferences from batch
                aw = float(batch.get("adaptive_weight", 1.0))
                
                # If there is historical preference data, extract for embedding fusion
                historical_preference = ""
                if "his_interaction" in batch:
                    from .conditioning import extract_titles_from_his_interaction, build_preference_text
                    his_interaction = batch.get("his_interaction", "")
                    hist_titles = extract_titles_from_his_interaction(his_interaction)
                    historical_preference = build_preference_text(hist_titles)
                
                # Use enhanced embedding fusion: instruction + title + adaptive_weight * historical_preference
                fused_tok_embeds, _ = self.fuse_embeddings_enhanced(
                    instruction, title, historical_preference, aw
                )
                prompt_embeds = fused_tok_embeds
                
            except Exception as e:
                print(f"[DEBUG] Enhanced embedding fusion failed, fallback to text: {e}")
                prompt_embeds = None

        # For compatibility with old diffusers, uniformly use text prompt to avoid screen artifacts/noise caused by prompt_embeds
        from .conditioning import extract_titles_from_his_interaction, build_preference_text, build_enhanced_prompt
        aw_s = float(batch.get("adaptive_weight", 1.0))
        hist_titles_s = extract_titles_from_his_interaction(batch.get("his_interaction", ""))
        hist_pref_s = build_preference_text(hist_titles_s)
        prompt_text = build_enhanced_prompt(instruction, title, hist_pref_s, aw_s)
        images = self.pipe(
            prompt=prompt_text,
            negative_prompt="low quality, blurry, noisy, artifacts, distorted",
            num_inference_steps=30,
            guidance_scale=7.5,
        ).images

        # Smart image management: automatic saving, cleaning, and organizing images
        saved_paths = []
        if len(images) > 0:
            # Only save the first image (usually generate one during training)
            main_image = images[0]
            
            try:
                # Calculate loss and metrics first for use when saving
                image_tensor = self._pil_list_to_tensor(images).to(self.device)
                loss, metrics = self.compute_loss(image_tensor, instruction, topk_desc)

                
                # Use smart image manager to save
                saved_paths_dict = self.image_manager.save_training_image(
                    image=main_image,
                    step=self.step_count,
                    epoch=self.epoch_count,
                    loss=loss.item() if hasattr(loss, 'item') else float(loss),
                    metrics=metrics,
                    sample_info={
                        "instruction": instruction,
                        "title": title,
                        "adaptive_weight": batch.get("adaptive_weight", 1.0),
                        "fused_text": fused_pref_text[:100] if fused_pref_text else ""
                    },
                    force_save=(self.step_count % 100 == 0)  # Force save every 100 steps
                )
                saved_paths = list(saved_paths_dict.values())
                
                # Periodically display statistics
                if self.step_count % 50 == 0:
                    stats = self.image_manager.get_stats()
                    print(f"📊 Image management statistics: Total {stats['total_saved']} images | "
                          f"Recent {stats['recent_count']} images | "
                          f"Best {stats['best_count']} images | "
                          f"Used {stats['total_size_mb']:.1f}MB")
                
            except Exception as e:
                print(f"⚠️ Smart image saving failed: {e}")
                # Fall back to simple saving
                if save_dir is not None:
                    try:
                        from pathlib import Path
                        from datetime import datetime
                        import uuid
                        out_dir = Path(save_dir)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        name = f"fallback_{ts}_{uuid.uuid4().hex[:8]}.png"
                        simple_path = out_dir / name
                        main_image.save(simple_path)
                        saved_paths = [str(simple_path)]
                    except Exception:
                        pass
        else:
            # If no image generation, still need to calculate loss
            try:
                image_tensor = self._pil_list_to_tensor(images).to(self.device)
                loss, metrics = self.compute_loss(image_tensor, instruction, topk_desc)
            except Exception as e:
                print(f"⚠️ Loss calculation failed: {e}")
                loss = torch.tensor(0.0, device=self.device)
                metrics = {"consistency": 0.0, "accuracy": 0.0, "integrity": 0.0, "quality": 0.0}

        return {"loss": loss, "metrics": metrics, "images": saved_paths}

    @staticmethod
    def _pil_list_to_tensor(images):
        import torchvision.transforms as T
        tfm = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])
        tensors = [tfm(img) for img in images]
        return torch.stack(tensors, dim=0)


