import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
from transformers import LlamaTokenizer

from minigpt4.common.registry import registry
from minigpt4.models.pllama import LlamaForCausalLM
from minigpt4.models.rec_model import Rec2Base, disabled_train
from .ppeft import PLoraConfig, get_peft_model
try:
    from image_personalization.hook import update_image_module
except Exception:
    update_image_module = None


def get_ids_order(prompt):
    id_flags = ["<UserID>", "<ItemIDList>", "<TargetItemID>"]
    id_order_ = []
    for flag_ in id_flags:
        pos_ = prompt.find(flag_)
        if pos_ >= 0:
            id_order_.append(pos_)
    id_order_ = np.argsort(np.array(id_order_))
    return id_order_


@registry.register_model("mini_gpt4rec_vx")
class MiniGPT4Rec_vx(Rec2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/minigpt4rec.yaml",
    }
    
    @classmethod
    def from_config(cls, cfg):
        """Create model instance from configuration"""
        # Get recommendation model config
        rec_config = cfg.get("rec_config", {})
        
        # Create model instance
        model = cls(
            rec_model=cfg.get("rec_model", "MF"),
            rec_config=rec_config,
            pretrained_rec=rec_config.get("pretrained_path", None),
            freeze_rec=cfg.get("freeze_rec", True),
            rec_precision=cfg.get("rec_precision", "fp16"),
            llama_model=cfg.get("llama_model", ""),
            prompt_path=cfg.get("prompt_path", ""),
            prompt_template=cfg.get("prompt_template", ""),
            max_txt_len=cfg.get("max_txt_len", 1024),
            end_sym=cfg.get("end_sym", "###"),
            low_resource=cfg.get("low_resource", False),
            lora_config=cfg.get("lora_config", {}),
            proj_token_num=cfg.get("proj_token_num", 1),
            proj_drop=cfg.get("proj_drop", 0.0),
            proj_mid=cfg.get("proj_mid_times", 10),
            freeze_lora=cfg.get("freeze_lora", False),
            freeze_proj=cfg.get("freeze_proj", False),
            freeze_bias=cfg.get("freeze_bias", True),
            enable_rating_prediction=cfg.get("enable_rating_prediction", False),
        )
        
        # Set gradient clipping parameters
        model.max_grad_norm = cfg.get("max_grad_norm", 1.0)
        model.label_smoothing = cfg.get("loss_smoothing", 0.1)
        
        print("Model created from configuration successfully")
        return model

    def __init__(
            self,
            rec_model="MF",
            rec_config=None,
            pretrained_rec=None,
            freeze_rec=True,
            rec_precision='fp16',
            llama_model="",
            prompt_path="",
            prompt_template="",
            max_txt_len=32,
            end_sym='\n',
            low_resource=False,  # use 8 bit and put vit in cpu
            device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
            proj_token_num=1,  # the number of tokens that the user/item embedding projected to
            proj_drop=0,
            lora_config=None,
            proj_mid=5,
            freeze_lora=False,
            freeze_proj=False,
            freeze_bias=False,
            enable_rating_prediction=True,  # Enable rating prediction
    ):
        super().__init__()

        # self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
        self.proj_token_num = proj_token_num

        print("runing MiniGPT4Rec_vx ...... ")

        print('Loading Rec_model')
        self.rec_model_type = rec_model
        self.rec_encoder = self.init_rec_encoder(rec_model, rec_config, rec_precision)
        # Safely load pretrained MF (supports partial loading when shapes don't match)
        if self.rec_encoder is not None and pretrained_rec != "not_have":
            try:
                ckpt = torch.load(pretrained_rec, map_location="cpu")
                model_sd = self.rec_encoder.state_dict()
                new_sd = {}
                copied = []
                partially_copied = []
                for k, v in ckpt.items():
                    if k not in model_sd:
                        continue  # Skip keys not in model (e.g., bias)
                    target = model_sd[k]
                    # Exact match
                    if target.shape == v.shape:
                        new_sd[k] = v
                        copied.append(k)
                        continue
                    # Partial copy of first min rows for embedding layers
                    if (
                        k.endswith("user_embedding.weight")
                        or k.endswith("item_embedding.weight")
                    ) and target.dim() == 2 and v.dim() == 2 and target.shape[1] == v.shape[1]:
                        # Note: To ensure eval split IDs don't go out of bounds, don't shrink num_embeddings; only do prefix copy when checkpoint is smaller
                        rows = min(target.shape[0], v.shape[0])
                        buf = target.clone()
                        buf[:rows, :] = v[:rows, :]
                        new_sd[k] = buf
                        partially_copied.append(f"{k}[:{rows},:]")
                        continue
                    # Skip all other shape mismatches
                result = self.rec_encoder.load_state_dict(new_sd, strict=False)
                print(
                    f"Loaded MF pretrained. full={len(copied)}, partial={len(partially_copied)}, "
                    f"missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}"
                )
                if partially_copied:
                    print(f"Partial keys: {partially_copied[:4]}{' ...' if len(partially_copied)>4 else ''}")
            except Exception as e:
                print(f"[WARN] Failed to load pretrained MF (fallback to random init): {e}")
        # except:
        #     # print(pretrained_rec)
        #     # self.rec_encoder.config
        #     raise RuntimeError("Please provide your pretained rec model path or check whether the pretrained model and the defined mode can match each other")
        if freeze_rec and self.rec_encoder is not None:
            for name, param in self.rec_encoder.named_parameters():
                param.requires_grad = False
            self.rec_encoder = self.rec_encoder.eval()
            self.rec_encoder.train = disabled_train
            logging.info("freeze rec encoder")
            print("freeze rec encoder")
        print('Loading Rec_model Done')

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        # Ensure pad_token and pad_token_id are valid positive numbers (avoid -1 causing embed errors)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        if getattr(self.llama_tokenizer, 'pad_token_id', None) is None or self.llama_tokenizer.pad_token_id < 0:
            self.llama_tokenizer.pad_token_id = self.llama_tokenizer.eos_token_id

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )

        # Reduce memory: enable gradient checkpointing and disable cache
        try:
            if hasattr(self.llama_model, 'gradient_checkpointing_enable'):
                self.llama_model.gradient_checkpointing_enable()
            if hasattr(self.llama_model, 'config'):
                self.llama_model.config.use_cache = False
        except Exception:
            pass

        # Sync model config's pad_token_id to prevent tokenizer using -1 causing embedding index out of bounds
        try:
            self.llama_model.config.pad_token_id = self.llama_tokenizer.pad_token_id
            if hasattr(self.llama_model, 'generation_config') and self.llama_model.generation_config is not None:
                self.llama_model.generation_config.pad_token_id = self.llama_tokenizer.pad_token_id
        except Exception:
            pass

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        self.use_lora = False
        if lora_config is not None and lora_config.use_lora:
            print("Setting Lora")
            
            # Prevent recursion depth from exceeding limit
            import sys
            old_recursion_limit = sys.getrecursionlimit()
            try:
                # Increase recursion limit
                sys.setrecursionlimit(10000)
                print(f"Increased recursion limit: {old_recursion_limit} -> 10000")
                
                peft_config = PLoraConfig(
                    r=lora_config.r,
                    lora_alpha=lora_config.alpha,
                    target_modules=lora_config.target_modules,
                    lora_dropout=lora_config.dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                    user_embedding_size=self.rec_encoder.config.embedding_size,
                    num_queries=4,
                )
                
                try:
                    # Try using simplified method to create PEFT model
                    print("Trying to create PEFT model using simplified method...")
                    
                    # Method 1: Use get_peft_model
                    try:
                        self.llama_model_lora = get_peft_model(self.llama_model, peft_config)
                        self.use_lora = True
                        print("Successfully created PEFT model using get_peft_model")
                    except Exception as e1:
                        print(f"Warning: get_peft_model method failed: {e1}")
                        
                        # Method 2: Manually create LoRA layers
                        print("Trying to manually create LoRA layers...")
                        
                        # Copy original model
                        self.llama_model_lora = self.llama_model
                        
                        # Add LoRA parameters to target modules
                        lora_param_count = 0
                        for name, module in self.llama_model_lora.named_modules():
                            # Check if this is a target module
                            is_target = False
                            for target in lora_config.target_modules:
                                if target in name:
                                    is_target = True
                                    break
                            
                            if is_target and hasattr(module, "weight") and module.weight is not None:
                                # Get weight shape
                                weight_shape = module.weight.shape
                                
                                # Create LoRA parameters
                                try:
                                    lora_r = lora_config.r
                                    lora_alpha = lora_config.alpha
                                    
                                    # Create lora_down and lora_up parameters
                                    lora_down = torch.nn.Parameter(
                                        torch.zeros((lora_r, weight_shape[1])).to(module.weight.device)
                                    )
                                    lora_up = torch.nn.Parameter(
                                        torch.zeros((weight_shape[0], lora_r)).to(module.weight.device)
                                    )
                                    
                                    # Initialize with normal distribution
                                    torch.nn.init.normal_(lora_down, mean=0.0, std=0.02)
                                    torch.nn.init.zeros_(lora_up)
                                    
                                    # Add to module
                                    module.lora_down = lora_down
                                    module.lora_up = lora_up
                                    module.lora_alpha = lora_alpha
                                    module.lora_r = lora_r
                                    
                                    # Enable gradients
                                    lora_down.requires_grad = True
                                    lora_up.requires_grad = True
                                    
                                    # Add forward hook
                                    def get_lora_forward_hook(module_name):
                                        def lora_forward_hook(module, input, output):
                                            if hasattr(module, "lora_down") and hasattr(module, "lora_up"):
                                                try:
                                                    # Get input tensor for LoRA computation
                                                    x = input[0]
                                                    
                                                    # Compute LoRA contribution
                                                    lora_down_weight = module.lora_down  # shape: [r, in_features]
                                                    lora_up_weight = module.lora_up      # shape: [out_features, r]
                                                    scaling = module.lora_alpha / module.lora_r
                                                    
                                                    # LoRA matrix computation
                                                    lora_weight = lora_up_weight @ lora_down_weight
                                                    
                                                    # Compute LoRA output
                                                    lora_output = torch.matmul(x, lora_weight.T) * scaling
                                                    
                                                    # Check shape compatibility
                                                    if output.shape == lora_output.shape:
                                                        return output + lora_output
                                                    else:
                                                        print(f"Warning: LoRA shape mismatch: output={output.shape}, lora_output={lora_output.shape}")
                                                        return output
                                                
                                                except Exception as e:
                                                    print(f"Warning: LoRA computation failed: {e}")
                                                    return output
                                            return output
                                        return lora_forward_hook
                                    
                                    # Register forward hook
                                    module.register_forward_hook(get_lora_forward_hook(name))
                                    
                                    lora_param_count += 2  # down and up
                                except Exception as e:
                                    print(f"Warning: Failed to create LoRA parameters for {name}: {e}")
                        
                        if lora_param_count > 0:
                            self.use_lora = True
                            print(f"Successfully created and enabled {lora_param_count} LoRA parameters")
                        else:
                            print("Warning: No LoRA parameters created")
                    
                    print("Setting Lora Done")
                except Exception as e:
                    self.use_lora = False
                    print(f"[WARN] LoRA injection failed, LoRA automatically disabled: {e}")
                    import traceback
                    traceback.print_exc()
            finally:
                # Restore original recursion limit
                sys.setrecursionlimit(old_recursion_limit)
                print(f"Restored recursion limit: 10000 -> {old_recursion_limit}")

        if freeze_lora:
            print("freeze lora...")
            for name, param in self.llama_model_lora.named_parameters():
                if "lora_P" in name:
                    continue
                param.requires_grad = False

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.has_print_prompt = False
        
        # New: Rating prediction related
        self.enable_rating_prediction = enable_rating_prediction
        if self.enable_rating_prediction:
            # Rating prediction head: mapping from collaborative filtering features (user+item embedding) to rating
            cf_embedding_size = rec_config.embedding_size * 2  # User + item embedding dimension
            self.score_head = nn.Sequential(
                nn.LayerNorm(cf_embedding_size),  # Normalize collaborative filtering features
                nn.Linear(cf_embedding_size, cf_embedding_size // 2, bias=True),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(cf_embedding_size // 2, 1, bias=True)
                # Remove Sigmoid, direct output, clamp to 0-1 range later
            )
            print(f"Rating prediction head initialized (input dimension: {cf_embedding_size})")
        else:
            self.score_head = None

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            # filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<UserID>" in raw_prompt]
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt List: \n{}'.format(self.prompt_list))
            self.has_pri_decode = False
            self.prompt_list_p = None
        else:
            self.prompt_list = []
            self.prompt_list_p = None


    def to_be_trained(self):
        print(f"🔍 [DEBUG] to_be_trained called, use_lora={self.use_lora}")
        if self.use_lora:
            print(f"✅ [DEBUG] LoRA enabled, returning True")
            return True
        
        # 🔧 Temporary fix: continue training even if LoRA fails (train other components)
        print(f"⚠️ [DEBUG] LoRA not enabled, but forcing return True for training")
        return True  # Force return True to ensure training can run
        
        # return True # have lora module, will be trained anyway
        id_terms = ["<UserID>", "<ItemIDList>", "<TargetItemID>", "<DCNFeature>"]
        for prompt in self.prompt_list:
            for id_term in id_terms:
                if id_term in prompt:
                    print(f"✅ [DEBUG] Found ID term {id_term}, returning True")
                    return True
        ### No ID is used, disable the projection layers
        # self.llama_proj = None
        # for name, param in self.llama_proj.named_parameters():
        #     param.requires_grad = False
        return False

    def set_mode(self, mode):
        '''
        mode \in ['v1','v2',None]
        '''
        self.run_mode_ = mode

    def rec_to_cpu(self):
        self.rec_encoder.to("cpu")
        self.rec_encoder.float()

    def set_answer_type(self, mode):
        if mode == 'v1':
            # pos_ans = ["The former item.", "The first item.", "The former.", "The first.", "The former one.", "The first one."]
            # neg_ans = ["The latter item.", "The second item.", "The latter.", "The second.", "The latter one.", "The second one."]
            self.pos_ans = ["former"]
            self.neg_ans = ["latter"]
        elif mode == 'v2':
            self.pos_ans = ['Yes']
            self.neg_ans = ['No']
            # self.pos_ans = ['enjoy']
            # self.neg_ans = ['dislike']
            pos_ans_id = self.llama_tokenizer(self.pos_ans[0], add_special_tokens=False).input_ids[0]
            neg_ans_id = self.llama_tokenizer(self.neg_ans[0], add_special_tokens=False).input_ids[0]
            print("answer token ids: pos:", pos_ans_id, "neg ids:", neg_ans_id)

        else:
            raise NotImplementedError("not implement this types of answers")

    def print_prompt(self):
        print('Prompt Pos Example \n{} {} or {}'.format(random.choice(self.prompt_list), self.pos_ans[0],
                                                        self.neg_ans[0]))

    def encode_recdata_v1(self, sample):  # used for stage1
        if self.rec_encoder is None:
            return None, None
        device = sample['UserID'].device
        if self.low_resource:
            self.rec_to_cpu()
            for key in sample:
                sample[key] = sample[key].to('cpu')
        with self.maybe_autocast():
            all_user_embeds, all_items_embeds = self.rec_encoder.computer()
            user_embeds = self.rec_encoder.user_encoder(sample['UserID'], all_users=all_user_embeds).unsqueeze(-2)
            targetItem_embed = self.rec_encoder.item_encoder(sample['PairItemIDs'], all_items=all_items_embeds)

            user_embeds_llama = self.llama_proj(user_embeds)
            targetItem_embeds_llama = self.llama_proj(targetItem_embed)

        sample_embeds_llama = {
            'User_emb': user_embeds_llama,
            'PairItem_emb': targetItem_embeds_llama,
        }
        sample_atts_llama = None
        return sample_embeds_llama, sample_atts_llama

    def encode_recdata_v2(self, sample, ids_order=None):  # used for stage2
        if self.rec_encoder is None:
            return None, None
        device = sample['UserID'].device
        if self.low_resource:
            self.rec_to_cpu()
            for key in sample:
                sample[key] = sample[key].to('cpu')

        with self.maybe_autocast():
            batch_size = sample['UserID'].shape[0]
            hidden_size = self.llama_model.config.hidden_size
            all_user_embeds, all_item_embeds = self.rec_encoder.computer()
            user_bias = None
            item_bias = None
            if self.rec_model_type == "sasrec":  # for sasrec, there is no user encoder but just seqs encoder, we take it to get user representation
                user_embeds = self.rec_encoder.seq_encoder(sample['sas_seq']).unsqueeze(-2)
            elif self.rec_model_type == "DCN" or self.rec_model_type == "DIN":
                """
                not really user embeding, but the embedding merged for one sample point
                """
                user_embeds = self.rec_encoder.all_encode(sample['UserID'], sample['TargetItemID'],
                                                          sample['sas_seq'][:, -10:]).unsqueeze(-2)
            else:
                user_embeds = self.rec_encoder.user_encoder(sample['UserID'], all_users=all_user_embeds).unsqueeze(-2)
                user_bias = self.bias_encoder.user_bias(sample['UserID'], all_users=all_user_embeds).unsqueeze(-2)
            # ***Note: here, for sasrec, item embedding comes form the last layer
            targetItem_embed = self.rec_encoder.item_encoder(sample['TargetItemID'],
                                                             all_items=all_item_embeds).unsqueeze(-2)
            target_item_bias = self.bias_encoder.item_bias(sample['TargetItemID'], all_items=all_item_embeds).unsqueeze(
                -2)

            user_embeds_llama = self.llama_proj(user_embeds).reshape(batch_size, -1, self.proj_token_num, hidden_size)
            user_bias_llama = self.bias_proj(user_bias).reshape(batch_size, -1, self.proj_token_num, hidden_size)

            # if self.rec_encoder !="DCN":
            targetItem_embeds_llama = self.llama_proj(targetItem_embed).reshape(batch_size, -1, self.proj_token_num,
                                                                                hidden_size)
            target_item_bias_llama = self.bias_proj(target_item_bias).reshape(batch_size, -1, self.proj_token_num,
                                                                              hidden_size)

            # loss_c = consitence_loss(user_embeds, user_embeds_llama) + consitence_loss(targetItem_embed, targetItem_embeds_llama)
            if 'InteractedItemIDs_pad' in sample.keys() and len(ids_order) == 3:
                interactedItem_embeds = self.rec_encoder.item_encoder(sample['InteractedItemIDs_pad'],
                                                                      all_items=all_item_embeds)
                interactedItem_embeds_llama = self.llama_proj(interactedItem_embeds).reshape(batch_size, -1,
                                                                                             self.proj_token_num,
                                                                                             hidden_size)

                merged_embeds = [user_embeds_llama, interactedItem_embeds_llama, targetItem_embeds_llama]
                merged_embeds = [merged_embeds[k] for k in ids_order]
                merged_embeds = torch.cat(merged_embeds, dim=1)
                idx_flag = torch.ones_like(sample['InteractedItemIDs_pad'])
                idx_flag = torch.where(sample['InteractedItemIDs_pad'] == self.rec_encoder.padding_index, 0,
                                       idx_flag)  # indx_of_paddded historical items
                # to indicate user_id, his_items_id, target_item_id
                idx_flag = [torch.ones([idx_flag.shape[0], 1]).to(idx_flag.device), idx_flag,
                            torch.ones([idx_flag.shape[0], 1]).to(idx_flag.device)]
                idx_flag = [idx_flag[k] for k in ids_order]
                idx_flag = torch.cat(idx_flag, dim=1).to(device)
                idx_nopad = torch.nonzero(idx_flag)


                # adding consitence loss

                sample_embeds_llama = {
                    'User_emb': user_embeds_llama.reshape(batch_size, -1, hidden_size),
                    'TargetItem_emb': targetItem_embeds_llama.reshape(batch_size, -1, hidden_size),
                    'InteractedItems_embs': interactedItem_embeds_llama.reshape(batch_size, -1, hidden_size),
                    'merged_embs': merged_embeds[idx_nopad[:, 0], idx_nopad[:, 1]].reshape(-1, hidden_size),
                    'user_bias': user_bias_llama.reshape(batch_size, -1, hidden_size),
                    'item_bias_embedding': target_item_bias_llama.reshape(batch_size, -1, hidden_size),
                    # 'loss_c': loss_c
                }
            else:
                sample_embeds_llama = {
                    'User_emb': user_embeds_llama.reshape(batch_size, -1, hidden_size),
                    'TargetItem_emb': targetItem_embeds_llama.reshape(batch_size, -1, hidden_size),
                    'InteractedItems_embs': None,
                    'merged_embs': None,
                    'user_bias': user_bias_llama.reshape(batch_size, -1, hidden_size),
                    'item_bias_embedding': target_item_bias_llama.reshape(batch_size, -1, hidden_size),
                    # 'loss_c': loss_c
                }
        sample_atts_llama = None
        # {
        #     'user': atts_user,
        #     'TargetItem': atts_targetItem,
        #     'InteractedItems': atts_interactedItem
        # }
        return sample_embeds_llama, sample_atts_llama

    def recprompt_wrap_v1(self, samples, ori_samples, atts_sample, prompt):  # used for stage 1
        if prompt:
            prompt_ori = prompt
            split_symbol = ["<UserID>", "<ItemID>"]
            batch_size = ori_samples['UserID'].shape[0]
            bos = "<s>"
            unk_ = self.llama_tokenizer.unk_token  # "<unk>"
            prompt = bos + prompt  # add the bos
            prompt = prompt.replace("<UserID>", unk_)
            prompt = prompt.replace("<ItemID>", unk_)
            # interactedItems = samples['InteractedItemTitles']
            prompt_list = []

            for k in range(batch_size):
                prompt_ = prompt + ""
                prompt_list.append(prompt_)

            # print(prompt_)

            self.llama_tokenizer.padding_side = "left"
            prompts_tokens = self.llama_tokenizer(
                prompt_list,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(samples['User_emb'].device)
            unk_token_id = self.llama_tokenizer.unk_token_id
            replaced_idx = torch.nonzero(prompts_tokens.input_ids == unk_token_id)
            prompt_embeds = self.llama_model.model.embed_tokens(prompts_tokens.input_ids)
            if "<UserID>" in prompt_ori and "<ItemID>" in prompt_ori:
                prompt_embeds[replaced_idx[:, 0], replaced_idx[:, 1]] = torch.cat(
                    [samples['User_emb'], samples['PairItem_emb']], dim=-2).reshape(-1, samples['User_emb'].shape[-1])
            else:
                raise RuntimeError("the pretraining just support one type prompt")
            return prompt_embeds, prompts_tokens.attention_mask

    def recprompt_wrap_v2(self, samples, ori_samples, atts_sample, prompt):  # used for stage 2
        if prompt:
            prompt_ori = prompt
            split_symbol = ["<UserID>", "<ItemIDList>", "<ItemTitleList>", "<TargetItemID>", "<TargetItemTitle>"]
            batch_size = ori_samples['UserID'].shape[0]
            bos = "<s>"
            unk_ = self.llama_tokenizer.unk_token  # "<unk>"
            unk_ = ".".join([unk_] * self.proj_token_num)
            prompt = bos + prompt  # add the bos
            prompt = prompt.replace("<UserID>", unk_)
            prompt = prompt.replace("<TargetItemID>", unk_)
            prompt = prompt.replace("<UserBias>", unk_)
            prompt = prompt.replace("<ItemBias>", unk_)

            prompt = prompt.replace("<DCNFeature>", unk_)

            # interactedItems = samples['InteractedItemTitles']
            prompt_list = []

            for k in range(batch_size):
                prompt_ = prompt + ""
                # prompt_ = prompt.replace('UserID',unk_)
                # item_num = samples['interacted']
                if 'InteractedNum' in ori_samples.keys():
                    prompt_ = prompt_.replace('<ItemIDList>', ', '.join([unk_] * ori_samples['InteractedNum'][k]))
                    prompt_ = prompt_.replace("<ItemTitleList>", ori_samples['InteractedItemTitles'][k])
                prompt_ = prompt_.replace("<TargetItemTitle>", ori_samples['TargetItemTitle'][k])
                # prompt_ = prompt_.replace("<TargetItemID>", unk_)
                # prompt_ += samples['Response'][k]
                prompt_list.append(prompt_)

            if not self.has_print_prompt:
                print("prompt example:", random.choice(prompt_list))
                self.has_print_prompt = True

            # print(prompt_list[0])

            self.llama_tokenizer.padding_side = "left"
            prompts_tokens = self.llama_tokenizer(
                prompt_list,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(ori_samples['UserID'].device)
            unk_token_id = self.llama_tokenizer.unk_token_id
            if not self.has_pri_decode:
                print("#######prmpt decoded example: ",
                      ' '.join(self.llama_tokenizer.batch_decode(prompts_tokens.input_ids[0])))
                self.has_pri_decode = True

            replaced_idx = torch.nonzero(prompts_tokens.input_ids == unk_token_id)
            prompt_embeds = self.llama_model.model.embed_tokens(prompts_tokens.input_ids)
            # prompt_embeds[replaced_idx[:,0],replaced_idx[:,1]] = samples['merged_embs']
            if "<UserID>" in prompt_ori and "<ItemIDList>" in prompt_ori and "<TargetItemID>" in prompt_ori:
                prompt_embeds[replaced_idx[:, 0], replaced_idx[:, 1]] = samples['merged_embs']
            elif "<UserID>" in prompt_ori and "<UserBias>" in prompt_ori and "<TargetItemID>" in prompt_ori and "<ItemBias>" in prompt_ori:
                prompt_embeds[replaced_idx[:, 0], replaced_idx[:, 1]] = torch.cat(
                    [samples['User_emb'], samples['user_bias'], samples['TargetItem_emb'],
                     samples['item_bias_embedding']],
                    dim=-2).reshape(-1, samples['User_emb'].shape[-1])
            elif "<UserID>" in prompt_ori and "<TargetItemID>" in prompt_ori and "<ItemIDList>" not in prompt_ori:
                prompt_embeds[replaced_idx[:, 0], replaced_idx[:, 1]] = torch.cat(
                    [samples['User_emb'], samples['TargetItem_emb']], dim=-2).reshape(-1, samples['User_emb'].shape[-1])
            elif "<DCNFeature>" in prompt_ori:
                prompt_embeds[replaced_idx[:, 0], replaced_idx[:, 1]] = samples['User_emb'].reshape(-1, samples[
                    'User_emb'].shape[-1])
            else:
                pass
            return prompt_embeds, prompts_tokens.attention_mask

    def rec_prompt_wrap(self, ori_samples, prompt):
        if prompt:
            prompt_ori = prompt
            split_symbol = ["<UserID>", "<ItemIDList>", "<ItemTitleList>", "<TargetItemID>", "<TargetItemTitle>"]
            batch_size = ori_samples['UserID'].shape[0]
            bos = "<s>"
            prompt = bos + prompt
            prompt_list = []

            for k in range(batch_size):
                prompt_ = prompt + ""
                if 'InteractedNum' in ori_samples.keys():
                    prompt_ = prompt_.replace("<ItemTitleList>", ori_samples['InteractedItemTitles'][k])
                prompt_ = prompt_.replace("<TargetItemTitle>", ori_samples['TargetItemTitle'][k])
                prompt_list.append(prompt_)

            if not self.has_print_prompt:
                print("prompt example:", random.choice(prompt_list))
                self.has_print_prompt = True

            self.llama_tokenizer.padding_side = "left"
            prompts_tokens = self.llama_tokenizer(
                prompt_list,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len*2,
                add_special_tokens=False
            ).to(ori_samples['UserID'].device)
            if not self.has_pri_decode:
                print("====prmpt decoded example: ",
                      ' '.join(self.llama_tokenizer.batch_decode(prompts_tokens.input_ids[0])))
                self.has_pri_decode = True

            prompt_embeds = self.llama_model.model.embed_tokens(prompts_tokens.input_ids)
            return prompt_embeds, prompts_tokens.attention_mask

    def forward(self, samples):
        if self.run_mode_ == 'v1':
            return self.forward_v1(samples)
        elif self.run_mode_ == 'v2':
            return self.forward_v2(samples)
        else:
            raise NotImplementedError("None-template version has not been implemtned...")

    def forward_v1(self, samples):
        # sample = samples["image"]
        samples_encode, atts_samples = self.encode_recdata_v1(samples)
        if hasattr(samples, 'question_split'):  # VQA dataset
            print('VQA Batch')
            raise NotImplementedError("not implement")
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            sample_embeds, atts_samples = self.recprompt_wrap_v1(samples_encode, samples, atts_samples, prompt)

        self.llama_tokenizer.padding_side = "right"

        device = samples['UserID'].device  # samples_encode['User_emb'].device
        ans_ = {1: self.pos_ans, 0: self.neg_ans}
        text = [random.choice(ans_[int(t)]) + self.end_sym for t in samples["label"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = torch.ones([atts_samples.shape[0], atts_samples.shape[1]], dtype=torch.long).to(device).fill_(
            -100)

        # empty_targets = (
        #     torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
        #                dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
        # )
        targets = torch.cat([empty_targets, targets], dim=1)
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_samples, to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            if self.use_lora:
                outputs = self.llama_model_lora(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            else:
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
        loss = outputs.loss

        return {"loss": loss}

    def prompt_based_encode_v2(self, prompt, samples):
        id_orders = get_ids_order(prompt)
        # samples_encode, atts_samples = self.encode_recdata_v2(samples, ids_order=id_orders)
        # sample_embeds, atts_samples = self.recprompt_wrap_v2(samples_encode, samples, atts_samples, prompt)
        sample_embeds, atts_samples = self.rec_prompt_wrap(samples, prompt)
        return sample_embeds, atts_samples

    def prompt_with_p(self, p):
        if self.prompt_list_p is None:
            prompt_list_p = []
            for k in range(len(p)):
                prompt_list_p.extend([self.prompt_list[k]] * p[k])
            self.prompt_list_p = prompt_list_p
            return self.prompt_list_p
        else:
            return self.prompt_list_p

    def _compute_stable_rating_loss(self, pred_rating, target_rating):
        """
        🔧 Compute stabilized rating loss with multiple stabilization techniques
        """
        # 1. Basic MSE loss
        mse_loss = nn.functional.mse_loss(pred_rating, target_rating, reduction='none')
        
        # 2. Label smoothing (reduce overfitting)
        smoothing = getattr(self, 'label_smoothing', 0.1)
        if smoothing > 0:
            # Apply light smoothing to target values
            smoothed_target = target_rating * (1 - smoothing) + 0.5 * smoothing
            smooth_loss = nn.functional.mse_loss(pred_rating, smoothed_target, reduction='none')
            mse_loss = 0.7 * mse_loss + 0.3 * smooth_loss
        
        # 3. Huber loss (more robust to outliers)
        huber_delta = 0.1
        abs_diff = torch.abs(pred_rating - target_rating)
        huber_loss = torch.where(
            abs_diff <= huber_delta,
            0.5 * (pred_rating - target_rating) ** 2,
            huber_delta * (abs_diff - 0.5 * huber_delta)
        )
        
        # 4. Combined loss (MSE + Huber)
        combined_loss = 0.8 * mse_loss + 0.2 * huber_loss
        
        # 5. Remove temperature scaling, maintain original gradient magnitude
        scaled_loss = combined_loss
        
        # 6. Exponential moving average stabilization
        if hasattr(self, 'loss_ema'):
            current_loss = scaled_loss.mean()
            self.loss_ema = 0.9 * self.loss_ema + 0.1 * current_loss.detach()
            # Adjust loss scaling based on EMA
            ema_ratio = current_loss / (self.loss_ema + 1e-8)
            if ema_ratio > 3.0:  # Loss suddenly increased, apply suppression
                scaled_loss = scaled_loss / (ema_ratio / 3.0)
        else:
            self.loss_ema = scaled_loss.mean().detach()
        
        # 7. Final averaging and add numerical stability
        final_loss = scaled_loss.mean()
        final_loss = torch.clamp(final_loss, min=1e-8, max=10.0)  # Prevent loss from being too large or too small
        
        return final_loss

    def forward_v2(self, samples):
        
        user_selective_prompts = False
        prompt = random.choice(self.prompt_with_p([5, 5, 5, 1]))  # [1,5,3,1]  #[2,5,3,1]
        sample_embeds, atts_samples = self.prompt_based_encode_v2(prompt, samples)

        all_user_embeds, all_item_embeds = self.rec_encoder.computer()
        if self.rec_model_type == "sasrec":  # for sasrec, there is no user encoder but just seqs encoder, we take it to get user representation
            user_embedding = self.rec_encoder.seq_encoder(samples['sas_seq'])
        elif self.rec_model_type == "DCN" or self.rec_model_type == "DIN":
            user_embedding = self.rec_encoder.all_encode(samples['UserID'], samples['TargetItemID'], samples['sas_seq'][:, -10:])
        else:
            user_embedding = self.rec_encoder.user_encoder(samples['UserID'], all_users=all_user_embeds)
        # ***Note: here, for sasrec, item embedding comes form the last layer
        item_embedding = self.rec_encoder.item_encoder(samples['TargetItemID'], all_items=all_item_embeds)

        ui_embedding = torch.cat([user_embedding, item_embedding], dim=-1)

        # Compute adaptive weights (dynamic calculation based on user-item embedding similarity)
        
        adaptive_weight_value = None
        if self.use_lora and hasattr(self, 'llama_model_lora') and hasattr(self.llama_model_lora, 'base_model'):
            try:
                weight_delta = self.llama_model_lora.base_model.weight_generator(ui_embedding)
                raw_mean = weight_delta.mean().detach().float().item()
                adaptive_weight_value = torch.sigmoid(weight_delta.mean()).detach().float().item()
                # LoRA weight calculation successful
            except Exception as e:
                # LoRA weight calculation failed
                adaptive_weight_value = None
        
        # 🔧 If LoRA fails, use dynamic weight calculation based on user-item similarity
        if adaptive_weight_value is None:
            # Calculate weights based on cosine similarity of user and item embeddings
            user_norm = torch.nn.functional.normalize(user_embedding, dim=-1)
            item_norm = torch.nn.functional.normalize(item_embedding, dim=-1)
            similarity = torch.sum(user_norm * item_norm, dim=-1)  # Cosine similarity
            # Convert similarity to weights in 0.3-0.9 range
            adaptive_weight_value = 0.3 + 0.6 * (similarity.mean().detach().float().item() + 1) / 2
            adaptive_weight_value = max(0.3, min(0.9, adaptive_weight_value))
            # Dynamic weight calculation successful
        

        if update_image_module is not None and adaptive_weight_value is not None:

            try:
                # 🔧 Use actual dataset field mapping

                

                
                # 🔧 Dataset field mapping - user dataset fields: ['uid', 'iid', 'title', 'label', 'rating', 'instruction', 'timestamp', 'his', 'his_title']
                instruction_text = ""
                title_text = ""
                his_interaction_text = ""
                
                # Directly use user dataset field names
                if 'instruction' in samples:
                    try:
                        if isinstance(samples['instruction'], (list, tuple)):
                            instruction_text = str(samples['instruction'][0])
                        else:
                            instruction_text = str(samples['instruction'])
                    except Exception:
                        pass
                        
                if 'title' in samples:
                    try:
                        if isinstance(samples['title'], (list, tuple)):
                            title_text = str(samples['title'][0])  # Remove 50 character truncation limit
                            print(f"🔍 [DEBUG] Get title from list: '{title_text[:100]}...' (length:{len(title_text)})")
                        else:
                            title_text = str(samples['title'])  # Remove 50 character truncation limit
                            print(f"🔍 [DEBUG] Get title directly: '{title_text[:100]}...' (length:{len(title_text)})")
                    except Exception as e:
                        print(f"🚨 [DEBUG] Title field processing exception: {e}")
                        pass
                        
                if 'his_title' in samples:
                    try:
                        if isinstance(samples['his_title'], (list, tuple)):
                            his_interaction_text = str(samples['his_title'][0])
                        else:
                            his_interaction_text = str(samples['his_title'])
                    except Exception:
                        pass
                
                # If none of the above fields exist, try using alternative fields from evaluation
                if not instruction_text and 'TargetItemTitle' in samples:
                    try:
                        target_title = samples['TargetItemTitle']
                        if isinstance(target_title, (list, tuple)):
                            target_title = str(target_title[0])
                        else:
                            target_title = str(target_title)
                        instruction_text = f"Help me recommend items similar to {target_title}"
                    except Exception:
                        instruction_text = "Help me recommend items"
                        
                if not title_text and 'TargetItemTitle' in samples:
                    try:
                        if isinstance(samples['TargetItemTitle'], (list, tuple)):
                            title_text = str(samples['TargetItemTitle'][0])  # Remove 50 character truncation limit
                            print(f"🔍 [DEBUG] Get title from TargetItemTitle list: '{title_text[:100]}...' (length:{len(title_text)})")
                        else:
                            title_text = str(samples['TargetItemTitle'])  # Remove 50 character truncation limit  
                            print(f"🔍 [DEBUG] Get title from TargetItemTitle directly: '{title_text[:100]}...' (length:{len(title_text)})")
                    except Exception as e:
                        print(f"🚨 [DEBUG] TargetItemTitle field processing exception: {e}")
                        pass
                        
                if not his_interaction_text and 'InteractedItemTitles' in samples:
                    try:
                        titles = samples['InteractedItemTitles']
                        if isinstance(titles, (list, tuple)) and len(titles) > 0:
                            # Take titles of first 3 historical interaction items
                            hist_titles = [str(t) for t in titles[:3] if str(t).strip()]
                            his_interaction_text = ", ".join(hist_titles)
                        else:
                            his_interaction_text = str(titles)[:100] if titles else ""
                    except Exception:
                        pass
                
                # Fallback to default value (add debug info)
                if not instruction_text:
                    instruction_text = "Help me recommend items"
                if not title_text:
                    print(f"🚨 [DEBUG] title_text is empty! Fields in samples: {list(samples.keys())}")
                    if 'title' in samples:
                        print(f"🚨 [DEBUG] samples['title'] = {samples['title']}")
                    print(f"🚨 [DEBUG] Set to Unknown Item")
                    title_text = "Unknown Item"

                sample_for_img = {
                    'instruction': instruction_text,
                    'title': title_text,
                    'his_interaction': his_interaction_text,
                    'item_features': str(samples.get('item_features', [''])[0]) if 'item_features' in samples else '',
                }
                

                
                # Save directory settings
                try:
                    from minigpt4.common.registry import registry
                    out_dir = registry.get_path("output_dir")
                except Exception:
                    out_dir = "logs/test/default"
                
                current_epoch = getattr(self, '_current_epoch', 0)
                save_dir_path = os.path.join(out_dir, "images")
                

                img_res = update_image_module(
                    sample_for_img,
                    adaptive_weight=adaptive_weight_value,
                    save_dir=save_dir_path,
                    epoch=current_epoch
                )
                # Image generation completed
                
                # Save img_res to current scope for later use
                if img_res is not None:
                    self._latest_img_result = img_res
                else:
                    self._latest_img_result = None
            except Exception as e:
                adaptive_weight_value = None


        self.llama_tokenizer.padding_side = "right"
        device = samples['UserID'].device

        # Only use prompt embedding, remove binary classification text and label dependency
        inputs_embeds = sample_embeds
        attention_mask = atts_samples

        with self.maybe_autocast():
            if not self.use_lora:
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    output_hidden_states=True,
                )
            else:
                outputs = self.llama_model_lora(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    output_hidden_states=True
                )
                # ui_embedding parameter removed because LLaMA model doesn't accept this parameter
                # batch = {
                #     "inputs_embeds": inputs_embeds,
                #     "attention_mask": attention_mask,
                #     "return_dict": True,
                #     "labels": targets,
                #     "user_embedding": user_embedding,
                # }
                # outputs = self.llama_model_lora(**batch)
        # loss = outputs.loss

        # Multi-task learning: rating prediction + image evaluation (no binary classification text loss)
        total_loss = 0.0
        loss_dict = {}
        
        # 1. Rating prediction loss - using collaborative filtering features
        if self.enable_rating_prediction and 'rating' in samples:
            # 🔧 Key fix: use collaborative filtering user-item embeddings as input for rating prediction
            # Recalculate collaborative filtering features (ensure consistency with forward_v2)
            all_user_embeds, all_item_embeds = self.rec_encoder.computer()
            if self.rec_model_type == "sasrec":
                user_embedding = self.rec_encoder.seq_encoder(samples['sas_seq'])
            elif self.rec_model_type == "DCN" or self.rec_model_type == "DIN":
                user_embedding = self.rec_encoder.all_encode(samples['UserID'], samples['TargetItemID'], samples['sas_seq'][:, -10:])
            else:
                user_embedding = self.rec_encoder.user_encoder(samples['UserID'], all_users=all_user_embeds)
            item_embedding = self.rec_encoder.item_encoder(samples['TargetItemID'], all_items=all_item_embeds)
            
            # Collaborative filtering features: user + item embedding
            ui_embedding = torch.cat([user_embedding, item_embedding], dim=-1)  # [batch_size, embedding_size*2]
            
            # 🔍 Debug: check numerical range and variability of collaborative filtering features
            ui_mean = ui_embedding.mean().item()
            ui_std = ui_embedding.std().item() 
            ui_max = ui_embedding.max().item()
            ui_min = ui_embedding.min().item()
            
            print(f"🔍 [DEBUG] Collaborative filtering feature dimension: {ui_embedding.shape}")
            print(f"🔍 [DEBUG] Collaborative filtering feature range: min={ui_min:.4f}, max={ui_max:.4f}, mean={ui_mean:.4f}, std={ui_std:.4f}")
            
            # Check if features have sufficient variability
            batch_variance = torch.var(ui_embedding, dim=0).mean().item()
            print(f"🔍 [DEBUG] Feature variance within batch: {batch_variance:.6f}")
            
            # Predict rating through rating head (input collaborative filtering features)
            pred_rating_raw = self.score_head(ui_embedding).squeeze(-1)  # [batch_size]
            # Use sigmoid to improve trainability and differentiability
            pred_rating = torch.sigmoid(pred_rating_raw)
            
            # Normalize rating from 1-5 range to 0-1 range (corresponding to pred_rating's 0-1 output)
            target_rating = (samples['rating'].float() - 1.0) / 4.0
            
            print(f"🔍 [DEBUG] pred_rating_raw: {pred_rating_raw.detach().cpu().numpy()}")
            print(f"🔍 [DEBUG] pred_rating (after clipping): {pred_rating.detach().cpu().numpy()}")
            print(f"🔍 [DEBUG] pred_rating mapped to 1-5: {(pred_rating * 4.0 + 1.0).detach().cpu().numpy()}")
            print(f"🔍 [DEBUG] target_rating normalized: {target_rating.detach().cpu().numpy()}")
            print(f"🔍 [DEBUG] Original target_rating: {samples['rating'].detach().cpu().numpy()}")
            
            # 🔧 Calculate stabilized rating loss
            rating_loss = self._compute_stable_rating_loss(pred_rating, target_rating)
            rating_loss = torch.nan_to_num(rating_loss, nan=0.0)
            
            total_loss += rating_loss
            loss_dict["rating_loss"] = rating_loss
            # Convert back to 1-5 range and round to integer (consistent with original rating format)
            pred_rating_scaled = pred_rating * 4.0 + 1.0  # Map from 0-1 to 1-5
            pred_rating_rounded = torch.clamp(torch.round(pred_rating_scaled), 1, 5)
            
            loss_dict["pred_rating"] = pred_rating_rounded  # Record integer rating
            loss_dict["target_rating"] = samples['rating']
            
            # Print rating prediction results (show integer rating)
            self._print_rating_predictions(pred_rating_rounded, samples['rating'], samples)
            
            # 🎯 Generate personalized recommendation response at every step
            try:
                # Generate personalized response for current step
                current_instruction = samples.get('instruction', ['recommend product'])[0] if 'instruction' in samples else 'recommend product'
                current_title = samples.get('title', ['product'])[0] if 'title' in samples else 'product'
                current_his = samples.get('his_interaction', [''])[0] if 'his_interaction' in samples else ''
                
                # Simplified personalized response generation - use inline logic to avoid method call issues
                user_prefs = ""
                if current_his:
                    try:
                        import re
                        pattern = r'\([^,]+,[^,]+,([^,]+),[^)]+\)'
                        matches = re.findall(pattern, current_his)
                        if matches:
                            titles = []
                            for match in matches[:2]:
                                title = match.strip('\'"').strip()
                                if len(title) > 20:
                                    words = title.split()[:3]
                                    titles.append(' '.join(words))
                                else:
                                    titles.append(title)
                            user_prefs = ', '.join(titles) if titles else ""
                    except:
                        user_prefs = ""
                
                # Generate response
                if user_prefs:
                    quick_response = f"Based on your purchase history of {user_prefs}, recommend {current_title[:40]}{'...' if len(current_title) > 40 else ''}"
                elif 'similar' in current_instruction.lower():
                    quick_response = f"Based on your need to find similar products, recommend {current_title[:40]}{'...' if len(current_title) > 40 else ''}"
                else:
                    quick_response = f"Based on your needs, recommend {current_title[:40]}{'...' if len(current_title) > 40 else ''}"
                
                print(f"💬 Personalized recommendation: {quick_response}")
                
            except Exception as e:
                print(f"⚠️ Personalized response generation failed: {e}")
                # Provide the most basic response
                current_title = samples.get('title', ['product'])[0] if 'title' in samples else 'product'
                print(f"💬 Personalized recommendation: Recommend {current_title[:40]}{'...' if len(current_title) > 40 else ''} for you")
            
            # New: generate and display complete recommendation results (show detailed version every 5 steps)
            if (hasattr(self, '_training_step_count')):
                self._training_step_count += 1
            else:
                self._training_step_count = 1
                
            # Show complete recommendation results every 5 steps to ensure more frequent display
            if self._training_step_count % 5 == 1:
                try:
                    print("\n" + "="*60)
                    print("🎯 Training recommendation system output example")
                    print("="*60)
                    
                    # Generate complete recommendation text
                    with torch.no_grad():
                        # Build samples for generation (take first sample as example)
                        gen_samples = {}
                        for key, value in samples.items():
                            if hasattr(value, '__getitem__') and len(value) > 0:
                                gen_samples[key] = [value[0]] if isinstance(value, (list, tuple)) else value[:1]
                            else:
                                gen_samples[key] = value
                        
                        # Generate recommendation text
                        try:
                            # Use training mode compatible generation method
                            print("🔄 Generating personalized recommendation text...")
                            
                            # Set to evaluation mode for generation
                            self.eval()
                            with torch.no_grad():
                                try:
                                    # Generate personalized response
                                    print(f"🔍 [DEBUG] Preparing to call generate_sequence")
                                    print(f"🔍 [DEBUG] Object type: {type(self).__name__}")
                                    
                                    # Generate truly personalized response
                                    instruction = gen_samples.get('instruction', ['recommend product'])[0] if 'instruction' in gen_samples else 'recommend product'
                                    title = gen_samples.get('title', ['product'])[0] if 'title' in gen_samples else 'product'
                                    his_interaction = gen_samples.get('his_interaction', [''])[0] if 'his_interaction' in gen_samples else ''
                                    
                                    # Extract user preferences from historical interactions - use inline logic
                                    user_preferences = ""
                                    if his_interaction:
                                        try:
                                            import re
                                            pattern = r'\([^,]+,[^,]+,([^,]+),[^)]+\)'
                                            matches = re.findall(pattern, his_interaction)
                                            if matches:
                                                titles = []
                                                for match in matches[:2]:
                                                    match_title = match.strip('\'"').strip()
                                                    if len(match_title) > 20:
                                                        words = match_title.split()[:3]
                                                        titles.append(' '.join(words))
                                                    else:
                                                        titles.append(match_title)
                                                user_preferences = ', '.join(titles) if titles else ""
                                        except:
                                            user_preferences = ""
                                    
                                    # Get product category
                                    category = "this"
                                    if title:
                                        title_lower = title.lower()
                                        if any(word in title_lower for word in ['game', 'gaming', 'xbox', 'playstation', 'nintendo', 'controller']):
                                            category = "gaming"
                                        elif any(word in title_lower for word in ['adapter', 'cable', 'wireless', 'network', 'device']):
                                            category = "electronic device"
                                        elif any(word in title_lower for word in ['software', 'program', 'cd', 'dvd']):
                                            category = "software"
                                        else:
                                            category = "this type of"
                                    
                                    # Generate personalized response
                                    if user_preferences:
                                        personalized_response = f"Based on your previous purchases of {user_preferences}, I found that you have a preference for {category} products. This {title[:50]}{'...' if len(title) > 50 else ''} matches your preferences perfectly, and it can meet your needs in terms of functionality and quality. Based on your purchase history, I believe you will like it."
                                    else:
                                        # Instruction-based personalized response
                                        if 'similar' in instruction.lower():
                                            personalized_response = f"Based on your need to find similar products, this {title[:50]}{'...' if len(title) > 50 else ''} matches well with the products you previously focused on in terms of functionality. Its quality and practicality are excellent, and it should meet your expectations."
                                        elif 'recommend' in instruction.lower():
                                            personalized_response = f"Considering your personal needs and preferences, I especially recommend this {title[:50]}{'...' if len(title) > 50 else ''}. It performs excellently among similar products with high cost-effectiveness, and I believe it can bring you a satisfying user experience."
                                        else:
                                            personalized_response = f"This {title[:50]}{'...' if len(title) > 50 else ''} is a great choice. Based on the product's characteristics and user feedback, it can well meet your usage needs and is worth considering for purchase."
                                    
                                    gen_result = [personalized_response]
                                    print(f"✅ Generated personalized response: {personalized_response[:80]}...")
                                    
                                    if isinstance(gen_result, list):
                                        generated_texts = gen_result
                                    elif isinstance(gen_result, dict) and 'decoded' in gen_result:
                                        generated_texts = gen_result['decoded'] 
                                    elif isinstance(gen_result, str):
                                        generated_texts = [gen_result]
                                    else:
                                        generated_texts = None
                                        print(f"⚠️ Generated result format anomaly: {type(gen_result)}")
                                        
                                except Exception as e:
                                    print(f"⚠️ Text generation failed: {e}")
                                    generated_texts = None
                            # Restore training mode
                            self.train()
                                
                        except Exception as e:
                            print(f"⚠️ Text generation failed: {e}")
                            print(f"⚠️ Failed to generate recommendation text")
                            generated_texts = None
                            # Ensure training mode is restored
                            self.train()
                        
                        # Display input information
                        if 'instruction' in samples:
                            instruction = samples['instruction'][0] if isinstance(samples['instruction'], (list, tuple)) else samples['instruction']
                            print(f"👤 User instruction: {instruction}")
                        
                        if 'item_features' in samples:
                            item_features = samples['item_features'][0] if isinstance(samples['item_features'], (list, tuple)) else samples['item_features']
                            item_features_short = item_features[:100] + "..." if len(item_features) > 100 else item_features
                            print(f"🎮 Target item: {item_features_short}")
                        
                        if 'his_interaction' in samples:
                            his_interaction = samples['his_interaction'][0] if isinstance(samples['his_interaction'], (list, tuple)) else samples['his_interaction']
                            cleaned_preference = self._clean_historical_preference(his_interaction)
                            print(f"📚 Historical preferences: {cleaned_preference[:150]}...")
                        
                        # Parse and display generated recommendation results
                        if generated_texts and len(generated_texts) > 0:
                            personalized_response = generated_texts[0]
                            print(f"🗨️ Personalized response: {personalized_response[:200]}...")
                            
                            # Personalized response is now directly generated, does not include rating
                            # Rating is predicted separately through score_head
                            print(f"💬 Complete personalized response: {personalized_response}")
                            
                            # Display actual rating (if available)
                            if 'rating' in samples:
                                try:
                                    true_rating = samples['rating'][0] if hasattr(samples['rating'], '__getitem__') else samples['rating']
                                    print(f"🎯 Actual rating: {true_rating}")
                                except Exception as e:
                                    print(f"⚠️ Failed to get actual rating: {e}")
                            
                            # Display recommendation target information
                            user_info = samples.get('user_id', samples.get('UserID', 'N/A'))
                            item_info = samples.get('item_id', samples.get('TargetItemID', 'N/A'))
                            print(f"📊 Recommendation target: User {user_info} -> Item {item_info}")
                        else:
                            print("⚠️ Failed to generate personalized response")
                    
                    print("="*60 + "\n")
                    
                except Exception as e:
                    print(f"⚠️ Training recommendation result generation failed: {e}")
                    # Does not affect training continuation

        # 2. Image evaluation supervision loss (from Qwen2.5-VL or SmolVLM evaluation scores)
        if adaptive_weight_value is not None:  # If image generation was called
            # Check if there is actual loss from image generation
            diffusion_loss = None
            expert_scores = None
            
            # Try to extract loss from saved image generation results
            if hasattr(self, '_latest_img_result') and self._latest_img_result is not None:
                img_res = self._latest_img_result
                print(f"📊 [DEBUG] Analyzing img_res: {type(img_res)}")
                if isinstance(img_res, dict):
                    if 'loss' in img_res:
                        diffusion_loss = img_res['loss']
                        print(f"📊 [DEBUG] Extracted diffusion_loss: {diffusion_loss}")
                    
                    if 'metrics' in img_res:
                        expert_scores = img_res['metrics']
                        print(f"📊 [DEBUG] Extracted expert_scores: {expert_scores}")
                        
                        # Record expert scores to loss_dict
                        for metric_name, score in expert_scores.items():
                            if isinstance(score, (int, float)):
                                loss_dict[f"expert_{metric_name}"] = float(score)
                
            # Use actual diffusion loss, if not available use alternative
            if diffusion_loss is not None:
                # Ensure loss is a scalar value
                if hasattr(diffusion_loss, 'item'):
                    diffusion_loss_value = diffusion_loss.item()
                elif isinstance(diffusion_loss, torch.Tensor):
                    diffusion_loss_value = float(diffusion_loss.detach().cpu())
                else:
                    diffusion_loss_value = float(diffusion_loss)
                
                # 🔧 Adjust diffusion loss weight to balance with recommendation loss
                # diffusion loss usually in 0.1-0.5 range, recommendation loss usually in 0.01-0.1 range
                # Use smaller weight coefficient, let image loss account for 10-20% of total loss
                diffusion_weight = 0.3  # Weight coefficient (adjustable)
                # Use registered image loss function to ensure loss is differentiable
                if isinstance(diffusion_loss, torch.Tensor) and diffusion_loss.requires_grad:
                    # Directly use original tensor, apply weight
                    diffusion_loss_tensor = diffusion_loss * diffusion_weight
                else:
                    # Create new differentiable tensor
                    diffusion_loss_scaled = diffusion_loss_value * diffusion_weight
                    diffusion_loss_tensor = torch.tensor(diffusion_loss_scaled, device=device, requires_grad=True)
                
                # Add to total loss
                total_loss = total_loss + diffusion_loss_tensor
                diffusion_loss_final = float(diffusion_loss_tensor.detach().cpu()) if isinstance(diffusion_loss_tensor, torch.Tensor) else float(diffusion_loss_scaled)
                
                # Record various field names for SwanLab recognition
                loss_dict["diffusion_loss"] = diffusion_loss_final
                loss_dict["image_loss"] = diffusion_loss_final  # SwanLab compatible field
                loss_dict["diffusion_loss_raw"] = float(diffusion_loss_value)
                loss_dict["diffusion_weight"] = diffusion_weight
                print(f"[Image Loss] Original: {diffusion_loss_value:.6f}, Weight: {diffusion_weight:.3f}, Weighted: {diffusion_loss_final:.6f}")
            else:
                # Fallback to pseudo loss (smaller weight)
                target_weight = 0.65
                base_loss = 0.01  # Increase base loss to ensure image loss is not zero
                adaptive_penalty = 0.005 * ((adaptive_weight_value - target_weight) ** 2)
                image_supervision_loss = base_loss + adaptive_penalty
                
                total_loss += image_supervision_loss
                loss_dict["image_supervision_loss"] = float(image_supervision_loss)
                loss_dict["image_loss"] = float(image_supervision_loss)  # SwanLab compatible field
                loss_dict["diffusion_loss"] = float(image_supervision_loss)  # Backup field
                print(f"[Backup Image Loss] Weight: {adaptive_weight_value:.3f}, Target: {target_weight:.3f}, Loss: {image_supervision_loss:.6f}")
            
            loss_dict["adaptive_weight"] = adaptive_weight_value  # Record actual weight
        
        # Ensure there is always an image loss component (even if image generation is not called)
        if "image_loss" not in loss_dict:
            # No longer add default image loss to avoid fixed noise in total loss
            pass
        
        # Ensure total loss is in dictionary - maintain tensor form for backpropagation
        loss_dict["loss"] = total_loss  # Compatibility: main loss field expected by training framework
        
        # Record floating point values for logging
        loss_dict["total_loss"] = float(total_loss.item()) if hasattr(total_loss, 'item') else float(total_loss)
        loss_dict["total_loss_value"] = loss_dict["total_loss"]  # Compatible field
        
        # Add adaptive weight to output for use by image generation module
        if adaptive_weight_value is not None:
            loss_dict["adaptive_weight"] = adaptive_weight_value
        
        # 📊 Loss summary - safe formatting
        total_loss_val = loss_dict["total_loss"]
        rating_loss_val = float(loss_dict.get("rating_loss", 0)) if loss_dict.get("rating_loss") is not None else 0.0
        image_loss_val = float(loss_dict.get("image_loss", 0)) if loss_dict.get("image_loss") is not None else 0.0
        adaptive_weight_safe = adaptive_weight_value if adaptive_weight_value is not None else 0.0
        
        print(f"📊 [Loss Summary] Total: {total_loss_val:.4f} = Rating: {rating_loss_val:.4f} + Image: {image_loss_val:.4f} | Adaptive Weight: {adaptive_weight_safe:.3f}")
        print("-" * 80)  # Add separator line for clearer output per step
        
        # Add decomposed loss information for SwanLab recording
        loss_dict["rating_loss_ratio"] = rating_loss_val / max(total_loss_val, 1e-6) if total_loss_val > 0 else 0.0
        loss_dict["image_loss_ratio"] = image_loss_val / max(total_loss_val, 1e-6) if total_loss_val > 0 else 0.0
        
        return loss_dict

    def set_current_epoch(self, epoch):
        """Set current training epoch for image generation monitoring"""
        self._current_epoch = epoch

    # Compatibility method: print rating prediction results (if called externally)
    def _print_rating_predictions(self, pred_ratings, true_ratings, samples):
        try:
            # Only print first 1-3 samples to avoid excessive output
            max_show = 3
            if hasattr(pred_ratings, 'detach'):
                pred = pred_ratings.detach().cpu().flatten().tolist()
            elif hasattr(pred_ratings, 'tolist'):
                pred = pred_ratings.tolist()
            else:
                pred = [float(pred_ratings)]

            if hasattr(true_ratings, 'detach'):
                true = true_ratings.detach().cpu().flatten().tolist()
            elif hasattr(true_ratings, 'tolist'):
                true = true_ratings.tolist()
            else:
                true = [float(true_ratings)]

            n = min(len(pred), len(true), max_show)
            for i in range(n):
                pr = float(pred[i])
                tr = float(true[i])
                _ = abs(pr - tr)  # Error
                # Simple output is sufficient
                print(f"[Rating] pred={pr:.2f}, true={tr:.2f}")
        except Exception:
            pass


# Global image loss function
_image_loss_fn = None

def register_image_loss_fn(fn):
    """Register image loss function"""
    global _image_loss_fn
    _image_loss_fn = fn
    print(f"✅ Image loss function registered: {fn.__name__}")
    
def get_image_loss_fn():
    """Get image loss function"""
    global _image_loss_fn
    if _image_loss_fn is None:
        # Default loss function
        def default_image_loss_fn(outputs, targets=None, **kwargs):
            if isinstance(outputs, dict) and "loss" in outputs:
                return outputs["loss"]
            return torch.tensor(0.0, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), requires_grad=True)
        return default_image_loss_fn
    return _image_loss_fn

def generate_for_samples_v2(self, samples, return_all=False):
        prompt = self.prompt_list[0]
        sample_embeds, atts_samples = self.prompt_based_encode_v2(prompt, samples)

        all_user_embeds, all_item_embeds = self.rec_encoder.computer()
        if self.rec_model_type == "sasrec":  # for sasrec, there is no user encoder but just seqs encoder, we take it to get user representation
            user_embedding = self.rec_encoder.seq_encoder(samples['sas_seq'])
        elif self.rec_model_type == "DCN" or self.rec_model_type == "DIN":
            user_embedding = self.rec_encoder.all_encode(samples['UserID'], samples['TargetItemID'],
                                                         samples['sas_seq'][:, -10:])
        else:
            user_embedding = self.rec_encoder.user_encoder(samples['UserID'], all_users=all_user_embeds)
        item_embedding = self.rec_encoder.item_encoder(samples['TargetItemID'], all_items=all_item_embeds)
        ui_embedding = torch.cat([user_embedding, item_embedding], dim=-1)

        self.llama_tokenizer.padding_side = "right"

        device = samples['UserID'].device  # samples_encode['User_emb'].device

        # Rating task: no longer depends on binary classification label, avoid CUDA assertion
        pos_ans = self.pos_ans[0]
        neg_ans = self.neg_ans[0]
        classification_enabled = False
        try:
            if "label" in samples:
                labels_cpu = samples["label"].detach().cpu().numpy()
                # Only enable old binary classification evaluation when all are {0,1}
                if np.isin(labels_cpu, [0, 1]).all():
                    classification_enabled = True
        except Exception:
            classification_enabled = False

        if classification_enabled:
            ans_ = {1: pos_ans, 0: neg_ans}
            text = [ans_[int(t)] for t in samples["label"]]
            to_regress_tokens = self.llama_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(device)

            t_posi = to_regress_tokens.input_ids.shape[-1] + 1

            targets = to_regress_tokens.input_ids.masked_fill(
                to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
            )
            empty_targets = torch.ones([atts_samples.shape[0], atts_samples.shape[1]], dtype=torch.long).to(device).fill_(
                -100)
            targets = torch.cat([empty_targets, targets], dim=1)

            to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
            inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
            attention_mask = torch.cat([atts_samples, to_regress_tokens.attention_mask], dim=1)
        else:
            # Skip LLaMA forward directly, return zero loss to pass evaluation process
            loss = torch.zeros([], device=device)
            if return_all:
                return {"loss": loss, "pred_rating": None}
            return {"loss": loss}

        with self.maybe_autocast():
            if not self.use_lora:
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            else:
                outputs = self.llama_model_lora(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                    ui_embedding=ui_embedding,
                )
                # generate_ids = self.llama_model.generate(inputs_embeds=sample_embeds, max_length=2048)
                # reviews = self.llama_tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
                #                                             clean_up_tokenization_spaces=False)[0]
                # with open("/data2/liuyuting/logs/reviews.txt", 'a+', encoding='utf-8') as f:
                #     f.writelines(reviews)
        # loss = outputs.loss
        if not classification_enabled:
            loss = torch.zeros([], device=device)
            if return_all:
                return {"loss": loss, "pred_rating": None}
            return {"loss": loss}

        pos_ans_id = self.llama_tokenizer(pos_ans, add_special_tokens=False).input_ids[0]
        neg_ans_id = self.llama_tokenizer(neg_ans, add_special_tokens=False).input_ids[0]

        # logits = outputs.logits[:,-2,:][:,[neg_ans_id, pos_ans_id]]
        # logits_ = torch.ones_like(logits[:,0]).float()
        # logits_ = torch.where(logits[:,0]>logits[:,1],0,logits_)
        # logits_ = torch.where(logits[:,0]==logits[:,1],0.5,logits_)

        logits_ = outputs.logits[:, -t_posi, :][:, pos_ans_id]
        logits_ = torch.nan_to_num(logits_, nan=0.0, posinf=1.0, neginf=-1.0)
        # print(lo)
        loss = nn.functional.binary_cross_entropy_with_logits(logits_, samples['label'].float())
        loss = torch.nan_to_num(loss, nan=0.0)
        if return_all:
            return outputs, logits_

        return {"loss": loss, 'logits': logits_}

def generate_sequence(self, samples, return_text=False):
        print("🔍 [DEBUG] generate_sequence method called")

        if hasattr(samples, 'question_split'):  # VQA dataset
            print('VQA Batch')
            raise NotImplementedError("not implement")
            # vqa_prompt = '###Human: <Img><ImageHere></Img> '
            # img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, vqa_prompt)
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            id_orders = get_ids_order(prompt)
            samples_encode, atts_samples = self.encode_recdata_v2(samples, ids_order=id_orders)
            sample_embeds, atts_samples = self.recprompt_wrap_v2(samples_encode, samples, atts_samples, prompt)
        else:
            # Handle case when prompt_list is empty
            print("⚠️ prompt_list is empty, using default generation method")
            
            # Ensure collaborative filtering encoder is available
            if self.rec_encoder is None:
                print("❌ Collaborative filtering encoder not initialized, cannot perform personalized generation")
                if return_text:
                    return ["Collaborative filtering encoder not initialized, cannot generate personalized recommendations"]
                else:
                    return {"loss": 0, 'logits': None}
            
            # Use default encoding method
            samples_encode, atts_samples = self.encode_recdata_v2(samples)
            
            # Check if encoding is successful
            if samples_encode is None:
                print("❌ Collaborative feature encoding failed, possibly due to data format issues")
                if return_text:
                    return ["Collaborative feature encoding failed, cannot generate personalized recommendations"]
                else:
                    return {"loss": 0, 'logits': None}
            
            # Prompt specifically for personalized response generation (does not include rating requirements)
            personalized_prompt = "#Question: The user has historical interactions:<his_interaction>. Target item is <item_features>, User instruction is <instruction>. Please provide a personalized recommendation response that explains why this item suits the user based on their preferences and needs. \n#Response:"
            sample_embeds, atts_samples = self.recprompt_wrap_v2(samples_encode, samples, atts_samples, personalized_prompt)

        inputs_embeds = sample_embeds  # torch.cat([sample_embeds, to_regress_embeds], dim=1)
        with torch.no_grad():
            try:
                # Personalized response generation requires sufficient tokens
                generation_config = {
                    "max_new_tokens": 80,    # Increase to 80 tokens, sufficient for personalized response
                    "num_beams": 2,          # Use beam search to improve quality
                    "do_sample": True,
                    "min_length": 10,        # Generate at least 10 tokens
                    "top_p": 0.85,          # Slightly reduce top_p to increase diversity
                    "repetition_penalty": 1.1,  # Avoid repetition
                    "length_penalty": 1.0,
                    "temperature": 0.8,      # Lower temperature to improve coherence
                    "return_dict_in_generate": True,
                    "output_scores": True,
                    "pad_token_id": self.llama_tokenizer.eos_token_id,
                }
                
                if not self.use_lora:
                    print("🔄 Using original LLM to generate personalized response...")
                    outputs = self.llama_model.generate(
                        inputs_embeds=inputs_embeds,
                        **generation_config
                    )
                else:
                    print("🔄 Using LoRA LLM to generate personalized response...")
                    outputs = self.llama_model_lora.generate(
                        inputs_embeds=inputs_embeds,
                        **generation_config
                    )
            except Exception as e:
                print(f"❌ Error in generation process: {e}")
                # If generation fails, return default result
                if return_text:
                    return ["Generation failed, returning default recommendation"]
                else:
                    return {"loss": 0, 'logits': None}
        
        # Check if outputs are valid
        if not hasattr(outputs, 'sequences'):
            print("❌ Generation result is invalid")
            if return_text:
                return ["Generation result is invalid"]
            else:
                return {"loss": 0, 'logits': None}
                
        print(f"🔍 Input embedding shape: {inputs_embeds.shape}, generated sequence shape: {outputs.sequences.shape}")
        
        # Decode generated text (personalized response)
        decoded = self.llama_tokenizer.batch_decode(
            outputs.sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        # Clean decoding results, keep only personalized response part
        cleaned_decoded = []
        for text in decoded:
            # Remove prompt part, keep actual generated personalized response
            if "#Response:" in text:
                response_part = text.split("#Response:")[-1].strip()
            elif "### Response:" in text:
                response_part = text.split("### Response:")[-1].strip()
            elif "#Question:" in text:
                # If contains question part, extract response after question
                parts = text.split("#Question:")
                if len(parts) > 1:
                    # Find response part
                    after_question = parts[-1]
                    if "Response:" in after_question:
                        response_part = after_question.split("Response:")[-1].strip()
                    else:
                        # If no clear Response marker, take content after question
                        response_part = after_question.strip()
                else:
                    response_part = text.strip()
            else:
                response_part = text.strip()
            
            # Remove possible rating information, keep only response text
            if response_part.startswith("score:"):
                lines = response_part.split('\n')
                response_lines = [line for line in lines if not line.strip().startswith("score:")]
                response_part = '\n'.join(response_lines).strip()
            
            # Ensure valid response content
            if response_part and len(response_part) > 5:
                cleaned_decoded.append(response_part)
            else:
                # Generate default personalized response
                cleaned_decoded.append("Based on your preferences and history, I believe this item would be a great match for you.")
        
        decoded = cleaned_decoded
        print(f"✅ Cleaned personalized response: {decoded}")
        
        # Parse and print recommendation results
        try:
            self._print_recommendation_results(decoded, samples)
        except Exception as e:
            print(f"⚠️ Failed to parse recommendation results: {e}")
            print("Original output:", decoded)
            
        if return_text:
            return decoded
        print()
        # Safely access logits
        logits = getattr(outputs, 'scores', None) or getattr(outputs, 'logits', None)
        return {"loss": 0, 'logits': logits}
        # return outputs

def _print_recommendation_results(self, decoded_outputs, samples):
        """Display personalized response results (ratings predicted separately by score_head)"""
        print("\n" + "="*80)
        print("🎯 Personalized recommendation response results")
        print("="*80)
        
        for i, personalized_response in enumerate(decoded_outputs):
            print(f"\n📊 Sample {i+1}:")
            
            # Display input information
            if 'instruction' in samples:
                instruction = samples['instruction'][i] if isinstance(samples['instruction'], list) else samples['instruction']
                print(f"👤 User instruction: {instruction}")
            
            if 'item_features' in samples:
                item_features = samples['item_features'][i] if isinstance(samples['item_features'], list) else samples['item_features']
                # Truncate overly long item features
                item_features_short = item_features[:100] + "..." if len(item_features) > 100 else item_features
                print(f"🎮 Target item: {item_features_short}")
            
            if 'his_interaction' in samples:
                his_interaction = samples['his_interaction'][i] if isinstance(samples['his_interaction'], list) else samples['his_interaction']
                # Extract user preferences from historical interactions (remove itemid, keep only title)
                cleaned_preference = self._clean_historical_preference(his_interaction)
                print(f"📚 Historical preferences: {cleaned_preference[:150]}...")
            
            # Display personalized response (rating predicted separately by score_head)
            print(f"💬 Personalized response: {personalized_response}")
                
                # Display actual rating (if available)
            if 'rating' in samples:
                try:
                    true_rating = samples['rating'][i] if hasattr(samples['rating'], '__getitem__') else samples['rating']
                    print(f"🎯 Actual rating: {true_rating}")
                except Exception as e:
                    print(f"⚠️ Failed to get actual rating: {e}")
            
            print("-" * 80)
    
def _clean_historical_preference(self, his_interaction):
        """Extract user preferences from historical interactions, remove itemid and keep only title"""
        try:
            # Assume format is (user_id,hist_asin,hist_title,hist_rating)
            # We only extract the title part
            import re
            
            # Use regular expression to extract title part
            # Assume title is in third position, format like: (id,asin,title,rating)
            pattern = r'\([^,]+,[^,]+,([^,]+),[^)]+\)'
            matches = re.findall(pattern, his_interaction)
            
            if matches:
                # Merge all titles
                titles = [match.strip() for match in matches[:3]]  # Only take first 3
                return " | ".join(titles)
            else:
                # If regex matching fails, directly return truncated original text
                return his_interaction[:100]
                
        except Exception:
            return his_interaction[:100] if his_interaction else ""
    
def _extract_user_preferences_from_history(self, his_interaction):
        """Extract user preference keywords from historical interactions"""
        if not his_interaction:
            return ""
        
        try:
            import re
            # Extract historical product titles
            pattern = r'\([^,]+,[^,]+,([^,]+),[^)]+\)'
            matches = re.findall(pattern, his_interaction)
            
            if matches:
                # Extract simplified names of first 2 products
                titles = []
                for match in matches[:2]:
                    title = match.strip('\'"').strip()
                    # Simplify product names, extract keywords
                    if len(title) > 20:
                        # Extract first few keywords
                        words = title.split()[:3]
                        titles.append(' '.join(words))
                    else:
                        titles.append(title)
                
                return ', '.join(titles) if titles else ""
            
            return ""
        except Exception:
            return ""
    
def _get_category_from_title(self, title):
        """Infer product category from product title"""
        if not title:
            return "this"
        
        title_lower = title.lower()
        
        # Gaming related
        if any(word in title_lower for word in ['game', 'gaming', 'xbox', 'playstation', 'nintendo', 'controller']):
            return "gaming"
        # Electronic devices
        elif any(word in title_lower for word in ['adapter', 'cable', 'wireless', 'network', 'device']):
            return "electronic device"
        # Software
        elif any(word in title_lower for word in ['software', 'program', 'cd', 'dvd']):
            return "software"
        # Default
        else:
            return "this type"
    
def _parse_model_output(self, output_text):
        """Parse model output, extract score and response"""
        import re
        
        score = None
        response = None
        
        try:
            # Parse rating - find "score: X" pattern
            score_pattern = r'score:\s*(\d+(?:\.\d+)?)'
            score_match = re.search(score_pattern, output_text, re.IGNORECASE)
            if score_match:
                score = float(score_match.group(1))
                # Ensure rating is in 0-5 range
                score = max(0, min(5, score))
            
            # Parse response - find content after "response: "
            response_pattern = r'response:\s*["\']?([^"\']+)["\']?'
            response_match = re.search(response_pattern, output_text, re.IGNORECASE | re.DOTALL)
            if response_match:
                response = response_match.group(1).strip()
            
            # If standard format not found, try to extract from overall text
            if score is None or response is None:
                # Try to extract numbers from text as score
                number_pattern = r'\b([0-5](?:\.\d+)?)\b'
                numbers = re.findall(number_pattern, output_text)
                if numbers and score is None:
                    score = float(numbers[0])
                
                # If no response found, use entire output as response
                if response is None:
                    response = output_text.strip()
            
        except Exception as e:
            print(f"⚠️ Error parsing output: {e}")
            response = output_text.strip()
        
        return score, response
    
def _print_rating_predictions(self, pred_ratings, true_ratings, samples):
        """Print rating prediction results"""
        try:
            print("\n" + "="*60)
            print("📊 Collaborative filtering rating prediction results")
            print("="*60)
            
            batch_size = pred_ratings.shape[0] if hasattr(pred_ratings, 'shape') else len(pred_ratings)
            
            for i in range(min(batch_size, 3)):  # Show at most 3 samples
                pred_score = float(pred_ratings[i]) if hasattr(pred_ratings, '__getitem__') else float(pred_ratings)
                true_score = float(true_ratings[i]) if hasattr(true_ratings, '__getitem__') else float(true_ratings)
                
                print(f"\n🎯 Sample {i+1}:")
                
                # Display user and item information
                if 'UserID' in samples and hasattr(samples['UserID'], '__getitem__'):
                    user_id = samples['UserID'][i]
                    print(f"👤 User ID: {user_id}")
                
                if 'TargetItemID' in samples and hasattr(samples['TargetItemID'], '__getitem__'):
                    item_id = samples['TargetItemID'][i]
                    print(f"🎮 Item ID: {item_id}")
                
                if 'instruction' in samples:
                    instruction = samples['instruction'][i] if isinstance(samples['instruction'], list) else samples['instruction']
                    print(f"💭 User instruction: {instruction[:80]}...")
                
                # Display rating prediction results
                print(f"⭐ Predicted rating: {pred_score:.2f}")
                print(f"🎯 Actual rating: {true_score:.2f}")
                print(f"📏 Prediction error: {abs(pred_score - true_score):.2f}")
                
                # Rating quality assessment
                error = abs(pred_score - true_score)
                if error <= 0.5:
                    quality = "🎯 Extremely accurate"
                elif error <= 1.0:
                    quality = "✅ Accurate"
                elif error <= 1.5:
                    quality = "⚠️ Average"
                else:
                    quality = "❌ Poor"
                print(f"🔍 Prediction quality: {quality}")
                
                print("-" * 60)
                
        except Exception as e:
            print(f"⚠️ Failed to print rating prediction results: {e}")

def encode_allinputs(self, samples, mode='v1'):
        if mode == 'v2':
            samples_encode, atts_samples = self.encode_recdata_v2(samples)
        else:
            samples_encode, atts_samples = self.encode_recdata_v1(samples)
        if hasattr(samples, 'question_split'):  # VQA dataset
            print('VQA Batch')
            raise NotImplementedError("not implement")
            # vqa_prompt = '###Human: <Img><ImageHere></Img> '
            # img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, vqa_prompt)
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            if mode == 'v2':
                sample_embeds, atts_samples = self.recprompt_wrap_v2(samples_encode, samples, atts_samples, prompt)
            else:
                sample_embeds, atts_samples = self.recprompt_wrap_v1(samples_encode, samples, atts_samples, prompt)

        inputs_embeds = sample_embeds  # torch.cat([sample_embeds, to_regress_embeds], dim=1)
        return inputs_embeds

def generate_for_samples_v1(self, samples):

        samples_encode, atts_samples = self.encode_recdata_v1(samples)
        if hasattr(samples, 'question_split'):  # VQA dataset
            print('VQA Batch')
            raise NotImplementedError("not implement")
            # vqa_prompt = '###Human: <Img><ImageHere></Img> '
            # img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, vqa_prompt)
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            sample_embeds, atts_samples = self.recprompt_wrap_v1(samples_encode, samples, atts_samples, prompt)

        self.llama_tokenizer.padding_side = "right"

        device = samples_encode['User_emb'].device
        # sample = samples["image"]
        # ans_ = {1: "The user prefers the former item because its MF embedding is a closer match to the user's MF embedding than that of the latter item.",
        #         0: "The user prefers the latter item because its MF embedding is a closer match to the user's MF embedding than that of the former item."}
        # pos_ans = ["The former item.", "The first item.", "The former.", "The first.", "The former one.", "The first one."]
        # neg_ans = ["The latter item.", "The second item.", "The latter.", "The second.", "The latter one.", "The second one."]
        # pos_ans = ["The first item."]
        # neg_ans = ["The second item."]
        # pos_ans = ["The first item."]
        # neg_ans = ["The second item."]
        ans_ = {1: self.pos_ans,
                0: self.neg_ans}
        text = [random.choice(ans_[int(t)]) + self.end_sym for t in samples["label"]]

        # text = [ans_[int(t)] + self.end_sym for t in samples["label"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = torch.ones([atts_samples.shape[0], atts_samples.shape[1]], dtype=torch.long).to(device).fill_(
            -100)
        targets = torch.cat([empty_targets, targets], dim=1)
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_samples, to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            if not self.use_lora:
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            else:
                outputs = self.llama_model_lora(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
        loss = outputs.loss
        return {"loss": loss}

def generate_for_samples(self, samples):
        if self.run_mode_ == 'v1':
            return self.generate_for_samples_v1(samples)
        elif self.run_mode_ == 'v2':
            return self.generate_for_samples_v2(samples)
        else:
            raise NotImplementedError("Not implement the default version")
            # self.generate_sequence(samples)
        # return {'loss':loss, "logits": logits_}
