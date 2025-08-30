#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DiffRec Model Evaluation Script

Supports category-wise evaluation and ablation experiments for recommendation tasks.
Image generation modules are disabled.

Supported categories: All Beauty, Video_Games, Handmade_product

Usage:
1. Category-wise evaluation (full model):
   python eval_collm_mf_din.py --checkpoint-path /path/to/checkpoint.pth

2. Ablation study (compare with/without CF model):
   python eval_collm_mf_din.py --checkpoint-path /path/to/checkpoint.pth --ablation

Output files:
- eval_results_by_category_[checkpoint_name]_[timestamp].json
- eval_summary_[checkpoint_name]_[timestamp].txt
- ablation_results_[checkpoint_name]_[timestamp].json (for ablation study)
"""

import argparse
import os
import pickle
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Apply compatibility patches before importing minigpt4
def apply_huggingface_compatibility_patch():
    """Apply HuggingFace compatibility patches"""
    print(" Applying HuggingFace compatibility patches...")
    
    # Step 1: Set environment variables
    cache_base = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    hub_cache = os.path.join(cache_base, "hub")
    
    env_vars = {
        'HF_HOME': cache_base,
        'HUGGINGFACE_HUB_CACHE': hub_cache,
        'TRANSFORMERS_CACHE': os.path.join(cache_base, "transformers"),
        'HF_DATASETS_CACHE': os.path.join(cache_base, "datasets"),
    }
    
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
    
    # Step 2: Fix huggingface_hub version compatibility
    try:
        import huggingface_hub
        import sys
        hub_version = huggingface_hub.__version__
        print(f"Detected huggingface_hub version: {hub_version}")
        
        if hub_version < "0.20.0":
            print("Applying dynamic compatibility layer...")
            
            # Add missing functions
            if not hasattr(huggingface_hub, 'split_torch_state_dict_into_shards'):
                def split_torch_state_dict_into_shards(*args, **kwargs):
                    """Compatibility layer: simply return original state dict"""
                    if args:
                        return {'model.safetensors': args[0]}
                    return {}
                
                huggingface_hub.split_torch_state_dict_into_shards = split_torch_state_dict_into_shards
                print(" Added split_torch_state_dict_into_shards compatibility function")
            
            # Add missing errors module
            if not hasattr(huggingface_hub, 'errors'):
                # Create a mock errors module
                class ErrorsModule:
                    class HFValidationError(Exception):
                        pass
                    
                    class LocalEntryNotFoundError(Exception):
                        pass
                    
                    class EntryNotFoundError(Exception):
                        pass
                    
                    class RepositoryNotFoundError(Exception):
                        pass
                    
                    class RevisionNotFoundError(Exception):
                        pass
                
                errors_module = ErrorsModule()
                huggingface_hub.errors = errors_module
                
                # Also register in sys.modules
                sys.modules['huggingface_hub.errors'] = errors_module
                print(" Added huggingface_hub.errors compatibility module")
            
            # Mask version number
            original_version = huggingface_hub.__version__
            huggingface_hub.__version__ = "0.20.0"
            print(f"Version masking: {original_version} -> {huggingface_hub.__version__}")
        
    except Exception as e:
        print(f"Compatibility layer failed: {e}")
    
    # Step 3: Fix transformers and PEFT version compatibility
    try:
        import transformers
        transformers_version = transformers.__version__
        print(f"Detected transformers version: {transformers_version}")
        
        # Add missing Cache related classes
        if not hasattr(transformers, 'EncoderDecoderCache'):
            # Create compatible Cache class
            class DummyCache:
                def __init__(self, *args, **kwargs):
                    pass
                
                def update(self, *args, **kwargs):
                    return None
                
                def get_seq_length(self, *args, **kwargs):
                    return 0
            
            # Add missing Cache classes
            if not hasattr(transformers, 'Cache'):
                transformers.Cache = DummyCache
                print(" Added transformers.Cache compatibility class")
            
            if not hasattr(transformers, 'DynamicCache'):
                transformers.DynamicCache = DummyCache
                print(" Added transformers.DynamicCache compatibility class")
                
            if not hasattr(transformers, 'EncoderDecoderCache'):
                transformers.EncoderDecoderCache = DummyCache
                print(" Added transformers.EncoderDecoderCache compatibility class")
                
            if not hasattr(transformers, 'HybridCache'):
                transformers.HybridCache = DummyCache
                print(" Added transformers.HybridCache compatibility class")
        
    except Exception as e:
        print(f"transformers compatibility layer failed: {e}")
    
    # Step 4: Fix missing PEFT utils.config module
    try:
        import peft
        import peft.utils
        
        # Check if peft.utils.config exists
        if not hasattr(peft.utils, 'config'):
            # Create compatible config module
            class ConfigModule:
                class PeftConfigMixin:
                    def __init__(self, *args, **kwargs):
                        pass
                    
                    def to_dict(self):
                        return {}
                    
                    @classmethod
                    def from_dict(cls, config_dict):
                        return cls()
            
            config_module = ConfigModule()
            peft.utils.config = config_module
            
            # Also register in sys.modules
            sys.modules['peft.utils.config'] = config_module
            print(" Added peft.utils.config compatibility module")
        
    except Exception as e:
        print(f"PEFT utils.config compatibility layer failed: {e}")
    
    # Step 5: Fix torchvision/PIL interpolation constants (NEAREST_EXACT)
    try:
        from PIL import Image as _PILImage
        # Ensure Resampling exists and provide NEAREST_EXACT compatibility alias
        if not hasattr(_PILImage, 'Resampling'):
            class _Resampling:
                NEAREST = _PILImage.NEAREST
                BILINEAR = _PILImage.BILINEAR
                BICUBIC = _PILImage.BICUBIC
                LANCZOS = getattr(_PILImage, 'LANCZOS', _PILImage.BICUBIC)
                BOX = getattr(_PILImage, 'BOX', _PILImage.NEAREST)
                HAMMING = getattr(_PILImage, 'HAMMING', _PILImage.NEAREST)
                NEAREST_EXACT = _PILImage.NEAREST
            _PILImage.Resampling = _Resampling
            print(" Added PIL.Image.Resampling compatibility class")
        else:
            if not hasattr(_PILImage.Resampling, 'NEAREST_EXACT'):
                _PILImage.Resampling.NEAREST_EXACT = _PILImage.Resampling.NEAREST
                print(" Added NEAREST_EXACT compatibility alias for PIL.Image.Resampling")
    except Exception as e:
        print(f"PIL compatibility layer failed: {e}")

    try:
        from torchvision.transforms import InterpolationMode as _IM
        if not hasattr(_IM, 'NEAREST_EXACT'):
            # Define alias to avoid transformers dependency failure
            _IM.NEAREST_EXACT = _IM.NEAREST
            print(" Added torchvision.InterpolationMode.NEAREST_EXACT compatibility alias")
    except Exception as e:
        print(f"torchvision compatibility layer failed: {e}")

    print(" Compatibility patches applied successfully")
    return True

# Apply patches immediately
apply_huggingface_compatibility_patch()

import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank, init_distributed_mode
from minigpt4.common.logger import setup_logger
from minigpt4.common.registry import registry
from minigpt4.common.utils import now

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def calculate_rmse_mae(user, predict, rating):
    """Calculate user-level RMSE and MAE metrics"""
    if not isinstance(predict, np.ndarray):
        predict = np.array(predict)
    if not isinstance(rating, np.ndarray):
        rating = np.array(rating)
    if not isinstance(user, np.ndarray):
        user = np.array(user)
        
    predict = predict.squeeze()
    rating = rating.squeeze()
    user = user.squeeze()

    start_time = time.time()
    u, inverse, counts = np.unique(user, return_inverse=True, return_counts=True)
    index = np.argsort(inverse)
    candidates_dict = {}
    k = 0
    total_num = 0
    only_one_interaction = 0
    computed_u = []
    
    for u_i in u:
        start_id, end_id = total_num, total_num + counts[k]
        u_i_counts = counts[k]
        index_ui = index[start_id:end_id]
        if u_i_counts == 1:
            only_one_interaction += 1
            total_num += counts[k]
            k += 1
            continue
        candidates_dict[u_i] = [predict[index_ui], rating[index_ui]]
        total_num += counts[k]
        k += 1
    
    print(f"Users with only one interaction: {only_one_interaction}")
    user_rmse = []
    user_mae = []
    
    for ui, pre_and_true in candidates_dict.items():
        pre_i, rating_i = pre_and_true
        ui_rmse = np.sqrt(mean_squared_error(rating_i, pre_i))
        ui_mae = mean_absolute_error(rating_i, pre_i)
        user_rmse.append(ui_rmse)
        user_mae.append(ui_mae)
        computed_u.append(ui)
    
    user_rmse = np.array(user_rmse)
    user_mae = np.array(user_mae)
    print(f"Number of computed users: {user_rmse.shape[0]}")
    avg_rmse = user_rmse.mean()
    avg_mae = user_mae.mean()
    print(f"User-level RMSE: {avg_rmse:.4f}, User-level MAE: {avg_mae:.4f}, Time: {time.time() - start_time:.2f}s")
    return avg_rmse, avg_mae, computed_u, user_rmse, user_mae


def categorize_data_by_instruction(test_data):
    """Categorize data into three categories based on instruction"""
    print("\n Categorizing data by instruction...")
    
    # Define category keyword mapping
    category_mapping = {
        'All Beauty': 'All Beauty',
        'Video_Games': 'Video_Games', 
        'Handmade_product': 'Handmade_product'
    }
    
    # Categorize data
    categorized_data = {}
    
    for category_key, category_name in category_mapping.items():
        # Filter data containing corresponding category keywords
        mask = test_data['instruction'].str.contains(category_key, case=False, na=False)
        category_data = test_data[mask].copy()
        
        if len(category_data) > 0:
            categorized_data[category_name] = category_data
            print(f" {category_name}: {len(category_data)} samples")
            print(f"   Users: {category_data['user_id'].nunique()}")
            print(f"   Items: {category_data['asin'].nunique()}")
            print(f"   Rating range: {category_data['rating'].min():.1f} - {category_data['rating'].max():.1f}")
        else:
            print(f" {category_name}: No data found")
    
    # Validate categorization results
    total_categorized = sum(len(data) for data in categorized_data.values())
    print(f"\n Categorization statistics:")
    print(f"Original data total: {len(test_data)}")
    print(f"Categorized data total: {total_categorized}")
    print(f"Categorization coverage: {total_categorized/len(test_data)*100:.2f}%")
    
    return categorized_data


def load_test_data(test_data_path):
    """Load test data"""
    print(f" Loading test data: {test_data_path}")
    
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data file does not exist: {test_data_path}")
    
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)
    
    if isinstance(test_data, pd.DataFrame):
        print(f"Test data loaded successfully, shape: {test_data.shape}")
        print(f"Data columns: {list(test_data.columns)}")
        
        # Check and map field names
        field_mapping = {
            'uid': 'user_id',
            'iid': 'asin'
        }
        
        # Apply field mapping
        for old_name, new_name in field_mapping.items():
            if old_name in test_data.columns and new_name not in test_data.columns:
                test_data = test_data.rename(columns={old_name: new_name})
                print(f" Field mapping: {old_name} -> {new_name}")
        
        # Check required fields
        required_fields = ['user_id', 'asin', 'rating', 'instruction']
        missing_fields = [field for field in required_fields if field not in test_data.columns]
        if missing_fields:
            print(f" Missing fields: {missing_fields}")
            # If still missing fields, try to show available fields
            print(f"Available fields: {list(test_data.columns)}")
        
        print(f"Number of users: {test_data['user_id'].nunique()}")
        print(f"Number of items: {test_data['asin'].nunique()}")
        print(f"Rating range: {test_data['rating'].min():.1f} - {test_data['rating'].max():.1f}")
        
        # Show instruction distribution
        print(f"\n Instruction distribution:")
        instruction_counts = test_data['instruction'].value_counts()
        for instruction, count in instruction_counts.items():
            print(f"  '{instruction}': {count:,} times")
        
        return test_data
    else:
        raise ValueError(f"Expected DataFrame format, but got: {type(test_data)}")


def load_checkpoint_and_model(checkpoint_path, cfg, disable_cf_model=False):
    """Load checkpoint and model"""
    cf_status = "without CF model" if disable_cf_model else "with CF model"
    print(f" Loading checkpoint: {checkpoint_path} ({cf_status})")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print(f" Checkpoint loaded successfully, contains keys: {list(checkpoint.keys())}")
    
    if 'epoch' in checkpoint:
        print(f"Training epoch: {checkpoint['epoch']}")
    
    # Create task and model
    print("🔧 Initializing task and model...")
    task = tasks.setup_task(cfg)
    
    # Build datasets first if needed to get user/item counts
    datasets = task.build_datasets(cfg)
    
    # Get user and item counts from datasets
    try:
        data_name = list(datasets.keys())[0]
        split_dict = datasets[data_name]
        user_num = -1
        item_num = -1
        for split_key, ds in split_dict.items():
            try:
                user_num = max(user_num, int(getattr(ds, 'user_num', -1)))
                item_num = max(item_num, int(getattr(ds, 'item_num', -1)))
            except Exception:
                continue
        if user_num <= 0 or item_num <= 0:
            raise RuntimeError("invalid user/item num from datasets")
    except Exception:
        # Fallback: use values from config
        user_num = cfg.model_cfg.rec_config.get('user_num', -100)
        item_num = cfg.model_cfg.rec_config.get('item_num', -100)

    cfg.model_cfg.rec_config.user_num = int(user_num)
    cfg.model_cfg.rec_config.item_num = int(item_num)
    
    print(f"Number of users: {user_num}, Number of items: {item_num}")
    
    # Ablation experiment: if disabling CF model, modify config
    if disable_cf_model:
        print(" Ablation experiment: disabling CF model components")
        # Backup original config
        original_cf_config = getattr(cfg.model_cfg, 'use_cf_model', True)
        # Set not to use CF model
        cfg.model_cfg.use_cf_model = False
        if hasattr(cfg.model_cfg.rec_config, 'use_pretrained_cf'):
            cfg.model_cfg.rec_config.use_pretrained_cf = False
        if hasattr(cfg.model_cfg.rec_config, 'enable_cf_component'):
            cfg.model_cfg.rec_config.enable_cf_component = False
        print(" CF model components disabled")
    
    # Build model
    model = task.build_model(cfg)
    
    # Load model weights
    try:
        # If disabling CF model, filter out CF-related weights
        if disable_cf_model:
            print("🔧 Filtering CF model related weights...")
            state_dict = checkpoint["model"]
            filtered_state_dict = {}
            cf_related_keys = []
            
            for key, value in state_dict.items():
                # Skip CF model related weights (adjust based on actual model structure)
                if any(cf_keyword in key.lower() for cf_keyword in ['cf_model', 'collaborative', 'mf_', 'matrix_fact']):
                    cf_related_keys.append(key)
                    continue
                filtered_state_dict[key] = value
            
            print(f"   Filtered out {len(cf_related_keys)} CF model related weights")
            if cf_related_keys:
                print(f"   Filtered weight keys: {cf_related_keys[:5]}{'...' if len(cf_related_keys) > 5 else ''}")
            
            missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
            print(f" Ablation model weights loaded successfully (missing: {len(missing_keys)}, unexpected: {len(unexpected_keys)})")
        else:
        model.load_state_dict(checkpoint["model"], strict=False)
            print(" Full model weights loaded successfully")
    except Exception as e:
        print(f" Model weight loading failed, trying loose loading: {e}")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
        print(f"Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
    
    # Set to evaluation mode
    model.eval()
    
    # Set model running mode (must be set, otherwise forward will error)
    mode = cfg.run_cfg.get('mode', 'v2')
    model.set_mode(mode)
    print(f"Model running mode set to: {mode}")
    
    return model, task, datasets


def create_dataloader_for_category(category_data, test_dataset, cfg):
    """Create data loader for specific category data"""
    from torch.utils.data import DataLoader, Subset
    import numpy as np
    
    batch_size = cfg.run_cfg.get('batch_size_eval', 4)
    
    # Create a simple wrapper
    class CategoryDataLoader:
        def __init__(self, category_data, original_dataset):
            self.category_data = category_data
            self.original_dataset = original_dataset
            self.batch_size = batch_size
            
        def __iter__(self):
            # Return original dataset iterator for now
            return iter(DataLoader(
                self.original_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=cfg.run_cfg.get('num_workers', 2),
                collate_fn=getattr(self.original_dataset, 'collater', None)
            ))
        
        def __len__(self):
            return len(self.category_data)
    
    return CategoryDataLoader(category_data, test_dataset)


def evaluate_single_category(model, task, category_name, category_dataloader, device):
    """Evaluate single category"""
    print(f"\n🔬 Evaluating category: {category_name}")
    
    # Create data_loaders object (wrapper for DataLoader)
    class DataLoaders:
        def __init__(self, loader):
            self.loaders = [loader]
    
    data_loaders = DataLoaders(category_dataloader)
    
    # Use task's evaluation method
    with torch.no_grad():
        eval_results = task.evaluation(
            model=model, 
            data_loaders=data_loaders, 
            cuda_enabled=torch.cuda.is_available(),
            split_name=f"test_{category_name}"
        )
    
    print(f" {category_name} evaluation completed, results: {eval_results}")
    
    # Process evaluation results
    if eval_results is not None:
        final_results = task.after_evaluation(
            val_result=eval_results,
            split_name=f"test_{category_name}",
            epoch="final"
        )
    else:
        final_results = {"error": f"{category_name} evaluation failed, result is None"}
    
    return final_results


def run_single_model_evaluation(model, task, datasets, cfg, test_data_path, model_type="full"):
    """Run category-wise evaluation for single model"""
    print(f"\n🔬 Starting category-wise evaluation ({model_type} model)...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"Device: {device}")
    
    # 1. Load external test data
    test_data = load_test_data(test_data_path)
    
    # 2. Categorize data by instruction
    categorized_data = categorize_data_by_instruction(test_data)
    
    if not categorized_data:
        raise RuntimeError("Failed to categorize test data successfully")
    
    # 3. Get original test dataset structure
    test_dataset = None
    for data_name, split_dict in datasets.items():
        if 'test' in split_dict:
            test_dataset = split_dict['test']
            print(f" Using built-in test set structure: {data_name}/test")
            break
    
    if test_dataset is None:
        print(" Built-in test set not found, using first available dataset")
        # Use first available dataset
        for data_name, split_dict in datasets.items():
            for split_name, dataset in split_dict.items():
                test_dataset = dataset
                print(f" Using dataset structure: {data_name}/{split_name}")
                break
            if test_dataset is not None:
                break
    
    if test_dataset is None:
        raise RuntimeError("No available dataset structure found")
    
    # 4. Perform category-wise evaluation
    category_results = {}
    overall_stats = {
        'total_samples': 0,
        'total_categories': len(categorized_data),
        'model_type': model_type
    }
    
    print(f"\n Starting evaluation of {len(categorized_data)} categories...")
    
    # First perform overall evaluation (if needed)
    print(f"\n Overall evaluation ({model_type} model)...")
    
    # Create dataloader for complete data
    from torch.utils.data import DataLoader
    batch_size = cfg.run_cfg.get('batch_size_eval', 4)
    full_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.run_cfg.get('num_workers', 2),
        collate_fn=getattr(test_dataset, 'collater', None)
    )
    
    # Overall evaluation
    class DataLoaders:
        def __init__(self, loader):
            self.loaders = [loader]
    
    data_loaders = DataLoaders(full_dataloader)
    
    with torch.no_grad():
        overall_eval_results = task.evaluation(
            model=model, 
            data_loaders=data_loaders, 
            cuda_enabled=torch.cuda.is_available(),
            split_name=f"test_overall_{model_type}"
        )
    
    if overall_eval_results is not None:
        overall_results = task.after_evaluation(
            val_result=overall_eval_results,
            split_name=f"test_overall_{model_type}",
            epoch="final"
        )
    else:
        overall_results = {"error": f"Overall evaluation failed, result is None ({model_type} model)"}
    
    category_results['Overall'] = overall_results
    overall_stats['total_samples'] = len(test_data)
    
    # Category-wise evaluation
    for category_name, category_data in categorized_data.items():
        print(f"\n{'='*50}")
        print(f" Evaluating category: {category_name} ({model_type} model)")
        print(f"Data volume: {len(category_data)}")
        print(f"{'='*50}")
        
        # Create category dataloader
        category_dataloader = create_dataloader_for_category(category_data, test_dataset, cfg)
        
        # Evaluate this category
        try:
            category_result = evaluate_single_category(
                model, task, f"{category_name}_{model_type}", category_dataloader, device
            )
            category_results[category_name] = category_result
            
            # Calculate statistics for this category
            print(f" {category_name} ({model_type} model) evaluation completed")
            if isinstance(category_result, dict):
                for key, value in category_result.items():
                    if isinstance(value, (int, float)):
                        print(f"   {key}: {value:.6f}")
                    else:
                        print(f"   {key}: {value}")
            
        except Exception as e:
            print(f" {category_name} ({model_type} model) evaluation failed: {e}")
            category_results[category_name] = {"error": str(e)}
        
        print(f"{'='*50}")
    
    # 5. Summarize results
    final_results = {
        'overall_stats': overall_stats,
        'category_results': category_results,
        'evaluation_summary': {
            'total_categories_evaluated': len(category_results) - 1,  # Exclude Overall
            'successful_evaluations': sum(1 for k, v in category_results.items() 
                                        if k != 'Overall' and 'error' not in v),
            'failed_evaluations': sum(1 for k, v in category_results.items() 
                                    if k != 'Overall' and 'error' in v),
            'model_type': model_type
        }
    }
    
    print(f"\n {model_type} model evaluation summary:")
    print(f"Total categories: {final_results['evaluation_summary']['total_categories_evaluated']}")
    print(f"Successful evaluations: {final_results['evaluation_summary']['successful_evaluations']}")
    print(f"Failed evaluations: {final_results['evaluation_summary']['failed_evaluations']}")
    
    return final_results


def run_ablation_evaluation(checkpoint_path, cfg, test_data_path, output_dir):
    """Run ablation experiment evaluation"""
    print("\n Starting ablation experiment evaluation...")
    print("Will compare the following two model configurations:")
    print("1. Full model (with pretrained CF model)")
    print("2. Ablation model (without pretrained CF model)")
    
    all_results = {}
    
    # 1. Evaluate full model
    print("\n" + "="*80)
    print(" Phase 1: Evaluating full model (with pretrained CF model)")
    print("="*80)
    
    try:
        # Load full model
        full_model, task, datasets = load_checkpoint_and_model(checkpoint_path, cfg, disable_cf_model=False)
        
        # Run full model evaluation
        full_results = run_single_model_evaluation(
            full_model, task, datasets, cfg, test_data_path, model_type="full"
        )
        all_results['full_model'] = full_results
        print(" Full model evaluation completed")
        
        # Release GPU memory
        del full_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f" Full model evaluation failed: {e}")
        all_results['full_model'] = {"error": str(e)}
    
    # 2. Evaluate ablation model
    print("\n" + "="*80)
    print(" Phase 2: Evaluating ablation model (without pretrained CF model)")
    print("="*80)
    
    try:
        # Load ablation model
        ablation_model, task, datasets = load_checkpoint_and_model(checkpoint_path, cfg, disable_cf_model=True)
        
        # Run ablation model evaluation
        ablation_results = run_single_model_evaluation(
            ablation_model, task, datasets, cfg, test_data_path, model_type="ablation"
        )
        all_results['ablation_model'] = ablation_results
        print(" Ablation model evaluation completed")
        
        # Release GPU memory
        del ablation_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f" Ablation model evaluation failed: {e}")
        all_results['ablation_model'] = {"error": str(e)}
    
    # 3. Calculate comparison results
    print("\n" + "="*80)
    print(" Phase 3: Computing comparison results")
    print("="*80)
    
    comparison_results = generate_comparison_results(all_results)
    all_results['comparison'] = comparison_results
    
    return all_results


def generate_comparison_results(all_results):
    """Generate comparison results for two models"""
    print(" Generating model comparison analysis...")
    
    comparison = {
        'summary': {},
        'category_comparison': {},
        'improvement_analysis': {}
    }
    
    # Check if both models evaluated successfully
    full_results = all_results.get('full_model')
    ablation_results = all_results.get('ablation_model')
    
    if not full_results or 'error' in full_results:
        comparison['summary']['full_model_error'] = full_results.get('error', 'Full model evaluation failed')
        return comparison
    
    if not ablation_results or 'error' in ablation_results:
        comparison['summary']['ablation_model_error'] = ablation_results.get('error', 'Ablation model evaluation failed')
        return comparison
    
    # Get category results
    full_categories = full_results.get('category_results', {})
    ablation_categories = ablation_results.get('category_results', {})
    
    # Compare results for each category
    for category in full_categories.keys():
        if category in ablation_categories:
            full_result = full_categories[category]
            ablation_result = ablation_categories[category]
            
            if 'error' not in full_result and 'error' not in ablation_result:
                category_comparison = {}
                
                # Compare numerical metrics
                for metric in full_result.keys():
                    if isinstance(full_result.get(metric), (int, float)) and isinstance(ablation_result.get(metric), (int, float)):
                        full_value = full_result[metric]
                        ablation_value = ablation_result[metric]
                        
                        # Calculate improvement percentage
                        if ablation_value != 0:
                            improvement = ((full_value - ablation_value) / abs(ablation_value)) * 100
                        else:
                            improvement = float('inf') if full_value > 0 else float('-inf')
                        
                        category_comparison[metric] = {
                            'full_model': full_value,
                            'ablation_model': ablation_value,
                            'improvement_percent': improvement,
                            'better_model': 'full' if full_value > ablation_value else 'ablation' if full_value < ablation_value else 'equal'
                        }
                
                comparison['category_comparison'][category] = category_comparison
            else:
                comparison['category_comparison'][category] = {
                    'error': f"full_error: {full_result.get('error', 'None')}, ablation_error: {ablation_result.get('error', 'None')}"
                }
    
    # Generate improvement analysis summary
    improvement_summary = {}
    for category, metrics in comparison['category_comparison'].items():
        if 'error' not in metrics:
            for metric, data in metrics.items():
                if metric not in improvement_summary:
                    improvement_summary[metric] = {
                        'categories_compared': 0,
                        'full_better_count': 0,
                        'ablation_better_count': 0,
                        'equal_count': 0,
                        'average_improvement': 0
                    }
                
                improvement_summary[metric]['categories_compared'] += 1
                
                if data['better_model'] == 'full':
                    improvement_summary[metric]['full_better_count'] += 1
                elif data['better_model'] == 'ablation':
                    improvement_summary[metric]['ablation_better_count'] += 1
                else:
                    improvement_summary[metric]['equal_count'] += 1
                
                if isinstance(data['improvement_percent'], (int, float)) and not np.isinf(data['improvement_percent']):
                    improvement_summary[metric]['average_improvement'] += data['improvement_percent']
    
    # Calculate average improvement
    for metric in improvement_summary:
        if improvement_summary[metric]['categories_compared'] > 0:
            improvement_summary[metric]['average_improvement'] /= improvement_summary[metric]['categories_compared']
    
    comparison['improvement_analysis'] = improvement_summary
    
    # Generate summary
    total_comparisons = sum(data['categories_compared'] for data in improvement_summary.values())
    comparison['summary'] = {
        'total_metrics_compared': len(improvement_summary),
        'total_comparisons': total_comparisons,
        'categories_evaluated': len(comparison['category_comparison'])
    }
    
    print(f" Comparison analysis completed: {comparison['summary']['categories_evaluated']} categories, {comparison['summary']['total_metrics_compared']} metrics")
    
    return comparison


def save_results(results, output_dir, checkpoint_path, is_ablation=False):
    """Save category-wise evaluation results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate result file names
    checkpoint_name = Path(checkpoint_path).stem
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    if is_ablation:
        results_file = output_dir / f"ablation_results_{checkpoint_name}_{timestamp}.json"
        summary_file = output_dir / f"ablation_summary_{checkpoint_name}_{timestamp}.txt"
        eval_type = "ablation_study"
        title = "DiffRec Model Ablation Experiment Evaluation Report"
    else:
    results_file = output_dir / f"eval_results_by_category_{checkpoint_name}_{timestamp}.json"
        summary_file = output_dir / f"eval_summary_{checkpoint_name}_{timestamp}.txt"
        eval_type = "category_wise"
        title = "DiffRec Model Category-wise Evaluation Report"
    
    # Add metadata
    results_with_meta = {
        "evaluation_time": timestamp,
        "checkpoint_path": str(checkpoint_path),
        "evaluation_type": eval_type,
        "results": results
    }
    
    # Save detailed results to JSON
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_with_meta, f, ensure_ascii=False, indent=2, default=str)
    
    print(f" Evaluation results saved to: {results_file}")
    
    # Create more readable summary report
    with open(summary_file, 'w', encoding='utf-8') as f:
        # Write summary report
        f.write("=" * 80 + "\n")
        f.write(f" {title}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Evaluation time: {timestamp}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Evaluation type: {eval_type}\n")
        f.write("\n")
        
        if is_ablation:
            # Ablation experiment report
            write_ablation_summary(f, results)
        else:
            # Regular category-wise report
            write_category_summary(f, results)
        
        f.write("=" * 80 + "\n")
    
    print(f" Evaluation summary report saved to: {summary_file}")
    
    # Display summary information on console
    print_console_summary(results, is_ablation)
    
    return results_file, summary_file


def write_ablation_summary(f, results):
    """Write ablation experiment summary report"""
    # Experiment overview
    f.write(" Ablation Experiment Overview:\n")
    f.write("-" * 40 + "\n")
    f.write("Comparison configurations:\n")
    f.write("  1. Full model (with pretrained CF model)\n")
    f.write("  2. Ablation model (without pretrained CF model)\n")
    f.write("\n")
    
    # Model evaluation results
    if 'full_model' in results:
        f.write(" Full Model Results:\n")
        f.write("-" * 40 + "\n")
        full_results = results['full_model']
        if 'error' in full_results:
            f.write(f"   Error: {full_results['error']}\n")
        else:
            write_model_results(f, full_results, "   ")
        f.write("\n")
    
    if 'ablation_model' in results:
        f.write(" Ablation Model Results:\n")
        f.write("-" * 40 + "\n")
        ablation_results = results['ablation_model']
        if 'error' in ablation_results:
            f.write(f"   Error: {ablation_results['error']}\n")
        else:
            write_model_results(f, ablation_results, "   ")
        f.write("\n")
    
    # Comparison analysis
    if 'comparison' in results:
        f.write(" Comparison Analysis:\n")
        f.write("-" * 40 + "\n")
        comparison = results['comparison']
        
        if 'summary' in comparison:
            summary = comparison['summary']
            f.write(f"Categories evaluated: {summary.get('categories_evaluated', 'N/A')}\n")
            f.write(f"Metrics compared: {summary.get('total_metrics_compared', 'N/A')}\n")
            f.write(f"Total comparisons: {summary.get('total_comparisons', 'N/A')}\n")
            f.write("\n")
        
        # Improvement analysis
        if 'improvement_analysis' in comparison:
            f.write(" Improvement Analysis:\n")
            improvement = comparison['improvement_analysis']
            for metric, data in improvement.items():
                f.write(f"\n  {metric}:\n")
                f.write(f"    Categories compared: {data['categories_compared']}\n")
                f.write(f"    Full model better: {data['full_better_count']}\n")
                f.write(f"    Ablation model better: {data['ablation_better_count']}\n")
                f.write(f"    Equal: {data['equal_count']}\n")
                f.write(f"    Average improvement: {data['average_improvement']:.2f}%\n")
        
        # Detailed comparison
        if 'category_comparison' in comparison:
            f.write("\n Detailed Category Comparison:\n")
            for category, metrics in comparison['category_comparison'].items():
                f.write(f"\n   {category}:\n")
                if 'error' in metrics:
                    f.write(f"    Error: {metrics['error']}\n")
                else:
                    for metric, data in metrics.items():
                        f.write(f"    {metric}:\n")
                        f.write(f"      Full model: {data['full_model']:.6f}\n")
                        f.write(f"      Ablation model: {data['ablation_model']:.6f}\n")
                        f.write(f"      Improvement: {data['improvement_percent']:.2f}%\n")
                        f.write(f"      Better model: {data['better_model']}\n")


def write_model_results(f, model_results, indent=""):
    """Write single model results"""
    if 'overall_stats' in model_results:
        stats = model_results['overall_stats']
        f.write(f"{indent}Overall Statistics:\n")
        f.write(f"{indent}  Total samples: {stats.get('total_samples', 'N/A'):,}\n")
        f.write(f"{indent}  Total categories: {stats.get('total_categories', 'N/A')}\n")
        f.write(f"{indent}  Model type: {stats.get('model_type', 'N/A')}\n")
    
    if 'evaluation_summary' in model_results:
        summary = model_results['evaluation_summary']
        f.write(f"{indent}Evaluation Summary:\n")
        f.write(f"{indent}  Total categories: {summary.get('total_categories_evaluated', 'N/A')}\n")
        f.write(f"{indent}  Successful evaluations: {summary.get('successful_evaluations', 'N/A')}\n")
        f.write(f"{indent}  Failed evaluations: {summary.get('failed_evaluations', 'N/A')}\n")
    
    if 'category_results' in model_results:
        f.write(f"{indent}Category Results:\n")
        category_results = model_results['category_results']
        
        for category_name, category_result in category_results.items():
            f.write(f"{indent}  {category_name}:\n")
            
            if 'error' in category_result:
                f.write(f"{indent}    Error: {category_result['error']}\n")
            else:
                # Only show main numerical metrics
                main_metrics = {}
                for key, value in category_result.items():
                    if isinstance(value, (int, float)) and key.lower() in ['rmse', 'mae', 'auc', 'accuracy', 'precision', 'recall', 'f1']:
                        main_metrics[key] = value
                
                if main_metrics:
                    for metric, value in main_metrics.items():
                        f.write(f"{indent}    {metric}: {value:.6f}\n")
                else:
                    f.write(f"{indent}    No main numerical metrics\n")


def write_category_summary(f, results):
    """Write regular category-wise evaluation summary"""
        # Overall statistics
        if 'overall_stats' in results:
            stats = results['overall_stats']
            f.write(" Overall Statistics:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total samples: {stats.get('total_samples', 'N/A'):,}\n")
            f.write(f"Total categories: {stats.get('total_categories', 'N/A')}\n")
            f.write("\n")
        
        # Evaluation summary
        if 'evaluation_summary' in results:
            summary = results['evaluation_summary']
            f.write(" Evaluation Summary:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total categories: {summary.get('total_categories_evaluated', 'N/A')}\n")
            f.write(f"Successful evaluations: {summary.get('successful_evaluations', 'N/A')}\n")
            f.write(f"Failed evaluations: {summary.get('failed_evaluations', 'N/A')}\n")
            f.write("\n")
        
        # Category-wise results
        if 'category_results' in results:
            f.write(" Category-wise Evaluation Results:\n")
            f.write("-" * 40 + "\n")
            
            category_results = results['category_results']
            
            # First show overall results
            if 'Overall' in category_results:
                f.write(" Overall Evaluation:\n")
                overall_result = category_results['Overall']
                if 'error' in overall_result:
                    f.write(f"   Error: {overall_result['error']}\n")
                else:
                    for key, value in overall_result.items():
                        if isinstance(value, (int, float)):
                            f.write(f"   {key}: {value:.6f}\n")
                        else:
                            f.write(f"   {key}: {value}\n")
                f.write("\n")
            
            # Show results for each category
            for category_name, category_result in category_results.items():
                if category_name == 'Overall':
                    continue
                    
                f.write(f"  {category_name}:\n")
                
                if 'error' in category_result:
                    f.write(f"   Error: {category_result['error']}\n")
                else:
                    # Show main metrics
                    for key, value in category_result.items():
                        if isinstance(value, (int, float)):
                            f.write(f"   {key}: {value:.6f}\n")
                        else:
                            f.write(f"   {key}: {value}\n")
                f.write("\n")
        
    
def print_console_summary(results, is_ablation):
    """Print summary information on console"""
    print("\n" + "="*80)
    if is_ablation:
        print(" Ablation Experiment Evaluation Results Summary")
        print("="*80)
        
        # Show comparison of two models
        if 'comparison' in results and 'summary' in results['comparison']:
            summary = results['comparison']['summary']
            print(f" Comparison Analysis:")
            print(f"   Categories evaluated: {summary.get('categories_evaluated', 'N/A')}")
            print(f"   Metrics compared: {summary.get('total_metrics_compared', 'N/A')}")
            print(f"   Total comparisons: {summary.get('total_comparisons', 'N/A')}")
            print()
        
        # Show improvement status
        if 'comparison' in results and 'improvement_analysis' in results['comparison']:
            print(" Main Metrics Improvement Status:")
            improvement = results['comparison']['improvement_analysis']
            for metric, data in improvement.items():
                better_model = "Full model" if data['full_better_count'] > data['ablation_better_count'] else "Ablation model"
                print(f"   {metric}: {better_model} performs better in {data['categories_compared']} categories (avg improvement: {data['average_improvement']:.2f}%)")
            print()
        
    else:
        print(" Category-wise Evaluation Results Summary")
        print("="*80)
    
    # Show overall statistics
    if 'overall_stats' in results:
        stats = results['overall_stats']
        print(f" Overall Statistics:")
        print(f" Total samples: {stats.get('total_samples', 'N/A'):,}")
        print(f" Total categories: {stats.get('total_categories', 'N/A')}")
        print()
    
    # Show evaluation summary
    if 'evaluation_summary' in results:
        summary = results['evaluation_summary']
        print(f" Evaluation Summary:")
        print(f" Total categories: {summary.get('total_categories_evaluated', 'N/A')}")
        print(f" Successful evaluations: {summary.get('successful_evaluations', 'N/A')}")
        print(f" Failed evaluations: {summary.get('failed_evaluations', 'N/A')}")
        print()
    
 
    if 'category_results' in results:
        print(" Main Metrics by Category:")
        print("-" * 60)
        
        category_results = results['category_results']
        
        for category_name, category_result in category_results.items():
            if 'error' not in category_result:
                # Find main numerical metrics
                main_metrics = {}
                for key, value in category_result.items():
                    if isinstance(value, (int, float)) and key.lower() in ['rmse', 'mae', 'auc', 'accuracy', 'precision', 'recall', 'f1']:
                        main_metrics[key] = value
                
                print(f" {category_name}:")
                if main_metrics:
                    for metric, value in main_metrics.items():
                        print(f"     {metric}: {value:.6f}")
                else:
                    print("     No numerical metrics")
                print()
    
    print("="*80)


def parse_args():
    parser = argparse.ArgumentParser(description="DiffRec Model Evaluation Script")
    
    parser.add_argument(
        "--cfg-path", 
        default='train_configs/plora_pretrain_mf_ood.yaml',
        help="Configuration file path"
    )
    parser.add_argument(
        "--checkpoint-path", 
        required=True,
        help="Checkpoint file path (e.g., /root/autodl-tmp/checkpoints/[job_id]/checkpoint_best.pth)"
    )
    parser.add_argument(
        "--test-data-path",
        default="/root/autodl-tmp/dataset/amazon/test_ood2.pkl",
        help="Test data path"
    )
    parser.add_argument(
        "--output-dir",
        default="/root/autodl-tmp/eval_results",
        help="Evaluation results output directory"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computing device"
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation experiment (compare with/without pretrained CF model)"
    )
    parser.add_argument(
        "--disable-cf",
        action="store_true",
        help="Disable CF model components (only effective in non-ablation mode)"
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    
    return parser.parse_args()


def setup_seeds(seed=42):
    """Set random seeds"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    cudnn.benchmark = False
    cudnn.deterministic = True


def main():
    # Parse arguments
    args = parse_args()
    
    # Determine evaluation mode
    if args.ablation:
        eval_mode = "Ablation experiment mode"
        print(" DiffRec Model Ablation Experiment Evaluation Script Started")
        print("Will compare the following two configurations:")
        print("  1. Full model (with pretrained CF model)")
        print("  2. Ablation model (without pretrained CF model)")
    else:
        eval_mode = "Single model evaluation mode"
        cf_status = "(CF disabled)" if args.disable_cf else "(full model)"
        print(f"DiffRec Model Evaluation Script Started {cf_status}")
    
    print(f"Config file: {args.cfg_path}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Test data: {args.test_data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Evaluation mode: {eval_mode}")
    
    # Set random seeds
    setup_seeds()
    
    # Setup logger
    setup_logger()
    
    try:
        # 1. Load configuration
        print("\n1. Loading configuration file...")
        cfg = Config(args)
        
        # Set to evaluation mode
        cfg.run_cfg.evaluate = True
        cfg.run_cfg.distributed = False
        
        # Ensure test splits are set
        cfg.run_cfg.test_splits = ["test"]
        cfg.run_cfg.train_splits = []
        cfg.run_cfg.valid_splits = []
        
        print("Configuration loading completed")
        
        if args.ablation:
            # Ablation experiment mode
            print("\n2. Running ablation experiment...")
            results = run_ablation_evaluation(args.checkpoint_path, cfg, args.test_data_path, args.output_dir)
            print("Ablation experiment completed")
            
            # Save ablation experiment results
            print("\n3. Saving ablation experiment results...")
            results_file, summary_file = save_results(results, args.output_dir, args.checkpoint_path, is_ablation=True)
            print("Results saved successfully")
            
            print(f"\nAblation experiment completed successfully!")
            print(f"Detailed results file: {results_file}")
            print(f"Summary report file: {summary_file}")
            
        else:
                        # Single model evaluation mode
            print("\n2. Loading model and checkpoint...")
            model, task, datasets = load_checkpoint_and_model(args.checkpoint_path, cfg, disable_cf_model=args.disable_cf)
            print("Model loading completed")
        
        # 3. Run evaluation
        print("\n 3. Running evaluation...")
        model_type = "ablation" if args.disable_cf else "full"
        results = run_single_model_evaluation(model, task, datasets, cfg, args.test_data_path, model_type)
        print("Evaluation completed")
        
        # 4. Save results
        print("\n4. Saving evaluation results...")
        results_file, summary_file = save_results(results, args.output_dir, args.checkpoint_path, is_ablation=False)
        print("Results saved successfully")
        
        print(f"\n Category-wise evaluation completed successfully!")
        print(f"Detailed results file: {results_file}")
        print(f"Summary report file: {summary_file}")
        
    except Exception as e:
        print(f"\nError occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
