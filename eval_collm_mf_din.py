#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoRA模型分类别评估脚本 - 基于训练脚本train_collm_mf_din.py
专注于推荐任务评估，支持按instruction分类别评估和消融实验
图像生成模块已禁用

支持的类别：
- All Beauty
- Video_Games  
- Handmade_product

根据测试数据的instruction列自动分类和评估。

功能特性：
1. 分类别评估：根据instruction自动分类，对每个类别分别评估
2. 消融实验：对比使用和不使用预训练CF模型的效果
3. 详细报告：生成JSON和可读性文本两种格式的报告

用法：
1. 普通分类别评估（完整模型）：
   python eval_collm_mf_din.py --checkpoint-path /path/to/checkpoint.pth

2. 普通分类别评估（禁用CF模型）：
   python eval_collm_mf_din.py --checkpoint-path /path/to/checkpoint.pth --disable-cf

3. 消融实验（对比两种模型）：
   python eval_collm_mf_din.py --checkpoint-path /path/to/checkpoint.pth --ablation

输出文件：
普通评估模式：
- eval_results_by_category_[checkpoint_name]_[timestamp].json: 详细的JSON格式结果
- eval_summary_[checkpoint_name]_[timestamp].txt: 可读性好的汇总报告

消融实验模式：
- ablation_results_[checkpoint_name]_[timestamp].json: 详细的消融实验结果
- ablation_summary_[checkpoint_name]_[timestamp].txt: 消融实验对比报告

消融实验会产生6种结果：3个类别 × 2种模型版本（完整/消融）
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

# 🔧 兼容性补丁：必须在导入 minigpt4 之前应用
def apply_huggingface_compatibility_patch():
    """应用 HuggingFace 兼容性补丁"""
    print("🔧 应用 HuggingFace 兼容性补丁...")
    
    # 步骤1: 设置环境变量
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
    
    # 步骤2: 修复 huggingface_hub 版本兼容性
    try:
        import huggingface_hub
        import sys
        hub_version = huggingface_hub.__version__
        print(f"检测到 huggingface_hub 版本: {hub_version}")
        
        if hub_version < "0.20.0":
            print("应用动态兼容层...")
            
            # 添加缺失的函数
            if not hasattr(huggingface_hub, 'split_torch_state_dict_into_shards'):
                def split_torch_state_dict_into_shards(*args, **kwargs):
                    """兼容层：简单返回原始状态字典"""
                    if args:
                        return {'model.safetensors': args[0]}
                    return {}
                
                huggingface_hub.split_torch_state_dict_into_shards = split_torch_state_dict_into_shards
                print("✅ 添加 split_torch_state_dict_into_shards 兼容函数")
            
            # 添加缺失的 errors 模块
            if not hasattr(huggingface_hub, 'errors'):
                # 创建一个模拟的 errors 模块
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
                
                # 同时在 sys.modules 中注册
                sys.modules['huggingface_hub.errors'] = errors_module
                print("✅ 添加 huggingface_hub.errors 兼容模块")
            
            # 伪装版本号
            original_version = huggingface_hub.__version__
            huggingface_hub.__version__ = "0.20.0"
            print(f"版本伪装: {original_version} -> {huggingface_hub.__version__}")
        
    except Exception as e:
        print(f"兼容层应用失败: {e}")
    
    # 步骤3: 修复 transformers 和 PEFT 版本兼容性
    try:
        import transformers
        transformers_version = transformers.__version__
        print(f"检测到 transformers 版本: {transformers_version}")
        
        # 添加缺失的 Cache 相关类
        if not hasattr(transformers, 'EncoderDecoderCache'):
            # 创建兼容的 Cache 类
            class DummyCache:
                def __init__(self, *args, **kwargs):
                    pass
                
                def update(self, *args, **kwargs):
                    return None
                
                def get_seq_length(self, *args, **kwargs):
                    return 0
            
            # 添加缺失的 Cache 类
            if not hasattr(transformers, 'Cache'):
                transformers.Cache = DummyCache
                print("✅ 添加 transformers.Cache 兼容类")
            
            if not hasattr(transformers, 'DynamicCache'):
                transformers.DynamicCache = DummyCache
                print("✅ 添加 transformers.DynamicCache 兼容类")
                
            if not hasattr(transformers, 'EncoderDecoderCache'):
                transformers.EncoderDecoderCache = DummyCache
                print("✅ 添加 transformers.EncoderDecoderCache 兼容类")
                
            if not hasattr(transformers, 'HybridCache'):
                transformers.HybridCache = DummyCache
                print("✅ 添加 transformers.HybridCache 兼容类")
        
    except Exception as e:
        print(f"transformers 兼容层失败: {e}")
    
    # 步骤4: 修复 PEFT utils.config 模块缺失
    try:
        import peft
        import peft.utils
        
        # 检查 peft.utils.config 是否存在
        if not hasattr(peft.utils, 'config'):
            # 创建兼容的 config 模块
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
            
            # 同时在 sys.modules 中注册
            sys.modules['peft.utils.config'] = config_module
            print("✅ 添加 peft.utils.config 兼容模块")
        
    except Exception as e:
        print(f"PEFT utils.config 兼容层失败: {e}")
    
    # 步骤5: 兼容 torchvision/PIL 插值常量 (NEAREST_EXACT)
    try:
        from PIL import Image as _PILImage
        # 确保存在 Resampling，并提供 NEAREST_EXACT 兼容别名
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
            print("✅ 添加 PIL.Image.Resampling 兼容类")
        else:
            if not hasattr(_PILImage.Resampling, 'NEAREST_EXACT'):
                _PILImage.Resampling.NEAREST_EXACT = _PILImage.Resampling.NEAREST
                print("✅ 为 PIL.Image.Resampling 添加 NEAREST_EXACT 兼容别名")
    except Exception as e:
        print(f"PIL兼容层失败: {e}")

    try:
        from torchvision.transforms import InterpolationMode as _IM
        if not hasattr(_IM, 'NEAREST_EXACT'):
            # 定义别名，避免 transformers 依赖失败
            _IM.NEAREST_EXACT = _IM.NEAREST
            print("✅ 添加 torchvision.InterpolationMode.NEAREST_EXACT 兼容别名")
    except Exception as e:
        print(f"torchvision兼容层失败: {e}")

    print("✅ 兼容性补丁应用完成")
    return True

# 立即应用补丁
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
    """计算用户级RMSE和MAE指标"""
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
    
    print(f"只有一个交互的用户数: {only_one_interaction}")
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
    print(f"计算的用户数: {user_rmse.shape[0]}")
    avg_rmse = user_rmse.mean()
    avg_mae = user_mae.mean()
    print(f"用户级 RMSE: {avg_rmse:.4f}, 用户级 MAE: {avg_mae:.4f}, 耗时: {time.time() - start_time:.2f}s")
    return avg_rmse, avg_mae, computed_u, user_rmse, user_mae


def categorize_data_by_instruction(test_data):
    """根据instruction将数据分为三个类别"""
    print("\n📊 根据instruction分类数据...")
    
    # 定义类别关键词映射
    category_mapping = {
        'All Beauty': 'All Beauty',
        'Video_Games': 'Video_Games', 
        'Handmade_product': 'Handmade_product'
    }
    
    # 分类数据
    categorized_data = {}
    
    for category_key, category_name in category_mapping.items():
        # 筛选包含对应类别关键词的数据
        mask = test_data['instruction'].str.contains(category_key, case=False, na=False)
        category_data = test_data[mask].copy()
        
        if len(category_data) > 0:
            categorized_data[category_name] = category_data
            print(f"✅ {category_name}: {len(category_data)} 条数据")
            print(f"   用户数: {category_data['user_id'].nunique()}")
            print(f"   物品数: {category_data['asin'].nunique()}")
            print(f"   评分范围: {category_data['rating'].min():.1f} - {category_data['rating'].max():.1f}")
        else:
            print(f"⚠️ {category_name}: 未找到数据")
    
    # 验证分类结果
    total_categorized = sum(len(data) for data in categorized_data.values())
    print(f"\n📈 分类统计:")
    print(f"原始数据总数: {len(test_data)}")
    print(f"分类后总数: {total_categorized}")
    print(f"分类覆盖率: {total_categorized/len(test_data)*100:.2f}%")
    
    return categorized_data


def load_test_data(test_data_path):
    """加载测试数据"""
    print(f"📂 加载测试数据: {test_data_path}")
    
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"测试数据文件不存在: {test_data_path}")
    
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)
    
    if isinstance(test_data, pd.DataFrame):
        print(f"✅ 测试数据加载成功，形状: {test_data.shape}")
        print(f"数据列: {list(test_data.columns)}")
        
        # 检查并映射字段名
        field_mapping = {
            'uid': 'user_id',
            'iid': 'asin'
        }
        
        # 应用字段映射
        for old_name, new_name in field_mapping.items():
            if old_name in test_data.columns and new_name not in test_data.columns:
                test_data = test_data.rename(columns={old_name: new_name})
                print(f"🔄 字段映射: {old_name} -> {new_name}")
        
        # 检查必要字段
        required_fields = ['user_id', 'asin', 'rating', 'instruction']
        missing_fields = [field for field in required_fields if field not in test_data.columns]
        if missing_fields:
            print(f"⚠️ 缺少字段: {missing_fields}")
            # 如果仍然缺少字段，尝试显示可用字段
            print(f"可用字段: {list(test_data.columns)}")
        
        print(f"用户数: {test_data['user_id'].nunique()}")
        print(f"物品数: {test_data['asin'].nunique()}")
        print(f"评分范围: {test_data['rating'].min():.1f} - {test_data['rating'].max():.1f}")
        
        # 显示instruction的分布情况
        print(f"\n📋 Instruction分布:")
        instruction_counts = test_data['instruction'].value_counts()
        for instruction, count in instruction_counts.items():
            print(f"  '{instruction}': {count:,}次")
        
        return test_data
    else:
        raise ValueError(f"期望DataFrame格式，但得到: {type(test_data)}")


def load_checkpoint_and_model(checkpoint_path, cfg, disable_cf_model=False):
    """加载checkpoint和模型"""
    cf_status = "不使用CF模型" if disable_cf_model else "使用CF模型"
    print(f"📂 加载checkpoint: {checkpoint_path} ({cf_status})")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint文件不存在: {checkpoint_path}")
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print(f"✅ Checkpoint加载成功，包含键: {list(checkpoint.keys())}")
    
    if 'epoch' in checkpoint:
        print(f"训练epoch: {checkpoint['epoch']}")
    
    # 创建任务和模型
    print("🔧 初始化任务和模型...")
    task = tasks.setup_task(cfg)
    
    # 如果需要数据集来获取用户/物品数量，先构建数据集
    datasets = task.build_datasets(cfg)
    
    # 从数据集获取用户和物品数量
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
        # 回退：使用配置中的值
        user_num = cfg.model_cfg.rec_config.get('user_num', -100)
        item_num = cfg.model_cfg.rec_config.get('item_num', -100)

    cfg.model_cfg.rec_config.user_num = int(user_num)
    cfg.model_cfg.rec_config.item_num = int(item_num)
    
    print(f"用户数: {user_num}, 物品数: {item_num}")
    
    # 消融实验：如果禁用CF模型，修改配置
    if disable_cf_model:
        print("🚫 消融实验：禁用CF模型组件")
        # 备份原始配置
        original_cf_config = getattr(cfg.model_cfg, 'use_cf_model', True)
        # 设置不使用CF模型
        cfg.model_cfg.use_cf_model = False
        if hasattr(cfg.model_cfg.rec_config, 'use_pretrained_cf'):
            cfg.model_cfg.rec_config.use_pretrained_cf = False
        if hasattr(cfg.model_cfg.rec_config, 'enable_cf_component'):
            cfg.model_cfg.rec_config.enable_cf_component = False
        print("✅ CF模型组件已禁用")
    
    # 构建模型
    model = task.build_model(cfg)
    
    # 加载模型权重
    try:
        # 如果禁用CF模型，过滤掉CF相关的权重
        if disable_cf_model:
            print("🔧 过滤CF模型相关权重...")
            state_dict = checkpoint["model"]
            filtered_state_dict = {}
            cf_related_keys = []
            
            for key, value in state_dict.items():
                # 跳过CF模型相关的权重（根据实际模型结构调整）
                if any(cf_keyword in key.lower() for cf_keyword in ['cf_model', 'collaborative', 'mf_', 'matrix_fact']):
                    cf_related_keys.append(key)
                    continue
                filtered_state_dict[key] = value
            
            print(f"   过滤掉 {len(cf_related_keys)} 个CF模型相关权重")
            if cf_related_keys:
                print(f"   过滤的权重键: {cf_related_keys[:5]}{'...' if len(cf_related_keys) > 5 else ''}")
            
            missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
            print(f"✅ 消融模型权重加载完成 (缺少: {len(missing_keys)}, 意外: {len(unexpected_keys)})")
        else:
            model.load_state_dict(checkpoint["model"], strict=False)
            print("✅ 完整模型权重加载成功")
    except Exception as e:
        print(f"⚠️ 模型权重加载失败，尝试宽松加载: {e}")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
        print(f"缺少的键: {len(missing_keys)}, 意外的键: {len(unexpected_keys)}")
    
    # 设置为评估模式
    model.eval()
    
    # 设置模型运行模式（必须设置，否则forward会报错）
    mode = cfg.run_cfg.get('mode', 'v2')
    model.set_mode(mode)
    print(f"✅ 模型运行模式设置为: {mode}")
    
    return model, task, datasets


def create_dataloader_for_category(category_data, test_dataset, cfg):
    """为特定类别的数据创建数据加载器"""
    from torch.utils.data import DataLoader, Subset
    import numpy as np
    
    # 这里需要根据具体的数据集实现来创建子集
    # 由于我们无法直接从外部数据创建DataLoader，
    # 我们需要使用现有的test_dataset的结构
    
    batch_size = cfg.run_cfg.get('batch_size_eval', 4)
    
    # 创建一个简单的包装器
    class CategoryDataLoader:
        def __init__(self, category_data, original_dataset):
            self.category_data = category_data
            self.original_dataset = original_dataset
            self.batch_size = batch_size
            
        def __iter__(self):
            # 这里我们需要实现按类别筛选的迭代逻辑
            # 暂时返回原始数据集的迭代器
            # 在实际使用中，需要根据具体的数据集结构来实现
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
    """评估单个类别"""
    print(f"\n🔬 评估类别: {category_name}")
    
    # 创建data_loaders对象（包装DataLoader）
    class DataLoaders:
        def __init__(self, loader):
            self.loaders = [loader]
    
    data_loaders = DataLoaders(category_dataloader)
    
    # 使用任务的evaluation方法
    with torch.no_grad():
        eval_results = task.evaluation(
            model=model, 
            data_loaders=data_loaders, 
            cuda_enabled=torch.cuda.is_available(),
            split_name=f"test_{category_name}"
        )
    
    print(f"✅ {category_name} 评估完成，结果: {eval_results}")
    
    # 处理评估结果
    if eval_results is not None:
        final_results = task.after_evaluation(
            val_result=eval_results,
            split_name=f"test_{category_name}",
            epoch="final"
        )
    else:
        final_results = {"error": f"{category_name} 评估失败，结果为None"}
    
    return final_results


def run_single_model_evaluation(model, task, datasets, cfg, test_data_path, model_type="full"):
    """运行单个模型的分类别评估"""
    print(f"\n🔬 开始分类别评估 ({model_type}模型)...")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"设备: {device}")
    
    # 1. 加载外部测试数据
    test_data = load_test_data(test_data_path)
    
    # 2. 按instruction分类数据
    categorized_data = categorize_data_by_instruction(test_data)
    
    if not categorized_data:
        raise RuntimeError("未能成功分类测试数据")
    
    # 3. 获取原始测试数据集（用于创建数据加载器的结构）
    test_dataset = None
    for data_name, split_dict in datasets.items():
        if 'test' in split_dict:
            test_dataset = split_dict['test']
            print(f"✅ 使用内置测试集结构: {data_name}/test")
            break
    
    if test_dataset is None:
        print("⚠️ 未找到内置测试集，使用第一个可用数据集")
        # 使用第一个可用的数据集
        for data_name, split_dict in datasets.items():
            for split_name, dataset in split_dict.items():
                test_dataset = dataset
                print(f"✅ 使用数据集结构: {data_name}/{split_name}")
                break
            if test_dataset is not None:
                break
    
    if test_dataset is None:
        raise RuntimeError("未找到可用的数据集结构")
    
    # 4. 分类别进行评估
    category_results = {}
    overall_stats = {
        'total_samples': 0,
        'total_categories': len(categorized_data),
        'model_type': model_type
    }
    
    print(f"\n🚀 开始评估 {len(categorized_data)} 个类别...")
    
    # 先进行整体评估（如果需要）
    print(f"\n📊 整体评估 ({model_type}模型)...")
    
    # 创建完整数据的数据加载器
    from torch.utils.data import DataLoader
    batch_size = cfg.run_cfg.get('batch_size_eval', 4)
    full_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.run_cfg.get('num_workers', 2),
        collate_fn=getattr(test_dataset, 'collater', None)
    )
    
    # 整体评估
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
        overall_results = {"error": f"整体评估失败，结果为None ({model_type}模型)"}
    
    category_results['Overall'] = overall_results
    overall_stats['total_samples'] = len(test_data)
    
    # 分类别评估
    for category_name, category_data in categorized_data.items():
        print(f"\n{'='*50}")
        print(f"🎯 评估类别: {category_name} ({model_type}模型)")
        print(f"数据量: {len(category_data)}")
        print(f"{'='*50}")
        
        # 创建类别数据加载器
        category_dataloader = create_dataloader_for_category(category_data, test_dataset, cfg)
        
        # 评估该类别
        try:
            category_result = evaluate_single_category(
                model, task, f"{category_name}_{model_type}", category_dataloader, device
            )
            category_results[category_name] = category_result
            
            # 计算该类别的统计信息
            print(f"✅ {category_name} ({model_type}模型) 评估完成")
            if isinstance(category_result, dict):
                for key, value in category_result.items():
                    if isinstance(value, (int, float)):
                        print(f"   {key}: {value:.6f}")
                    else:
                        print(f"   {key}: {value}")
            
        except Exception as e:
            print(f"❌ {category_name} ({model_type}模型) 评估失败: {e}")
            category_results[category_name] = {"error": str(e)}
        
        print(f"{'='*50}")
    
    # 5. 汇总结果
    final_results = {
        'overall_stats': overall_stats,
        'category_results': category_results,
        'evaluation_summary': {
            'total_categories_evaluated': len(category_results) - 1,  # 除去Overall
            'successful_evaluations': sum(1 for k, v in category_results.items() 
                                        if k != 'Overall' and 'error' not in v),
            'failed_evaluations': sum(1 for k, v in category_results.items() 
                                    if k != 'Overall' and 'error' in v),
            'model_type': model_type
        }
    }
    
    print(f"\n📊 {model_type}模型评估汇总:")
    print(f"总类别数: {final_results['evaluation_summary']['total_categories_evaluated']}")
    print(f"成功评估: {final_results['evaluation_summary']['successful_evaluations']}")
    print(f"失败评估: {final_results['evaluation_summary']['failed_evaluations']}")
    
    return final_results


def run_ablation_evaluation(checkpoint_path, cfg, test_data_path, output_dir):
    """运行消融实验评估"""
    print("\n🧪 开始消融实验评估...")
    print("将对比以下两种模型配置：")
    print("1. 完整模型 (包含预训练CF模型)")
    print("2. 消融模型 (不使用预训练CF模型)")
    
    all_results = {}
    
    # 1. 评估完整模型
    print("\n" + "="*80)
    print("🔵 第一阶段：评估完整模型 (包含预训练CF模型)")
    print("="*80)
    
    try:
        # 加载完整模型
        full_model, task, datasets = load_checkpoint_and_model(checkpoint_path, cfg, disable_cf_model=False)
        
        # 运行完整模型评估
        full_results = run_single_model_evaluation(
            full_model, task, datasets, cfg, test_data_path, model_type="full"
        )
        all_results['full_model'] = full_results
        print("✅ 完整模型评估完成")
        
        # 释放GPU内存
        del full_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"❌ 完整模型评估失败: {e}")
        all_results['full_model'] = {"error": str(e)}
    
    # 2. 评估消融模型
    print("\n" + "="*80)
    print("🔴 第二阶段：评估消融模型 (不使用预训练CF模型)")
    print("="*80)
    
    try:
        # 加载消融模型
        ablation_model, task, datasets = load_checkpoint_and_model(checkpoint_path, cfg, disable_cf_model=True)
        
        # 运行消融模型评估
        ablation_results = run_single_model_evaluation(
            ablation_model, task, datasets, cfg, test_data_path, model_type="ablation"
        )
        all_results['ablation_model'] = ablation_results
        print("✅ 消融模型评估完成")
        
        # 释放GPU内存
        del ablation_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"❌ 消融模型评估失败: {e}")
        all_results['ablation_model'] = {"error": str(e)}
    
    # 3. 计算比较结果
    print("\n" + "="*80)
    print("📊 第三阶段：计算对比结果")
    print("="*80)
    
    comparison_results = generate_comparison_results(all_results)
    all_results['comparison'] = comparison_results
    
    return all_results


def generate_comparison_results(all_results):
    """生成两种模型的对比结果"""
    print("🔍 生成模型对比分析...")
    
    comparison = {
        'summary': {},
        'category_comparison': {},
        'improvement_analysis': {}
    }
    
    # 检查是否两种模型都评估成功
    full_results = all_results.get('full_model')
    ablation_results = all_results.get('ablation_model')
    
    if not full_results or 'error' in full_results:
        comparison['summary']['full_model_error'] = full_results.get('error', '完整模型评估失败')
        return comparison
    
    if not ablation_results or 'error' in ablation_results:
        comparison['summary']['ablation_model_error'] = ablation_results.get('error', '消融模型评估失败')
        return comparison
    
    # 获取类别结果
    full_categories = full_results.get('category_results', {})
    ablation_categories = ablation_results.get('category_results', {})
    
    # 对比各类别结果
    for category in full_categories.keys():
        if category in ablation_categories:
            full_result = full_categories[category]
            ablation_result = ablation_categories[category]
            
            if 'error' not in full_result and 'error' not in ablation_result:
                category_comparison = {}
                
                # 比较数值指标
                for metric in full_result.keys():
                    if isinstance(full_result.get(metric), (int, float)) and isinstance(ablation_result.get(metric), (int, float)):
                        full_value = full_result[metric]
                        ablation_value = ablation_result[metric]
                        
                        # 计算改进幅度
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
    
    # 生成改进分析摘要
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
    
    # 计算平均改进
    for metric in improvement_summary:
        if improvement_summary[metric]['categories_compared'] > 0:
            improvement_summary[metric]['average_improvement'] /= improvement_summary[metric]['categories_compared']
    
    comparison['improvement_analysis'] = improvement_summary
    
    # 生成总结
    total_comparisons = sum(data['categories_compared'] for data in improvement_summary.values())
    comparison['summary'] = {
        'total_metrics_compared': len(improvement_summary),
        'total_comparisons': total_comparisons,
        'categories_evaluated': len(comparison['category_comparison'])
    }
    
    print(f"✅ 对比分析完成：{comparison['summary']['categories_evaluated']} 个类别，{comparison['summary']['total_metrics_compared']} 个指标")
    
    return comparison


def save_results(results, output_dir, checkpoint_path, is_ablation=False):
    """保存分类别评估结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成结果文件名
    checkpoint_name = Path(checkpoint_path).stem
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    if is_ablation:
        results_file = output_dir / f"ablation_results_{checkpoint_name}_{timestamp}.json"
        summary_file = output_dir / f"ablation_summary_{checkpoint_name}_{timestamp}.txt"
        eval_type = "ablation_study"
        title = "CoRA模型消融实验评估报告"
    else:
        results_file = output_dir / f"eval_results_by_category_{checkpoint_name}_{timestamp}.json"
        summary_file = output_dir / f"eval_summary_{checkpoint_name}_{timestamp}.txt"
        eval_type = "category_wise"
        title = "CoRA模型分类别评估报告"
    
    # 添加元信息
    results_with_meta = {
        "evaluation_time": timestamp,
        "checkpoint_path": str(checkpoint_path),
        "evaluation_type": eval_type,
        "results": results
    }
    
    # 保存详细结果到JSON
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_with_meta, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"📁 评估结果已保存到: {results_file}")
    
    # 创建可读性更好的汇总报告
    with open(summary_file, 'w', encoding='utf-8') as f:
        # 写入汇总报告
        f.write("=" * 80 + "\n")
        f.write(f"📊 {title}\n")
        f.write("=" * 80 + "\n")
        f.write(f"评估时间: {timestamp}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"评估类型: {eval_type}\n")
        f.write("\n")
        
        if is_ablation:
            # 消融实验报告
            write_ablation_summary(f, results)
        else:
            # 普通分类别报告
            write_category_summary(f, results)
        
        f.write("=" * 80 + "\n")
    
    print(f"📄 评估汇总报告已保存到: {summary_file}")
    
    # 在控制台显示汇总信息
    print_console_summary(results, is_ablation)
    
    return results_file, summary_file


def write_ablation_summary(f, results):
    """写入消融实验汇总报告"""
    # 实验概述
    f.write("🧪 消融实验概述:\n")
    f.write("-" * 40 + "\n")
    f.write("对比配置:\n")
    f.write("  1. 完整模型 (包含预训练CF模型)\n")
    f.write("  2. 消融模型 (不使用预训练CF模型)\n")
    f.write("\n")
    
    # 模型评估结果
    if 'full_model' in results:
        f.write("🔵 完整模型结果:\n")
        f.write("-" * 40 + "\n")
        full_results = results['full_model']
        if 'error' in full_results:
            f.write(f"   ❌ 错误: {full_results['error']}\n")
        else:
            write_model_results(f, full_results, "   ")
        f.write("\n")
    
    if 'ablation_model' in results:
        f.write("🔴 消融模型结果:\n")
        f.write("-" * 40 + "\n")
        ablation_results = results['ablation_model']
        if 'error' in ablation_results:
            f.write(f"   ❌ 错误: {ablation_results['error']}\n")
        else:
            write_model_results(f, ablation_results, "   ")
        f.write("\n")
    
    # 对比分析
    if 'comparison' in results:
        f.write("📊 对比分析:\n")
        f.write("-" * 40 + "\n")
        comparison = results['comparison']
        
        if 'summary' in comparison:
            summary = comparison['summary']
            f.write(f"评估类别数: {summary.get('categories_evaluated', 'N/A')}\n")
            f.write(f"对比指标数: {summary.get('total_metrics_compared', 'N/A')}\n")
            f.write(f"总对比次数: {summary.get('total_comparisons', 'N/A')}\n")
            f.write("\n")
        
        # 改进分析
        if 'improvement_analysis' in comparison:
            f.write("📈 改进分析:\n")
            improvement = comparison['improvement_analysis']
            for metric, data in improvement.items():
                f.write(f"\n  {metric}:\n")
                f.write(f"    对比类别数: {data['categories_compared']}\n")
                f.write(f"    完整模型更优: {data['full_better_count']}\n")
                f.write(f"    消融模型更优: {data['ablation_better_count']}\n")
                f.write(f"    相等: {data['equal_count']}\n")
                f.write(f"    平均改进: {data['average_improvement']:.2f}%\n")
        
        # 详细对比
        if 'category_comparison' in comparison:
            f.write("\n📋 各类别详细对比:\n")
            for category, metrics in comparison['category_comparison'].items():
                f.write(f"\n  🏷️ {category}:\n")
                if 'error' in metrics:
                    f.write(f"    ❌ 错误: {metrics['error']}\n")
                else:
                    for metric, data in metrics.items():
                        f.write(f"    {metric}:\n")
                        f.write(f"      完整模型: {data['full_model']:.6f}\n")
                        f.write(f"      消融模型: {data['ablation_model']:.6f}\n")
                        f.write(f"      改进幅度: {data['improvement_percent']:.2f}%\n")
                        f.write(f"      更优模型: {data['better_model']}\n")


def write_model_results(f, model_results, indent=""):
    """写入单个模型的结果"""
    if 'overall_stats' in model_results:
        stats = model_results['overall_stats']
        f.write(f"{indent}整体统计:\n")
        f.write(f"{indent}  总样本数: {stats.get('total_samples', 'N/A'):,}\n")
        f.write(f"{indent}  类别数量: {stats.get('total_categories', 'N/A')}\n")
        f.write(f"{indent}  模型类型: {stats.get('model_type', 'N/A')}\n")
    
    if 'evaluation_summary' in model_results:
        summary = model_results['evaluation_summary']
        f.write(f"{indent}评估汇总:\n")
        f.write(f"{indent}  总类别数: {summary.get('total_categories_evaluated', 'N/A')}\n")
        f.write(f"{indent}  成功评估: {summary.get('successful_evaluations', 'N/A')}\n")
        f.write(f"{indent}  失败评估: {summary.get('failed_evaluations', 'N/A')}\n")
    
    if 'category_results' in model_results:
        f.write(f"{indent}类别结果:\n")
        category_results = model_results['category_results']
        
        for category_name, category_result in category_results.items():
            f.write(f"{indent}  {category_name}:\n")
            
            if 'error' in category_result:
                f.write(f"{indent}    ❌ 错误: {category_result['error']}\n")
            else:
                # 只显示主要数值指标
                main_metrics = {}
                for key, value in category_result.items():
                    if isinstance(value, (int, float)) and key.lower() in ['rmse', 'mae', 'auc', 'accuracy', 'precision', 'recall', 'f1']:
                        main_metrics[key] = value
                
                if main_metrics:
                    for metric, value in main_metrics.items():
                        f.write(f"{indent}    {metric}: {value:.6f}\n")
                else:
                    f.write(f"{indent}    无主要数值指标\n")


def write_category_summary(f, results):
    """写入普通分类别评估汇总"""
        # 整体统计
    if 'overall_stats' in results:
            stats = results['overall_stats']
            f.write("📈 整体统计:\n")
            f.write("-" * 40 + "\n")
            f.write(f"总样本数: {stats.get('total_samples', 'N/A'):,}\n")
            f.write(f"类别数量: {stats.get('total_categories', 'N/A')}\n")
            f.write("\n")
        
        # 评估汇总
    if 'evaluation_summary' in results:
            summary = results['evaluation_summary']
            f.write("🎯 评估汇总:\n")
            f.write("-" * 40 + "\n")
            f.write(f"总类别数: {summary.get('total_categories_evaluated', 'N/A')}\n")
            f.write(f"成功评估: {summary.get('successful_evaluations', 'N/A')}\n")
            f.write(f"失败评估: {summary.get('failed_evaluations', 'N/A')}\n")
            f.write("\n")
        
        # 分类别结果
    if 'category_results' in results:
            f.write("📋 分类别评估结果:\n")
            f.write("-" * 40 + "\n")
            
            category_results = results['category_results']
            
            # 首先显示整体结果
            if 'Overall' in category_results:
                f.write("🌟 整体评估:\n")
                overall_result = category_results['Overall']
                if 'error' in overall_result:
                    f.write(f"   ❌ 错误: {overall_result['error']}\n")
                else:
                    for key, value in overall_result.items():
                        if isinstance(value, (int, float)):
                            f.write(f"   {key}: {value:.6f}\n")
                        else:
                            f.write(f"   {key}: {value}\n")
                f.write("\n")
            
            # 显示各类别结果
            for category_name, category_result in category_results.items():
                if category_name == 'Overall':
                    continue
                    
                f.write(f"🏷️  {category_name}:\n")
                
                if 'error' in category_result:
                    f.write(f"   ❌ 错误: {category_result['error']}\n")
                else:
                    # 显示主要指标
                    for key, value in category_result.items():
                        if isinstance(value, (int, float)):
                            f.write(f"   {key}: {value:.6f}\n")
                        else:
                            f.write(f"   {key}: {value}\n")
                f.write("\n")
        
    
def print_console_summary(results, is_ablation):
    """在控制台打印汇总信息"""
    print("\n" + "="*80)
    if is_ablation:
        print("🧪 消融实验评估结果汇总")
        print("="*80)
        
        # 显示两种模型的对比
        if 'comparison' in results and 'summary' in results['comparison']:
            summary = results['comparison']['summary']
            print(f"📊 对比分析:")
            print(f"   评估类别数: {summary.get('categories_evaluated', 'N/A')}")
            print(f"   对比指标数: {summary.get('total_metrics_compared', 'N/A')}")
            print(f"   总对比次数: {summary.get('total_comparisons', 'N/A')}")
            print()
        
        # 显示改进情况
        if 'comparison' in results and 'improvement_analysis' in results['comparison']:
            print("📈 主要指标改进情况:")
            improvement = results['comparison']['improvement_analysis']
            for metric, data in improvement.items():
                better_model = "完整模型" if data['full_better_count'] > data['ablation_better_count'] else "消融模型"
                print(f"   {metric}: {better_model}在{data['categories_compared']}个类别中表现更优 (平均改进: {data['average_improvement']:.2f}%)")
            print()
        
    else:
        print("📊 分类别评估结果汇总")
        print("="*80)
    
    # 显示整体统计
    if 'overall_stats' in results:
        stats = results['overall_stats']
        print(f"📈 整体统计:")
        print(f"   总样本数: {stats.get('total_samples', 'N/A'):,}")
        print(f"   类别数量: {stats.get('total_categories', 'N/A')}")
        print()
    
    # 显示评估汇总
    if 'evaluation_summary' in results:
        summary = results['evaluation_summary']
        print(f"🎯 评估汇总:")
        print(f"   总类别数: {summary.get('total_categories_evaluated', 'N/A')}")
        print(f"   成功评估: {summary.get('successful_evaluations', 'N/A')}")
        print(f"   失败评估: {summary.get('failed_evaluations', 'N/A')}")
        print()
    
    # 显示关键指标对比
    if 'category_results' in results:
        print("📋 各类别主要指标:")
        print("-" * 60)
        
        category_results = results['category_results']
        
        for category_name, category_result in category_results.items():
            if 'error' not in category_result:
                # 查找主要的数值指标
                main_metrics = {}
                for key, value in category_result.items():
                    if isinstance(value, (int, float)) and key.lower() in ['rmse', 'mae', 'auc', 'accuracy', 'precision', 'recall', 'f1']:
                        main_metrics[key] = value
                
                print(f"🏷️  {category_name}:")
                if main_metrics:
                    for metric, value in main_metrics.items():
                        print(f"     {metric}: {value:.6f}")
                else:
                    print("     无数值指标")
                print()
    
    print("="*80)


def parse_args():
    parser = argparse.ArgumentParser(description="CoRA模型评估脚本")
    
    parser.add_argument(
        "--cfg-path", 
        default='train_configs/plora_pretrain_mf_ood.yaml',
        help="配置文件路径"
    )
    parser.add_argument(
        "--checkpoint-path", 
        required=True,
        help="checkpoint文件路径 (如: /root/autodl-tmp/checkpoints/[job_id]/checkpoint_best.pth)"
    )
    parser.add_argument(
        "--test-data-path",
        default="/root/autodl-tmp/dataset/amazon/test_ood2.pkl",
        help="测试数据路径"
    )
    parser.add_argument(
        "--output-dir",
        default="/root/autodl-tmp/eval_results",
        help="评估结果保存目录"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="计算设备"
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="运行消融实验 (对比使用和不使用预训练CF模型的效果)"
    )
    parser.add_argument(
        "--disable-cf",
        action="store_true",
        help="禁用CF模型组件 (仅在非消融实验模式下有效)"
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
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    cudnn.benchmark = False
    cudnn.deterministic = True


def main():
    # 解析参数
    args = parse_args()
    
    # 确定评估模式
    if args.ablation:
        eval_mode = "消融实验模式"
        print("🧪 CoRA模型消融实验评估脚本启动")
        print("将对比以下两种配置:")
        print("  1. 完整模型 (包含预训练CF模型)")
        print("  2. 消融模型 (不使用预训练CF模型)")
    else:
        eval_mode = "单模型评估模式"
        cf_status = "(禁用CF模型)" if args.disable_cf else "(完整模型)"
        print(f"🚀 CoRA模型评估脚本启动 {cf_status}")
    
    print(f"配置文件: {args.cfg_path}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"测试数据: {args.test_data_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"评估模式: {eval_mode}")
    
    # 设置随机种子
    setup_seeds()
    
    # 设置日志
    setup_logger()
    
    try:
        # 1. 加载配置
        print("\n📋 1. 加载配置文件...")
        cfg = Config(args)
        
        # 设置为评估模式
        cfg.run_cfg.evaluate = True
        cfg.run_cfg.distributed = False
        
        # 确保测试分割被设置
        cfg.run_cfg.test_splits = ["test"]
        cfg.run_cfg.train_splits = []
        cfg.run_cfg.valid_splits = []
        
        print("✅ 配置加载完成")
        
        if args.ablation:
            # 消融实验模式
            print("\n🧪 2. 运行消融实验...")
            results = run_ablation_evaluation(args.checkpoint_path, cfg, args.test_data_path, args.output_dir)
            print("✅ 消融实验完成")
            
            # 保存消融实验结果
            print("\n💾 3. 保存消融实验结果...")
            results_file, summary_file = save_results(results, args.output_dir, args.checkpoint_path, is_ablation=True)
            print("✅ 结果保存完成")
            
            print(f"\n🎉 消融实验成功完成！")
            print(f"📁 详细结果文件: {results_file}")
            print(f"📄 汇总报告文件: {summary_file}")
            
        else:
            # 单模型评估模式
            print("\n🤖 2. 加载模型和checkpoint...")
            model, task, datasets = load_checkpoint_and_model(args.checkpoint_path, cfg, disable_cf_model=args.disable_cf)
            print("✅ 模型加载完成")
        
        # 3. 运行评估
        print("\n🔬 3. 运行评估...")
        model_type = "ablation" if args.disable_cf else "full"
        results = run_single_model_evaluation(model, task, datasets, cfg, args.test_data_path, model_type)
        print("✅ 评估完成")
        
        # 4. 保存结果
        print("\n💾 4. 保存评估结果...")
        results_file, summary_file = save_results(results, args.output_dir, args.checkpoint_path, is_ablation=False)
        print("✅ 结果保存完成")
        
        print(f"\n🎉 分类别评估成功完成！")
        print(f"📁 详细结果文件: {results_file}")
        print(f"📄 汇总报告文件: {summary_file}")
        
    except Exception as e:
        print(f"\n❌ 评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
