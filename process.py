#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
划分Amazon数据集为训练集、验证集和测试集
用于协同过滤模型训练
"""

import json
import gzip
import pandas as pd
import numpy as np
import os
import re
from sklearn.preprocessing import LabelEncoder
import argparse

def load_amazon_data(file_path):
    """加载Amazon数据"""
    print(f"Loading data from {file_path}...")
    data = []
    
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line.strip():
                data.append(json.loads(line.strip()))
            
            # 每10万条显示一次进度
            if (i + 1) % 100000 == 0:
                print(f"Loaded {i + 1:,} records...")
    
    print(f"Total loaded: {len(data):,} records")
    return data

def parse_his_interaction(h: str, max_len: int = 15):
    """将 his_interaction 解析为 (asins, titles)
    期望形如 (user,asin,title,rating)(user,asin,title,rating) 的字符串，
    兼容中间 title 含有逗号的情况（用聚合合并 2..-2 段）。
    """
    if not isinstance(h, str) or not h:
        return [], []
    parts = re.split(r"\)\s*\(", h.strip().strip("()"))
    asins, titles = [], []
    for p in parts:
        toks = [t.strip() for t in p.split(",")]
        if len(toks) >= 4:
            asin = toks[1]
            title = ",".join(toks[2:-1]).strip()
            if asin:
                asins.append(asin)
                titles.append(title)
        if len(asins) >= max_len:
            break
    return asins, titles


def convert_split_df(df: pd.DataFrame, user_encoder: LabelEncoder, item_encoder: LabelEncoder,
                     title_max_len: int = 256, rating_threshold: float = 4.0) -> pd.DataFrame:
    """将一个拆分后的 DataFrame 转换为 *_ood2.pkl 需要的字段集。
    输出字段：uid,iid,title,label,rating,instruction,timestamp,his,his_title
    """
    df = df.copy()
    # 数值化 rating 并保留
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df[df["rating"].notna()]

    # 全局编码（LabelEncoder 只接受已见过的类别）
    # 对主 uid/iid 可用 transform；历史 asin 需通过映射字典处理未知
    df["uid"] = user_encoder.transform(df["user_id"])
    df["iid"] = item_encoder.transform(df["asin"])

    # title：从 item_features 截断
    df["title"] = df["item_features"].astype(str).str.slice(0, title_max_len)
    # label：占位标签，兼容旧构建器（训练仅用 rating）
    df["label"] = (df["rating"] >= rating_threshold).astype(int)
    # instruction/timestamp 保留
    df["instruction"] = df.get("instruction", "").astype(str)
    df["timestamp"] = df.get("timestamp", "")

    # 历史交互解析
    his_asins_titles = df.get("his_interaction", "").map(parse_his_interaction)
    df["his_title"] = his_asins_titles.map(lambda x: x[1])

    # 历史 asin -> 历史 iid（未知 asin 丢弃）
    item_classes = list(item_encoder.classes_)
    item_to_id = {a: i for i, a in enumerate(item_classes)}
    def map_his_asins_to_iids(x):
        asins = x[0]
        out = []
        for a in asins:
            if a in item_to_id:
                out.append(int(item_to_id[a]))
        return out
    df["his"] = his_asins_titles.map(map_his_asins_to_iids)

    return df[["uid", "iid", "title", "label", "rating", "instruction", "timestamp", "his", "his_title"]]


def split_amazon_dataset(file_path, output_dir="dataset/amazon/",
                        train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15,
                        title_max_len: int = 256, rating_threshold: float = 4.0):
    """
    划分Amazon数据集
    
    Args:
        file_path: Amazon数据文件路径
        output_dir: 输出目录
        train_ratio: 训练集比例
        valid_ratio: 验证集比例  
        test_ratio: 测试集比例
    """
    
    # 检查比例是否合理
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + valid_ratio + test_ratio must equal 1.0")
    
    # 加载数据
    amazon_data = load_amazon_data(file_path)
    df = pd.DataFrame(amazon_data)
    
    print(f"\n=== 数据预处理 ===")
    print(f"原始数据大小: {len(df):,}")
    
    # 显示字段信息
    print(f"数据字段: {list(df.columns)}")
    
    # 只保留有评分的数据（rating > 0）
    df_filtered = df[pd.to_numeric(df['rating'], errors='coerce') > 0].copy()
    print(f"过滤后数据大小: {len(df_filtered):,} (保留rating > 0的记录)")
    
    # 显示评分分布
    print(f"\n=== 评分分布 ===")
    rating_counts = df_filtered['rating'].value_counts().sort_index()
    for rating, count in rating_counts.items():
        print(f"评分 {rating}: {count:,} ({count/len(df_filtered)*100:.1f}%)")
    
    print(f"评分范围: {df_filtered['rating'].min()} - {df_filtered['rating'].max()}")
    print(f"平均评分: {df_filtered['rating'].mean():.2f}")
    
    # 统计用户和物品信息
    print(f"\n=== 用户和物品统计 ===")
    print(f"唯一用户数: {df_filtered['user_id'].nunique():,}")
    print(f"唯一物品数: {df_filtered['asin'].nunique():,}")
    
    # 按时间戳排序（确保时间顺序）
    df_filtered["timestamp"] = pd.to_numeric(df_filtered["timestamp"], errors="coerce")
    df_filtered = df_filtered.sort_values('timestamp')
    print(f"数据按时间戳排序完成")
    
    # 划分数据集
    print(f"\n=== 数据集划分 ===")
    total_size = len(df_filtered)
    train_end = int(total_size * train_ratio)
    valid_end = int(total_size * (train_ratio + valid_ratio))
    
    # 拆分 DataFrame（保留全部字段）
    train_df = df_filtered.iloc[:train_end].copy()
    valid_df = df_filtered.iloc[train_end:valid_end].copy()
    test_df = df_filtered.iloc[valid_end:].copy()

    print(f"训练集: {len(train_df):,} ({len(train_df)/total_size*100:.1f}%)")
    print(f"验证集: {len(valid_df):,} ({len(valid_df)/total_size*100:.1f}%)")
    print(f"测试集: {len(test_df):,} ({len(test_df)/total_size*100:.1f}%)")

    # 全量拟合编码器，确保各拆分共享同一映射
    user_encoder = LabelEncoder().fit(df_filtered['user_id'])
    item_encoder = LabelEncoder().fit(df_filtered['asin'])

    # 转换为 *_ood2.pkl 所需结构
    out_train = convert_split_df(train_df, user_encoder, item_encoder, title_max_len, rating_threshold)
    out_valid = convert_split_df(valid_df, user_encoder, item_encoder, title_max_len, rating_threshold)
    out_test = convert_split_df(test_df, user_encoder, item_encoder, title_max_len, rating_threshold)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存为 pkl
    train_pkl = os.path.join(output_dir, "train_ood2.pkl")
    valid_pkl = os.path.join(output_dir, "valid_ood2.pkl")
    test_pkl = os.path.join(output_dir, "test_ood2.pkl")
    out_train.to_pickle(train_pkl)
    out_valid.to_pickle(valid_pkl)
    out_test.to_pickle(test_pkl)

    print(f"训练集保存到: {train_pkl}")
    print(f"验证集保存到: {valid_pkl}")
    print(f"测试集保存到: {test_pkl}")
    
    # 保存数据集统计信息
    stats = {
        'total_records': len(df_filtered),
        'user_num': df_filtered['user_id'].nunique(),
        'item_num': df_filtered['asin'].nunique(),
        'rating_min': float(df_filtered['rating'].min()),
        'rating_max': float(df_filtered['rating'].max()),
        'rating_mean': float(df_filtered['rating'].mean()),
        'train_size': len(train_df),
        'valid_size': len(valid_df),
        'test_size': len(test_df),
        'train_ratio': train_ratio,
        'valid_ratio': valid_ratio,
        'test_ratio': test_ratio,
        'rating_distribution': df_filtered['rating'].value_counts().sort_index().to_dict()
    }
    
    with open(output_dir + "dataset_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"统计信息保存到: {output_dir}dataset_stats.json")
    
    # 显示最终统计
    print(f"\n=== 最终统计 ===")
    for key, value in stats.items():
        if key == 'rating_distribution':
            continue
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value:,}")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="按时间划分 Amazon 合并数据，并直接导出 *_ood2.pkl")
    parser.add_argument("--input", type=str, default=r"/root/autodl-tmp/dataset/amazon/amazon_merged_training_data.jsonl.gz", help="输入合并的 jsonl.gz")
    parser.add_argument("--output-dir", type=str, default="dataset/amazon/", help="输出目录")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--title-max-len", type=int, default=256)
    parser.add_argument("--rating-threshold", type=float, default=4.0)
    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 找不到文件 {args.input}")
        return

    print("=== Amazon数据集划分工具 ===")
    print(f"输入文件: {args.input}")
    
    try:
        stats = split_amazon_dataset(
            file_path=args.input,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            test_ratio=args.test_ratio,
            title_max_len=args.title_max_len,
            rating_threshold=args.rating_threshold,
        )

        print(f"\n✅ 数据集划分完成！")
        print(f"已导出: train_ood2.pkl / valid_ood2.pkl / test_ood2.pkl 到 {args.output_dir}")
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 