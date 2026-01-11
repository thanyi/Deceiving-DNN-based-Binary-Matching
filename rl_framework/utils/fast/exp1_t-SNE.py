#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import random

import sys 
sys.path.insert(0, '/home/ycy/ours/Deceiving-DNN-based-Binary-Matching')    
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_framework.env_wrapper import BinaryPerturbationEnv

# ================= 配置 =================
DATASET_PATH = "dataset_train.json"
SAVE_DIR = "chapter3_results_final"
os.makedirs(SAVE_DIR, exist_ok=True)

# 绘图风格
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    sns.set_style("whitegrid")

plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['font.size'] = 14

# ================= 工具函数 =================

def load_dataset():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: {DATASET_PATH} not found!")
        sys.exit(1)
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)
    
    # 【修改点1】过滤掉 main 函数，过滤掉太小的函数
    filtered_data = []
    for d in data:
        if d['func_name'] == 'main': continue # main 函数逻辑各异，不适合聚类
        if d.get('size', 0) < 50: continue    # 太小的函数特征不明显
        filtered_data.append(d)
        
    print(f"[*] Loaded {len(filtered_data)} valid samples (filtered 'main').")
    return filtered_data

def get_feature_extractor():
    return BinaryPerturbationEnv(save_path="/tmp/test_env", dataset_path=DATASET_PATH)

# ================= 实验 1: t-SNE (去除 main 干扰版) =================

def exp1_discriminability(data, extractor, input_perplexity=10, input_candidates_num=20, jitter_strength=0.01):
    print("\n[Experiment 1] Running t-SNE Visualization (Multi-Binary Mode)...")
    
    from collections import Counter
    
    # === 1. 智能筛选逻辑 ===
    valid_candidates = []
    
    # 黑名单：名字相同但语义不同的函数
    BANNED_NAMES = {'main', 'usage', 'emit_bug_reporting_address', 'version_etc'}
    
    for d in data:
        fname = d['func_name']
        
        # 1. 剔除黑名单
        if fname in BANNED_NAMES: continue
        
        # 2. 剔除太简单或太复杂的函数 (保持特征的代表性)
        # size < 100: 可能是 wrapper，特征太少容易重叠
        # size > 2000: 可能是超大函数，容易 outlier
        if d.get('size', 0) < 500 or d.get('size', 0) > 1500: continue
        
        valid_candidates.append(fname)
        
    # 统计出现频率最高的函数
    # 既然是多二进制文件，像 'xstrdup' 这种通用函数会出现很多次，这正是我们想要的
    func_counts = Counter(valid_candidates)
    
    # 选取前 4 类，且每类样本数要足够多 (比如 > 15 个，保证簇的丰满度)
    top_funcs = [name for name, cnt in func_counts.most_common(20) if cnt >= 15][:6]
    
    print(f"[*] Selected Target Functions for Clustering: {top_funcs}")
    # 打印一下来源，确认是跨文件的
    # 示例输出: xstrdup (from ls, cp, mv...)
    
    selected_samples = []
    labels = []
    opt_levels = []
    
    for target_func in top_funcs:
        # 从所有二进制文件中搜集这个函数
        candidates = [d for d in data if d['func_name'] == target_func]
        
        # 采样 (样本多一点，簇更好看)
        if len(candidates) > input_candidates_num: 
            candidates = random.sample(candidates, input_candidates_num)
            
        selected_samples.extend(candidates)
        labels.extend([target_func] * len(candidates))
        opt_levels.extend([d.get('opt_level', 'unknown') for d in candidates])
        
    print(f"[*] Extracting features for {len(selected_samples)} samples...")
    # print(f"[*] Selected samples: {selected_samples}")
    features = []
    for sample in tqdm(selected_samples):
        # 提取 256维 特征，去掉前16维 RL历史
        vec = extractor.extract_features_from_function(sample['binary_path'], sample['func_name'])[16:]
        features.append(vec)
        
    print(f"[*] Features example: {features[1]}")
    print(f"[*] Features name: {selected_samples[1]['func_name']}, binary_name: {selected_samples[1]['binary_name']}, opt_level: {selected_samples[1]['opt_level']}, version: {selected_samples[1]['version']}")
    X = np.array(features)
    
    # 归一化是 t-SNE 成功的关键
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    
    # t-SNE
    # 【优化参数】
    # perplexity: 通常选择样本数的 5%-10%，160-240 样本推荐 10-25
    # n_iter: 根据 perplexity 动态调整，确保收敛
    # learning_rate: 'auto' 让 sklearn 自动计算（1.2版本默认）
    n_samples = X_norm.shape[0]
    n_iter_optimal = max(1000, min(5000, input_perplexity * 150))  # perplexity*150，但不超过5000
    
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=input_perplexity,
        n_iter=n_iter_optimal,
        learning_rate='auto',  # 自动计算，通常 = max(200, n_samples / 4)
        init='pca',
        verbose=0
    )
    X_embedded = tsne.fit_transform(X_norm)
    

    # === 添加抖动 (Jitter) ===
    # 【修复】根据实际坐标范围动态计算抖动强度
    # t-SNE 的坐标范围通常在 100-200 之间（例如 [-50, 50] 或 [-100, 100]）
    # jitter_strength 作为比例（例如 0.02 表示坐标范围的 2%，默认 2% 可见但不过度）
    coord_range = np.max(X_embedded) - np.min(X_embedded)
    # 如果 jitter_strength >= 1.0，视为绝对数值；否则视为比例
    if jitter_strength >= 1.0:
        dynamic_jitter_std = jitter_strength
    else:
        dynamic_jitter_std = coord_range * jitter_strength
    jitter = np.random.normal(0, dynamic_jitter_std, X_embedded.shape)
    X_plot = X_embedded + jitter
    
    if dynamic_jitter_std > 0:
        print(f"[*] 抖动强度: {dynamic_jitter_std:.2f} (坐标范围: {coord_range:.2f}, 比例: {jitter_strength*100:.1f}%)")

    # 绘图：使用函数名作为颜色（hue），优化等级作为样式（style）
    plt.figure(figsize=(14, 10))
    
    # 使用 seaborn 绘制，hue 用函数名，style 用优化等级
    ax = sns.scatterplot(
        x=X_plot[:, 0], y=X_plot[:, 1],  # 使用抖动后的坐标
        hue=labels,
        style=opt_levels,
        # 【修改】调整点的大小和透明度
        s=150,          # 点稍微大一点
        alpha=0.7,      # 透明度 0.7，重叠处会变深
        palette="deep", 
        edgecolor='k',  # 给点加个黑边，更容易区分
        linewidth=0.5
    )
    
    plt.title("Semantic Clustering of Binary Functions (t-SNE)\nwith Optimization Levels", 
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel("Dimension 1", fontsize=14)
    plt.ylabel("Dimension 2", fontsize=14)
    
    # 调整图例位置和样式
    # seaborn 会自动创建两个图例：函数名（hue）和优化等级（style）
    legend = ax.get_legend()
    if legend is not None:
        # 将图例移到右侧（bbox_to_anchor 已经设置了位置）
        legend.set_bbox_to_anchor((1.02, 1.0))
        # 设置图例框架样式
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        # 调整图例中的字体大小
        for text in legend.get_texts():
            text.set_fontsize(10)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(SAVE_DIR, f"exp1_tsne_perplexity_{input_perplexity}_candidates_{input_candidates_num}.png"), dpi=300)
    print(f"[+] Saved exp1_tsne_perplexity_{input_perplexity}_candidates_{input_candidates_num}.png")



if __name__ == "__main__":
    np.random.seed(2024)
    random.seed(2024)
    
    data = load_dataset()
    extractor = get_feature_extractor()
    exp1_discriminability(data, extractor, input_perplexity=5, input_candidates_num=20)
    exp1_discriminability(data, extractor, input_perplexity=10, input_candidates_num=20)
    exp1_discriminability(data, extractor, input_perplexity=15, input_candidates_num=20)  # 最佳
    
    
    exp1_discriminability(data, extractor, input_perplexity=10, input_candidates_num=30)
    exp1_discriminability(data, extractor, input_perplexity=15, input_candidates_num=30)  # 最佳
    exp1_discriminability(data, extractor, input_perplexity=20, input_candidates_num=30)


    print("\n[+] All Done!")