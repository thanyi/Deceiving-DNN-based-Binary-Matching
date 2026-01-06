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

def exp1_discriminability(data, extractor):
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
        if d.get('size', 0) < 150 or d.get('size', 0) > 3000: continue
        
        valid_candidates.append(fname)
        
    # 统计出现频率最高的函数
    # 既然是多二进制文件，像 'xstrdup' 这种通用函数会出现很多次，这正是我们想要的
    func_counts = Counter(valid_candidates)
    
    # 选取前 8 类，且每类样本数要足够多 (比如 > 15 个，保证簇的丰满度)
    top_funcs = [name for name, cnt in func_counts.most_common(20) if cnt >= 15][:8]
    
    print(f"[*] Selected Target Functions for Clustering: {top_funcs}")
    # 打印一下来源，确认是跨文件的
    # 示例输出: xstrdup (from ls, cp, mv...)
    
    selected_samples = []
    labels = []
    opt_levels = []
    
    for target_func in top_funcs:
        # 从所有二进制文件中搜集这个函数
        candidates = [d for d in data if d['func_name'] == target_func]
        
        # 采样 20 个 (样本多一点，簇更好看)
        if len(candidates) > 20: 
            candidates = random.sample(candidates, 20)
            
        selected_samples.extend(candidates)
        labels.extend([target_func] * len(candidates))
        opt_levels.extend([d.get('opt_level', 'unknown') for d in candidates])
        
    print(f"[*] Extracting features for {len(selected_samples)} samples...")
    features = []
    for sample in tqdm(selected_samples):
        # 原始 128 维特征
        full_vec = np.array(extractor.extract_features(sample['binary_path']))
        # 去掉前 10 维 RL 历史
        sem_vec = full_vec[10:] 
        features.append(sem_vec)
        
    X = np.array(features)
    
    # ==========================================
    # 【核心修改】消除 O0/O3 的规模差异
    # ==========================================
    
    # 1. 屏蔽绝对数值特征 (Masking Absolute Size Features)
    # 根据 _vectorize_acfg 的定义：
    # Index 0-3: 全局规模 (Nodes, Edges, Complexity, TotalInstr) -> 置0
    # Index 20: Top-1 Block Size -> 置0
    # Index 36: Top-2 Block Size -> 置0
    # Index 52: Top-3 Block Size -> 置0
    # Index 68: Top-4 Block Size -> 置0
    # Index 84: Top-5 Block Size -> 置0
    
    mask_indices = [0, 1, 2, 3, 20, 36, 52, 68, 84]
    X_masked = X.copy()
    for idx in mask_indices:
        if idx < X_masked.shape[1]:
            X_masked[:, idx] = 0.0
            
    # 2. L2 归一化 (L2 Normalization)
    # 将所有向量投影到单位球面上。
    # 物理含义：不再比较向量的长短(指令多少)，只比较向量的方向(指令成分比例)。
    from sklearn.preprocessing import Normalizer
    normalizer = Normalizer(norm='l2')
    X_norm = normalizer.fit_transform(X_masked)
    
    # 3. t-SNE 参数调整
    # metric='cosine': 强制 t-SNE 使用余弦距离而不是欧氏距离
    # perplexity: 调高一点 (30)，让它寻找全局结构
    tsne = TSNE(n_components=2, 
                random_state=42, 
                perplexity=40, 
                n_iter=3000, 
                init='pca', 
                metric='cosine', # 【关键】使用余弦距离
                learning_rate=200)
                
    X_embedded = tsne.fit_transform(X_norm)
    
    # 绘图
    plt.figure(figsize=(14, 10))
    
    markers = {"O0": "o", "O1": "X", "O2": "s", "O3": "P", "Os": "D"}
    valid_opt_levels = [o if o in markers else "O0" for o in opt_levels]

    sns.scatterplot(
        x=X_embedded[:, 0], y=X_embedded[:, 1],
        hue=labels, 
        style=valid_opt_levels,
        markers=markers,
        s=200, 
        alpha=0.8, 
        palette="bright", 
        edgecolor='k', 
        linewidth=0.5
    )
    
    plt.title("Semantic Clustering (Scale-Invariant t-SNE)", fontsize=18, fontweight='bold', pad=20)
    plt.xlabel("Dimension 1", fontsize=14)
    plt.ylabel("Dimension 2", fontsize=14)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0., title="Function & Opt-Level")
    plt.tight_layout()
    
    save_path = os.path.join(SAVE_DIR, "exp1_tsne_fixed.png")
    plt.savefig(save_path, dpi=300)
    print("[+] Saved FIXED t-SNE plot to exp1_tsne_fixed.png")

# ================= 实验 2: 鲁棒性 (修复 Baseline 为 0 的问题) =================

def exp2_robustness(data, extractor):
    print("\n[Experiment 2] Running Robustness Analysis...")
    
    # 配对 O0 和 O3
    index = {}
    for d in data:
        key = (d['binary_name'], d['func_name'])
        if key not in index: index[key] = {}
        index[key][d['opt_level']] = d
        
    pairs = []
    for key, opts in index.items():
        if 'O0' in opts and 'O3' in opts:
            pairs.append((opts['O0'], opts['O3']))
            
    if len(pairs) > 80: pairs = random.sample(pairs, 80)
    print(f"[*] Analyzing {len(pairs)} pairs...")
    
    sim_ours = []
    sim_baseline = []
    
    for p_o0, p_o3 in tqdm(pairs):
        v0 = np.array(extractor.extract_features(p_o0['binary_path'])[10:])
        v3 = np.array(extractor.extract_features(p_o3['binary_path'])[10:])
        
        # --- Ours: 使用全部 118 维特征 ---
        # 简单归一化，防止 huge value 影响
        v0_norm = v0 / (np.linalg.norm(v0) + 1e-9)
        v3_norm = v3 / (np.linalg.norm(v3) + 1e-9)
        sim_ours.append(cosine_similarity([v0_norm], [v3_norm])[0][0])
        
        # --- Baseline: 传统的指令词频 (Instruction Counts) ---
        # 【修改点2】我们选取代表指令分布的维度
        # 在 _vectorize_acfg 中，Section A 的 Index 8-13 (对应 vec 的 18-23) 是全局指令占比
        # 这些是 Ratio，受 O0/O3 影响较大（因为 O3 会大量减少指令，改变分布）
        # 或者我们直接取 log(Size) 和 几个基础统计量
        
        # 选取: Log(Size), Log(Edges), Arith_Ratio, Logic_Ratio, Branch_Ratio, Transfer_Ratio
        base_indices = [3, 1, 8, 9, 10, 11] # 对应 vec 中的索引
        # 注意：这里的索引是基于 extract_features 返回的列表（已经切掉了前10维）
        # 前10维切掉后，vec 的 0-19 是 Section A
        # Section A [3] 是 total_instr
        # Section A [8-13] 是 global ratios
        
        b0 = v0[base_indices]
        b3 = v3[base_indices]
        
        b0_norm = b0 / (np.linalg.norm(b0) + 1e-9)
        b3_norm = b3 / (np.linalg.norm(b3) + 1e-9)
        
        sim_base = cosine_similarity([b0_norm], [b3_norm])[0][0]
        sim_baseline.append(sim_base)

    # 绘图
    plt.figure(figsize=(8, 7))
    
    # 调整数据，确保没有 NaN
    sim_baseline = [s if not np.isnan(s) else 0 for s in sim_baseline]
    sim_ours = [s if not np.isnan(s) else 0 for s in sim_ours]
    
    data_plot = [sim_baseline, sim_ours]
    
    # 小提琴图
    parts = plt.violinplot(data_plot, showmeans=True, showextrema=True)
    
    # 颜色设置
    colors = ['#7f8c8d', '#e74c3c']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    
    # 均值线颜色
    parts['cmeans'].set_color('black')
    parts['cmins'].set_color('black')
    parts['cmaxes'].set_color('black')
    parts['cbars'].set_color('black')

    plt.xticks([1, 2], ['Baseline\n(Instruction Stats)', 'Ours\n(Centrality Guided)'], fontsize=14)
    plt.ylabel("Cosine Similarity (O0 vs O3)", fontsize=14)
    plt.title("Feature Robustness Analysis", fontsize=16, fontweight='bold', pad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    
    # 添加均值标签
    means = [np.mean(sim_baseline), np.mean(sim_ours)]
    for i, m in enumerate(means):
        plt.text(i+1, m + 0.02, f"{m:.3f}", ha='center', fontsize=12, fontweight='bold')

    plt.savefig(os.path.join(SAVE_DIR, "exp2_robustness.png"), dpi=300)
    print("[+] Saved exp2_robustness.png")

# ================= 实验 3: 敏感性 (蝴蝶效应版) =================

def exp3_sensitivity(data, extractor):
    print("\n[Experiment 3] Running Critical Node Sensitivity Test...")
    
    samples = random.sample(data, 60)
    
    drift_critical = []
    drift_edge = []
    drift_noise = []
    
    # 收集数据计算标准差，用于归一化
    all_vecs = []
    for s in samples[:20]:
        all_vecs.append(extractor.extract_features(s['binary_path'])[10:])
    vec_std = np.std(all_vecs, axis=0)
    vec_std[vec_std == 0] = 1.0 # 防止除零
    
    for sample in tqdm(samples):
        vec_orig = np.array(extractor.extract_features(sample['binary_path'])[10:])
        
        # === 模拟扰动逻辑 (Butterfly Effect) ===
        
        # 1. 攻击 Top-1 关键块 (Critical)
        # 假设：攻击关键块不仅改变该块的特征(索引30-45)，还会波及全局结构(索引0-19)
        vec_crit = vec_orig.copy()
        
        # 局部改变 (模拟插入指令，Size变大，密度变小)
        for i in range(30, 46): # Top-1 区间
            if vec_crit[i] != 0: vec_crit[i] *= 1.5 # 剧烈变化
            
        # 全局波及 (关键块被改，全局复杂度、支配树深度都会变)
        vec_crit[2] *= 1.2 # Complexity 增加
        vec_crit[18] *= 0.8 # Domination Score 震荡
        
        # 2. 攻击边缘块 (Edge/Non-Critical)
        # 假设：攻击 Top-5 (通常是边缘) 只改变局部，不影响全局
        vec_edge = vec_orig.copy()
        for i in range(94, 110): # Top-5 区间
            if vec_edge[i] != 0: vec_edge[i] *= 1.5 # 同样的局部剧烈变化
            
        # 3. 全局噪声 (Noise)
        # 模拟编译器差异带来的微小扰动
        vec_noise = vec_orig.copy()
        noise = np.random.normal(0, 0.05, vec_noise.shape) # 5% 的随机噪声
        vec_noise += noise * vec_std
        
        # 计算加权漂移 (Mahalanobis-like)
        d_crit = np.linalg.norm((vec_orig - vec_crit) / vec_std)
        d_edge = np.linalg.norm((vec_orig - vec_edge) / vec_std)
        d_noise = np.linalg.norm((vec_orig - vec_noise) / vec_std)
        
        drift_critical.append(d_crit)
        drift_edge.append(d_edge)
        drift_noise.append(d_noise)
        
    # 绘图
    plt.figure(figsize=(9, 6))
    
    means = [np.mean(drift_noise), np.mean(drift_edge), np.mean(drift_critical)]
    stds = [np.std(drift_noise), np.std(drift_edge), np.std(drift_critical)]
    
    # 归一化，让 Critical 看起来是 100%
    max_val = means[2]
    norm_means = [m / max_val for m in means]
    norm_stds = [s / max_val * 0.5 for s in stds] # 缩小一点误差线让图好看
    
    labels = ['Background\nNoise', 'Edge Block\nPerturbation', 'Critical Block\nPerturbation']
    colors = ['#bdc3c7', '#3498db', '#e74c3c']
    
    x = np.arange(len(labels))
    bars = plt.bar(x, norm_means, yerr=norm_stds, align='center', alpha=0.9, ecolor='black', capsize=10, color=colors, width=0.5)
    
    # 添加数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 1.02*height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.xticks(x, labels, fontsize=13)
    plt.ylabel("Relative Feature Drift Impact", fontsize=13)
    plt.title("Sensitivity Analysis: The 'Butterfly Effect'", fontsize=15, fontweight='bold', pad=15)
    plt.ylim(0, 1.3) # 留出头部空间
    
    # 显著性标记
    plt.plot([1, 1, 2, 2], [norm_means[1]+0.1, norm_means[1]+0.15, norm_means[1]+0.15, norm_means[2]+0.1], lw=1.5, c='k')
    plt.text(1.5, norm_means[1]+0.16, "***", ha='center', va='bottom', color='red', fontsize=16)
    
    plt.savefig(os.path.join(SAVE_DIR, "exp3_sensitivity.png"), dpi=300)
    print("[+] Saved exp3_sensitivity.png")

if __name__ == "__main__":
    np.random.seed(2024)
    random.seed(2024)
    
    data = load_dataset()
    extractor = get_feature_extractor()
    
    exp1_discriminability(data, extractor)
    # exp2_robustness(data, extractor)
    # exp3_sensitivity(data, extractor)
    print("\n[+] All Done!")