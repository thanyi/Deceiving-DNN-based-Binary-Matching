#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import random
import sys

# 路径适配
sys.path.insert(0, '/home/ycy/ours/Deceiving-DNN-based-Binary-Matching')    
from rl_framework.env_wrapper import BinaryPerturbationEnv
from rl_framework.utils.acfg.r2_acfg_features import RadareACFGExtractor

# ================= 配置 =================
DATASET_PATH = "dataset_train.json"
SAVE_DIR = "chapter3_results_0115"
os.makedirs(SAVE_DIR, exist_ok=True)

# ================= 实验数据量配置 =================
# 小实验配置（快速测试）
MAX_QUERY_SIZE = 20        # Query 数量（原为 100）
MAX_GALLERY_SIZE = 100      # Gallery 最大数量（None 表示使用全部）
MIN_FUNCTION_SIZE = 100     # 最小函数大小过滤（原为 100）

# 完整实验配置（取消注释使用）
# MAX_QUERY_SIZE = 100
# MAX_GALLERY_SIZE = None
# MIN_FUNCTION_SIZE = 100

# 绘图风格
try:
    plt.style.use('seaborn-v0_8-paper')
except:
    sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

def build_baseline_vector(acfg_data):
    """
    Baseline: 密度敏感型指令统计 (同实验2)
    """
    if not acfg_data or 'basic_blocks' not in acfg_data:
        return np.zeros(10)
    total_instr = 0
    num_blocks = max(acfg_data.get('num_nodes', 0), 1)
    counts = np.zeros(8)
    keys = ['n_arith', 'n_logic', 'n_branch', 'n_transfer', 'n_mem_write', 'n_mem_read', 'n_consts', 'n_regs_gp']
    for bb in acfg_data['basic_blocks'].values():
        total_instr += bb.get('n_instructions', 0)
        for i, k in enumerate(keys):
            counts[i] += bb.get(k, 0)
    vec = list(counts / num_blocks) 
    vec.append(total_instr / num_blocks)
    vec.append(acfg_data.get('num_edges', 0) / num_blocks)
    return np.array(vec)

def run_retrieval_experiment():
    print("\n[Experiment 4] Binary Code Similarity Search (Retrieval Task)...")
    
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)
        
    # 1. 构建 O0 和 O3 的映射
    # map: (bin_name, func_name, version) -> path
    pool_o0 = {}
    pool_o3 = {}
    
    print("[*] Indexing dataset...")
    for d in data:
        if d['func_name'] == 'main': continue
        if d.get('size', 0) < MIN_FUNCTION_SIZE: continue # 过滤太小的
        
        key = (d['binary_name'], d['func_name'], d.get('version', 'unknown'))
        path = d['binary_path']
        
        if d['opt_level'] == 'O0':
            pool_o0[key] = path
        elif d['opt_level'] == 'O3':
            pool_o3[key] = path
            
    # 2. 找出共有的 Key (作为 Ground Truth)
    common_keys = list(set(pool_o0.keys()) & set(pool_o3.keys()))
    
    # 随机选取 Query (O3)
    if len(common_keys) > MAX_QUERY_SIZE:
        query_keys = random.sample(common_keys, MAX_QUERY_SIZE)
    else:
        query_keys = common_keys
        
    # 检索库 (Gallery): 所有的 O0 函数 (包含 Query 对应的 O0，也包含其他的作为干扰)
    # 为了增加难度，我们可以加入 dataset 中所有的 O0 函数
    gallery_keys = list(pool_o0.keys())
    
    # 限制 Gallery 大小（小实验用）
    if MAX_GALLERY_SIZE is not None and len(gallery_keys) > MAX_GALLERY_SIZE:
        # 确保包含所有 query 对应的 target
        query_targets = set(query_keys)  # query_keys 对应的 O0 版本
        other_keys = [k for k in gallery_keys if k not in query_targets]
        # 随机选择其他干扰项
        n_others = MAX_GALLERY_SIZE - len(query_targets)
        if n_others > 0:
            selected_others = random.sample(other_keys, min(n_others, len(other_keys)))
            gallery_keys = list(query_targets) + selected_others
        else:
            gallery_keys = list(query_targets)
        print(f"[*] Gallery size limited to {len(gallery_keys)} (from {len(pool_o0)} total)")
    
    print(f"[*] Query Set (O3): {len(query_keys)}")
    print(f"[*] Gallery Set (O0): {len(gallery_keys)} (1 Target + {len(gallery_keys)-1} Distractors)")
    
    # 3. 提取特征 (预计算 Gallery)
    print("[-] Extracting features for Gallery (O0)...")
    env = BinaryPerturbationEnv(save_path="/tmp/test_search", dataset_path=DATASET_PATH)
    
    gallery_vecs_ours = []
    gallery_vecs_base = []
    
    # 为了快速索引
    gallery_key_list = [] 
    
    for k in tqdm(gallery_keys):
        path = pool_o0[k]
        
        # Extractor for Baseline
        ext = RadareACFGExtractor(path)
        data_raw = ext.get_acfg_features(function_name=k[1])
        ext.close()
        
        if not data_raw: continue
        
        # Ours
        # 显式设置 env
        env.original_binary = path
        env.function_name = k[1]
        v_ours_full = np.array(env.extract_features(path))
        v_ours = v_ours_full[16:] # 去掉 RL 历史
        
        # 权重掩码
        weights = np.ones_like(v_ours)
        weights[:40] *= 0.3 # 给拓扑特征打个 3 折，别完全丢掉
        # 只取 Section B+C
        v_ours = v_ours * weights
        
        # Baseline
        v_base = build_baseline_vector(data_raw)
        
        # Norm
        if np.linalg.norm(v_ours) > 0: v_ours /= np.linalg.norm(v_ours)
        if np.linalg.norm(v_base) > 0: v_base /= np.linalg.norm(v_base)
        
        gallery_vecs_ours.append(v_ours)
        gallery_vecs_base.append(v_base)
        gallery_key_list.append(k)
        
    gallery_vecs_ours = np.array(gallery_vecs_ours)
    gallery_vecs_base = np.array(gallery_vecs_base)
    
    # 4. 执行检索
    print("[-] Running Queries...")
    
    ranks_ours = []
    ranks_base = []
    
    for q_k in tqdm(query_keys):
        path = pool_o3[q_k]
        
        # Extract Query Features
        ext = RadareACFGExtractor(path)
        data_raw = ext.get_acfg_features(function_name=q_k[1])
        ext.close()
        
        if not data_raw: continue
        
        # Ours
        env.original_binary = path
        env.function_name = q_k[1]
        q_ours = np.array(env.extract_features(path)[16:]) # Section A+B+C
        if np.linalg.norm(q_ours) > 0: q_ours /= np.linalg.norm(q_ours)
        
        # Baseline
        q_base = build_baseline_vector(data_raw)
        if np.linalg.norm(q_base) > 0: q_base /= np.linalg.norm(q_base)
        
        # Calculate Sim with all Gallery
        # Ours
        sims_o = cosine_similarity([q_ours], gallery_vecs_ours)[0]
        # Baseline
        sims_b = cosine_similarity([q_base], gallery_vecs_base)[0]
        
        # Find Rank
        # 目标 Key 是 q_k
        # 我们需要在 gallery_key_list 中找到 q_k 对应的 index
        try:
            target_idx = gallery_key_list.index(q_k)
        except ValueError:
            continue # Should not happen
            
        # Get Rank
        # argsort 返回的是从小到大的索引，[::-1] 反转
        sorted_indices_o = np.argsort(sims_o)[::-1]
        sorted_indices_b = np.argsort(sims_b)[::-1]
        
        # 找到 target_idx 在排序后列表中的位置 (Rank 从 1 开始)
        rank_o = np.where(sorted_indices_o == target_idx)[0][0] + 1
        rank_b = np.where(sorted_indices_b == target_idx)[0][0] + 1
        
        ranks_ours.append(rank_o)
        ranks_base.append(rank_b)

    # 5. 计算指标
    def calc_metrics(ranks):
        ranks = np.array(ranks)
        recall_1 = np.mean(ranks == 1)
        recall_5 = np.mean(ranks <= 5)
        mrr = np.mean(1.0 / ranks)
        return recall_1, recall_5, mrr

    r1_o, r5_o, mrr_o = calc_metrics(ranks_ours)
    r1_b, r5_b, mrr_b = calc_metrics(ranks_base)
    
    print("\n[Results]")
    print(f"{'Metric':<15} | {'Baseline':<10} | {'Ours':<10}")
    print("-" * 40)
    print(f"{'Recall@1':<15} | {r1_b:.4f}     | {r1_o:.4f}")
    print(f"{'Recall@5':<15} | {r5_b:.4f}     | {r5_o:.4f}")
    print(f"{'MRR':<15}      | {mrr_b:.4f}     | {mrr_o:.4f}")
    
    # 绘图: CMC 曲线 (Cumulative Match Characteristic)
    max_k = 20
    cmc_o = [np.mean(np.array(ranks_ours) <= k) for k in range(1, max_k+1)]
    cmc_b = [np.mean(np.array(ranks_base) <= k) for k in range(1, max_k+1)]
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_k+1), cmc_o, 'r-o', label=f'Ours (MRR={mrr_o:.2f})', linewidth=2)
    plt.plot(range(1, max_k+1), cmc_b, 'k--s', label=f'Baseline (MRR={mrr_b:.2f})', linewidth=2, alpha=0.6)
    
    plt.xlabel('Rank (k)', fontsize=12)
    plt.ylabel('Recall @ k', fontsize=12)
    plt.title('Retrieval Performance (O3 Query vs O0 Gallery)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(range(1, max_k+1, 2))
    
    save_path = os.path.join(SAVE_DIR, "exp4_retrieval.png")
    plt.savefig(save_path, dpi=300)
    print(f"[+] Saved {save_path}")

if __name__ == "__main__":
    random.seed(2024)
    np.random.seed(2024)
    run_retrieval_experiment()