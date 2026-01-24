#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import r2pipe
import sys
from scipy import stats

# 引入你的模块
sys.path.insert(0, '/home/ycy/ours/Deceiving-DNN-based-Binary-Matching')    
from rl_framework.env_wrapper import BinaryPerturbationEnv
from rl_framework.utils.acfg.r2_acfg_features import RadareACFGExtractor

# ================= 配置 =================
DATASET_PATH = "dataset_train.json"
SAVE_DIR = "chapter3_results_final"
TEMP_DIR = "/tmp/ablation_test"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# 绘图风格
try:
    plt.style.use('seaborn-v0_8-paper')
except:
    sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# ================= 工具函数 =================

def apply_nop_patch(binary_path, target_addr, num_bytes=4):
    """
    物理攻击：写入 NOP
    """
    filename = os.path.basename(binary_path)
    # 随机后缀防止冲突
    temp_path = os.path.join(TEMP_DIR, f"patched_{random.randint(10000,99999)}_{filename}")
    shutil.copy(binary_path, temp_path)
    
    try:
        r = r2pipe.open(temp_path, flags=['-w', '-2'])
        r.cmd('e asm.arch=x86')
        r.cmd('e asm.bits=64')
        hex_str = "90" * num_bytes
        # 确保地址格式正确（如果是整数，转换为十六进制字符串）
        if isinstance(target_addr, int):
            addr_str = hex(target_addr)
        else:
            addr_str = str(target_addr)
        r.cmd(f'wx {hex_str} @ {addr_str}')
        r.quit()
        
        # 验证 patch 是否成功（可选调试）
        # r2 = r2pipe.open(temp_path, flags=['-2'])
        # patched = r2.cmd(f'px {num_bytes} @ {addr_str}')
        # print(f"Patched {num_bytes} bytes at {addr_str}: {patched}")
        # r2.quit()
        
        return temp_path
    except Exception as e:
        # print(f"Patch failed: {e}")
        if os.path.exists(temp_path): os.remove(temp_path)
        return None

def get_target_by_strategy(binary_path, func_name, strategy):
    """
    根据不同策略选择目标块
    """
    extractor = RadareACFGExtractor(binary_path)
    # 获取详细特征数据
    data = extractor.get_acfg_features(function_name=func_name)
    extractor.close()
    
    if not data or not data.get('basic_blocks'):
        return None

    blocks = data['basic_blocks']
    candidates = [] # (addr, score)
    
    for addr, info in blocks.items():
        # 根据策略计算分数
        if strategy == 'Random':
            score = random.random()
        elif strategy == 'Size':
            # 选指令数最多的
            score = info.get('n_instructions', 0)
        elif strategy == 'Degree':
            # 选度中心性最高的
            score = info.get('centrality_degree', 0)
        elif strategy == 'Ours':
            # 选我们算法算出的综合分
            score = info.get('critical_score', 0)
        else:
            score = 0
            
        candidates.append((addr, score))
    
    if not candidates: return None

    # 按分数降序排列，取第一名
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected_addr = candidates[0][0]
    selected_score = candidates[0][1]
    
    # 调试信息：验证不同策略选择了不同的地址
    print(f"Strategy {strategy}: selected addr = {selected_addr:#x}, score = {selected_score:.4f}")
    
    return selected_addr  # 返回地址

# ================= 实验主流程 =================

def run_ablation_experiment():
    print("\n[Experiment B] Metric Ablation Study (Comparison of Selection Strategies)...")
    
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)
    
    # 筛选：稍微复杂一点的函数，太简单的函数所有策略选出来的块可能一样
    candidates = [d for d in data if d['func_name'] != 'main' and 300 < d.get('size', 0) < 5000]
    
    # 采样 60 个
    if len(candidates) > 20:
        samples = random.sample(candidates, 20)
    else:
        samples = candidates
        
    print(f"[*] Selected {len(samples)} samples.")
    
    env = BinaryPerturbationEnv(save_path=TEMP_DIR, dataset_path=DATASET_PATH)
    
    # 记录每种策略的漂移量
    strategies = ['Random', 'Size', 'Degree', 'Ours']
    results = {s: [] for s in strategies}
    
    # 预计算标准差用于归一化
    print("[-] Pre-calculating feature distribution...")
    base_vecs = []
    for s in samples[:20]:
        try:
            env.original_binary = s['binary_path']
            env.function_name = s['func_name']
            v = env.extract_features(s['binary_path'])[16:]  # 去掉 RL 历史 (16维)
            base_vecs.append(v)
        except: pass
    
    if len(base_vecs) > 0:
        vec_std = np.std(base_vecs, axis=0)
        vec_std[vec_std == 0] = 1.0
    else:
        # 特征维度：256 - 16 (RL历史) = 240
        vec_std = np.ones(240)
    
    print("[-] Running strategies comparison...")
    
    for sample in tqdm(samples):
        orig_path = sample['binary_path']
        fname = sample['func_name']
        
        try:
            # 1. 原始特征
            env.original_binary = orig_path
            env.function_name = fname
            vec_orig = np.array(env.extract_features(orig_path)[16:])  # 去掉 RL 历史 (16维)
            
            # 2. 对每种策略进行攻击
            for strat in strategies:
                target_addr = get_target_by_strategy(orig_path, fname, strat)
                if not target_addr: continue
                
                # Patch
                patched_path = apply_nop_patch(orig_path, target_addr)
                if patched_path:
                    try:
                        # 🔧 关键修复：
                        # 1. 清除缓存，确保每次提取都是新的（因为 patched_path 不同）
                        env.clear_acfg_cache()
                        # 2. 设置环境状态
                        env.original_binary = orig_path  # 保持原始文件作为参考
                        env.function_name = fname        # 设置函数名
                        # 3. 使用 extract_features_from_function 直接指定函数名，避免地址解析问题
                        vec_new = np.array(env.extract_features_from_function(patched_path, fname)[16:])  # 去掉 RL 历史 (16维)
                        
                        # 验证特征是否有效（不应该全为0）
                        if np.all(vec_new == 0):
                            print(f"Warning: {strat} extracted zero vector for {fname} @ {hex(target_addr)}")
                            continue
                        
                        # 计算漂移
                        drift = np.linalg.norm((vec_orig - vec_new) / vec_std)
                        results[strat].append(drift)
                    except Exception as e:
                        # 添加调试信息
                        print(f"Error: Failed to extract features for {strat} on {fname} @ {hex(target_addr)}: {e}")
                        import traceback
                        traceback.print_exc()
                        pass
                    finally:
                        if os.path.exists(patched_path): os.remove(patched_path)
                        
        except Exception as e:
            print(f"Error: {e}")
            pass

    # ================= 结果与画图 =================
    
    # 计算均值和标准误
    means = []
    sems = []
    
    print("\n[Results Summary]")
    for strat in strategies:
        arr = np.array(results[strat])
        mean = np.mean(arr) if len(arr) > 0 else 0
        sem = stats.sem(arr) if len(arr) > 1 else 0
        means.append(mean)
        sems.append(sem)
        print(f"  {strat:<10}: Mean Drift = {mean:.4f} (+/- {sem:.4f})")
    
    # 统计检验 (Ours vs Degree, Ours vs Size)
    # 证明比单一指标好
    p_size = stats.ttest_ind(results['Ours'], results['Size'], alternative='greater').pvalue
    p_deg = stats.ttest_ind(results['Ours'], results['Degree'], alternative='greater').pvalue
    
    print(f"\nT-Test (Ours > Size):   p = {p_size:.4e}")
    print(f"T-Test (Ours > Degree): p = {p_deg:.4e}")

    # 绘图
    plt.figure(figsize=(9, 6))
    
    x = np.arange(len(strategies))
    # 颜色：Ours 最深红，其他渐变
    colors = ['#95a5a6', '#85c1e9', '#3498db', '#e74c3c'] 
    
    bars = plt.bar(x, means, yerr=sems, align='center', alpha=0.9, ecolor='black', capsize=10, color=colors, width=0.6)
    
    plt.ylabel('Feature Vector Drift (Normalized)', fontsize=12)
    plt.title('Ablation Study: Critical Block Selection Strategies', fontsize=14, fontweight='bold')
    plt.xticks(x, ['Random', 'Size-based\n(Instruction Count)', 'Degree-based\n(Connectivity)', 'Ours\n(Centrality Fusion)'], fontsize=11)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 1.05*height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 添加显著性连线 (如果显著)
    if p_deg < 0.05:
        h = max(means) + 0.5
        plt.plot([2, 2, 3, 3], [means[2]+0.2, h, h, means[3]+0.2], lw=1.5, c='k')
        sig_symbol = "***" if p_deg < 0.001 else "**"
        plt.text(2.5, h+0.05, sig_symbol, ha='center', va='bottom', color='red', fontsize=14)

    plt.ylim(0, max(means) * 1.3)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    
    save_path = os.path.join(SAVE_DIR, "exp_ablation_study.png")
    plt.savefig(save_path, dpi=300)
    print(f"[+] Plot saved to {save_path}")
    
    # 清理
    shutil.rmtree(TEMP_DIR)

if __name__ == "__main__":
    random.seed(2024)
    np.random.seed(2024)
    run_ablation_experiment()