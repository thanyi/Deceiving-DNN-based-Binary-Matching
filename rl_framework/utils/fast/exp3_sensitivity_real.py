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
TEMP_DIR = "tmp/binary_patch_test"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# 绘图风格设置
try:
    plt.style.use('seaborn-v0_8-paper')
except:
    sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# ================= 核心工具函数 =================

def select_drift_subvector(vec, mode="no_section_b"):
    """
    从 240 维 ACFG 向量（去掉前 16 维 RL 历史后）选择漂移计算子向量。

    布局:
    - Section A: [0:40]
    - Section B: [40:200]
    - Section C: [200:240]
    """
    arr = np.asarray(vec)
    if arr.shape[0] < 240:
        return arr
    if mode == "full":
        return arr
    if mode == "no_section_b":
        return np.concatenate([arr[:40], arr[200:240]])
    if mode == "section_c_only":
        return arr[200:240]
    raise ValueError(f"Unsupported drift mode: {mode}")


def apply_nop_patch(binary_path, target_addr, num_bytes=4):
    """
    【物理攻击】在二进制文件的指定虚拟地址处写入 NOP 指令 (0x90)
    """
    filename = os.path.basename(binary_path)
    # 创建临时文件，避免破坏原始数据集
    temp_path = os.path.join(TEMP_DIR, f"patched_{random.randint(1000,9999)}_{filename}")
    shutil.copy(binary_path, temp_path)
    
    try:
        # 使用 r2 以写模式打开 (-w)
        r = r2pipe.open(temp_path, flags=['-w', '-2'])
        # 强制配置架构，防止写入错误的指令集
        r.cmd('e asm.arch=x86')
        r.cmd('e asm.bits=64')
        
        # 写入 NOP (x86: 90)
        # wx 90909090 @ target_addr
        hex_str = "90" * num_bytes
        r.cmd(f'wx {hex_str} @ {target_addr}')
        r.quit()
        return temp_path
    except Exception as e:
        print(f"[!] Patch failed: {e}")
        if os.path.exists(temp_path): os.remove(temp_path)
        return None

def get_target_blocks(binary_path, func_name):
    """
    利用我们第三章的算法，找出 Top-1 关键块和 Bottom-1 边缘块
    """
    extractor = RadareACFGExtractor(binary_path)
    # 获取详细特征数据（包含 block 列表和 scores）
    # 注意：这里需要 r2_acfg_features.py 返回 basic_blocks 和 critical_score
    # 假设我们修改后的代码已经将 critical_score 注入到了 basic_blocks 的 info 中
    
    # 为了准确获取地址，我们需要先拿到函数入口
    # 复用 get_acfg_features 的逻辑
    data = extractor.get_acfg_features(function_name=func_name)
    extractor.close()
    
    if not data or not data.get('basic_blocks'):
        return None, None, None

    blocks = data['basic_blocks']
    # 1. 提取分项指标 (注意：特征提取器已经归一化过了吗？
    # V4.0里是归一化过的，但为了保险，这里取出来的值最好再确认一下)
    # 这里假设 extractor 返回的 bbs_features 里已经有了 centrality 等值
    
    scored_blocks = []
    all_addrs = []
    
    for addr, info in blocks.items():
        # 获取三个核心指标
        bet = info.get('centrality_betweenness', 0)
        dom = info.get('dominator_score', 0) # 注意：这里可能是原始计数值
        deg = info.get('centrality_degree', 0)
        
        # 存原始数据以便后续归一化（如果需要）
        # 但简单起见，我们直接利用提取器里已经归一化的值（如果有）
        # 如果 V4.0 代码没把归一化后的值存进去，只存了原始值，那这里直接加权会有偏。
        # 既然是做实验，我们简单粗暴一点，直接用原始值排序也是可以的，
        # 或者我们只用 bet (介数) 作为核心指标，因为它最能代表“枢纽”。
        
        # 建议：复用论文公式 0.4*Bet + 0.3*Dom + 0.3*Deg
        # 注意量纲：Bet是0-1，Deg是0-1，Dom是整数(0-N)。
        # 必须在这里做一次 Log 平滑或者归一化，否则 Dom 会主导。
        
        import math
        score = 0.4 * bet + 0.3 * deg + 0.3 * math.log1p(dom)
        
        all_addrs.append(addr)
        scored_blocks.append((addr, score))
    
    if not scored_blocks: return None, None, None

    # 排序：分数从高到低
    scored_blocks.sort(key=lambda x: x[1], reverse=True)
    
    # Top-1 (Critical)
    crit_addr = scored_blocks[0][0]
    
    # Bottom-1 (Edge) - 找分数最低的
    edge_addr = scored_blocks[-1][0]
    
    # Random - 随机找一个
    rand_addr = random.choice(all_addrs)
    
    return crit_addr, edge_addr, rand_addr

# ================= 实验主流程 =================

def run_experiment(drift_mode="no_section_b"):
    print("\n[Experiment 3] Real Binary Patching Sensitivity Test...")
    print(f"[*] Drift mode: {drift_mode}")
    
    # 1. 加载并筛选数据
    if not os.path.exists(DATASET_PATH):
        print("Dataset not found.")
        return
        
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)
    
    # 筛选大小适中的函数 (太小没法区分边缘和核心，太大跑得慢)
    candidates = [d for d in data if d['func_name'] != 'main' and 200 < d.get('size', 0) < 5000]
    
    # 随机采样 50 个样本
    if len(candidates) > 50:
        samples = random.sample(candidates, 50)
    else:
        samples = candidates
        
    print(f"[*] Selected {len(samples)} samples for physical patching.")
    
    # 初始化环境 wrapper (用于提取特征向量)
    env = BinaryPerturbationEnv(save_path=TEMP_DIR, dataset_path=DATASET_PATH)
    
    # 记录漂移量
    drifts = {
        'Critical': [],
        'Edge': [],
        'Random': []
    }
    
    # 预计算标准差用于归一化 (Mahalanobis distance)
    # 这是一个 trick：为了让距离有意义，我们需要知道特征分布的尺度
    print("[-] Pre-calculating feature distribution...")
    base_vecs = []
    for s in samples[:50]:
        raw_v = np.array(env.extract_features_from_function(s['binary_path'], s['func_name'])[16:]) # 去掉 RL 历史
        v = select_drift_subvector(raw_v, mode=drift_mode)
        base_vecs.append(v)
    vec_std = np.std(base_vecs, axis=0)
    vec_std[vec_std == 0] = 1.0 # 防止除零
    
    print("[-] Starting patching process...")
    
    for sample in tqdm(samples):
        orig_path = sample['binary_path']
        fname = sample['func_name']
        
        # 1. 获取原始特征
        # 注意：env.extract_features 内部会处理地址解析
        # 为了保证对比公平，我们在这里显式指定 function_name
        env.original_binary = orig_path
        env.function_name = fname
        raw_orig = np.array(env.extract_features_from_function(orig_path, fname)[16:])
        vec_orig = select_drift_subvector(raw_orig, mode=drift_mode)
        
        # 2. 定位目标块
        crit_addr, edge_addr, rand_addr = get_target_blocks(orig_path, fname)
        print(f"[exp3_sensitivity_real:run_experiment] crit_addr: {crit_addr}, edge_addr: {edge_addr}, rand_addr: {rand_addr}")
        if crit_addr is None: continue
        
        # 定义三次攻击任务
        tasks = [
            ('Critical', crit_addr),
            ('Edge', edge_addr),
            ('Random', rand_addr)
        ]
        
        for label, target_addr in tasks:
            # 3. 物理打补丁
            patched_path = apply_nop_patch(orig_path, target_addr, num_bytes=4)
            # print(f"[exp3_sensitivity_real:run_experiment] patched_path: {patched_path}")
            # print(f"[exp3_sensitivity_real:run_experiment] fname: {label}")
            if patched_path:
                try:
                    # 4. 提取变异后特征
                    # 注意：这是对新文件提取，特征提取器会重新计算中心性等
                    print(f"[exp3_sensitivity_real:run_experiment] patched_path: {patched_path}, fname: {fname}")
                    raw_new = np.array(env.extract_features_from_function(patched_path, fname)[16:])
                    vec_new = select_drift_subvector(raw_new, mode=drift_mode)
                    print(f"[exp3_sensitivity_real:run_experiment] vec_new shape: {vec_new.shape}, vec_orig shape: {vec_orig.shape}")
                    
                    # 确保特征维度一致
                    if vec_new.shape != vec_orig.shape:
                        print(f"[!] Feature dimension mismatch: {vec_new.shape} vs {vec_orig.shape}, skipping...")
                        continue
                    
                    # 5. 计算漂移 (Normalized Euclidean Distance)
                    # d = sqrt( sum( (x_i - y_i)^2 / std_i^2 ) )
                    dist = np.linalg.norm((vec_orig - vec_new) / vec_std)
                    print(f"[exp3_sensitivity_real:run_experiment] {label} drift: {dist}")
                    
                    drifts[label].append(dist)
                except Exception as e:
                    print(f"[!] Failed to extract features for {label} @ {hex(target_addr)}: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    # 清理临时文件
                    if os.path.exists(patched_path):
                        os.remove(patched_path)

    # ================= 结果统计与可视化 =================
    print(f"[exp3_sensitivity_real:run_experiment] drifts: {drifts}")
    
    # 检查是否有有效数据
    if not any(drifts.values()):
        print("[!] No valid data collected! All feature extractions failed.")
        print("[!] Check if binaries are corrupted or if radare2 can analyze them.")
        return
    
    # 转换为数组
    d_crit = np.array(drifts['Critical'])
    d_edge = np.array(drifts['Edge'])
    d_rand = np.array(drifts['Random'])
    
    # 去除无效值 (极少数情况下可能提取失败)
    d_crit = d_crit[d_crit > 0]
    d_edge = d_edge[d_edge > 0]
    d_rand = d_rand[d_rand > 0]
    
    # 打印统计信息
    print("\n[Results]")
    print(f"Critical Mean Drift: {np.mean(d_crit):.4f} (std: {np.std(d_crit):.4f})")
    print(f"Edge     Mean Drift: {np.mean(d_edge):.4f} (std: {np.std(d_edge):.4f})")
    print(f"Random   Mean Drift: {np.mean(d_rand):.4f} (std: {np.std(d_rand):.4f})")
    
    # 进行 T检验 (T-Test) 证明显著性
    # 证明 Critical > Edge 是统计显著的
    t_stat, p_val = stats.ttest_ind(d_crit, d_edge, alternative='greater')
    print(f"\nT-Test (Critical > Edge): p-value = {p_val:.4e}")
    significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
    
    # 绘图
    plt.figure(figsize=(10, 7))
    
    means = [np.mean(d_rand), np.mean(d_edge), np.mean(d_crit)]
    errors = [np.std(d_rand)/np.sqrt(len(d_rand)), 
              np.std(d_edge)/np.sqrt(len(d_edge)), 
              np.std(d_crit)/np.sqrt(len(d_crit))] # 使用标准误 SE
    
    labels = ['Random\nPatching', 'Edge Block\nPatching', 'Critical Block\nPatching']
    colors = ['#95a5a6', '#3498db', '#e74c3c'] # 灰、蓝、红
    
    x_pos = np.arange(len(labels))
    bars = plt.bar(x_pos, means, yerr=errors, align='center', alpha=0.85, 
                   ecolor='black', capsize=8, color=colors, width=0.65, edgecolor='white', linewidth=1.5)
    
    # 在柱子上标数值 - 增加间距，调整位置
    for bar in bars:
        height = bar.get_height()
        # 增加距离：从 1.05 改为 1.12，让数字离柱子更远
        plt.text(bar.get_x() + bar.get_width()/2., height + max(errors) * 1.3,
                f'{height:.2f}', ha='center', va='bottom', 
                fontsize=13, fontweight='bold', color='#2c3e50')
    
    # 画显著性标记线 - 调整位置，避免和数字重叠
    max_h = max(means) + max(errors)
    sig_y_start = max_h + max(errors) * 2.5  # 增加起始高度
    sig_y_end = sig_y_start + max(errors) * 0.8
    plt.plot([1, 1, 2, 2], [sig_y_start, sig_y_end, sig_y_end, sig_y_start], 
             lw=1.8, c='black')
    plt.text(1.5, sig_y_end + max(errors) * 0.3, f"{significance}\n(p = {p_val:.4e})", 
             ha='center', va='bottom', color='#e74c3c', fontsize=11, fontweight='bold')

    plt.ylabel('Feature Vector Drift (Normalized L2 Distance)', fontsize=13, fontweight='bold')
    plt.title('Sensitivity Analysis of Critical Region Perception', fontsize=15, fontweight='bold', pad=20)
    plt.xticks(x_pos, labels, fontsize=12)
    plt.ylim(0, max_h * 1.8) # 增加顶部空间，从 1.4 改为 1.8
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(SAVE_DIR, "exp3_real_sensitivity.png")
    plt.savefig(save_path, dpi=300)
    print(f"[+] Plot saved to {save_path}")
    
    # 清理
    shutil.rmtree(TEMP_DIR)

if __name__ == "__main__":
    # 固定随机种子
    random.seed(2024)
    np.random.seed(2024)
    # 可选: "full" / "no_section_b" / "section_c_only"
    DRIFT_MODE = "no_section_b"
    run_experiment(drift_mode=DRIFT_MODE)
