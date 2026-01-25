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
import sys
from scipy import stats
import r2pipe

# 路径适配
sys.path.insert(0, '/home/ycy/ours/Deceiving-DNN-based-Binary-Matching')
from rl_framework.env_wrapper import BinaryPerturbationEnv
from rl_framework.utils.acfg.r2_acfg_features import RadareACFGExtractor
from run_utils import run_one

# ================= 配置 =================
DATASET_PATH = "dataset_train.json"
SAVE_DIR = "chapter3_results_final"
os.makedirs(SAVE_DIR, exist_ok=True)

# 绘图风格
try:
    plt.style.use('seaborn-v0_8-paper')
except:
    sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# ================= 工具函数 =================

def generate_junk_instructions(target_bytes=64):
    """
    生成指定字节数的垃圾指令（无副作用，不改变程序语义）

    包含的指令类型：
    - push/pop 配对 (2 bytes each)
    - nop (1 byte)
    - xchg rax, rax (3 bytes)
    - lea rax, [rax+0] (4 bytes)
    """
    junk_patterns = [
        "90",              # nop (1 byte)
        "50 58",           # push rax; pop rax (2 bytes)
        "51 59",           # push rcx; pop rcx (2 bytes)
        "52 5a",           # push rdx; pop rdx (2 bytes)
        "53 5b",           # push rbx; pop rbx (2 bytes)
        "4887c0",          # xchg rax, rax (3 bytes)
        "488d4000",        # lea rax, [rax+0] (4 bytes)
        "488d4900",        # lea rcx, [rcx+0] (4 bytes)
        "488d5200",        # lea rdx, [rdx+0] (4 bytes)
    ]

    result = []
    current_bytes = 0

    while current_bytes < target_bytes:
        pattern = random.choice(junk_patterns)
        pattern_bytes = len(pattern.replace(" ", "")) // 2

        if current_bytes + pattern_bytes <= target_bytes:
            result.append(pattern)
            current_bytes += pattern_bytes
        else:
            # 填充剩余空间用 NOP
            remaining = target_bytes - current_bytes
            result.append("90" * remaining)
            break

    return "".join(result).replace(" ", "")

def apply_junk_patch(binary_path, target_addr, num_bytes=64):
    """
    在指定地址插入垃圾指令，返回修改后的文件路径

    Args:
        binary_path: 原始二进制文件路径
        target_addr: 目标地址
        num_bytes: 插入的字节数 (默认 64)
    """
    filename = os.path.basename(binary_path)
    temp_path = os.path.join("/tmp", f"junk_patch_{random.randint(10000,99999)}_{filename}")
    shutil.copy(binary_path, temp_path)

    try:
        r = r2pipe.open(temp_path, flags=['-w', '-2'])
        r.cmd('e asm.arch=x86')
        r.cmd('e asm.bits=64')

        # 生成垃圾指令
        junk_hex = generate_junk_instructions(num_bytes)

        addr_str = hex(target_addr) if isinstance(target_addr, int) else str(target_addr)
        r.cmd(f'wx {junk_hex} @ {addr_str}')
        r.quit()

        # 验证文件
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            return None

        return temp_path
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return None

def get_degree_and_random_blocks(binary_path, func_name):
    """获取度中心性最大的块和随机块地址"""
    extractor = RadareACFGExtractor(binary_path)
    data = extractor.get_acfg_features(function_name=func_name)
    extractor.close()

    if not data or not data.get('basic_blocks'):
        return None, None

    # 从 basic_blocks 中提取度中心性
    blocks = []
    for addr, bb_info in data['basic_blocks'].items():
        degree_score = bb_info.get('centrality_degree', 0)
        blocks.append((addr, degree_score))

    if not blocks:
        return None, None

    # 按度中心性排序，取最大的
    blocks.sort(key=lambda x: x[1], reverse=True)
    degree_addr = blocks[0][0]

    # 随机选一个基本块（排除度中心性最大的块，确保不同）
    other_blocks = [addr for addr, score in blocks[1:] if len(blocks) > 1]
    if other_blocks:
        random_addr = random.choice(other_blocks)
    else:
        # 如果只有一个块，就用同一个
        random_addr = degree_addr

    return degree_addr, random_addr

# ================= 实验主流程 =================

def run_correlation_experiment(junk_bytes=64):
    print("\n[Experiment] Proxy Correlation Analysis (Feature Drift vs Similarity Drop)...")
    print(f"[*] Using junk instruction injection ({junk_bytes} bytes)")
    print("[*] Comparing: Highest Degree Centrality vs Random Block")

    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)

    # 筛选中等大小函数
    candidates = [d for d in data if d['func_name'] != 'main' and 200 < d.get('size', 0) < 3000]

    # 采样 60 个函数，每个 2 次变异（度中心性最大块/随机块），共 120 个点
    samples = random.sample(candidates, min(60, len(candidates)))

    print(f"[*] Selected {len(samples)} samples, targeting ~{len(samples)*2} data points...")

    temp_dir = "/tmp/proxy_corr_exp_degree"
    os.makedirs(temp_dir, exist_ok=True)
    env = BinaryPerturbationEnv(save_path=temp_dir, dataset_path=DATASET_PATH)

    # 用于 run_one 的缓存
    asm_cache = {}

    # 预计算特征标准差用于归一化
    print("[-] Pre-calculating feature std...")
    base_vecs = []
    for s in samples[:20]:
        try:
            # 直接提取特征，不调用 reset（避免污染 env.original_binary）
            v = np.array(env.extract_features_from_function(s['binary_path'], s['func_name'])[16:])
            base_vecs.append(v)
        except:
            pass

    if not base_vecs:
        print("[!] Failed to extract features for std calculation")
        env.close()
        shutil.rmtree(temp_dir, ignore_errors=True)
        return

    vec_std = np.std(base_vecs, axis=0)
    vec_std[vec_std == 0] = 1.0

    # 数据收集
    x_ours_drift = []
    y_target_drop = []
    point_types = []

    for sample in tqdm(samples, desc="Collecting data"):
        orig_path = sample['binary_path']
        fname = sample['func_name']

        try:
            # 直接提取特征
            vec_orig = np.array(env.extract_features_from_function(orig_path, fname)[16:])

            # 获取度中心性最大的块和随机块
            degree_addr, random_addr = get_degree_and_random_blocks(orig_path, fname)
            if not degree_addr or not random_addr:
                continue

            # 对两种块分别插入垃圾指令
            for label, target_addr in [('Degree', degree_addr), ('Random', random_addr)]:
                patched_path = None
                try:
                    # 插入垃圾指令
                    patched_path = apply_junk_patch(orig_path, target_addr, num_bytes=junk_bytes)
                    if not patched_path:
                        continue

                    # 计算特征漂移 (X轴)
                    vec_new = np.array(env.extract_features_from_function(patched_path, fname)[16:])
                    drift = np.linalg.norm((vec_orig - vec_new) / vec_std)

                    # 计算目标模型相似度下降 (Y轴)
                    # 使用 simple_mode，不依赖 pickle 文件
                    target_score, _ = run_one(
                        orig_path,
                        patched_path,
                        None,  # model_original
                        None,  # checkdict
                        fname,
                        simple_mode=True,
                        asm_work_dir=temp_dir,
                        original_asm_cache=asm_cache
                    )

                    if target_score is None:
                        continue

                    # 相似度下降 = 1.0 - 当前相似度
                    drop = 1.0 - target_score

                    x_ours_drift.append(drift)
                    y_target_drop.append(drop)
                    point_types.append(label)

                except Exception as inner_e:
                    pass
                finally:
                    # 清理临时 patch 文件
                    if patched_path and os.path.exists(patched_path):
                        os.remove(patched_path)

        except Exception as e:
            pass

    # 统计与绘图
    if len(x_ours_drift) < 10:
        print(f"[!] Not enough data points: {len(x_ours_drift)}")
        env.close()
        shutil.rmtree(temp_dir, ignore_errors=True)
        return

    x = np.array(x_ours_drift)
    y = np.array(y_target_drop)

    r_val, p_val = stats.pearsonr(x, y)
    print(f"\n[Results]")
    print(f"Data points: {len(x)}")
    print(f"Pearson r: {r_val:.4f}")
    print(f"P-value: {p_val:.4e}")

    # 绘图
    plt.figure(figsize=(9, 7))

    colors = {'Degree': '#e74c3c', 'Random': '#3498db'}
    c_map = [colors[t] for t in point_types]

    plt.scatter(x, y, c=c_map, alpha=0.7, s=60, edgecolors='w', linewidth=0.5)
    sns.regplot(x=x, y=y, scatter=False, color='#2c3e50', line_kws={'linestyle': '--', 'linewidth': 2})

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Highest Degree Centrality',
               markerfacecolor='#e74c3c', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Random Block',
               markerfacecolor='#3498db', markersize=10),
        Line2D([0], [0], color='#2c3e50', lw=2, linestyle='--',
               label=f'Trend (r={r_val:.2f})')
    ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=11, frameon=True)

    plt.xlabel("Feature Space Drift (Normalized L2)", fontsize=13)
    plt.ylabel("Target Model Similarity Drop", fontsize=13)
    plt.title("Proxy Correlation: Highest Degree Centrality vs Random Block",
              fontsize=15, fontweight='bold', pad=15)
    plt.grid(True, linestyle='--', alpha=0.4)

    save_path = os.path.join(SAVE_DIR, "exp8_degree_vs_random.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[+] Saved to {save_path}")

    # env.close()
    shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    random.seed(2024)
    np.random.seed(2024)

    # 实验参数：垃圾指令字节数
    # 推荐: 64 字节 (平衡扰动强度和函数识别成功率)
    # 可选: 32, 64, 128
    JUNK_BYTES = 64

    run_correlation_experiment(junk_bytes=JUNK_BYTES)
