#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import r2pipe
import json
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys
from graphviz import Digraph

# ================= 配置区域 =================
# 你想分析的二进制文件路径 (建议选一个 O0 或者 O1 的，结构清晰)
# 这里以 dataset_train.json 里的某一个为例，你可以手动替换成你觉得逻辑复杂的那个
BINARY_PATH = "/home/ycy/ours/Deceiving-DNN-based-Binary-Matching/rl_framework/datasets/coreutils/bin/coreutils-8.32-O0/ls"
# 目标函数名 (建议选 sort_files, print_current_files 这种有逻辑的)
TARGET_FUNC = "sym.print_current_files" 
SAVE_DIR = "chapter3_results_final"
os.makedirs(SAVE_DIR, exist_ok=True)

# ================= 核心逻辑 =================

class CriticalityVisualizer:
    def __init__(self, binary_path):
        self.r2 = r2pipe.open(binary_path, flags=['-2'])
        self.r2.cmd('e asm.arch=x86')
        self.r2.cmd('e asm.bits=64')
        self.r2.cmd('aaa')
        
    def analyze_function(self, func_name):
        print(f"[*] Analyzing {func_name}...")
        self.r2.cmd(f's {func_name}')
        info = self.r2.cmdj('afij')
        if not info:
            print("[-] Function not found.")
            return None
        
        addr = info[0]['addr']
        blocks = self.r2.cmdj(f'afbj @ {addr}')
        if not blocks: return None
        
        # 1. 构建图
        G = nx.DiGraph()
        entry_node = None
        min_addr = float('inf')
        
        block_map = {} # addr -> info
        
        for b in blocks:
            b_addr = b['addr']
            if b_addr < min_addr:
                min_addr = b_addr
                entry_node = b_addr
            
            G.add_node(b_addr, size=b['size'])
            block_map[b_addr] = b
            if 'jump' in b: G.add_edge(b_addr, b['jump'])
            if 'fail' in b: G.add_edge(b_addr, b['fail'])
            
        # 2. 计算指标 (复用 r2_acfg_features.py 的逻辑)
        try:
            betweenness = nx.betweenness_centrality(G)
            degree = nx.degree_centrality(G)
        except:
            return None
            
        dominator_scores = {n: 0 for n in G.nodes()}
        try:
            if entry_node:
                idom = nx.immediate_dominators(G, entry_node)
                dom_tree = nx.DiGraph()
                for node, dom in idom.items():
                    if node != dom: dom_tree.add_edge(dom, node)
                for node in G.nodes():
                    if node in dom_tree:
                        dominator_scores[node] = len(nx.descendants(dom_tree, node))
        except: pass
        
        # 3. 归一化与聚合
        def normalize(d):
            vals = list(d.values())
            if not vals: return d
            mx = max(vals)
            mn = min(vals)
            if mx == mn: return {k: 0 for k in d}
            return {k: (v - mn) / (mx - mn) for k, v in d.items()}
            
        n_bet = normalize(betweenness)
        n_deg = normalize(degree)
        n_dom = normalize(dominator_scores)
        
        final_scores = {}
        for n in G.nodes():
            # 权重公式
            score = 0.4 * n_bet.get(n, 0) + 0.3 * n_dom.get(n, 0) + 0.3 * n_deg.get(n, 0)
            final_scores[n] = score
            
        return G, final_scores, block_map

    def render_graph(self, G, scores, block_map, output_name):
        print("[*] Rendering Graph...")
        dot = Digraph(comment='CFG Criticality Heatmap', format='png')
        dot.attr(rankdir='TB', dpi='300')
        
        # 创建颜色映射 (白 -> 红)
        cmap = plt.get_cmap('Reds')
        
        # 找出 Top-3 用于加粗边框
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_k = set([x[0] for x in sorted_nodes[:3]])
        
        for node in G.nodes():
            score = scores.get(node, 0)
            # 获取颜色 hex
            rgba = cmap(score) # score 0~1
            # 为了让字看清，如果颜色太深，字变白
            font_color = 'white' if score > 0.6 else 'black'
            hex_color = mcolors.to_hex(rgba)
            
            # 构建 Label
            # 显示：地址 \n 评分 \n 指令数
            label = f"Addr: 0x{node:x}\nScore: {score:.2f}\nSize: {block_map[node]['size']}"
            
            # 样式
            penwidth = '3.0' if node in top_k else '1.0'
            color = 'red' if node in top_k else 'black'
            
            dot.node(str(node), label=label, 
                     style='filled', fillcolor=hex_color, 
                     fontcolor=font_color, shape='box',
                     penwidth=penwidth, color=color)
            
        for u, v in G.edges():
            dot.edge(str(u), str(v))
            
        output_path = os.path.join(SAVE_DIR, output_name)
        dot.render(output_path, cleanup=True)
        print(f"[+] Graph saved to {output_path}.png")

    def close(self):
        self.r2.quit()

if __name__ == "__main__":
    # 如果你想自动从 dataset.json 里挑一个复杂的函数
    # dataset_path = "dataset_train.json"
    # if os.path.exists(dataset_path):
    #     with open(dataset_path, 'r') as f:
    #         data = json.load(f)
    #     # 筛选: 复杂度高一点的，O0 版本的 (结构完整)
    #     candidates = [d for d in data if d.get('size', 0) > 500 and 'O0' in d.get('opt_level', '')]
    #     if candidates:
    #         # 选一个看起来比较典型的
    #         target = candidates[0] # 或者 random.choice
    #         BINARY_PATH = target['binary_path']
    #         TARGET_FUNC = "sym." + target['func_name'] if not target['func_name'].startswith("sym.") else target['func_name']
    #         print(f"[*] Auto-selected target: {TARGET_FUNC} in {(BINARY_PATH)}")
    
    viz = CriticalityVisualizer(BINARY_PATH)
    res = viz.analyze_function(TARGET_FUNC)
    
    if res:
        G, scores, bmap = res
        viz.render_graph(G, scores, bmap, "case_study_heatmap")
    else:
        print("Analysis failed.")
    print(f"[*] Auto-selected target: {TARGET_FUNC} in {(BINARY_PATH)}")
    viz.close()