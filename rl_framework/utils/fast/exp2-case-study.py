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
# 建议选一个 O0 版本的，结构比较清晰
BINARY_PATH = "/home/ycy/ours/Deceiving-DNN-based-Binary-Matching/uroboros_testing/0114/test_mutant_pwd_815.bin"
TARGET_FUNC = "fcn.0040264a" 
SAVE_DIR = "chapter3_results_final"
os.makedirs(SAVE_DIR, exist_ok=True)

class CriticalityVisualizer:
    def __init__(self, binary_path):
        if not os.path.exists(binary_path):
            print(f"[-] Binary not found: {binary_path}")
            sys.exit(1)
        self.r2 = r2pipe.open(binary_path, flags=['-2'])
        self.r2.cmd('e asm.arch=x86')
        self.r2.cmd('e asm.bits=64')
        self.r2.cmd('aaa')
        
    def analyze_function(self, func_name):
        print(f"[*] Analyzing {func_name}...")
        self.r2.cmd(f's {func_name}')
        info = self.r2.cmdj('afij')
        
        # 自动纠错：尝试 sym. 前缀
        if not info or info[0]['name'] == 'entry0':
            if not func_name.startswith('sym.'):
                print(f"[*] Retrying with sym.{func_name}...")
                self.r2.cmd(f's sym.{func_name}')
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
        block_map = {} 
        
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
            betweenness = {n: 0 for n in G.nodes()}
            degree = {n: 0 for n in G.nodes()}
            
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
            score = 0.4 * n_bet.get(n, 0) + 0.3 * n_dom.get(n, 0) + 0.3 * n_deg.get(n, 0)
            final_scores[n] = score
            
        return G, final_scores, block_map

    def render_graph_color(self, G, scores, block_map, output_name):
        """
        【彩色热力图风格】
        - 颜色：根据 Critical Score 从 白(0.0) -> 红(1.0) 渐变
        - 形状：圆角矩形
        - 重点：Top-3 节点加粗边框
        """
        print("[*] Rendering Graph (Color Heatmap)...")
        dot = Digraph(comment='CFG Criticality Heatmap', format='png')
        
        # 全局属性: 使用正交线(ortho)看起来更像电路图/流程图，很整齐
        dot.attr(rankdir='TB', dpi='300', splines='ortho')
        dot.attr('node', shape='box', style='rounded,filled', 
                 fontname='Times-Roman', fontsize='11', margin='0.1')
        dot.attr('edge', fontname='Times-Roman', fontsize='10')
        
        # 1. 准备颜色映射 (使用 matplotlib 的 Reds 色盘)
        # 0.0 -> 白色/极浅红
        # 1.0 -> 深红
        cmap = plt.get_cmap('Reds')
        
        # 2. 找出 Top-3 关键节点 (用于加粗边框)
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_k_addrs = set([x[0] for x in sorted_nodes[:3]])
        
        for node in G.nodes():
            score = scores.get(node, 0)
            
            # --- 颜色计算 ---
            # 为了让低分节点不完全是纯白(看不见)，可以加一点底色偏移，或者直接用 score
            # 这里直接映射 score，效果最直观
            rgba = cmap(score) 
            hex_color = mcolors.to_hex(rgba)
            
            # --- 字体颜色自适应 ---
            # 如果背景太深(红色)，字就变白；否则字是黑的
            font_color = 'white' if score > 0.6 else 'black'
            
            # --- 边框逻辑 ---
            # 关键节点：边框加粗 (3.0)，颜色用深红或黑
            # 普通节点：边框细 (1.0)，颜色浅灰
            if node in top_k_addrs:
                penwidth = '3.0'
                border_color = 'black' # 关键节点黑框更醒目
            else:
                penwidth = '1.0'
                border_color = 'grey'
            
            # --- 标签内容 (保持学术格式) ---
            # 格式：
            # Memory: 0x...
            # Score: ...
            # Size: ...
            label = (f"Memory: 0x{node:x}\\n"
                     f"Score: {score:.2f}\\n"
                     f"Size: {block_map[node]['size']}")
            
            dot.node(str(node), label=label, 
                     fillcolor=hex_color, 
                     fontcolor=font_color,
                     color=border_color,
                     penwidth=penwidth)
            
        for u, v in G.edges():
            dot.edge(str(u), str(v), color='dimgrey')
            
        output_path = os.path.join(SAVE_DIR, output_name)
        dot.render(output_path, cleanup=True)
        print(f"[+] Color Graph saved to {output_path}.png")

    def close(self):
        self.r2.quit()

if __name__ == "__main__":
    # 自动寻找目标
    target_bin = BINARY_PATH
    target_func = TARGET_FUNC

    # 如果默认路径不存在，尝试从 dataset_train.json 读取第一个合适的
    if not os.path.exists(target_bin):
        print(f"[-] Default binary {target_bin} not found. Trying dataset...")
        try:
            with open("dataset_train.json", 'r') as f:
                data = json.load(f)
            # 找一个 O0 的，大小适中的函数
            candidates = [d for d in data if 'O0' in d.get('opt_level', '') and 200 < d.get('size', 0) < 1000]
            if candidates:
                target = candidates[0]
                target_bin = target['binary_path']
                target_func = target['func_name']
                print(f"[*] Switched to: {target_func} in {os.path.basename(target_bin)}")
        except: pass

    viz = CriticalityVisualizer(target_bin)
    res = viz.analyze_function(target_func)
    
    if res:
        G, scores, bmap = res
        # 使用新的学术风格渲染函数
        viz.render_graph_color(G, scores, bmap, "case_study_academic")
    else:
        print("Analysis failed.")
    
    viz.close()