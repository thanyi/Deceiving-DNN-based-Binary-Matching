#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import r2pipe
import json
import os
import networkx as nx
import re
import numpy as np
from loguru import logger

def print(message):
    pass

# 全局寄存器定义
X86_GP_REGS = {
    'rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'rbp', 'rsp', 'rip',
    'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15',
    'eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'ebp', 'esp', 'eip',
    'ax', 'bx', 'cx', 'dx', 'si', 'di', 'bp', 'sp',
    'al', 'bl', 'cl', 'dl', 'sil', 'dil', 'bpl', 'spl',
    'ah', 'bh', 'ch', 'dh',
    'r8d', 'r9d', 'r10d', 'r11d', 'r12d', 'r13d', 'r14d', 'r15d',
    'r8w', 'r9w', 'r10w', 'r11w', 'r12w', 'r13w', 'r14w', 'r15w',
    'r8b', 'r9b', 'r10b', 'r11b', 'r12b', 'r13b', 'r14b', 'r15b'
}

X86_VEC_REGS = {
    'xmm0', 'xmm1', 'xmm2', 'xmm3', 'xmm4', 'xmm5', 'xmm6', 'xmm7',
    'xmm8', 'xmm9', 'xmm10', 'xmm11', 'xmm12', 'xmm13', 'xmm14', 'xmm15',
    'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm5', 'ymm6', 'ymm7',
    'ymm8', 'ymm9', 'ymm10', 'ymm11', 'ymm12', 'ymm13', 'ymm14', 'ymm15'
}

# 【新增】只读指令白名单 (即使操作数是 [dst], src 也不算写)
READ_ONLY_INSTRS = {'cmp', 'test', 'ucomisd', 'ucomiss', 'comisd', 'comiss'}

class RadareACFGExtractor:
    """
    【第三章核心创新实现 - V3.0 优化版】
    基于多粒度图中心性的 ACFG 特征提取器
    修正：移除高复杂度计算，增强操作数统计
    """

    def __init__(self, binary_path):
        if not os.path.exists(binary_path):
            raise FileNotFoundError(f"Binary not found: {binary_path}")
        self.r2 = r2pipe.open(binary_path, flags=['-2'])
        self.r2.cmd('e asm.arch=x86')
        self.r2.cmd('e asm.bits=64')
        self.r2.cmd('e anal.hasnext=1') 
        self.r2.cmd('aaa') 

    def get_acfg_features(self, function_name=None, function_addr=None):
        seek_cmd = None
        if function_addr is not None:
            target_addr = hex(function_addr) if isinstance(function_addr, int) else str(function_addr)
            seek_cmd = f's {target_addr}'
        elif function_name:
            seek_cmd = f's {function_name}'
        else:
            seek_cmd = 's entry0'

        if seek_cmd: self.r2.cmd(seek_cmd)
        print(f"seek_cmd: {seek_cmd}")

        func_info = self.r2.cmdj('afij')
        # 检查 func_info 是否为空
        if not func_info or len(func_info) == 0:
            print(f"func_info is empty after seek, function_name: {function_name}, function_addr: {function_addr}")
            # 尝试分析函数
            self.r2.cmd('af')
            func_info = self.r2.cmdj('afij')
            if not func_info or len(func_info) == 0:
                print(f"func_info is still empty after 'af', function_name: {function_name}, function_addr: {function_addr}")
                return None
        
        # 没有识别到函数，进入entry0
        if func_info[0].get('name') == 'entry0':
            print(f"func_info.get('name') == 'entry0', function_name: {function_name}, function_addr: {function_addr}")
            if function_name is not None and function_addr is None:
                function_name = 'sym.' + function_name
                print(f"[r2_acfg_features.py:RadareACFGExtractor:get_acfg_features] function_name: {function_name}")
                seek_cmd = f's {function_name}'
                if seek_cmd: self.r2.cmd(seek_cmd)
                func_info = self.r2.cmdj('afij')
                # 检查重新获取的 func_info 是否为空
                if not func_info or len(func_info) == 0:
                    print(f"func_info is empty after 'sym.' retry, function_name: {function_name}, function_addr: {function_addr}")
                    return None
            else:
                return None

        
        print(f"[r2_acfg_features.py:RadareACFGExtractor:get_acfg_features] func_info: {func_info}")
        if not func_info:
            print(f"func_info is None, function_name: {function_name}, function_addr: {function_addr}")
            self.r2.cmd('af')
            func_info = self.r2.cmdj('afij')
        
        if not func_info or len(func_info) == 0: 
            print(f"func_info is None or length is 0, function_name: {function_name}, function_addr: {function_addr}")
            return None
        
        first_func = func_info[0]
        real_addr = first_func.get('offset') or first_func.get('addr') or first_func.get('vaddr')
        print(f"real_addr: {real_addr}")
        if real_addr is None: 
            print(f"real_addr is None, function_name: {function_name}, function_addr: {function_addr}")
            return None
        
        try:
            blocks_json = self.r2.cmd(f'afbj @ {real_addr}')
            if not blocks_json: 
                print(f"blocks_json is None, function_name: {function_name}, function_addr: {function_addr}")
                return None
            basic_blocks = json.loads(blocks_json)
            # print(f"basic_blocks: {basic_blocks}, length: {len(basic_blocks)}")
        except: return None

        if not basic_blocks: return None

        # --- 构建 NetworkX 图 ---
        G = nx.DiGraph()
        block_map = {}   
        entry_node = None
        min_addr = float('inf')

        for bb in basic_blocks:
            addr = bb['addr']
            if addr < min_addr:
                min_addr = addr
                entry_node = addr
            block_map[addr] = bb
            G.add_node(addr, size=bb['size'])
            if 'jump' in bb: G.add_edge(addr, bb['jump'])
            if 'fail' in bb: G.add_edge(addr, bb['fail'])

        # --- 3.3.1 & 3.3.2 拓扑计算 ---
        try:
            betweenness = nx.betweenness_centrality(G)
            degree = nx.degree_centrality(G)
        except:
            betweenness = {n: 0 for n in G.nodes()}
            degree = {n: 0 for n in G.nodes()}

        dominator_scores = {n: 0 for n in G.nodes()}
        try:
            if entry_node is not None:
                idom = nx.immediate_dominators(G, entry_node)
                dom_tree = nx.DiGraph()
                for node, dom in idom.items():
                    if node != dom: dom_tree.add_edge(dom, node)
                for node in G.nodes():
                    if node in dom_tree:
                        dominator_scores[node] = len(nx.descendants(dom_tree, node))
        except: pass

        # --- 3.3.3 关键区域筛选 (仅用于排序，不作为特征输出) ---
        def simple_norm(d):
            vals = list(d.values())
            if not vals: return d
            mx = max(vals)
            return {k: v/mx if mx>0 else 0 for k,v in d.items()}

        n_bet = simple_norm(betweenness)
        n_dom = simple_norm(dominator_scores)
        n_deg = simple_norm(degree)
        
        critical_scores = []
        for addr in G.nodes():
            # 权重聚合
            score = 0.4*n_bet.get(addr,0) + 0.3*n_dom.get(addr,0) + 0.3*n_deg.get(addr,0)
            critical_scores.append((addr, score))
        
        critical_scores.sort(key=lambda x: x[1], reverse=True)
        top_k_blocks = [x[0] for x in critical_scores[:5]]

        # --- B. 语义指纹提取 ---
        fingerprints = {
            'n_calls': 0, 'n_strings': 0, 'api_types': set(), 'consts': [],
            # 新增：全局操作数分布统计
            'n_ops_imm': 0, 'n_ops_reg': 0, 'n_ops_mem': 0
        }
        
        try:
            xrefs = self.r2.cmdj(f'axfj @ {real_addr}')
            if xrefs:
                for ref in xrefs:
                    ref_type = ref.get('type', '')
                    ref_name = ref.get('name', '').lower()
                    if ref_type == 'call':
                        fingerprints['n_calls'] += 1
                        if any(k in ref_name for k in ['print', 'write', 'read', 'open', 'close']): fingerprints['api_types'].add('io')
                        elif any(k in ref_name for k in ['alloc', 'free', 'map']): fingerprints['api_types'].add('mem')
                        elif any(k in ref_name for k in ['str', 'mem', 'cpy', 'cmp']): fingerprints['api_types'].add('str')
                        elif any(k in ref_name for k in ['exit', 'abort', 'signal', 'fork']): fingerprints['api_types'].add('sys')
                        elif any(k in ref_name for k in ['sock', 'connect', 'bind', 'recv']): fingerprints['api_types'].add('net')
                        elif any(k in ref_name for k in ['crypt', 'hash', 'sha', 'md5', 'aes']): fingerprints['api_types'].add('crypto')
                        elif any(k in ref_name for k in ['err', 'warn', 'fail']): fingerprints['api_types'].add('error')
                        else: fingerprints['api_types'].add('other')
                    if ref_type == 'data':
                        fingerprints['n_strings'] += 1
        except: pass

        # --- C. 提取块特征 ---
        bbs_features = {}
        for bb_addr in block_map:
            bb = block_map[bb_addr]
            # 传入 fingerprints 以便累加全局操作数统计
            feats = self._extract_bb_features(bb_addr, bb['size'], fingerprints)
            
            feats['centrality_betweenness'] = betweenness.get(bb_addr, 0)
            feats['centrality_degree'] = degree.get(bb_addr, 0)
            feats['dominator_score'] = dominator_scores.get(bb_addr, 0)
            
            bbs_features[bb_addr] = feats

        result = {
            'num_nodes': len(G.nodes()),
            'num_edges': len(G.edges()),
            'cyclomatic_complexity': len(G.edges()) - len(G.nodes()) + 2,
            # 移除 graph_diameter
            'basic_blocks': bbs_features,
            'top_critical_blocks': top_k_blocks,
            'fingerprints': fingerprints
        }
        return result

    def _extract_bb_features(self, addr, size, fingerprints):
        """
        提取微观特征，同时统计操作数类型
        """
        try:
            instrs_json = self.r2.cmd(f'pDj {size} @ {addr}')
            instrs = json.loads(instrs_json)
        except: instrs = []
            
        stats = {
            'n_instructions': len(instrs),
            'n_arith': 0, 'n_logic': 0, 'n_branch': 0, 'n_transfer': 0,
            'n_xor': 0, 'n_shift': 0, 'n_cmp': 0,
            'n_mem_write': 0, 'n_mem_read': 0, 
            'n_regs_gp': 0, 'n_regs_vec': 0, 'n_consts': 0
        }
        
        type_arith = {'add', 'sub', 'mul', 'div', 'inc', 'dec', 'imul', 'idiv'}
        type_logic = {'and', 'or', 'not'} # test 是 cmp 类
        type_branch = {'jmp', 'cjmp', 'call', 'ret'}
        type_trans = {'mov', 'lea', 'push', 'pop', 'movzx', 'movsx', 'cmov'}
        
        for ins in instrs:
            itype = ins.get('type', 'unk')
            opcode = ins.get('opcode', '')
            mnemonic = ins.get('mnemonic', '').lower()
            val = ins.get('val')
            
            # 1. 语义统计 (增强版)
            # 支持向量指令 (vpxor, vxorps)
            if 'xor' in mnemonic: stats['n_xor'] += 1
            # 支持 cmp/test
            elif mnemonic.startswith('cmp') or mnemonic == 'test' or mnemonic.startswith('ucom'): stats['n_cmp'] += 1
            # 支持 sal/sar/shl/shr/rol/ror
            elif mnemonic.startswith('sh') or mnemonic.startswith('sa') or mnemonic.startswith('ro'): stats['n_shift'] += 1
            elif itype in type_arith: stats['n_arith'] += 1
            elif itype in type_logic: stats['n_logic'] += 1
            elif itype in type_branch: stats['n_branch'] += 1
            elif any(t in itype for t in type_trans): stats['n_transfer'] += 1
            
            # 2. 数据流分析 (修复逻辑)
            # 必须区分读写。CMP/TEST 是只读的。
            is_read_only_instr = (mnemonic in READ_ONLY_INSTRS)
            
            operands = ins.get('operands', [])
            if operands:
                for idx, op in enumerate(operands):
                    otype = op.get('type')
                    if otype == 'imm': fingerprints['n_ops_imm'] += 1
                    elif otype == 'reg': fingerprints['n_ops_reg'] += 1
                    elif otype == 'mem': 
                        fingerprints['n_ops_mem'] += 1
                        # 内存读写细分
                        if mnemonic == 'lea': pass # LEA 只是算地址
                        elif is_read_only_instr:
                            stats['n_mem_read'] += 1
                        else:
                            # 默认：第一个操作数如果是内存，且不是只读指令，通常是写
                            # (Intel语法: op dest, src)
                            if idx == 0: stats['n_mem_write'] += 1
                            else: stats['n_mem_read'] += 1
            
            # 回退机制：字符串匹配 (当 R2 解析 operands 失败时)
            else:
                if '[' in opcode:
                    if mnemonic == 'lea': pass
                    elif mnemonic in {'push', 'call'}: stats['n_mem_write'] += 1 # Push/Call 写栈
                    elif mnemonic in {'pop', 'ret'}:   stats['n_mem_read'] += 1  # Pop/Ret 读栈
                    elif mnemonic.startswith('stos'):  stats['n_mem_write'] += 1
                    elif mnemonic.startswith('lods'):  stats['n_mem_read'] += 1
                    elif is_read_only_instr:
                        stats['n_mem_read'] += 1 # cmp [rax], 1 是读
                    else:
                        # 简单启发式
                        comma_pos = opcode.find(',')
                        bracket_pos = opcode.find('[')
                        if comma_pos == -1: # 单操作数 (如 inc [rax]) -> 读改写，算写
                            stats['n_mem_write'] += 1
                        elif bracket_pos < comma_pos: # [dst], src -> 写
                            stats['n_mem_write'] += 1
                        else: # dst, [src] -> 读
                            stats['n_mem_read'] += 1
            
            # 3. 寄存器匹配 (Register Pressure)
            # 使用预定义的全集进行匹配
            words = re.findall(r'\b([a-z0-9]+)\b', opcode.lower())
            for w in words:
                if w in X86_GP_REGS: stats['n_regs_gp'] += 1
                elif w in X86_VEC_REGS: stats['n_regs_vec'] += 1

            # 4. 常量
            if val is not None and abs(val) > 64:
                stats['n_consts'] += 1
                fingerprints['consts'].append(val)

        return stats

    def close(self):
        try: self.r2.quit()
        except: pass


if __name__ == '__main__':
    extractor = RadareACFGExtractor('/home/ycy/ours/Deceiving-DNN-based-Binary-Matching/rl_framework/datasets/coreutils/bin/coreutils-8.32-O0/pwd')
    features = extractor.get_acfg_features(function_name='usage')
    print(f"features: {features}")