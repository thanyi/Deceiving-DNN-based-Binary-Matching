#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import r2pipe
import json
import sys
import os

class RadareACFGExtractor:
    """
    使用 radare2 提取 ACFG 特征，替代 IDA Pro 脚本。
    特点：速度快，无需保存文件，直接内存交互。
    """

    def __init__(self, binary_path):
        """
        初始化提取器。
        :param binary_path: 二进制文件路径
        """
        if not os.path.exists(binary_path):
            raise FileNotFoundError(f"Binary not found: {binary_path}")
        
        # 打开二进制文件，-2 参数禁止 stderr 输出
        self.r2 = r2pipe.open(binary_path, flags=['-2'])
        
        # 分析二进制 (aaa = analyze all)
        # 这一步是必须的，否则无法识别函数和 CFG
        # print(f"Analyzing {binary_path}...")
        self.r2.cmd('aaa')

    def get_acfg_features(self, function_name=None, function_addr=None):
        """
        提取 ACFG。
        注意：对于 Strip 的文件，强烈建议传入 function_addr (int)。
        """
        seek_cmd = None
        
        # 1. 优先使用地址 (最靠谱)
        if function_addr is not None:
            # 确保是十六进制字符串或整数
            if isinstance(function_addr, int):
                target_addr = hex(function_addr)
            else:
                target_addr = str(function_addr)
            
            seek_cmd = f's {target_addr}'
            print(f"[+] Seeking to address: {target_addr}")
            
        # 2. 其次尝试名字 (Strip 后通常失效，除非是用 r2 的 fcn.xxx 格式)
        elif function_name:
            # 尝试直接跳转
            seek_cmd = f's {function_name}'
        
        else:
            # 如果都没传，默认分析 Entrypoint (通常是 _start，不是 main)
            seek_cmd = 's entry0'

        if seek_cmd:
            self.r2.cmd(seek_cmd)
            print(f"[+] Seeking to address: {seek_cmd}")
        # === 新增关键逻辑：验证是否真的跳到了函数上 ===
        # 有时候 s 0x123456 跳过去了，但 r2 没识别出那是函数起始点
        # 我们用 afi (analyze function info) 检查一下
        
        func_info = self.r2.cmdj('afij') # 返回当前位置的函数信息
        if not func_info:
            # 如果当前位置没有识别出函数，尝试强行定义一个函数
            # print(f"Warning: No function identified at current location. Trying 'af'...")
            self.r2.cmd('af') # analyze function force
            func_info = self.r2.cmdj('afij')
            
        if not func_info:
            # 仍然无法识别，说明地址可能错了，或者是垃圾数据
            return None
            
        # 此时 func_info 是一个列表，取第一个
        # 真正的函数起始地址
        real_addr = func_info[0]['addr']
        print(f"[+] Real address: {real_addr}")
        # 3. 获取基本块 (afbj)
        try:
            # 确保我们在正确的函数起始位置获取 blocks
            blocks_json = self.r2.cmd(f'afbj @ {real_addr}')
            # print(f"[+] Blocks json: {blocks_json}")
            if not blocks_json:
                return None
            basic_blocks = json.loads(blocks_json)
            # print(f"[+] Basic blocks: {basic_blocks}")
        except Exception as e:
            return None

        if not basic_blocks:
            return None

        # 4. 构建特征容器
        nodes = []
        edges = []
        bbs_features = {}
        
        # 5. 遍历基本块提取局部特征
        for bb in basic_blocks:
            bb_addr = bb['addr']
            bb_size = bb['size']
            
            nodes.append(bb_addr)
            
            # 处理边 (Edge)
            # radare2 的 afbj 输出包含 jump (true branch) 和 fail (false branch)
            if 'jump' in bb and bb['jump'] != 0:
                edges.append([bb_addr, bb['jump']])
            if 'fail' in bb and bb['fail'] != 0:
                edges.append([bb_addr, bb['fail']])
                
            # 提取块内的指令特征
            features = self._extract_bb_features(bb_addr, bb_size)
            bbs_features[bb_addr] = features

        # 6. 计算函数级统计特征 (Function Level Features)
        # 这里为了保持和 ACFG 格式一致，返回类似的结构
        # 但在 RL 中，你可能需要手动把这些字典压扁成向量
        
        result = {
            'nodes': nodes,
            'edges': edges,
            'num_nodes': len(nodes),
            'num_edges': len(edges),
            'basic_blocks': bbs_features
        }
        # print(f"[+] Result: {result}")
        return result

    def _extract_bb_features(self, addr, size):
        """
        提取单个基本块的指令统计特征
        """
        # 反汇编该块的指令 (pDj = print disassembly json)
        try:
            instrs_json = self.r2.cmd(f'pDj {size} @ {addr}')
            instrs = json.loads(instrs_json)
        except:
            instrs = []
            
        # 初始化计数器
        stats = {
            'n_instructions': len(instrs),
            'n_arith_instrs': 0,
            'n_logic_instrs': 0,
            'n_transfer_instrs': 0, # mov, push, pop
            'n_redirect_instrs': 0, # jmp, ret
            'n_call_instrs': 0,
            'n_numeric_consts': 0,
            'n_string_consts': 0, # r2 较难直接精确提取字符串引用，这里暂用 numeric 代替或简化
        }
        
        # 指令分类映射 (根据 radare2 的 type 字段)
        # 这是一个 heuristic 映射，可能需要根据 r2 版本微调
        type_arith = {'add', 'sub', 'mul', 'div', 'mod', 'inc', 'dec', 'abs', 'neg'}
        type_logic = {'and', 'or', 'xor', 'not', 'shl', 'shr', 'sal', 'sar', 'rol', 'ror'}
        type_trans = {'mov', 'lea', 'push', 'pop', 'xchg', 'cmov'} 
        type_redirect = {'jmp', 'cjmp', 'ret', 'ujmp'}
        type_call = {'call', 'ucall'}
        
        for ins in instrs:
            itype = ins.get('type', 'unk')
            
            # 1. 统计指令类型
            if itype in type_arith:
                stats['n_arith_instrs'] += 1
            elif itype in type_logic:
                stats['n_logic_instrs'] += 1
            elif any(t in itype for t in type_trans): # 包含 mov, push 等
                stats['n_transfer_instrs'] += 1
            elif itype in type_redirect:
                stats['n_redirect_instrs'] += 1
            elif itype in type_call:
                stats['n_call_instrs'] += 1
                
            # 2. 统计数值常量
            # r2 的 ops 通常有 'val' 字段表示立即数
            if 'val' in ins:
                stats['n_numeric_consts'] += 1
                
            # 3. 字符串常量 (简化处理：如果 comment 里有字符串引用)
            # 这是一个近似值
            if 'string' in ins.get('opcode', '') or 'str' in ins.get('type', ''):
                stats['n_string_consts'] += 1

        return stats

    def close(self):
        """关闭 r2 管道"""
        try:
            self.r2.quit()
        except:
            pass

# ==========================================
# 单元测试 / 使用示例
# ==========================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('binary', help='Path to binary file')
    parser.add_argument('function', help='Function name or address')
    args = parser.parse_args()
    
    try:
        extractor = RadareACFGExtractor(args.binary)
        print(f"[+] Loaded {args.binary}")
        print(f"[+] Function: {args.function}")
        import time
        t1 = time.time()
        
        # 提取特征
        features = extractor.get_acfg_features(function_addr=int(args.function, 16))
        # print(f"[+] Features: {features}")
        t2 = time.time()
        
        if features:
            print(f"[+] Extraction time: {t2 - t1:.4f}s")
            print(f"[+] Nodes: {features['num_nodes']}, Edges: {features['num_edges']}")
            # 打印第一个基本块的特征作为示例
            first_bb = list(features['basic_blocks'].keys())[0]
            print(f"[+] Features of BB {hex(first_bb)}:")
            print(json.dumps(features['basic_blocks'][first_bb], indent=2))
        else:
            print("[-] Failed to extract features (function not found?)")
            
        extractor.close()
        
    except Exception as e:
        print(f"[-] Error: {e}")