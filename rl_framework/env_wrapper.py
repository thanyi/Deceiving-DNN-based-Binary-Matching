#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environment Wrapper for Binary Code Perturbation
二进制代码变异环境包装器（Python 3）

功能：
- 调用 uroboros (Python 2) 进行代码变异
- 使用 run_utils (Python 3) 进行相似度评估
- 提供标准 RL 环境接口
"""

import sys
import os
import json
import time
import subprocess
import numpy as np
import pickle
import hashlib
from loguru import logger
import random

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入现有模块
from run_utils import run_one
import run_objdump
from rl_framework.utils.acfg.r2_acfg_features import RadareACFGExtractor


class BinaryPerturbationEnv:
    """
    二进制代码变异环境 (Python 3)
    
    与 PPO Agent 在同一进程中运行，通过函数调用通信
    """
    
    def __init__(self, save_path, dataset_path, sample_hold_interval=3):
        """
        参数:
            original_binary: 原始二进制文件路径
            function_name: 目标函数名
            save_path: 保存变异结果的路径
        """
        self.save_path = os.path.abspath(save_path)
        # 项目根目录（uroboros 所在目录）
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    

        # 加载数据集
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
            
        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)
        
        logger.info(f"已加载数据集: {len(self.dataset)} 个样本")
        
        # 切换策略控制
        self.sample_hold_interval = sample_hold_interval
        self.episodes_on_current = 0
        self.current_sample_data = None # 存储当前样本的元数据
        
        # 当前环境状态变量
        self.original_binary = None # 原始二进制文件路径
        self.function_name = None # 目标函数名
        self.current_binary = None # 当前变异后的二进制文件路径
        
        # 不再需要加载模型（使用 asm2vec 方法）
        self.model_original = None
        logger.info("Using asm2vec detection method (no model loading required)")
        
        # 变异历史
        self.mutation_history = []
        self.step_count = 0
        self.target_score = 0.40
        self.state_dim = 128  # 默认状态维度（128维），可以通过参数修改
        
        # 【性能优化】Radare2 特征提取缓存
        # 缓存键: (binary_path, function_name, function_addr)
        # 缓存值: acfg_data (dict)
        self._acfg_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # 【性能优化】原始文件汇编缓存（原始文件不变，可复用）
        # 缓存键: (original_binary, function_name, ori_sym_addr)
        # 缓存值: 汇编文件路径
        self._original_asm_cache = {}
        
        # 【性能优化】复用临时目录，避免频繁创建删除
        # 在 save_path 下创建固定工作目录
        self._asm_work_dir = os.path.join(self.save_path, '_asm_work')
        os.makedirs(self._asm_work_dir, exist_ok=True)
        
        logger.info(f"Environment initialized (Hold Strategy: {self.sample_hold_interval} eps)")
    
    def set_state_dim(self, state_dim):
        """
        设置状态维度（用于与 PPO Agent 保持一致）
        
        参数:
            state_dim: 状态维度
        """
        self.state_dim = state_dim
        logger.info(f"状态维度设置为: {state_dim}")


    def _resolve_mutated_address(self, binary_path):
        """
        核心辅助函数：解析变异后函数的真实地址
        解决 Strip 文件无法通过函数名定位的问题
        """
        # 1. 如果是原始文件，我们需要知道原始地址
        # 这里假设原始文件未 Strip，或者你能通过函数名找到
        if binary_path == self.original_binary:
            return None, self.function_name

        # 2. 寻找 sym_to_addr.pickle 映射文件
        mutated_dir = os.path.dirname(binary_path)
        pickle_path = os.path.join(mutated_dir, "sym_to_addr.pickle")
        
        # 有时候 uroboros 会生成在上一级
        if not os.path.exists(pickle_path):
            pickle_path = os.path.join(os.path.dirname(mutated_dir), "sym_to_addr.pickle")

        if not os.path.exists(pickle_path):
            # 如果找不到映射，只能返回 None，后续逻辑会尝试盲猜入口点
            logger.warning(f"Map file missing for {binary_path}")
            return None, None

        try:
            with open(pickle_path, 'rb') as f:
                addr_map = pickle.load(f)
            
            # 尝试获取目标函数的地址
            # Uroboros 的 map key 可能是原始函数名
            if self.function_name in addr_map:
                addr_str = addr_map[self.function_name]
                if isinstance(addr_str, str):
                    return int(addr_str, 16), None
                return int(addr_str), None
            
            # 如果找不到直接匹配，尝试寻找 'func_xxxx' 格式
            # 这里简化处理，返回 None 让 r2 尝试 entry0
            return None, None
            
        except Exception as e:
            logger.error(f"Error resolving address: {e}")
            return None, None
    
    def extract_features(self, binary_path):
        """
        特征提取函数 (128维)
        组成: [历史特征(10)] + [ACFG特征(118)]
        """
        features = []
        
        # ==========================================
        # Part 1: 变异历史与环境状态 (10维)
        # ==========================================
        # 1. 当前分数
        last_score = self.mutation_history[-1].get('score', 1.0) if self.mutation_history else 1.0
        features.append(min(last_score, 1.0))
        
        # 2. 分数趋势 (Delta)
        if len(self.mutation_history) >= 2:
            delta = self.mutation_history[-1]['score'] - self.mutation_history[-2]['score']
            features.append(max(-1.0, min(1.0, delta)))
        else:
            features.append(0.0)
            
        # 3. 进度 (Step / Max)
        features.append(min(self.step_count / 50.0, 1.0))
        
        # 4. 变异动作分布 (6类动作的频率)
        action_counts = {1: 0, 2: 0, 7: 0, 8: 0, 9: 0, 11: 0}
        total_acts = len(self.mutation_history) if self.mutation_history else 1
        for m in self.mutation_history:
            a = m.get('action')
            if a in action_counts:
                action_counts[a] += 1
        for a in [1, 2, 7, 8, 9, 11]:
            features.append(action_counts[a] / total_acts)
            
        # 补齐到 10 维 (目前是 1+1+1+6 = 9维，补1个0)
        features.append(0.0)

        # ==========================================
        # Part 2: 基于 Radare2 的 ACFG 特征 (核心)
        # ==========================================
        
        # 初始化默认向量 (全0) 用于失败情况
        acfg_vec = [0.0] * (self.state_dim - len(features))
        
        try:
            # 1. 解析地址
            target_addr, target_name = self._resolve_mutated_address(binary_path)
            
            # 2. 【性能优化】检查缓存
            cache_key = (os.path.abspath(binary_path), target_name, target_addr)
            acfg_data = self._acfg_cache.get(cache_key)
            
            if acfg_data is None:
                # 缓存未命中：调用 R2 提取
                self._cache_misses += 1
                r2_ext = RadareACFGExtractor(binary_path)
                acfg_data = r2_ext.get_acfg_features(function_name=target_name, function_addr=target_addr)
                r2_ext.close()
                
                # 存入缓存（只缓存成功提取的数据）
                if acfg_data:
                    self._acfg_cache[cache_key] = acfg_data
            else:
                # 缓存命中
                self._cache_hits += 1
                if self._cache_hits % 100 == 0:
                    total = self._cache_hits + self._cache_misses
                    hit_rate = self._cache_hits / total if total > 0 else 0.0
                    logger.debug(f"ACFG 缓存统计: 命中率={hit_rate:.2%} (命中={self._cache_hits}, 未命中={self._cache_misses})")
            
            if acfg_data:
                acfg_vec = self._vectorize_acfg(acfg_data)
                
        except (FileNotFoundError, KeyError, ValueError, AttributeError) as e:
            logger.warning(f"Feature extraction failed for {binary_path}: {e}")
            # 保持全0
        
        features.extend(acfg_vec)
        
        # 最终截断或补齐到 128 维
        if len(features) > self.state_dim:
            features = features[:self.state_dim]
        elif len(features) < self.state_dim:
            features.extend([0.0] * (self.state_dim - len(features)))

        # === 【核心修复】数据清洗 ===
        # 1. 转为 numpy 数组
        features = np.array(features, dtype=np.float32)
        
        # 2. 替换 NaN 为 0，替换 Infinity 为最大/最小有限值
        # 防止任何计算错误产生的 NaN 传入神经网络
        features = np.nan_to_num(features, nan=0.0, posinf=100.0, neginf=-100.0)
        
        # 3. 裁剪数值范围 (Clip)
        # 防止某些特征数值过大（比如 total_instr 突然很大），导致梯度爆炸
        # 将所有特征限制在 [-10, 100] 之间通常足够了
        features = np.clip(features, -10.0, 100.0)
            
        return features
    

    def _vectorize_acfg(self, data):
        """
        【128维 增强版】全景式关键区域感知特征提取
        Part 1 (10维) 由 extract_features 填充，这里生成剩下的 118 维
        """
        vec = []
        
        # 基础数据准备
        n_nodes = max(data.get('num_nodes', 0), 1.0)
        n_edges = data.get('num_edges', 0)
        complexity = data.get('cyclomatic_complexity', 0)
        bbs = list(data.get('basic_blocks', {}).values())
        
        # 获取 Top-5 关键块 (之前是 Top-3)
        top_critical_addrs = data.get('top_critical_blocks', [])
        
        # 计算全局统计量
        total_instr = sum(b['n_instructions'] for b in bbs)
        safe_total = max(total_instr, 1.0)
        
        # 辅助函数：安全除法
        def safe_div(a, b): return a / b if b > 0 else 0

        # =========================================================
        # Section A: 全局宏观特征 (20维) [Index 10-29]
        # =========================================================
        # 1. 基础规模 (4维)
        vec.append(np.log1p(n_nodes))
        vec.append(np.log1p(n_edges))
        vec.append(np.log1p(complexity))
        vec.append(np.log1p(total_instr))
        
        # 2. 图拓扑密度 (4维)
        vec.append(safe_div(n_edges, n_nodes))       # 边点比
        vec.append(safe_div(total_instr, n_nodes))   # 平均块大小
        vec.append(safe_div(complexity, n_nodes))    # 平均复杂度
        # 悬挂节点比例 (Leaf Nodes Ratio) - 反映控制流深度
        leaf_nodes = sum(1 for b in bbs if b.get('n_transfer', 0) == 0 and b.get('n_branch', 0) == 0) # 简化估算
        vec.append(safe_div(leaf_nodes, n_nodes))

        # 3. 全局指令分布 (6维)
        global_keys = ['n_arith', 'n_logic', 'n_branch', 'n_transfer', 
                       'n_mem_write', 'n_regs_used']
        global_sums = {k: sum(b.get(k, 0) for b in bbs) for k in global_keys}
        
        for k in global_keys:
            vec.append(safe_div(global_sums[k], safe_total))
            
        # 4. 统计异质性 (4维) - 论文加分项
        # 反映代码是否均匀，还是有巨大的核心块
        instr_counts = [b['n_instructions'] for b in bbs]
        if instr_counts:
            vec.append(np.std(instr_counts))           # 标准差
            vec.append(np.max(instr_counts))           # 最大块大小
            vec.append(safe_div(np.max(instr_counts), safe_total)) # 最大块占比
            vec.append(np.min(instr_counts))           # 最小块大小
        else:
            vec.extend([0.0] * 4)
            
        # 补齐 Section A (确保是 20 维)
        current_A_len = 4 + 4 + 6 + 4
        if current_A_len < 20:
            vec.extend([0.0] * (20 - current_A_len))

        # =========================================================
        # Section B: 关键区域感知 (Top-5 Blocks) (80维) [Index 30-109]
        # 核心创新：深入感知 5 个最重要的节点，每个节点 16 维特征
        # =========================================================
        # 特征列表 (16维/块):
        # [0] Size(Log)
        # [1-6] 6类指令占比 (Arith, Logic, Branch, Transfer, Mem, Regs)
        # [7-8] 中心性 (Betweenness, Degree)
        # [9]   是否是 Leaf Node (出度估算)
        # [10]  是否是 Entry Node (入度估算)
        # [11-15] 预留/扩展 (使用数据流强度填充)
        
        for i in range(5): # 扩大到 Top-5
            if i < len(top_critical_addrs):
                addr = top_critical_addrs[i]
                # 保护：检查地址是否存在于 basic_blocks 中
                if addr not in data.get('basic_blocks', {}):
                    # 如果地址不存在，填充0并继续下一个
                    vec.extend([0.0] * 16)
                    continue
                bb = data['basic_blocks'][addr]
                
                # --- 基础特征 (7维) ---
                safe_bb_total = max(bb['n_instructions'], 1.0)
                vec.append(np.log1p(bb['n_instructions'])) # Size
                
                vec.append(safe_div(bb['n_arith'], safe_bb_total))
                vec.append(safe_div(bb['n_logic'], safe_bb_total))
                vec.append(safe_div(bb['n_branch'], safe_bb_total))
                vec.append(safe_div(bb['n_transfer'], safe_bb_total))
                vec.append(safe_div(bb['n_mem_write'], safe_bb_total))
                vec.append(safe_div(bb['n_regs_used'], 16.0)) # 归一化寄存器数
                
                # --- 拓扑特征 (2维) ---
                vec.append(bb.get('centrality_betweenness', 0))
                vec.append(bb.get('centrality_degree', 0))
                
                # --- 高级结构特征 (3维) ---
                # 假设 r2_acfg_features 里我们没法直接拿到出入度，用指令估算
                is_branch = 1.0 if bb['n_branch'] > 0 else 0.0
                is_mem_heavy = 1.0 if bb['n_mem_write'] > 2 else 0.0
                is_compute_heavy = 1.0 if (bb['n_arith'] + bb['n_logic']) > 5 else 0.0
                
                vec.append(is_branch)
                vec.append(is_mem_heavy)
                vec.append(is_compute_heavy)
                
                # --- 补齐到 16 维 (4维) ---
                vec.extend([0.0] * 4) 
                
            else:
                # 填充 0 (Padding)
                vec.extend([0.0] * 16)

        # =========================================================
        # Section C: 数据流与上下文 (18维) [Index 110-127]
        # =========================================================
        # 1. 寄存器压力详情
        vec.append(safe_div(global_sums['n_regs_used'], 16.0)) # 通用寄存器使用率
        
        # 2. 内存交互强度
        mem_ops = global_sums['n_mem_write'] + global_sums.get('n_mem_read', 0)
        vec.append(safe_div(mem_ops, safe_total))
        
        # 3. 算术逻辑密度 (ALU Density) - 反映混淆潜能
        alu_ops = global_sums['n_arith'] + global_sums['n_logic']
        vec.append(safe_div(alu_ops, safe_total))
        
        # 计算剩余 needed
        # 当前 vec 长度 = 20(A) + 80(B) + 3(C) = 103
        # 目标总长度 = 118 (128 - 10个历史特征)
        needed = 118 - len(vec)
        
        if needed > 0:
            vec.extend([0.0] * needed)
        elif needed < 0:
            vec = vec[:118]
            
        return vec


    
    def apply_mutation(self, seed_binary, action):
        """
        应用变异操作
        
        参数:
            seed_binary: 种子二进制文件路径
            action: 变异模式 (1,2,3,5,7,8,9,11)
        
        返回:
            mutated_binary: 变异后的二进制文件路径
        """
        try:
            logger.info("Applying mutation {} to {}".format(action, seed_binary))
            
            # 确保输出目录存在
            output_dir = '/home/ycy/ours/Deceiving-DNN-based-Binary-Matching/rl_framework/rl_output'
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成临时二进制文件名
            tmp_bin = os.path.join(output_dir, 'mutant_' + str(int(time.time() * 1000)) + '.bin')
            
            # 确定模式
            fmode = 'mutated' if seed_binary != self.original_binary else 'original'
            
            # 为每次变异创建独立的临时目录
            tmp_id = str(int(time.time() * 1000))
            tmp_dir = os.path.join(self.save_path, 'tmp_' + tmp_id)
            os.makedirs(tmp_dir, exist_ok=True)
            
            # 调用 uroboros (Python 2) 进行变异
            # 注意：这里必须使用 python2，因为 uroboros 是 Python 2 代码
            cmd = [
                'python2',
                os.path.join(self.project_root, 'uroboros_automate-func-name.py'),
                seed_binary,
                '-i', '1',  # 迭代次数
                '-o', tmp_bin,
                '-d', str(action),
                '-m', fmode,
                '-f', tmp_dir,
                '--function', self.function_name
            ]
            
            # logger.debug("Command: " + " ".join(cmd))
            
            # 在项目根目录执行命令
            try:
                output = subprocess.check_output(
                    cmd, 
                    stderr=subprocess.STDOUT,
                    cwd=self.project_root,
                    universal_newlines=True  # 返回字符串而不是字节
                )
                logger.debug("Uroboros output: {}".format(output))
            except subprocess.CalledProcessError as e:
                # 捕获详细的错误信息
                error_output = e.output if hasattr(e, 'output') else str(e)
                logger.error("Uroboros command failed with exit code {}: {}".format(e.returncode, error_output))
                logger.error("Command was: {}".format(" ".join(cmd)))
                raise Exception("Uroboros mutation failed: {}\nOutput: {}".format(e, error_output))
            
            # 验证输出文件是否存在
            if not os.path.exists(tmp_bin):
                raise FileNotFoundError("Mutation output not found: {}".format(tmp_bin))
            
            # 计算 hash 并移动文件
            h = hashlib.md5(open(tmp_bin, 'rb').read()).hexdigest()
            container_path = os.path.join(self.save_path, h + '_container')
            
            if not os.path.exists(container_path):
                # 移动 tmp 目录到 container
                import shutil
                shutil.move(tmp_dir, container_path)
                # 移动二进制文件
                shutil.move(tmp_bin, os.path.join(container_path, h))
            
            mutated_binary = os.path.join(container_path, h)
            logger.info("Mutation successful: {}".format(mutated_binary))
            
            return mutated_binary, h
            
        except Exception as e:
            logger.error("Mutation failed: {}".format(e))
            return None, None
    
    def evaluate(self, mutated_binary, checkdict):
        """
        评估变异后的二进制文件
        
        返回:
            score: 相似度分数
            grad: 梯度值
        """
        try:
            # 【性能优化】传递工作目录和缓存，避免每次创建临时目录
            score, grad = run_one(
                self.original_binary,
                mutated_binary,
                self.model_original,
                checkdict,
                self.function_name,
                asm_work_dir=self._asm_work_dir,
                original_asm_cache=self._original_asm_cache
            )
            
            if score is None or grad is None:
                logger.warning("Evaluation returned None")
                return 1.0, 0.0  # 默认最差值
            
            return abs(score), abs(grad)
            
        except Exception as e:
            logger.error("Evaluation failed: {}".format(e))
            return 1.0, 0.0
    
    def step(self, action):
        """
        执行一步环境交互
        
        参数:
            action: 变异模式
        
        返回:
            state: 新状态特征
            reward: 奖励
            done: 是否完成
            info: 额外信息
        """
        self.step_count += 1
        # 记录上一步分数，用于计算差分奖励
        prev_score = self.mutation_history[-1]['score'] if self.mutation_history else 1.0
        
        # 应用变异
        mutated_binary, hash_val = self.apply_mutation(self.current_binary, action)
        
        if mutated_binary is None:
            # 变异失败：标记需要重置环境并切换文件
            logger.warning("Mutation failed, will reset environment and switch to new file")
            state = self.extract_features(self.current_binary)
            return state, -10.0, True, {
                'error': 'mutation_failed',
                'should_reset': True,  # 标志：需要重置并切换文件
                'score': 1.0,  # 默认最差分数
                'grad': 0.0
            }
        
        # 评估
        # TODO: 获取正确的 checkdict
        checkdict = {}  # 需要从实际文件中加载
        score, grad = self.evaluate(mutated_binary, checkdict)
        
        # 更新状态
        self.current_binary = mutated_binary
        self.mutation_history.append({
            'step': self.step_count,
            'action': action,
            'binary': mutated_binary,
            'hash': hash_val,
            'score': score,
            'grad': grad
        })
        
        # 提取新状态特征
        state = self.extract_features(mutated_binary)
        
        # 计算奖励
        reward = self.compute_reward_diff(prev_score, score, grad)
        # reward = self.compute_reward(score, grad)
        
        # 判断是否完成
        done = score < self.target_score or self.step_count >= 50
        
        info = {
            'score': score,
            'grad': grad,
            'step': self.step_count,
            'binary': mutated_binary,
            'target_func': self.function_name # 记录当前目标函数名
        }
        
        logger.info("Step {}: action={}, score={:.4f}, reward={:.4f}".format(
            self.step_count, action, score, reward
        ))
        
        return state, reward, done, info
    
    def compute_reward_diff(self, prev_score, current_score, grad):
        """
        差分奖励函数：适合多样本训练
        """
        # 1. 进步奖励 (关键)：分数下降了多少
        improvement = prev_score - current_score
        
        # 如果进步了，给正奖励；退步了，给负奖励
        # 放大系数 20，让 Agent 对微小的进步也敏感
        reward = improvement * 20.0
        
        # 2. 成功奖励 (Jackpot)
        if current_score < self.target_score:
            reward += 50.0 
        
        # 3. 步数惩罚 (Time Penalty)
        reward -= 0.1
        
        # 【修复】限制奖励范围，防止梯度爆炸
        reward = np.clip(reward, -20.0, 50.0) 
        return reward




    def compute_reward(self, score, grad):
        """计算奖励"""
        # 基础奖励
        reward = -score
        
        # 成功奖励
        if score < self.target_score:
            reward += 10.0
        
        # 梯度奖励
        reward += abs(grad) * 0.1
        
        # 步数惩罚
        reward -= self.step_count * 0.01
        
        return reward
    
    # def reset(self):
    #     """重置环境"""
    #     self.current_binary = self.original_binary
    #     self.mutation_history = []
    #     self.step_count = 0
        
    #     # 提取初始特征
    #     state = self.extract_features(self.original_binary)
    #     return state

    def reset(self, force_switch=False):
        """
        重置环境：实现自动切换目标 (Hold-N Strategy)
        
        参数:
            force_switch: 如果为 True，强制切换目标（用于错误恢复）
        """
        # 强制切换（错误恢复）：忽略 Hold-N 策略，直接切换目标
        if force_switch:
            self.current_sample_data = random.choice(self.dataset)
            self.episodes_on_current = 0
            self.original_binary = self.current_sample_data['binary_path']
            self.function_name = self.current_sample_data['func_name']
            logger.warning(f"🔄 FORCE SWITCH (Error Recovery) -> {os.path.basename(self.original_binary)}::{self.function_name}")
            logger.info(f"   Version: {self.current_sample_data.get('version')} | Opt: {self.current_sample_data.get('opt_level')}")
        # 正常切换：检查是否需要切换目标
        elif self.current_sample_data is None or self.episodes_on_current >= self.sample_hold_interval:
            # 随机抽取一个新样本
            self.current_sample_data = random.choice(self.dataset)
            self.episodes_on_current = 0
            
            # 更新环境配置
            self.original_binary = self.current_sample_data['binary_path']
            self.function_name = self.current_sample_data['func_name']
            
            logger.success(f"🔄 SWITCH TARGET -> {os.path.basename(self.original_binary)}::{self.function_name}")
            logger.info(f"   Version: {self.current_sample_data.get('version')} | Opt: {self.current_sample_data.get('opt_level')}")
        else:
            # 保持当前目标，增加计数
            self.episodes_on_current += 1
            logger.info(f"🔄 KEEP TARGET ({self.episodes_on_current}/{self.sample_hold_interval}) -> {self.function_name}")

        # 重置环境状态
        self.current_binary = self.original_binary
        self.mutation_history = []
        self.step_count = 0
        
        # 提取初始特征
        state = self.extract_features(self.original_binary)
        return state
    
    def clear_acfg_cache(self):
        """
        清理 ACFG 特征缓存
        
        用于释放内存，通常在切换大量不同目标时调用
        """
        cache_size = len(self._acfg_cache)
        self._acfg_cache.clear()
        logger.info(f"已清理 ACFG 缓存: 释放 {cache_size} 个条目")
    
    def get_cache_stats(self):
        """
        获取缓存统计信息
        
        返回:
            dict: 包含命中率、命中数、未命中数等统计信息
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            'cache_size': len(self._acfg_cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate
        }

if __name__ == "__main__":
    # 测试用例
    import argparse
    
    parser = argparse.ArgumentParser(description='Binary Perturbation Environment')
    parser.add_argument('--binary', required=True, help='Original binary path')
    parser.add_argument('--function', required=True, help='Target function name')
    parser.add_argument('--save-path', required=True, help='Save path for mutations')
    
    args = parser.parse_args()
    
    env = BinaryPerturbationEnv(
        original_binary=args.binary,
        function_name=args.function,
        save_path=args.save_path
    )
    
    logger.info("Environment initialized successfully")
    
    # 测试重置
    state = env.reset()
    logger.info("Initial state shape: {}".format(len(state)))
    
    # 测试单步
    logger.info("Testing mutation with action=5...")
    next_state, reward, done, info = env.step(5)
    logger.info("Score: {:.4f}, Reward: {:.4f}".format(info.get('score', 0), reward))

