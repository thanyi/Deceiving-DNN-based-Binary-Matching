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
        self.state_dim = 64  # 默认状态维度（推荐 64），可以通过参数修改
        
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
        全新的特征提取函数 (64维)
        组成: [历史特征(10)] + [ACFG结构(6)] + [指令统计(16)] + [关键指令(10)] + [Padding(22)]
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
            
            # 2. 调用 R2 提取
            # 注意：这里每次实例化会有开销，但在 env.step 频率下是可以接受的
            # 如果追求极致性能，可以在 __init__ 维护 r2 实例，但这涉及文件句柄切换
            r2_ext = RadareACFGExtractor(binary_path)
            
            # 提取数据
            acfg_data = r2_ext.get_acfg_features(function_name=target_name, function_addr=target_addr)
            r2_ext.close() # 记得关闭
            
            if acfg_data:
                acfg_vec = self._vectorize_acfg(acfg_data)
                
        except Exception as e:
            logger.warning(f"Feature extraction failed for {binary_path}: {e}")
            # 保持全0
        
        features.extend(acfg_vec)
        
        # 最终截断或补齐到 64 维
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
        将 r2_acfg_features 返回的字典数据转换为向量
        目标长度: 54维 (因为 Part 1 占了 10 维)
        """
        vec = []
        
        n_nodes = data.get('num_nodes', 0)
        n_nodes = max(n_nodes, 1.0) # 分母至少是 1.0

        n_edges = data.get('num_edges', 0)
        bbs = data.get('basic_blocks', {}).values()
        
        # --- A. 全局图结构 (6维) ---
        # 1. 节点数 (Log缩放)
        vec.append(np.log1p(n_nodes))
        # 2. 边数 (Log缩放)
        vec.append(np.log1p(n_edges))
        # 3. 圈复杂度 (E - N + 2)
        complexity = max(0, n_edges - n_nodes + 2)
        vec.append(np.log1p(complexity))
        # 4. 密度 (E / N)
        vec.append(n_edges / n_nodes if n_nodes > 0 else 0)
        # 5. 平均指令数 per Block
        total_instr = sum(b['n_instructions'] for b in bbs)
        total_instr = max(total_instr, 1.0) # 分母至少是 1.0
        vec.append(total_instr / n_nodes if n_nodes > 0 else 0)
        # 6. 总指令数 (Log缩放)
        vec.append(np.log1p(total_instr))
        
        # --- B. 指令类型统计 (16维) ---
        # 包含 8 种类型的：总量(Log) 和 占比(Ratio)
        # 类型: arith, logic, transfer, redirect, call, numeric, string, total
        
        keys = ['n_arith_instrs', 'n_logic_instrs', 'n_transfer_instrs', 
                'n_redirect_instrs', 'n_call_instrs', 'n_numeric_consts', 
                'n_string_consts']
        
        # 统计总和
        sums = {k: sum(b.get(k, 0) for b in bbs) for k in keys}
        
        # B1. 总量特征 (7维)
        for k in keys:
            vec.append(np.log1p(sums[k]))
            
        # B2. 密度特征 (7维，该类型指令占总指令的比例)
        for k in keys:
            vec.append(sums[k] / total_instr if total_instr > 0 else 0)
            
        # 补齐 B 部分剩余维度 (16 - 14 = 2维)
        # 比如：逻辑指令 / 算术指令 (混淆度量)
        vec.append(sums['n_logic_instrs'] / (sums['n_arith_instrs'] + 1))
        # 比如：转移指令 / 节点数
        vec.append(sums['n_transfer_instrs'] / n_nodes if n_nodes > 0 else 0)

        # --- C. 关键特征 & 变异敏感度 (10维) ---
        # 统计每个 Block 的平均特征
        
        # C1-C7: 平均每个块有多少个某类指令
        for k in keys:
             vec.append(sums[k] / n_nodes if n_nodes > 0 else 0)
             
        # C8: 包含字符串引用的 Block 比例 (数据流特征)
        blocks_with_str = sum(1 for b in bbs if b.get('n_string_consts', 0) > 0)
        vec.append(blocks_with_str / n_nodes if n_nodes > 0 else 0)
        
        # C9: 包含 Call 的 Block 比例 (函数调用密集度)
        blocks_with_call = sum(1 for b in bbs if b.get('n_call_instrs', 0) > 0)
        vec.append(blocks_with_call / n_nodes if n_nodes > 0 else 0)
        
        # C10: 包含 Logic 的 Block 比例 (加密/混淆块比例)
        blocks_with_logic = sum(1 for b in bbs if b.get('n_logic_instrs', 0) > 0)
        vec.append(blocks_with_logic / n_nodes if n_nodes > 0 else 0)

        # --- D. Padding (剩余维度) ---
        # 目前用到: 6 + 16 + 10 = 32维
        # 需要补齐到 54维 (54 - 32 = 22)
        
        current_len = len(vec)
        needed = 54 - current_len # 54 + 10(History) = 64
        
        if needed > 0:
            vec.extend([0.0] * needed)
            
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
            score, grad = run_one(
                self.original_binary,
                mutated_binary,
                self.model_original,
                checkdict,
                self.function_name
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

    def reset(self):
        """
        重置环境：实现自动切换目标 (Hold-N Strategy)
        """
        # 检查是否需要切换目标
        if self.current_sample_data is None or self.episodes_on_current >= self.sample_hold_interval:
            # 随机抽取一个新样本
            self.current_sample_data = random.choice(self.dataset)
            self.episodes_on_current = 0
            
            # 更新环境配置
            self.original_binary = self.current_sample_data['binary_path']
            self.function_name = self.current_sample_data['func_name']
            
            logger.success(f"🔄 SWITCH TARGET -> {os.path.basename(self.original_binary)}::{self.function_name}")
            logger.info(f"   Version: {self.current_sample_data.get('version')} | Opt: {self.current_sample_data.get('opt_level')}")
        else:
            self.episodes_on_current += 1
            logger.info(f"🔄 KEEP TARGET ({self.episodes_on_current}/{self.sample_hold_interval}) -> {self.function_name}")

        # 重置环境状态
        self.current_binary = self.original_binary
        self.mutation_history = []
        self.step_count = 0
        
        # 提取初始特征
        state = self.extract_features(self.original_binary)
        return state

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

