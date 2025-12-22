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

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入现有模块
from run_utils import run_one
import run_objdump

class BinaryPerturbationEnv:
    """
    二进制代码变异环境 (Python 3)
    
    与 PPO Agent 在同一进程中运行，通过函数调用通信
    """
    
    def __init__(self, original_binary, function_name, save_path):
        """
        参数:
            original_binary: 原始二进制文件路径
            function_name: 目标函数名
            save_path: 保存变异结果的路径
        """
        # 转为绝对路径
        self.original_binary = os.path.abspath(original_binary)
        self.function_name = function_name
        self.save_path = os.path.abspath(save_path)
        
        # 项目根目录（uroboros 所在目录）
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 不再需要加载模型（使用 asm2vec 方法）
        self.model_original = None
        logger.info("Using asm2vec detection method (no model loading required)")
        
        # 变异历史
        self.mutation_history = []
        self.current_binary = original_binary
        self.step_count = 0
        self.target_score = 0.40
        self.state_dim = 64  # 默认状态维度（推荐 64），可以通过参数修改
        
        logger.info("Environment initialized successfully")
    
    def set_state_dim(self, state_dim):
        """
        设置状态维度（用于与 PPO Agent 保持一致）
        
        参数:
            state_dim: 状态维度
        """
        self.state_dim = state_dim
        logger.info(f"状态维度设置为: {state_dim}")
    
    def extract_features(self, binary_path):
        """
        提取二进制文件的特征向量（函数级别特征）
        
        返回:
            features: list, 特征向量 (state_dim维)
        """
        try:
            if not os.path.exists(binary_path):
                logger.warning(f"二进制文件不存在: {binary_path}")
                return [0.0] * self.state_dim
            
            features = []
            
            # ========== 第一部分：变异历史和相似度特征 (约10维) ==========
            
            # 1. 当前相似度分数（最重要的特征）
            if self.mutation_history:
                last_score = self.mutation_history[-1].get('score', 1.0)
                features.append(min(last_score, 1.0))
            else:
                features.append(1.0)  # 初始状态
            
            # 2. 相似度变化趋势（最近3步）
            if len(self.mutation_history) >= 2:
                recent_scores = [m.get('score', 1.0) for m in self.mutation_history[-3:]]
                # 计算趋势：是在改善还是变差
                score_delta = recent_scores[-1] - recent_scores[0]
                features.append(max(-1.0, min(1.0, score_delta)))  # 归一化到 [-1, 1]
                # 最近的最好分数
                features.append(min(recent_scores))
            else:
                features.extend([0.0, 1.0])
            
            # 3. 变异历史统计
            mutation_count = len(self.mutation_history)
            step_count = self.step_count
            features.extend([
                min(mutation_count / 50.0, 1.0),  # 归一化
                min(step_count / 50.0, 1.0),
            ])
            
            # 4. 已应用的变异类型统计（one-hot 编码）
            # 动作集: [1, 2, 7, 8, 9, 11]
            action_counts = {1: 0, 2: 0, 7: 0, 8: 0, 9: 0, 11: 0}
            for m in self.mutation_history:
                action = m.get('action')
                if action in action_counts:
                    action_counts[action] += 1
            # 归一化为频率
            total_actions = sum(action_counts.values()) if mutation_count > 0 else 1
            for action in [1, 2, 7, 8, 9, 11]:
                features.append(action_counts[action] / total_actions)
            
            # ========== 第二部分：函数级别的特征 (约20维) ==========
            
            # 5. 提取目标函数的汇编代码统计特征
            func_features = self._extract_function_features(binary_path)
            features.extend(func_features)
            
            # ========== 第三部分：函数内容哈希（提供唯一性）(剩余维度) ==========
            
            # 6. 函数内容的哈希值（确保不同函数有不同特征）
            func_hash = self._get_function_hash(binary_path)
            features.extend(func_hash)
            
            # ========== 填充和截断 ==========
            
            # 填充到 state_dim 维
            if len(features) < self.state_dim:
                features.extend([0.0] * (self.state_dim - len(features)))
            
            # 截断到 state_dim 维
            features = features[:self.state_dim]
            
            # 确保所有值在合理范围内
            features = [max(0.0, min(1.0, f)) for f in features]
            
            return features
            
        except Exception as e:
            logger.error(f"特征提取失败: {e}, 使用零向量")
            # 降级方案：返回零向量（但至少是确定性的）
            return [0.0] * self.state_dim
    
    def _extract_function_features(self, binary_path):
        """
        提取函数级别的统计特征（约20维）
        
        返回:
            features: list, 函数特征向量
        """
        try:
            from scripts.bin2asm_util import binfunc2asm
            
            # 获取函数名（处理变异后的函数名变化）
            mutated_folder = os.path.dirname(binary_path)
            checkdict_path = os.path.join(mutated_folder, "sym_to_addr.pickle")
            
            target_func_name = self.function_name
            if os.path.exists(checkdict_path):
                try:
                    with open(checkdict_path, 'rb') as f:
                        checkdict = pickle.load(f)
                    ori_sym_addr = checkdict.get(self.function_name)
                    if ori_sym_addr:
                        target_func_name = 'func_' + ori_sym_addr[2:].lower()
                except:
                    pass
            
            # 提取汇编代码
            asm_file = binfunc2asm(
                ipath=binary_path,
                target_func_name=target_func_name,
                opath='/tmp/rl_func_features/',
                verbose=False
            )
            
            if not asm_file or not os.path.exists(asm_file):
                logger.debug(f"无法提取函数汇编: {binary_path}")
                return self._default_function_features()
            
            # 读取汇编代码并提取统计特征
            with open(asm_file, 'r') as f:
                asm_lines = f.readlines()
            
            # 过滤掉注释和空行
            code_lines = [l.strip() for l in asm_lines if l.strip() and not l.strip().startswith('#')]
            
            # 1. 函数长度（指令数量）
            num_instructions = len(code_lines)
            features = [min(num_instructions / 1000.0, 1.0)]  # 归一化，假设最多1000条指令
            
            # 2. 基本块数量（通过标签估算）
            num_labels = sum(1 for l in code_lines if ':' in l and not l.startswith('\t'))
            features.append(min(num_labels / 100.0, 1.0))  # 归一化
            
            # 3. 跳转指令数量（控制流复杂度）
            jump_instrs = ['jmp', 'je', 'jne', 'jg', 'jl', 'jge', 'jle', 'ja', 'jb', 'jae', 'jbe', 'call', 'ret']
            num_jumps = sum(1 for l in code_lines for j in jump_instrs if j in l.lower())
            features.append(min(num_jumps / 100.0, 1.0))  # 归一化
            
            # 4. 调用指令数量
            num_calls = sum(1 for l in code_lines if 'call' in l.lower())
            features.append(min(num_calls / 50.0, 1.0))  # 归一化
            
            # 5. 返回指令数量
            num_rets = sum(1 for l in code_lines if 'ret' in l.lower())
            features.append(min(num_rets / 10.0, 1.0))  # 归一化
            
            # 6. 内存访问指令数量
            mem_instrs = ['mov', 'lea', 'push', 'pop', 'load', 'store']
            num_mem = sum(1 for l in code_lines for m in mem_instrs if m in l.lower())
            features.append(min(num_mem / 500.0, 1.0))  # 归一化
            
            # 7. 算术指令数量
            arith_instrs = ['add', 'sub', 'mul', 'div', 'inc', 'dec', 'neg']
            num_arith = sum(1 for l in code_lines for a in arith_instrs if a in l.lower())
            features.append(min(num_arith / 200.0, 1.0))  # 归一化
            
            # 8. 逻辑指令数量
            logic_instrs = ['and', 'or', 'xor', 'not', 'shl', 'shr']
            num_logic = sum(1 for l in code_lines for lg in logic_instrs if lg in l.lower())
            features.append(min(num_logic / 100.0, 1.0))  # 归一化
            
            # 9. 比较指令数量
            num_cmp = sum(1 for l in code_lines if 'cmp' in l.lower() or 'test' in l.lower())
            features.append(min(num_cmp / 100.0, 1.0))  # 归一化
            
            # 10. 函数复杂度：指令/基本块比例（指令密度）
            if num_labels > 0:
                instr_per_bb = num_instructions / num_labels
                features.append(min(instr_per_bb / 50.0, 1.0))  # 归一化
            else:
                features.append(0.5)  # 默认值
            
            # 11. 控制流复杂度：跳转/指令比例
            if num_instructions > 0:
                jump_ratio = num_jumps / num_instructions
                features.append(min(jump_ratio, 1.0))
            else:
                features.append(0.0)
            
            # 12-20. 函数长度变化率（相对于原始函数）
            if binary_path != self.original_binary:
                # 获取原始函数的指令数量
                original_asm = binfunc2asm(
                    ipath=self.original_binary,
                    target_func_name=self.function_name,
                    opath='/tmp/rl_func_features/',
                    verbose=False
                )
                if original_asm and os.path.exists(original_asm):
                    with open(original_asm, 'r') as f:
                        orig_lines = [l.strip() for l in f.readlines() if l.strip() and not l.strip().startswith('#')]
                    orig_count = len(orig_lines)
                    if orig_count > 0:
                        length_ratio = num_instructions / orig_count
                        features.append(min(length_ratio, 2.0) / 2.0)  # 归一化到 [0, 1]
                    else:
                        features.append(1.0)
                else:
                    features.append(1.0)
            else:
                features.append(1.0)  # 原始文件，比例为1
            
            # 填充到20维
            while len(features) < 20:
                features.append(0.0)
            
            return features[:20]
            
        except Exception as e:
            logger.debug(f"提取函数特征失败: {e}")
            return self._default_function_features()
    
    def _default_function_features(self):
        """返回默认的函数特征（20维零向量）"""
        return [0.0] * 20
    
    def _get_function_hash(self, binary_path):
        """
        获取函数内容的哈希值（提供唯一性）
        针对 64 维优化：返回约 33 维
        
        返回:
            features: list, 哈希特征向量
        """
        try:
            from scripts.bin2asm_util import binfunc2asm
            
            # 获取函数名
            mutated_folder = os.path.dirname(binary_path)
            checkdict_path = os.path.join(mutated_folder, "sym_to_addr.pickle")
            
            target_func_name = self.function_name
            if os.path.exists(checkdict_path):
                try:
                    with open(checkdict_path, 'rb') as f:
                        checkdict = pickle.load(f)
                    ori_sym_addr = checkdict.get(self.function_name)
                    if ori_sym_addr:
                        target_func_name = 'func_' + ori_sym_addr[2:].lower()
                except:
                    pass
            
            # 提取汇编代码
            asm_file = binfunc2asm(
                ipath=binary_path,
                target_func_name=target_func_name,
                opath='/tmp/rl_func_features/',
                verbose=False
            )
            
            if not asm_file or not os.path.exists(asm_file):
                # 降级：使用文件哈希
                with open(binary_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).digest()
                return [b / 255.0 for b in file_hash]  # 16 维
            
            # 计算函数内容的哈希
            with open(asm_file, 'r') as f:
                func_content = f.read()
            
            # 针对 64 维优化：MD5(16) + SHA256前半部分(16) = 32 维
            md5_hash = hashlib.md5(func_content.encode()).digest()  # 16 bytes
            sha256_hash = hashlib.sha256(func_content.encode()).digest()[:16]  # 取前 16 bytes
            
            # 组合哈希值，总共 32 维（适合 64 维特征空间）
            combined_hash = md5_hash + sha256_hash  # 32 bytes = 32 维
            
            return [b / 255.0 for b in combined_hash]
            
        except Exception as e:
            logger.debug(f"提取函数哈希失败: {e}")
            # 降级：使用文件哈希
            try:
                with open(binary_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).digest()
                return [b / 255.0 for b in file_hash]
            except:
                return [0.0] * 16
            
        except Exception as e:
            logger.error("Feature extraction failed: {}".format(e))
            return None
    
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
            
            # 生成临时二进制文件名
            tmp_bin = '/home/ycy/ours/Deceiving-DNN-based-Binary-Matching/rl_framework/rl_output/mutant_' + str(int(time.time() * 1000)) + '.bin'
            
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
            output = subprocess.check_output(
                cmd, 
                stderr=subprocess.STDOUT,
                cwd=self.project_root
            )
            
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
        
        # 应用变异
        mutated_binary, hash_val = self.apply_mutation(self.current_binary, action)
        
        if mutated_binary is None:
            # 变异失败：保持当前状态，给予负奖励
            state = self.extract_features(self.current_binary)
            return state, -10.0, False, {'error': 'mutation_failed'}
        
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
        reward = self.compute_reward(score, grad)
        
        # 判断是否完成
        done = score < self.target_score or self.step_count >= 50
        
        info = {
            'score': score,
            'grad': grad,
            'step': self.step_count,
            'binary': mutated_binary
        }
        
        logger.info("Step {}: action={}, score={:.4f}, reward={:.4f}".format(
            self.step_count, action, score, reward
        ))
        
        return state, reward, done, info
    
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
    
    def reset(self):
        """重置环境"""
        self.current_binary = self.original_binary
        self.mutation_history = []
        self.step_count = 0
        
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

