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
from run_utils import run_one, train_pickle
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
        
        # 加载模型
        logger.info("Loading model...")
        model_path = "/home/ycy/ours/Deceiving-DNN-based-Binary-Matching/gnu.pickle"
        with open(model_path, 'rb') as f:
            self.model_original = pickle.load(f)
        
        # 变异历史
        self.mutation_history = []
        self.current_binary = original_binary
        self.step_count = 0
        self.target_score = 0.40
        
        logger.info("Environment initialized successfully")
    
    def extract_features(self, binary_path):
        """
        提取二进制文件的特征向量
        
        返回:
            features: numpy array, 特征向量
        """
        try:
            # 这里需要根据实际情况实现特征提取
            # 可以使用 run_objdump + 特征工程
            # 简化版：返回固定维度的随机特征（需要替换为真实实现）
            
            # TODO: 实现真实的特征提取
            # 1. 运行 objdump 获取汇编代码
            # 2. 提取指令序列、控制流图等特征
            # 3. 使用模型嵌入层编码
            
            # 临时实现：返回128维特征
            features = np.random.randn(128).astype(np.float32)
            return features.tolist()
            
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
            
            logger.debug("Command: " + " ".join(cmd))
            
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

