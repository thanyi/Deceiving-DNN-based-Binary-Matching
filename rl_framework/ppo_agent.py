#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPO Agent for Binary Code Perturbation
基于 Proximal Policy Optimization 的二进制代码变异智能体

依赖: pip install torch numpy pandas
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import json
import os
from collections import deque

class PolicyNetwork(nn.Module):
    """策略网络：输出每个动作的概率分布"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """前向传播：返回动作概率和状态价值"""
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value


class PPOAgent:
    """PPO 智能体"""
    
    def __init__(self, state_dim, n_actions=8, lr=3e-4, gamma=0.99, 
                 epsilon=0.2, epochs=10, device='cpu'):
        """
        参数:
            state_dim: 状态维度（特征向量长度）
            n_actions: 动作数量（8种变异策略）
            lr: 学习率
            gamma: 折扣因子
            epsilon: PPO裁剪参数
            epochs: 每次更新的训练轮数
            device: 'cpu' 或 'cuda'
        """
        self.device = torch.device(device)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        
        # 动作映射：索引 -> 实际变异模式
        self.action_map = [1, 2, 3, 5, 7, 8, 9, 11]
        self.n_actions = n_actions
        
        # 初始化网络
        self.policy = PolicyNetwork(state_dim, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # 经验缓冲
        self.memory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': []
        }
    
    def select_action(self, state, explore=True):
        """
        选择动作
        
        参数:
            state: 当前状态特征向量
            explore: 是否探索（训练时为True，测试时为False）
        
        返回:
            action_idx: 动作索引 (0-7)
            actual_action: 实际变异模式 (1,2,3,5,7,8,9,11)
            log_prob: 对数概率
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, state_value = self.policy(state)
        
        # 创建分类分布
        dist = Categorical(action_probs)
        
        if explore:
            action = dist.sample()
        else:
            action = torch.argmax(action_probs, dim=-1)
        
        log_prob = dist.log_prob(action)
        
        action_idx = action.item()
        actual_action = self.action_map[action_idx]
        
        return action_idx, actual_action, log_prob.item(), state_value.item()
    
    def store_transition(self, state, action, reward, log_prob, value):
        """存储单步经验"""
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['rewards'].append(reward)
        self.memory['log_probs'].append(log_prob)
        self.memory['values'].append(value)
    
    def compute_returns(self, next_value=0):
        """计算回报（使用 GAE - Generalized Advantage Estimation）"""
        returns = []
        advantages = []
        
        R = next_value
        gae = 0
        
        rewards = self.memory['rewards']
        values = self.memory['values'] + [next_value]
        
        # 反向计算 GAE
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] - values[i]
            gae = delta + self.gamma * 0.95 * gae  # lambda=0.95
            returns.insert(0, gae + values[i])
            advantages.insert(0, gae)
        
        return returns, advantages
    
    def update(self):
        """PPO 更新策略"""
        if len(self.memory['states']) == 0:
            return 0.0
        
        # 计算回报和优势
        returns, advantages = self.compute_returns()
        
        # 转换为 tensor
        states = torch.FloatTensor(np.array(self.memory['states'])).to(self.device)
        actions = torch.LongTensor(self.memory['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(self.memory['log_probs']).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # 标准化优势（增强数值稳定性）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.clamp(advantages, -10.0, 10.0)  # 裁剪防止极端值
        
        total_loss = 0
        
        # PPO 多轮更新
        for _ in range(self.epochs):
            # 前向传播
            action_probs, state_values = self.policy(states)
            
            # 数值稳定性检查
            action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)  # 重新归一化
            
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # 计算比率（裁剪防止溢出）
            log_ratio = new_log_probs - old_log_probs
            log_ratio = torch.clamp(log_ratio, -20, 20)  # 防止 exp 溢出
            ratio = torch.exp(log_ratio)
            
            # PPO 裁剪目标
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic 损失
            critic_loss = 0.5 * (returns - state_values.squeeze()).pow(2).mean()
            
            # 总损失（加入熵正则化鼓励探索）
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            # 更新网络
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # 清空缓冲
        self.clear_memory()
        
        return total_loss / self.epochs
    
    def clear_memory(self):
        """清空经验缓冲"""
        self.memory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': []
        }
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"模型已保存到: {path}")
    
    def load(self, path):
        """加载模型"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"模型已从 {path} 加载")
        else:
            print(f"模型文件 {path} 不存在")


class RewardShaper:
    """奖励塑形器：设计更好的奖励函数"""
    
    def __init__(self, target_score=0.40):
        self.target_score = target_score
        self.best_score = float('inf')
    
    def compute_reward(self, score, grad, done, step_count):
        """
        计算奖励（修复数值稳定性）
        
        参数:
            score: 相似度分数（越低越好）
            grad: 梯度值
            done: 是否达到目标
            step_count: 当前步数
        
        返回:
            reward: 奖励值（归一化到合理范围）
        """
        # 基础奖励：归一化分数变化
        reward = (1.0 - score) * 10.0  # 映射到 [0, 10]
        
        # 成功奖励
        if done and score < self.target_score:
            reward += 50.0  # 大奖励
        
        # 进步奖励（相比历史最优）
        if score < self.best_score:
            improvement = self.best_score - score
            reward += improvement * 100.0  # 放大进步信号
            self.best_score = score
        
        # 梯度引导奖励（归一化）
        # grad 通常在 416~417，归一化到 [0, 1]
        normalized_grad = (grad - 410) / 10.0  # 假设范围 [410, 420]
        reward += normalized_grad * 2.0  # 贡献 [0, 2]
        
        # 步数惩罚（避免过长序列）
        reward -= step_count * 0.05
        
        # 裁剪奖励到合理范围，避免溢出
        reward = max(-10.0, min(reward, 100.0))
        
        return reward
    
    def reset(self):
        """重置最优分数"""
        self.best_score = float('inf')


if __name__ == "__main__":
    # 测试代码
    print("PPO Agent 初始化测试...")
    
    state_dim = 128  # 假设特征维度
    agent = PPOAgent(state_dim=state_dim, n_actions=8)
    
    # 模拟一个状态
    dummy_state = np.random.randn(state_dim)
    action_idx, actual_action, log_prob, value = agent.select_action(dummy_state)
    
    print(f"选择的动作索引: {action_idx}")
    print(f"实际变异模式: {actual_action}")
    print(f"对数概率: {log_prob:.4f}")
    print(f"状态价值: {value:.4f}")
    
    print("\n测试通过! ✓")

