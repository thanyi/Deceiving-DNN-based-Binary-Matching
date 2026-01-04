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
from loguru import logger

class PolicyNetwork(nn.Module):
    """策略网络：输出每个动作的概率分布（改进版）"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        
        # Actor 网络（更深，带归一化）
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic 网络（独立架构）
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state):
        """前向传播：返回动作概率和状态价值"""
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value


class PPOAgent:
    """PPO 智能体"""
    
    def __init__(self, state_dim, n_actions=6, lr=1e-4, gamma=0.99, 
                 epsilon=0.2, epochs=10, device='cpu'):
        """
        参数:
            state_dim: 状态维度（特征向量长度）
            n_actions: 动作数量（6种变异策略）
            lr: 学习率（降低到 1e-4）
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
        self.action_map = [1, 2, 7, 8, 9, 11]
        self.n_actions = n_actions
        
        # 初始化网络
        self.policy = PolicyNetwork(state_dim, n_actions).to(self.device)
        
        # 分离 Actor 和 Critic 优化器（降低学习率）
        self.actor_optimizer = optim.Adam(self.policy.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.policy.critic.parameters(), lr=lr * 2)  # Critic 学习稍快
        
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
            
            # 检查计算结果
            if np.isnan(gae) or np.isinf(gae):
                logger.error(f"[ppo_agent.py:compute_returns]: NaN/Inf in gae at step {i}")
                logger.error(f"  delta={delta}, gae={gae}, rewards[{i}]={rewards[i]}, values[{i}]={values[i]}, values[{i+1}]={values[i+1]}")
                # 使用 0 作为后备
                gae = 0.0
            
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
        # 检查 advantages 是否包含 NaN
        if torch.isnan(advantages).any():
            logger.error(f"[ppo_agent.py:update]: NaN in advantages before normalization: {advantages}")
            logger.error(f"  returns: {returns}, memory values: {self.memory['values']}")
        
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        
        # 检查统计量
        if torch.isnan(adv_mean) or torch.isnan(adv_std):
            logger.error(f"[ppo_agent.py:update]: NaN in advantages statistics: mean={adv_mean}, std={adv_std}")
            logger.error(f"  advantages: {advantages}")
            # 使用零均值单位方差作为后备
            advantages = torch.zeros_like(advantages)
        else:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
            advantages = torch.clamp(advantages, -10.0, 10.0)  # 裁剪防止极端值
        
        total_loss = 0
        
        # PPO 多轮更新（分离 Actor 和 Critic 更新）
        total_actor_loss = 0
        total_critic_loss = 0
        
        for epoch in range(self.epochs):
            # 前向传播
            action_probs, state_values = self.policy(states)
            
            # 数值稳定性检查
            action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)
            prob_sum = action_probs.sum(dim=-1, keepdim=True)
            if (prob_sum == 0).any() or torch.isnan(prob_sum).any():
                logger.warning("[ppo_agent.py:update]: prob_sum is zero or NaN, skipping this epoch")
                continue
            action_probs = action_probs / prob_sum  # 重新归一化
            # logger.info(f"[ppo_agent.py:update]: action_probs: {action_probs}")
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
            
            # 加入熵正则化鼓励探索
            actor_loss = actor_loss - 0.02 * entropy  # 提高熵系数到 0.02
            
            # 更新 Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)  # 保留计算图供 Critic 使用
            torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            # Critic 损失（使用 Huber Loss 更鲁棒）
            # 修复尺寸不匹配问题
            state_values_squeezed = state_values.squeeze()
            if state_values_squeezed.dim() == 0:
                state_values_squeezed = state_values_squeezed.unsqueeze(0)
            if state_values_squeezed.shape != returns.shape:
                # 如果形状不匹配，调整 returns
                if returns.dim() == 0:
                    returns = returns.unsqueeze(0)
                if state_values_squeezed.shape[0] != returns.shape[0]:
                    # 取较小的长度
                    min_len = min(state_values_squeezed.shape[0], returns.shape[0])
                    state_values_squeezed = state_values_squeezed[:min_len]
                    returns = returns[:min_len]
            
            critic_loss = nn.functional.smooth_l1_loss(state_values_squeezed, returns)
            
            # 检查 critic_loss 是否包含 NaN
            if torch.isnan(critic_loss):
                logger.error("[ppo_agent.py:update]: NaN in critic_loss!")
                logger.error(f"  state_values: {state_values}, returns: {returns}")
                continue
            
            # 更新 Critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), 0.5)
            self.critic_optimizer.step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
        
        total_loss = total_actor_loss + total_critic_loss
        
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
        """保存模型（包含两个优化器）"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
        print(f"模型已保存到: {path}")
    
    def load(self, path):
        """加载模型（兼容旧版本）"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            
            # 兼容旧版本（单优化器）
            if 'actor_optimizer_state_dict' in checkpoint:
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            elif 'optimizer_state_dict' in checkpoint:
                print("警告: 加载旧版本模型，优化器状态未完全恢复")
            
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
        计算奖励（改进版：降低尺度，提高稳定性）
        
        参数:
            score: 相似度分数（越低越好，目标 < 0.40）
            grad: 梯度值
            done: 是否达到目标
            step_count: 当前步数
        
        返回:
            reward: 奖励值（归一化到 [-5, 10] 范围）
        """
        # 基础奖励：使用对数缩放，降低极端值
        # score 从 1.0 → 0.4，reward 从 0 → 6
        reward = -np.log(score + 0.01) * 2.0  # 对数缩放，更平滑
        
        # 成功奖励（降低到 10）
        if done and score < self.target_score:
            reward += 10.0
        
        # 进步奖励（大幅降低系数）
        if score < self.best_score:
            improvement = self.best_score - score
            reward += improvement * 20.0  # 从 100 降低到 20
            self.best_score = score
        else:
            # 如果没有进步，小惩罚
            reward -= 0.5
        
        # 移除梯度奖励（前面分析表明梯度信息冗余）
        
        # 步数惩罚（降低系数）
        reward -= step_count * 0.02  # 从 0.05 降低到 0.02
        
        # 裁剪奖励到合理范围（收窄范围）
        reward = np.clip(reward, -5.0, 10.0)
        
        return reward
    
    def reset(self):
        """重置最优分数"""
        self.best_score = float('inf')


if __name__ == "__main__":
    # 测试代码
    print("PPO Agent 初始化测试...")
    
    state_dim = 64  # 特征维度（推荐 64）
    agent = PPOAgent(state_dim=state_dim, n_actions=6)
    
    # 模拟一个状态
    dummy_state = np.random.randn(state_dim)
    action_idx, actual_action, log_prob, value = agent.select_action(dummy_state)
    
    print(f"选择的动作索引: {action_idx}")
    print(f"实际变异模式: {actual_action}")
    print(f"对数概率: {log_prob:.4f}")
    print(f"状态价值: {value:.4f}")
    
    print("\n测试通过! ✓")

