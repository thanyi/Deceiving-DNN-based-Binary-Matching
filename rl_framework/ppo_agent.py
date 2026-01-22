#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPO Agent for Binary Code Perturbation
基于 Proximal Policy Optimization 的二进制代码变异智能体

依赖: pip install torch numpy pandas
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import json
import os
from collections import deque
from loguru import logger

class DualHeadPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super(DualHeadPolicyNetwork, self).__init__()
        
        # 定义切片位置 (根据上面的特征工程)
        self.topk_dim = 32
        self.num_candidates = 3
        # 假设前 16+40=56 维是历史+拓扑
        # 56 到 56+96 是 Top-1/2/3
        self.start_idx = 56 
        
        # 1. Context Encoder (编码除了 Top-3 以外的所有全局特征)
        # context_dim = total - (32*3)
        self.context_dim = state_dim - (self.topk_dim * self.num_candidates)
        self.context_net = nn.Sequential(
            nn.Linear(self.context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 2. Candidate Encoder (编码 Top-1/2/3 每个块的特征)
        # 这是一个共享权重的层，处理每个候选块
        self.candidate_net = nn.Sequential(
            nn.Linear(self.topk_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )
        
        # 3. Location Head (决定选哪个块)
        # 输入：Context + 某个Candidate
        # 输出：Score
        self.location_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1) # 给每个块打分
        )
        
        # 4. Action Head (决定做什么动作)
        # 输入：Context + 被选中的Candidate
        # 【修复】移除 Softmax，让 Categorical 自己处理
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
            # Softmax 由 Categorical 自动处理，不需要在这里添加
        )
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, sampled_loc_idx=None):
        """
        前向传播
        
        参数:
            state: [B, state_dim] 状态特征
            sampled_loc_idx: [B] 或 None
                - 训练时：传入采样的 location 索引（硬选择）
                - 推理时：使用 None（软注意力）
        
        返回:
            action_probs: [B, n_actions] 动作概率分布（logits）
            loc_probs: [B, 3] 位置概率分布
            state_value: [B, 1] 状态价值
        """
        batch_size = state.size(0)
        
        # === 1. 拆分数据 ===
        # 提取 Top-3 Raw Features: [Batch, 3, 32]
        flat_candidates = state[:, self.start_idx : self.start_idx + 96]
        candidates = flat_candidates.view(batch_size, 3, 32)
        
        # 提取 Context: [Batch, Rest]
        ctx_part1 = state[:, :self.start_idx]
        ctx_part2 = state[:, self.start_idx + 96:]
        context_raw = torch.cat([ctx_part1, ctx_part2], dim=1)
        
        # === 2. 编码 ===
        ctx_emb = self.context_net(context_raw) # [B, Hidden]
        # 对 3 个候选块分别编码
        cand_embs = self.candidate_net(candidates) # [B, 3, Hidden/2]
        
        # === 3. Location Decision (Attention) ===
        # 把 Context 扩展后和每个 Candidate 拼接
        ctx_expanded = ctx_emb.unsqueeze(1).expand(-1, 3, -1) # [B, 3, Hidden]
        loc_input = torch.cat([ctx_expanded, cand_embs], dim=-1) # [B, 3, Hidden*1.5]
        
        # 计算每个位置的分数
        loc_scores = self.location_head(loc_input).squeeze(-1) # [B, 3]
        
        # 【修复】更鲁棒的 Masking：使用绝对值和避免正负抵消
        is_pad = (candidates.abs().sum(dim=-1) < 1e-6)
        loc_scores = loc_scores.masked_fill(is_pad, -1e9)
        
        loc_probs = F.softmax(loc_scores, dim=-1) # [B, 3] -> Location Policy
        
        # === 4. Action Decision (条件依赖) ===
        # 【关键修复】：根据是否传入 sampled_loc_idx 决定使用硬选择还是软注意力
        if sampled_loc_idx is None:
            # 推理模式：使用软注意力（加权平均）
            selected_cand_emb = torch.bmm(loc_probs.unsqueeze(1), cand_embs).squeeze(1) # [B, Hidden/2]
        else:
            # 训练模式：使用硬选择（实际采样的块）
            # sampled_loc_idx: [B], cand_embs: [B, 3, Hidden/2]
            # 使用 gather 或索引选择
            batch_indices = torch.arange(batch_size, device=state.device)
            selected_cand_emb = cand_embs[batch_indices, sampled_loc_idx]  # [B, Hidden/2]
        
        # Action Head Input
        action_input = torch.cat([ctx_emb, selected_cand_emb], dim=-1)
        action_logits = self.action_head(action_input)  # [B, n_actions] 返回 logits，不是概率
        
        state_value = self.critic(state)
        
        return action_logits, loc_probs, state_value


class PPOAgent:
    """PPO 智能体"""
    
    def __init__(self, state_dim, n_actions=None, n_locs=3, lr=1e-4, gamma=0.99, 
                 epsilon=0.2, epochs=10, device='cpu', action_map=None):
        """
        参数:
            state_dim: 状态维度（特征向量长度）
            n_actions: 动作数量（默认使用 action_map 的长度）
            n_locs: 位置数量（3个候选块）
            lr: 学习率（降低到 1e-4）
            gamma: 折扣因子
            epsilon: PPO裁剪参数
            epochs: 每次更新的训练轮数
            device: 'cpu' 或 'cuda'
            action_map: 动作映射列表（索引 -> 实际变异模式）
        """
        self.device = torch.device(device)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        
        # 动作映射：索引 -> 实际变异模式（保留原有选择顺序）
        default_action_map = [1, 2, 4, 7, 8, 9, 11]
        if action_map is None:
            action_map = list(default_action_map)

        if n_actions is not None:
            if n_actions <= 0:
                raise ValueError("n_actions must be positive")
            if n_actions > len(action_map):
                logger.warning(
                    f"[ppo_agent.py:init]: n_actions ({n_actions}) > action_map size "
                    f"({len(action_map)}), clamping to {len(action_map)}"
                )
                n_actions = len(action_map)
            action_map = action_map[:n_actions]

        self.action_map = action_map
        self.n_actions = len(self.action_map) 
        self.n_locs = n_locs

        # 初始化网络
        self.policy = DualHeadPolicyNetwork(state_dim, self.n_actions, hidden_dim=512).to(self.device)
        
        # 分离 Actor 和 Critic 优化器（降低学习率）
        # 优化器
        self.actor_optimizer = optim.Adam(
            list(self.policy.context_net.parameters()) + 
            list(self.policy.candidate_net.parameters()) + 
            list(self.policy.action_head.parameters()) + 
            list(self.policy.location_head.parameters()), 
            lr=lr
        )

        self.critic_optimizer = optim.Adam(self.policy.critic.parameters(), lr=lr * 2)  # Critic 学习稍快
        
        # 经验缓冲
        self.memory = {
            'states': [],
            'actions': [],
            'locations': [],
            'rewards': [],
            'values': [],
            'log_probs': []
        }
    
    def select_action(self, state, explore=True):
        """
        选择动作（两阶段采样：先选位置，再选动作）
        
        参数:
            state: 当前状态特征向量
            explore: 是否探索（训练时为True，测试时为False）
        
        返回:
            action_idx: 动作索引 (0 ~ n_actions-1)
            actual_action: 实际变异模式 (action_map 中的值)
            loc_idx_val: 位置索引 (0-2)
            joint_log_prob: 联合对数概率
            state_value: 状态价值
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # === 第一阶段：选择位置 ===
        with torch.no_grad():
            # 只使用 location head（不依赖于具体选中的块）
            _, loc_probs, _ = self.policy(state, sampled_loc_idx=None)
        
        dist_loc = Categorical(loc_probs)
        
        if explore:
            loc_idx = dist_loc.sample()
        else:
            loc_idx = torch.argmax(loc_probs, dim=-1)
        
        # === 第二阶段：基于选中的位置选择动作 ===
        with torch.no_grad():
            # 传入选中的 location，让 action head 看到真实的块特征
            action_logits, _, state_value = self.policy(state, sampled_loc_idx=loc_idx)
        
        dist_action = Categorical(logits=action_logits)  # 使用 logits 参数
        
        if explore:
            action_idx = dist_action.sample()
        else:
            action_idx = torch.argmax(action_logits, dim=-1)
        
        # 计算联合对数概率：P(location, action | state) = P(location | state) × P(action | state, location)
        joint_log_prob = dist_loc.log_prob(loc_idx) + dist_action.log_prob(action_idx)
        
        action_idx_val = action_idx.item()
        actual_action = self.action_map[action_idx_val]
        loc_idx_val = loc_idx.item()
        
        return action_idx_val, actual_action, loc_idx_val, joint_log_prob.item(), state_value.item()

    def estimate_value(self, state):
        """估计单个状态的价值（用于截断回合的 bootstrap）"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, state_value = self.policy(state, sampled_loc_idx=None)
        return state_value.item()
    
    def store_transition(self, state, action, location, reward, log_prob, value):
        """存储单步经验"""
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['locations'].append(location)
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
    
    def update(self, next_value=0.0):
        """PPO 更新策略"""
        if len(self.memory['states']) == 0:
            return 0.0
        
        # 计算回报和优势
        returns, advantages = self.compute_returns(next_value=next_value)
        
        # 转换为 tensor
        states = torch.FloatTensor(np.array(self.memory['states'])).to(self.device)
        actions = torch.LongTensor(self.memory['actions']).to(self.device)
        locations = torch.LongTensor(self.memory['locations']).to(self.device) # 新增
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
            # 【关键修复】前向传播时传入真实的 locations
            # 这样 action head 会看到实际选中的块，而不是三个块的平均
            action_logits, loc_probs, state_values = self.policy(states, sampled_loc_idx=locations)
            
            # 数值稳定性检查（对 logits 和 probs 分别处理）
            loc_probs = torch.clamp(loc_probs, min=1e-8, max=1.0)
            
            loc_prob_sum = loc_probs.sum(dim=-1, keepdim=True)
            
            if (loc_prob_sum == 0).any() or torch.isnan(loc_prob_sum).any():
                logger.warning("[ppo_agent.py:update]: loc_prob_sum is zero or NaN, skipping this epoch")
                continue

            loc_probs = loc_probs / loc_prob_sum  # 重新归一化
            
            # 3. 构建分布（action 使用 logits，location 使用 probs）
            dist_action = Categorical(logits=action_logits)  # 使用 logits
            dist_loc = Categorical(loc_probs)

            # 4. 计算新的联合 Log Prob
            new_log_probs_act = dist_action.log_prob(actions)
            new_log_probs_loc = dist_loc.log_prob(locations)
            new_joint_log_probs = new_log_probs_act + new_log_probs_loc

            # 5. 计算联合熵 (鼓励两个维度都探索)
            entropy = dist_action.entropy().mean() + dist_loc.entropy().mean()
            
            # 6.计算比率（裁剪防止溢出）
            log_ratio = new_joint_log_probs - old_log_probs
            log_ratio = torch.clamp(log_ratio, -20, 20)  # 防止 exp 溢出
            ratio = torch.exp(log_ratio)
            
            # 7. 计算 Actor Loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 减去熵奖励 (Entropy Bonus)
            # 提高探索强度，防止早期收敛导致训练停滞
            actor_loss = actor_loss - 0.05 * entropy
            
            # 8. 更新 Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)  # 保留计算图供 Critic 使用
            torch.nn.utils.clip_grad_norm_(self.policy.context_net.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.policy.candidate_net.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.policy.location_head.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.policy.action_head.parameters(), 0.5)
            self.actor_optimizer.step()
            
            # 9. 计算 Critic Loss 并更新
            state_values_squeezed = state_values.squeeze()

            if state_values_squeezed.dim() == 0:
                state_values_squeezed = state_values_squeezed.unsqueeze(0)
            # 尺寸对齐保护
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
            'locations': [],
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
    action_idx, actual_action, loc_idx_val, log_prob, value = agent.select_action(dummy_state)
    
    print(f"选择的动作索引: {action_idx}")
    print(f"实际变异模式: {actual_action}")
    print(f"位置索引: {loc_idx_val}")
    print(f"对数概率: {log_prob:.4f}")
    print(f"状态价值: {value:.4f}")
    
    print("\n测试通过! ✓")

