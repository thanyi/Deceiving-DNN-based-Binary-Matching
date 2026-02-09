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

class StructuredJointNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, n_locs, hidden_dim=512):
        super(StructuredJointNetwork, self).__init__()
        
        # === 1. 特征切片定义 (根据 env_wrapper.py) ===
        # Part 1 (16): History
        # Part 2 (40): Topology
        # Part 3 (128): Critical Semantics (Top-3 * 32 + Context)
        # Part 4 (72): Global Semantics
        
        self.block_feat_start = 56
        self.block_feat_end = 152
        self.block_dim = 32
        self.num_blocks = 3

        # === 编码器 ===
        # 块特征编码器（共享权重）
        self.block_encoder = nn.Sequential(
            nn.Linear(self.block_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # 全局特征编码器
        self.global_input_dim = state_dim - (self.block_dim * self.num_blocks)
        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # === 联合决策头 ===
        self.fusion_dim = 256 + (64 * self.num_blocks)

        # ✅ 不加 Softmax，输出 logits
        self.actor_head = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, n_locs * n_actions)
        )
        
        # Critic：使用与 actor 相同的融合表征，增加一层表达能力
        self.critic = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
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

        # 1. 提取特征
        blocks_raw = state[:, self.block_feat_start:self.block_feat_end]
        blocks_view = blocks_raw.view(batch_size, self.num_blocks, self.block_dim)

        global_raw = torch.cat([
            state[:, :self.block_feat_start], 
            state[:, self.block_feat_end:]
        ], dim=1)
        
        # 2. 编码
        global_emb = self.global_encoder(global_raw)
        
        block_embs = []
        for i in range(self.num_blocks):
            b_emb = self.block_encoder(blocks_view[:, i, :])
            block_embs.append(b_emb)
        
        blocks_concat = torch.cat(block_embs, dim=1)
        
        # 3. 融合与输出
        fusion = torch.cat([global_emb, blocks_concat], dim=1)
        action_logits = self.actor_head(fusion)  # ✅ logits
        state_value = self.critic(fusion)
        
        return action_logits, state_value


class PPOAgent:
    """PPO 智能体"""
    
    def __init__(self, state_dim=256, n_actions=None, n_locs=3, lr=1e-4, gamma=0.99, 
                 epsilon=0.2, epochs=10, device='cpu', action_map=None):
        """
        参数:
            state_dim: 状态维度（特征向量长度）
            n_actions: 动作数量（默认使用 action_map 的长度）
            n_locs: 位置数量（默认3：Top-3）
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
        
        # 动作映射：索引 -> 实际变异模式（与 env_wrapper 保持一致）
        default_action_map = [1, 2, 4, 7, 8, 9, 11, 13, 14, 15, 16]
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
        self.policy = StructuredJointNetwork(state_dim, self.n_actions, self.n_locs).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.action_stats = np.zeros((self.n_locs, self.n_actions))
        
        # 经验缓冲
        self.memory = {
            'states': [],
            'joint_actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': [],
            'loc_masks': []
        }
    
    def select_action(self, state, explore=True, loc_mask=None):
        """
       
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_logits, state_value = self.policy(state)
        # 位置掩码（仅在采样时使用）
        action_logits = self._apply_loc_mask(action_logits, loc_mask)
        
        dist = Categorical(logits=action_logits)
        
        if explore:
            joint_action = dist.sample()
        else:
            joint_action = torch.argmax(action_logits, dim=-1)
        
        log_prob = dist.log_prob(joint_action)
        joint_idx = joint_action.item()
        
        # 解码
        loc_idx = joint_idx // self.n_actions
        act_idx = joint_idx % self.n_actions
        actual_action = self.action_map[act_idx]
        
        # 统计
        if explore:
            self.action_stats[loc_idx, act_idx] += 1
        
        return joint_idx, loc_idx, act_idx, actual_action, log_prob.item(), state_value.item()

    def estimate_value(self, state):
        """
        估计单个状态的价值（用于截断回合的 bootstrap）
        
        参数:
            state: 当前状态特征向量
        
        返回:
            value: 状态价值估计
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, state_value = self.policy(state)  # 只需要 value，不需要 logits
        return state_value.item()
    
    def store_transition(self, state, joint_action, reward, log_prob, value, done, loc_mask=None):
        self.memory['states'].append(state)
        self.memory['joint_actions'].append(joint_action)
        self.memory['rewards'].append(reward)
        self.memory['log_probs'].append(log_prob)
        self.memory['values'].append(value)
        self.memory['dones'].append(1.0 if done else 0.0)
        if loc_mask is None:
            loc_mask = [1] * self.n_locs
        self.memory['loc_masks'].append(loc_mask)
    
    def compute_returns(self, next_value=0):
        """计算回报（使用 GAE - Generalized Advantage Estimation）"""
        returns = []
        advantages = []
        gae = 0
        
        rewards = self.memory['rewards']
        values = self.memory['values'] + [next_value]
        dones = self.memory['dones']

        # 反向计算 GAE
        for i in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[i])
            delta = rewards[i] + self.gamma * mask * values[i + 1] - values[i]
            gae = delta + self.gamma * 0.95 * mask * gae  # lambda=0.95
            
            # 检查计算结果
            if np.isnan(gae) or np.isinf(gae):
                logger.error(f"[ppo_agent.py:compute_returns]: NaN/Inf in gae at step {i}")
                logger.error(f"  delta={delta}, gae={gae}, rewards[{i}]={rewards[i]}, values[{i}]={values[i]}, values[{i+1}]={values[i+1]}")
                # 使用 0 作为后备
                gae = 0.0
            
            returns.insert(0, gae + values[i])
            advantages.insert(0, gae)
        
        # 先记录未归一化的优势统计，用于诊断是否在“真学习”
        raw_adv = np.array(advantages, dtype=np.float32) if advantages else np.zeros(1, dtype=np.float32)
        adv_stats = {
            'adv_mean_raw': float(raw_adv.mean()),
            'adv_std_raw': float(raw_adv.std()),
            'adv_abs_mean_raw': float(np.abs(raw_adv).mean()),
            'adv_max_abs_raw': float(np.abs(raw_adv).max())
        }

        # 在这里归一化 Advantage（用于训练稳定性）
        advantages = torch.FloatTensor(advantages).to(self.device)
        # ✅ 归一化前检查方差
        if len(advantages) > 1:
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
            else:
                logger.warning("Advantages方差为0，跳过归一化")
        return returns, advantages, adv_stats
    
    def update(self, next_value=0.0):
        """PPO 更新策略"""
        if len(self.memory['states']) == 0:
            return {
                'loss': 0.0,
                'adv_mean_raw': 0.0,
                'adv_std_raw': 0.0,
                'adv_abs_mean_raw': 0.0,
                'adv_max_abs_raw': 0.0
            }
        
        # 计算回报和优势
        returns, advantages, adv_stats = self.compute_returns(next_value=next_value)
        
        # 转换为 tensor
        states = torch.FloatTensor(np.array(self.memory['states'])).to(self.device)
        joint_actions = torch.LongTensor(self.memory['joint_actions']).to(self.device)
        old_log_probs = torch.FloatTensor(self.memory['log_probs']).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        total_loss_val = 0
        
        for epoch in range(self.epochs):
            action_logits, state_values = self.policy(states)
            
            # ✅ 简单裁剪，不归一化
            action_logits = torch.clamp(action_logits, -20, 20)
            # ✅ 位置掩码：屏蔽无效位置
            if self.memory.get('loc_masks'):
                loc_masks = self.memory['loc_masks']
                loc_masks = torch.FloatTensor(np.array(loc_masks)).to(self.device)
                action_logits = self._apply_loc_mask(action_logits, loc_masks)
            
            # ✅ NaN 检查
            if torch.isnan(action_logits).any():
                logger.error(f"Epoch {epoch}: NaN in logits!")
                self.clear_memory()
                return {
                    'loss': 0.0,
                    **adv_stats
                }
            
            # ✅ 使用 logits
            dist = Categorical(logits=action_logits)
            new_log_probs = dist.log_prob(joint_actions)
            entropy = dist.entropy().mean()

           # Actor Loss (PPO Clip)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            
            # Stronger entropy bonus helps prevent early collapse to one action.
            actor_loss = -torch.min(surr1, surr2).mean() - 0.05 * entropy
            
            # Critic Loss
            state_values_sq = state_values.squeeze()
            if state_values_sq.dim() == 0:
                state_values_sq = state_values_sq.unsqueeze(0)

            min_len = min(state_values_sq.shape[0], returns.shape[0])
            critic_loss = nn.functional.smooth_l1_loss(
                state_values_sq[:min_len], 
                returns[:min_len]
            )

            # 总 Loss
            loss = actor_loss + 0.5 * critic_loss
            
            # ✅ 梯度检查
            if torch.isnan(loss):
                logger.error(f"Epoch {epoch}: NaN in loss!")
                self.clear_memory()
                return {
                    'loss': 0.0,
                    **adv_stats
                }
            

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss_val += loss.item()

        self.clear_memory()
        return {
            'loss': total_loss_val / self.epochs,
            **adv_stats
        }

    def log_action_distribution(self, episode):
        """
        诊断工具：分析动作分布
        用于检测策略是否退化成均匀分布
        输出写入 log/action_distribution.log 文件
        """
        if episode % 50 != 0 or episode == 0:
            return
        
        total = self.action_stats.sum()
        if total < 10:
            return
        
        # 确定日志文件路径
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'log')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'action_distribution.log')
        
        # 写入文件（追加模式）
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"📊 动作分布分析 (Episode {episode})\n")
            f.write("=" * 60 + "\n")
            
            # 计算熵
            probs = self.action_stats.flatten() / total
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log(probs))
            max_entropy = np.log(self.n_locs * self.n_actions)
            
            f.write(f"策略熵: {entropy:.3f} / {max_entropy:.3f} ({entropy/max_entropy:.1%})\n")
            
            # Top-5 组合
            flat_indices = np.argsort(self.action_stats.flatten())[::-1][:5]
            f.write("\n🏆 Top-5 最常用组合:\n")
            for rank, flat_idx in enumerate(flat_indices, 1):
                loc_idx = flat_idx // self.n_actions
                act_idx = flat_idx % self.n_actions
                count = self.action_stats.flatten()[flat_idx]
                ratio = count / total
                f.write(
                    f"  #{rank}: 位置{loc_idx} × 动作{act_idx} "
                    f"(实际动作={self.action_map[act_idx]}) | {ratio:.2%}\n"
                )
            
            # 位置偏好
            loc_dist = self.action_stats.sum(axis=1) / total
            f.write(f"\n📍 位置选择分布: {loc_dist}\n")
            
            # 动作偏好
            act_dist = self.action_stats.sum(axis=0) / total
            f.write(f"⚡ 动作选择分布: {act_dist}\n")
            
            # 警告
            if entropy > max_entropy * 0.95:
                f.write("⚠️ 熵过高！策略接近随机选择（可能未收敛）\n")
            elif entropy < max_entropy * 0.2:
                f.write("⚠️ 熵过低！策略可能过早收敛到次优解\n")
            else:
                f.write("✅ 策略熵正常，探索与利用平衡良好\n")
            
            f.write("=" * 60 + "\n\n")
        
        # 重置
        self.action_stats.fill(0)

    def clear_memory(self):
        """清空经验缓冲"""
        self.memory = {
            'states': [],
            'joint_actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': [],
            'loc_masks': []
        }

    def _apply_loc_mask(self, action_logits, loc_mask):
        """
        将位置掩码应用到 joint action logits 上。
        loc_mask: shape [B, n_locs] 或 [n_locs]
        """
        if loc_mask is None:
            return action_logits

        if not torch.is_tensor(loc_mask):
            loc_mask = torch.tensor(loc_mask, dtype=action_logits.dtype, device=action_logits.device)
        if loc_mask.dim() == 1:
            loc_mask = loc_mask.unsqueeze(0)
        if loc_mask.size(-1) != self.n_locs:
            logger.warning(
                f"[ppo_agent.py:_apply_loc_mask] loc_mask size {loc_mask.size(-1)} "
                f"!= n_locs {self.n_locs}, skip mask"
            )
            return action_logits

        # 防止全 0 掩码导致无法采样：全 0 行改为全 1
        mask_sum = loc_mask.sum(dim=1, keepdim=True)
        loc_mask = torch.where(mask_sum > 0, loc_mask, torch.ones_like(loc_mask))
        joint_mask = loc_mask.repeat_interleave(self.n_actions, dim=1)
        return action_logits.masked_fill(joint_mask <= 0, -1e9)
    
    def save(self, path, extra_state=None):
        """保存模型（完整版）"""
        payload = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'action_stats': self.action_stats,
            'hyperparams': {
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epochs': self.epochs
            }
        }
        if extra_state:
            payload['trainer_state'] = extra_state
        torch.save(payload, path)
        logger.info(f"✅ 模型已保存: {path}")
    
    def load(self, path):
        """加载模型（兼容旧版本）"""
        if not os.path.exists(path):
            logger.warning(f"❌ 模型文件不存在: {path}")
            return
        
        checkpoint = torch.load(path, map_location=self.device)

        # 加载网络权重（兼容动作空间变更导致的 head 尺寸不一致）
        ckpt_state = checkpoint.get('policy_state_dict', {})
        model_state = self.policy.state_dict()
        compatible = {}
        skipped = []
        for k, v in ckpt_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                compatible[k] = v
            else:
                skipped.append(k)

        model_state.update(compatible)
        self.policy.load_state_dict(model_state, strict=False)
        if skipped:
            logger.warning(
                f"⚠️ 检测到结构不兼容参数，已跳过加载 {len(skipped)} 项（常见于动作数变化）"
            )
        
        # 加载优化器状态（可选，结构变化时可能失败）
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                logger.warning(f"⚠️ 优化器状态不兼容，跳过加载: {e}")
        
        # 加载统计信息（可选）
        if 'action_stats' in checkpoint:
            try:
                stats = checkpoint['action_stats']
                if isinstance(stats, np.ndarray) and stats.shape == self.action_stats.shape:
                    self.action_stats = stats
            except Exception:
                pass
        

if __name__ == "__main__":
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'log')
    print(log_dir)
