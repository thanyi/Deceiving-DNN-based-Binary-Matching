#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPO Trainer for Binary Code Perturbation
PPO 训练器（直接调用环境）
"""

import os
import numpy as np
import torch
from ppo_agent import PPOAgent, RewardShaper
import argparse
from loguru import logger
import sys

# 导入环境
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_wrapper import BinaryPerturbationEnv


def train_ppo(args):
    """
    PPO 训练主函数
    """
    logger.info("PPO 训练启动")
    logger.info(f"原始二进制: {args.binary}")
    logger.info(f"目标函数: {args.function}")
    logger.info(f"保存路径: {args.save_path}")
    logger.info(f"最大回合数: {args.episodes}")
    logger.info(f"最大步数/回合: {args.max_steps}")
    
    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # 初始化环境（直接创建，无需进程通信）
    logger.info("初始化变异环境...")
    env = BinaryPerturbationEnv(
        original_binary=args.binary,
        function_name=args.function,
        save_path=args.save_path
    )
    logger.info("环境初始化完成 ✓")
    
    # 初始化 PPO Agent
    agent = PPOAgent(
        state_dim=args.state_dim,
        n_actions=8,
        lr=args.lr,
        gamma=args.gamma,
        epsilon=args.epsilon,
        device='cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'
    )
    
    # 如果存在预训练模型，则加载
    if args.resume and os.path.exists(args.resume):
        agent.load(args.resume)
    
    # 奖励塑形器
    reward_shaper = RewardShaper(target_score=0.40)
    
    # 训练日志
    log_file = os.path.join(args.model_dir, 'training_log.txt')
    best_score = float('inf')
    success_count = 0
    
    try:
        for episode in range(args.episodes):
            logger.info("=" * 80)
            logger.info(f"回合 {episode + 1}/{args.episodes}")
            logger.info("=" * 80)
            
            # 重置环境
            state = env.reset()
            reward_shaper.reset()
            
            episode_reward = 0
            episode_loss = 0
            
            for step in range(args.max_steps):
                # 选择动作
                action_idx, actual_action, log_prob, value = agent.select_action(state, explore=True)
                
                logger.debug(f"步骤 {step + 1}: 动作={actual_action} (索引={action_idx})")
                
                # 执行动作
                next_state, reward, done, info = env.step(actual_action)
                
                # 奖励塑形
                if 'score' in info:
                    shaped_reward = reward_shaper.compute_reward(
                        info['score'], 
                        info.get('grad', 0), 
                        done, 
                        step
                    )
                else:
                    shaped_reward = reward  # 如果没有 score，直接使用原始奖励
                
                if 'score' in info:
                    logger.info(f"  奖励: {reward:.4f} → {shaped_reward:.4f} | 分数: {info['score']:.4f} | 梯度: {info.get('grad', 0):.4f}")
                else:
                    logger.info(f"  奖励: {reward:.4f}")
                
                # 存储经验
                agent.store_transition(state, action_idx, shaped_reward, log_prob, value)
                
                episode_reward += shaped_reward
                state = next_state
                
                # 检查成功
                if done:
                    if 'score' in info and info['score'] < 0.40:
                        success_count += 1
                        logger.success(f"🎉 成功绕过检测! 分数: {info['score']:.4f}")
                        
                        # 保存成功样本信息
                        success_log = os.path.join(args.save_path, 'success.log')
                        with open(success_log, 'a') as f:
                            f.write(f"Episode {episode}, Step {step}, Score: {info['score']:.4f}\n")
                            f.write(f"Binary: {info.get('binary', 'unknown')}\n\n")
                    
                    logger.info(f"回合结束 (步数: {step + 1})")
                    break
            
            # PPO 更新
            loss = agent.update()
            episode_loss = loss
            
            # 记录训练信息
            avg_reward = episode_reward / (step + 1)
            
            logger.info(f"回合总结: 总奖励={episode_reward:.4f} | 平均奖励={avg_reward:.4f} | 策略损失={loss:.4f} | 成功次数={success_count}")
            
            # 保存到日志
            with open(log_file, 'a') as f:
                f.write(f"{episode},{step+1},{episode_reward:.4f},{avg_reward:.4f},{loss:.4f}\n")
            
            # 定期保存模型
            if (episode + 1) % args.save_interval == 0:
                model_path = os.path.join(args.model_dir, f'ppo_model_ep{episode+1}.pt')
                agent.save(model_path)
            
            # 保存最佳模型
            if 'score' in info and info['score'] < best_score:
                best_score = info['score']
                best_model_path = os.path.join(args.model_dir, 'ppo_model_best.pt')
                agent.save(best_model_path)
                logger.success(f"💾 保存最佳模型 (分数: {best_score:.4f})")
    
    except KeyboardInterrupt:
        logger.warning("训练被用户中断")
    
    finally:
        # 保存最终模型
        final_model_path = os.path.join(args.model_dir, 'ppo_model_final.pt')
        agent.save(final_model_path)
        
        logger.info("=" * 80)
        logger.success("训练完成")
        logger.info(f"成功绕过次数: {success_count} | 最佳分数: {best_score:.4f}")
        logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO Trainer for Binary Perturbation')
    
    # 环境参数
    parser.add_argument('--binary', required=True, help='原始二进制文件路径')
    parser.add_argument('--function', required=True, help='目标函数名')
    parser.add_argument('--save-path', required=True, help='变异结果保存路径')
    
    # PPO 参数
    parser.add_argument('--state-dim', type=int, default=128, help='状态维度')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--epsilon', type=float, default=0.2, help='PPO 裁剪参数')
    
    # 训练参数
    parser.add_argument('--episodes', type=int, default=100, help='训练回合数')
    parser.add_argument('--max-steps', type=int, default=50, help='每回合最大步数')
    parser.add_argument('--save-interval', type=int, default=10, help='保存间隔')
    parser.add_argument('--model-dir', default='./rl_models', help='模型保存目录')
    parser.add_argument('--resume', default=None, help='恢复训练的模型路径')
    parser.add_argument('--use-gpu', action='store_true', help='使用 GPU')
    
    args = parser.parse_args()
    
    train_ppo(args)

