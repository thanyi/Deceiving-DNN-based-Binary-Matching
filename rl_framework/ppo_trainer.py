#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPO Trainer for Binary Code Perturbation
PPO 训练器（直接调用环境）
"""
import os
import numpy as np
import torch
from ppo_agent import PPOAgent
import argparse
from loguru import logger
import sys
import shutil
import glob
from collections import deque
from torch.utils.tensorboard import SummaryWriter

# 导入环境
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_wrapper import BinaryPerturbationEnv


def cleanup_intermediate_files(save_path, episode_binaries=None):
    """
    清理训练过程中的中间文件
    """
    if not os.path.exists(save_path):
        return
    
    logger.info(f"开始清理中间文件: {save_path}")
    
    # 提取需要保留的二进制文件路径
    binaries_to_keep = set()
    if episode_binaries:
        for item in episode_binaries:
            if 'binary' in item and item['binary']:
                binaries_to_keep.add(os.path.abspath(item['binary']))
        logger.info(f"将保留 {len(binaries_to_keep)} 个回合的最终二进制文件")
    
    deleted_count = 0
    deleted_size = 0
    
    # 清理 tmp_* 临时目录
    tmp_pattern = os.path.join(save_path, 'tmp_*')
    for tmp_dir in glob.glob(tmp_pattern):
        if os.path.isdir(tmp_dir):
            try:
                size = sum(os.path.getsize(os.path.join(dirpath, filename)) for dirpath, _, filenames in os.walk(tmp_dir) for filename in filenames)
                shutil.rmtree(tmp_dir)
                deleted_count += 1
                deleted_size += size
            except Exception as e:
                logger.warning(f"  无法删除临时目录 {tmp_dir}: {e}")
    
    # 清理 *_container 容器目录
    container_pattern = os.path.join(save_path, '*_container')
    for container_dir in glob.glob(container_pattern):
        if os.path.isdir(container_dir):
            container_dir_abs = os.path.abspath(container_dir)
            should_keep = False
            
            if binaries_to_keep:
                for binary_path in binaries_to_keep:
                    if binary_path.startswith(container_dir_abs + os.sep) or binary_path == container_dir_abs:
                        should_keep = True
                        break
            
            if should_keep: continue
            
            try:
                size = sum(os.path.getsize(os.path.join(dirpath, filename)) for dirpath, _, filenames in os.walk(container_dir) for filename in filenames)
                shutil.rmtree(container_dir)
                deleted_count += 1
                deleted_size += size
            except Exception as e:
                logger.warning(f"  无法删除容器目录 {container_dir}: {e}")
    
    # 清理 rl_output 中的中间文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rl_output_dir = os.path.join(current_dir, 'rl_output')
    if os.path.exists(rl_output_dir):
        mutant_files = glob.glob(os.path.join(rl_output_dir, 'mutant_*.bin*'))
        for file_path in mutant_files:
            try:
                size = os.path.getsize(file_path)
                os.remove(file_path)
                deleted_count += 1
                deleted_size += size
            except Exception as e:
                pass
    
    # 格式化显示
    if deleted_size < 1024: size_str = f"{deleted_size} B"
    elif deleted_size < 1024**2: size_str = f"{deleted_size/1024:.2f} KB"
    else: size_str = f"{deleted_size/1024**2:.2f} MB"
    
    logger.success(f"✓ 清理完成: 删除 {deleted_count} 个项目，释放 {size_str} 空间")


def train_ppo(args):
    """
    PPO 训练主函数
    """
    logger.info("PPO 训练启动 (Multi-Sample Mode)")
    logger.info(f"数据集: {args.dataset}")
    logger.info(f"保存路径: {args.save_path}")
    
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    tensorboard_dir = os.path.join(args.model_dir, 'tensorboard')
    writer = SummaryWriter(log_dir=tensorboard_dir)
    logger.info(f"TensorBoard: {tensorboard_dir}")
    
    # 初始化环境
    env = BinaryPerturbationEnv(
        save_path=args.save_path,
        dataset_path=args.dataset,
        sample_hold_interval=args.sample_hold_interval # Hold-N 策略
        
    )
    env.set_state_dim(args.state_dim)
    
    agent = PPOAgent(
        state_dim=args.state_dim,
        lr=args.lr,
        device='cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'
    )
    
    if args.resume and os.path.exists(args.resume):
        agent.load(args.resume)
    
    log_file = os.path.join(args.model_dir, 'training_log.txt')
    
    episode_binaries = []
    
    # 滑动窗口统计
    success_window = deque(maxlen=50)
    similarity_drop_window = deque(maxlen=50)
    
    # 初始化统计变量
    success_count = 0
    best_score = float('inf')
    info = {}  # 初始化 info，避免作用域问题
   
    global_total_steps = 0 
    try:
        for episode in range(args.episodes):
            logger.info("=" * 60)
            logger.info(f"回合 {episode + 1}/{args.episodes}")
            
            state = env.reset()
            
            episode_actions = [] 
            initial_score = 1.0 # 【优化】默认初始为1.0，防止第一步没取到score导致计算错误
            
            episode_reward = 0
            last_binary_info = None
            should_skip_update = False
            episode_done = False  # 标记 episode 是否正常结束
            
            for step in range(args.max_steps):
                global_total_steps += 1 

                joint_idx, loc_idx, act_idx, actual_action, log_prob, value = agent.select_action(state, explore=True)
                episode_actions.append(actual_action)
                
                # 执行动作
                next_state, reward, done, info = env.step(actual_action, loc_idx)
                
                episode_reward += reward
                state = next_state
                
                
                if step % 10 == 0:
                    # 记录每步指标 (当前设置：每步都记，如果太慢可改为 if step % 5 == 0)
                    writer.add_scalar('Step/Shaped_Reward', reward, global_total_steps)            # Agent 每做一步动作得到的即时反馈（包含进步分、惩罚分等）。
                    writer.add_scalar('Step/Critic_Value', value, global_total_steps)                     # Critic 网络（裁判）认为“当前这个状态，未来能拿多少分”。
                    if 'score' in info:
                        writer.add_scalar('Step/Similarity_Score', info['score'], global_total_steps)     # 每一步变异后的代码与原代码的相似度。

                # 存储经验
                agent.store_transition(state, joint_idx, reward, log_prob, value, done)
 
                if 'binary' in info:
                    last_binary_info = {
                        'episode': episode, 'step': step,
                        'binary': info['binary'], 'score': info.get('score', 1.0),
                        'func': info.get('target_func', 'unknown') # 存一下函数名
                    }
                
                # ✅ 成功检查与错误处理
                if done:
                    if info.get('should_reset', False):
                        logger.warning("⚠️ 错误发生，强制切换目标")
                        should_skip_update = True
                        state = env.reset(force_switch=True)
                    
                    episode_done = True
                    break
            
            # ✅ 统一的统计逻辑
            # 1. 统计成功率和降分 (使用 last_binary_info 更安全)
            final_score = last_binary_info['score'] if last_binary_info else 1.0
            target_func = last_binary_info['func'] if last_binary_info else "unknown"

            is_success = final_score < 0.40
            success_window.append(1 if is_success else 0)
            similarity_drop_window.append(max(0.0, initial_score - final_score))

            if is_success:
                success_count += 1
                logger.success(f"🎉 攻破! 目标: {info.get('target_func')} | 分数: {final_score:.4f}")
                with open(os.path.join(args.save_path, 'success.log'), 'a') as f:
                    f.write(f"Ep {episode}, Func: {info.get('target_func')}, Score: {final_score:.4f}\n")

            if last_binary_info:
                episode_binaries.append(last_binary_info)

            # 2. 如果出错跳过更新
            if should_skip_update:
                agent.clear_memory()
                continue
            
            # 截断回合时做 bootstrap
            next_value = 0.0
            if not episode_done:
                next_value = agent.estimate_value(state)
            
            # PPO 更新
            loss = agent.update(next_value=next_value)

            # 打印动作分布
            agent.log_action_distribution(episode)
            
            # === Episode 级别记录 (核心) ===
            current_success_rate = np.mean(success_window) if success_window else 0.0
            avg_drop = np.mean(similarity_drop_window) if similarity_drop_window else 0.0
            
            logger.info(f"回合总结: 总奖={episode_reward:.2f} | 滑动成功率={current_success_rate:.2f} | 平均降分={avg_drop:.2f}")
            
            writer.add_scalar('Main/Success_Rate_MA50', current_success_rate, episode)      # 最近 50 个回合中，成功绕过检测（分数 < 0.4）的比例。
            writer.add_scalar('Main/Similarity_Drop_MA50', avg_drop, episode)               # 最近 50 个回合中，平均把相似度降低了多少（初始分 1.0 - 最终分）
            writer.add_scalar('Main/Episode_Reward', episode_reward, episode)               # Agent 在一个回合内拿到的所有奖励之和。
            writer.add_scalar('Main/Episode_Length', step + 1, episode)                      # 一个回合内总共执行了多少步。
            writer.add_histogram('Debug/Action_Distribution', np.array(episode_actions), episode)   # 在当前回合中，Agent 选择了哪些变异动作（Action 0-5）。
            writer.add_scalar('Debug/Policy_Loss', loss, episode)                           # PPO 算法更新时的 Loss 值。
            
            # 写日志文件
            with open(log_file, 'a') as f:
                f.write(f"{episode},{step+1},{episode_reward:.4f},{loss:.4f},{current_success_rate:.2f}\n")
            
            # 保存模型
            if (episode + 1) % args.save_interval == 0:
                agent.save(os.path.join(args.model_dir, f'ppo_model_ep{episode+1}.pt'))
            
            if 'score' in info and info['score'] < best_score:
                best_score = info['score']
                agent.save(os.path.join(args.model_dir, 'ppo_model_best.pt'))

            # 定期清理
            if episode % 40 == 0:   
                cleanup_intermediate_files(args.save_path, episode_binaries)
    
    except KeyboardInterrupt:
        logger.warning("训练中断")
    
    finally:
        agent.save(os.path.join(args.model_dir, 'ppo_model_final.pt'))
        writer.close()
        
        # 【优化】取消注释，确保退出时清理垃圾
        cleanup_intermediate_files(args.save_path, episode_binaries)
        
        # 保存清单
        manifest_path = os.path.join(args.model_dir, 'episode_binaries.txt')
        with open(manifest_path, 'w') as f:
            for item in episode_binaries:
                f.write(f"{item['episode']},{item['step']},{item['binary']},{item['score']:.4f}\n")
        logger.info(f"✓ 训练结束，数据已保存")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--save-path', required=True)
    parser.add_argument('--state-dim', type=int, default=256) # 默认256维
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--max-steps', type=int, default=30)
    parser.add_argument('--save-interval', type=int, default=5)
    parser.add_argument('--sample-hold-interval', type=int, default=10)
    parser.add_argument('--model-dir', default='./rl_models')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--use-gpu', action='store_true')
    
    args = parser.parse_args()
    
    # 清理旧日志
    log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'log/uroboro.log')
    if os.path.exists(log_path): os.remove(log_path)
        
    train_ppo(args)