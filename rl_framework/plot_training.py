#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练日志可视化工具
从 training_log.txt 生成训练曲线图
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

def plot_training_curves(log_file, output_dir):
    """
    从训练日志生成可视化图表
    
    Args:
        log_file: 训练日志文件路径 (training_log.txt)
        output_dir: 输出图表保存目录
    """
    # 读取日志数据
    if not os.path.exists(log_file):
        print(f"错误: 日志文件不存在: {log_file}")
        return
    
    print(f"读取训练日志: {log_file}")
    
    episodes = []
    steps = []
    total_rewards = []
    avg_rewards = []
    losses = []
    
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) == 5:
                episodes.append(int(parts[0]))
                steps.append(int(parts[1]))
                total_rewards.append(float(parts[2]))
                avg_rewards.append(float(parts[3]))
                losses.append(float(parts[4]))
    
    if len(episodes) == 0:
        print("警告: 日志文件为空")
        return
    
    print(f"已加载 {len(episodes)} 个回合的数据")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置中文字体（如果可用）
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass
    
    # 1. 总奖励曲线
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, total_rewards, 'b-', alpha=0.3, label='Total Reward')
    
    # 计算移动平均
    window_size = min(10, len(total_rewards) // 5 + 1)
    if len(total_rewards) >= window_size:
        moving_avg = np.convolve(total_rewards, np.ones(window_size)/window_size, mode='valid')
        moving_episodes = episodes[window_size-1:]
        plt.plot(moving_episodes, moving_avg, 'b-', linewidth=2, label=f'Moving Avg (window={window_size})')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'total_reward.png'), dpi=150)
    print(f"✓ 已保存: {os.path.join(output_dir, 'total_reward.png')}")
    plt.close()
    
    # 2. 平均奖励曲线
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, avg_rewards, 'g-', alpha=0.3, label='Average Reward')
    
    if len(avg_rewards) >= window_size:
        moving_avg = np.convolve(avg_rewards, np.ones(window_size)/window_size, mode='valid')
        moving_episodes = episodes[window_size-1:]
        plt.plot(moving_episodes, moving_avg, 'g-', linewidth=2, label=f'Moving Avg (window={window_size})')
    
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Average Reward per Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_reward.png'), dpi=150)
    print(f"✓ 已保存: {os.path.join(output_dir, 'average_reward.png')}")
    plt.close()
    
    # 3. 策略损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, losses, 'r-', alpha=0.3, label='Policy Loss')
    
    if len(losses) >= window_size:
        moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        moving_episodes = episodes[window_size-1:]
        plt.plot(moving_episodes, moving_avg, 'r-', linewidth=2, label=f'Moving Avg (window={window_size})')
    
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Policy Loss per Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'policy_loss.png'), dpi=150)
    print(f"✓ 已保存: {os.path.join(output_dir, 'policy_loss.png')}")
    plt.close()
    
    # 4. 每回合步数
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, steps, 'c-', alpha=0.5, label='Steps per Episode')
    
    if len(steps) >= window_size:
        moving_avg = np.convolve(steps, np.ones(window_size)/window_size, mode='valid')
        moving_episodes = episodes[window_size-1:]
        plt.plot(moving_episodes, moving_avg, 'c-', linewidth=2, label=f'Moving Avg (window={window_size})')
    
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Steps per Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'steps_per_episode.png'), dpi=150)
    print(f"✓ 已保存: {os.path.join(output_dir, 'steps_per_episode.png')}")
    plt.close()
    
    # 5. 综合面板（4 合 1）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 总奖励
    axes[0, 0].plot(episodes, total_rewards, 'b-', alpha=0.3)
    if len(total_rewards) >= window_size:
        moving_avg = np.convolve(total_rewards, np.ones(window_size)/window_size, mode='valid')
        axes[0, 0].plot(episodes[window_size-1:], moving_avg, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Total Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 平均奖励
    axes[0, 1].plot(episodes, avg_rewards, 'g-', alpha=0.3)
    if len(avg_rewards) >= window_size:
        moving_avg = np.convolve(avg_rewards, np.ones(window_size)/window_size, mode='valid')
        axes[0, 1].plot(episodes[window_size-1:], moving_avg, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].set_title('Average Reward')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 策略损失
    axes[1, 0].plot(episodes, losses, 'r-', alpha=0.3)
    if len(losses) >= window_size:
        moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        axes[1, 0].plot(episodes[window_size-1:], moving_avg, 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Policy Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 步数
    axes[1, 1].plot(episodes, steps, 'c-', alpha=0.5)
    if len(steps) >= window_size:
        moving_avg = np.convolve(steps, np.ones(window_size)/window_size, mode='valid')
        axes[1, 1].plot(episodes[window_size-1:], moving_avg, 'c-', linewidth=2)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Steps')
    axes[1, 1].set_title('Steps per Episode')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_summary.png'), dpi=150)
    print(f"✓ 已保存: {os.path.join(output_dir, 'training_summary.png')}")
    plt.close()
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("训练统计信息")
    print("=" * 60)
    print(f"总回合数: {len(episodes)}")
    print(f"总奖励 - 最小: {min(total_rewards):.4f}, 最大: {max(total_rewards):.4f}, 平均: {np.mean(total_rewards):.4f}")
    print(f"平均奖励 - 最小: {min(avg_rewards):.4f}, 最大: {max(avg_rewards):.4f}, 平均: {np.mean(avg_rewards):.4f}")
    print(f"策略损失 - 最小: {min(losses):.4f}, 最大: {max(losses):.4f}, 平均: {np.mean(losses):.4f}")
    print(f"步数 - 最小: {min(steps)}, 最大: {max(steps)}, 平均: {np.mean(steps):.2f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练日志可视化工具')
    parser.add_argument('--log-file', default='../rl_models/training_log.txt', 
                        help='训练日志文件路径')
    parser.add_argument('--output-dir', default='../rl_models/plots', 
                        help='输出图表保存目录')
    
    args = parser.parse_args()
    
    plot_training_curves(args.log_file, args.output_dir)
    print("\n✓ 可视化完成!")

