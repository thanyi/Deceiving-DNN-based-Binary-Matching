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
import shutil
import glob
from torch.utils.tensorboard import SummaryWriter

# 导入环境
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_wrapper import BinaryPerturbationEnv


def cleanup_intermediate_files(save_path, episode_binaries=None):
    """
    清理训练过程中的中间文件
    
    清理策略:
    1. save_path 中:
       - 保留 success.log
       - 保留每个回合最后一个二进制文件所在的 *_container 目录
       - 删除所有 tmp_* 临时目录
       - 删除其他临时文件
    2. rl_output 中:
       - 删除所有 mutant_*.bin* 文件（全部清空）
    
    参数:
        save_path: 保存路径（function_container_* 目录）
        episode_binaries: 每个回合的最后一个二进制文件列表（需要保留）
    """
    if not os.path.exists(save_path):
        return
    
    logger.info(f"开始清理中间文件: {save_path}")
    success_log_path = os.path.join(save_path, 'success.log')
    
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
                # 计算大小
                size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(tmp_dir)
                    for filename in filenames
                )
                shutil.rmtree(tmp_dir)
                deleted_count += 1
                deleted_size += size
                # logger.debug(f"  删除临时目录: {os.path.basename(tmp_dir)}")
            except Exception as e:
                logger.warning(f"  无法删除临时目录 {tmp_dir}: {e}")
    
    # 清理 *_container 容器目录（但保留每个回合的最后一个）
    container_pattern = os.path.join(save_path, '*_container')
    for container_dir in glob.glob(container_pattern):
        if os.path.isdir(container_dir):
            # 检查这个容器目录是否包含需要保留的二进制文件
            # 将容器目录转换为绝对路径进行比较
            container_dir_abs = os.path.abspath(container_dir)
            should_keep = False
            
            if binaries_to_keep:
                for binary_path in binaries_to_keep:
                    # 两边都是绝对路径，检查二进制文件是否在这个容器目录下
                    if binary_path.startswith(container_dir_abs + os.sep) or binary_path == container_dir_abs:
                        # logger.debug(f"  保留容器目录（包含回合最终文件）: {os.path.basename(container_dir)}")
                        # logger.debug(f"    -> 包含文件: {os.path.basename(binary_path)}")
                        should_keep = True
                        break
            
            if should_keep:
                continue
            
            try:
                # 计算大小
                size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(container_dir)
                    for filename in filenames
                )
                shutil.rmtree(container_dir)
                deleted_count += 1
                deleted_size += size
                logger.debug(f"  删除容器目录: {os.path.basename(container_dir)}")
            except Exception as e:
                logger.warning(f"  无法删除容器目录 {container_dir}: {e}")
    
    # 清理其他临时文件（除了 success.log）
    for item in os.listdir(save_path):
        item_path = os.path.join(save_path, item)
        
        # 跳过 success.log
        if item == 'success.log':
            continue
        
        # 跳过目录（已处理）
        if os.path.isdir(item_path):
            continue
        
        # 删除其他文件
        try:
            size = os.path.getsize(item_path)
            os.remove(item_path)
            deleted_count += 1
            deleted_size += size
            logger.debug(f"  删除文件: {item}")
        except Exception as e:
            logger.warning(f"  无法删除文件 {item}: {e}")
    
    # 清理 rl_output 中的所有中间 mutant 文件（全部删除）
    # rl_output 路径：从当前文件所在目录推导
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rl_output_dir = os.path.join(current_dir, 'rl_output')
    
    if os.path.exists(rl_output_dir):
        mutant_pattern = os.path.join(rl_output_dir, 'mutant_*.bin*')
        mutant_files = glob.glob(mutant_pattern)
        
        if mutant_files:
            logger.info(f"清理 rl_output 中间文件: {rl_output_dir} ({len(mutant_files)} 个文件)")
            for file_path in mutant_files:
                try:
                    size = os.path.getsize(file_path)
                    os.remove(file_path)
                    deleted_count += 1
                    deleted_size += size
                    # logger.debug(f"  删除 rl_output 文件: {os.path.basename(file_path)}")
                except Exception as e:
                    logger.warning(f"  无法删除 rl_output 文件 {file_path}: {e}")
            logger.info(f"✓ 已删除 rl_output 中的所有 mutant 文件 ({len(mutant_files)} 个)")
        else:
            logger.debug(f"  rl_output 中没有 mutant 文件")
    
    # 格式化文件大小
    if deleted_size < 1024:
        size_str = f"{deleted_size} B"
    elif deleted_size < 1024 * 1024:
        size_str = f"{deleted_size / 1024:.2f} KB"
    else:
        size_str = f"{deleted_size / (1024 * 1024):.2f} MB"
    
    logger.success(f"✓ 清理完成: 删除 {deleted_count} 个项目，释放 {size_str} 空间")
    logger.info(f"✓ 已保留: success.log")
    if binaries_to_keep:
        logger.info(f"✓ 已保留: {len(binaries_to_keep)} 个回合的最终二进制文件")


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
    
    # 初始化 TensorBoard
    tensorboard_dir = os.path.join(args.model_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    logger.info(f"TensorBoard 日志目录: {tensorboard_dir}")
    logger.info(f"启动 TensorBoard: tensorboard --logdir={tensorboard_dir}")
    
    # 初始化环境（直接创建，无需进程通信）
    logger.info("初始化变异环境...")
    env = BinaryPerturbationEnv(
        original_binary=args.binary,
        function_name=args.function,
        save_path=args.save_path
    )
    # 设置状态维度，与环境保持一致
    env.set_state_dim(args.state_dim)
    logger.info("环境初始化完成 ✓")
    
    # 初始化 PPO Agent（使用改进的网络结构）
    agent = PPOAgent(
        state_dim=args.state_dim,
        n_actions=6,
        lr=args.lr,  # 默认已改为 1e-4
        gamma=args.gamma,
        epsilon=args.epsilon,
        device='cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'
    )
    
    logger.info(f"网络结构: Actor-Critic 分离架构")
    logger.info(f"Actor 学习率: {args.lr:.2e}, Critic 学习率: {args.lr * 2:.2e}")
    
    # 如果存在预训练模型，则加载
    if args.resume and os.path.exists(args.resume):
        agent.load(args.resume)
    
    # 奖励塑形器
    reward_shaper = RewardShaper(target_score=0.40)
    
    # 训练日志
    log_file = os.path.join(args.model_dir, 'training_log.txt')
    if os.path.exists(log_file):
        os.remove(log_file) # 删除旧的日志文件
    best_score = float('inf')
    success_count = 0
    
    # 保存每个回合的最后一个二进制文件路径
    episode_binaries = []  # 存储每个回合的最后一个二进制文件
    
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
            global_step = episode * args.max_steps  # 全局步数计数器
            last_binary_info = None  # 追踪本回合最后一个二进制
            
            for step in range(args.max_steps):
                # input(f"In step {step + 1}, Press Enter to continue...")
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
                
                # 记录每步指标到 TensorBoard
                current_step = global_step + step
                writer.add_scalar('Step/Raw_Reward', reward, current_step)
                writer.add_scalar('Step/Shaped_Reward', shaped_reward, current_step)
                writer.add_scalar('Step/Value', value, current_step)
                writer.add_scalar('Step/Action', actual_action, current_step)
                if 'score' in info:
                    writer.add_scalar('Step/Similarity_Score', info['score'], current_step)
                if 'grad' in info:
                    writer.add_scalar('Step/Gradient', info['grad'], current_step)
                
                # 存储经验
                agent.store_transition(state, action_idx, shaped_reward, log_prob, value)
                
                episode_reward += shaped_reward
                state = next_state
                
                # 更新本回合最后一个二进制信息
                if 'binary' in info:
                    last_binary_info = {
                        'episode': episode,
                        'step': step,
                        'binary': info['binary'],
                        'score': info.get('score', 1.0)
                    }
                
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
            
            # 保存本回合的最后一个二进制文件路径
            if last_binary_info is not None:
                episode_binaries.append(last_binary_info)
                logger.debug(f"✓ 记录回合 {episode} 的最终二进制: {os.path.basename(last_binary_info['binary'])} (分数: {last_binary_info['score']:.4f})")
            
            # PPO 更新
            loss = agent.update()
            episode_loss = loss
            
            # 记录训练信息
            avg_reward = episode_reward / (step + 1)
            
            logger.info(f"回合总结: 总奖励={episode_reward:.4f} | 平均奖励={avg_reward:.4f} | 策略损失={loss:.4f} | 成功次数={success_count}")
            
            # 记录回合级别指标到 TensorBoard
            writer.add_scalar('Episode/Total_Reward', episode_reward, episode)
            writer.add_scalar('Episode/Average_Reward', avg_reward, episode)
            writer.add_scalar('Episode/Policy_Loss', loss, episode)
            writer.add_scalar('Episode/Steps', step + 1, episode)
            writer.add_scalar('Episode/Success_Count', success_count, episode)
            if 'score' in info:
                writer.add_scalar('Episode/Final_Score', info['score'], episode)
                writer.add_scalar('Episode/Best_Score', best_score, episode)
            
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
        
        # 关闭 TensorBoard writer
        writer.close()
        
        logger.info("=" * 80)
        logger.success("训练完成")
        logger.info(f"成功绕过次数: {success_count} | 最佳分数: {best_score:.4f}")
        logger.info(f"TensorBoard 日志: {tensorboard_dir}")
        logger.info("=" * 80)
        
        # 清理中间文件（保留 success.log 和每个回合的最终二进制）
        logger.info("")
        cleanup_intermediate_files(args.save_path, episode_binaries)
        
        # 保存回合二进制文件清单
        manifest_path = os.path.join(args.model_dir, 'episode_binaries.txt')
        with open(manifest_path, 'w') as f:
            f.write("# 每个回合的最终二进制文件（每个回合的最后一步生成的二进制）\n")
            f.write("# 格式: episode, step, binary_path, score\n")
            for item in episode_binaries:
                f.write(f"{item['episode']},{item['step']},{item['binary']},{item['score']:.4f}\n")
        logger.info(f"✓ 回合二进制清单已保存: {manifest_path} ({len(episode_binaries)} 个文件)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO Trainer for Binary Perturbation')
    
    # 环境参数
    parser.add_argument('--binary', required=True, help='原始二进制文件路径')
    parser.add_argument('--function', required=True, help='目标函数名')
    parser.add_argument('--save-path', required=True, help='变异结果保存路径')
    
    # PPO 参数
    parser.add_argument('--state-dim', type=int, default=64, help='状态维度（推荐 64）')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率（平衡学习速度和稳定性）')
    parser.add_argument('--gamma', type=float, default=0.95, help='折扣因子（降低以减少未来奖励影响）')
    parser.add_argument('--epsilon', type=float, default=0.2, help='PPO 裁剪参数')
    
    # 训练参数
    parser.add_argument('--episodes', type=int, default=50, help='训练回合数（减少但更稳定）')
    parser.add_argument('--max-steps', type=int, default=50, help='每回合最大步数（减少以加快迭代）')
    parser.add_argument('--save-interval', type=int, default=10, help='保存间隔')
    parser.add_argument('--model-dir', default='./rl_models', help='模型保存目录')
    parser.add_argument('--resume', default=None, help='恢复训练的模型路径')
    parser.add_argument('--use-gpu', action='store_true', help='使用 GPU')
    
    args = parser.parse_args()
    os.remove(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'log/uroboro.log'))
    train_ppo(args)

