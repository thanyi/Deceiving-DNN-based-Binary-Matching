#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用示例：如何在代码中使用训练好的PPO模型

这个脚本展示了如何在你的Python程序中集成训练好的PPO模型
"""

import os
import sys
import torch
import numpy as np

# 导入环境和Agent
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_wrapper import BinaryPerturbationEnv
from ppo_agent import PPOAgent


def load_model(model_path, state_dim=64, device='cpu'):
    """
    加载训练好的PPO模型
    
    参数:
        model_path: 模型文件路径
        state_dim: 状态维度（必须与训练时一致）
        device: 'cpu' 或 'cuda'
    
    返回:
        agent: 加载好的PPOAgent对象
    """
    agent = PPOAgent(
        state_dim=state_dim,
        n_actions=6,
        device=device
    )
    agent.load(model_path)
    agent.policy.eval()  # 设置为评估模式
    return agent


def mutate_binary(agent, env, max_steps=30, target_score=0.40):
    """
    使用PPO模型对二进制文件进行变异
    
    参数:
        agent: 训练好的PPOAgent
        env: BinaryPerturbationEnv环境
        max_steps: 最大步数
        target_score: 目标相似度分数
    
    返回:
        success: 是否成功
        best_score: 最佳分数
        best_binary: 最佳变异结果路径
        steps: 执行的步数
    """
    state = env.reset()
    best_score = float('inf')
    best_binary = None
    
    for step in range(max_steps):
        # 选择动作（不探索，选择最优）
        action_idx, actual_action, log_prob, value = agent.select_action(state, explore=False)
        
        # 执行动作
        next_state, reward, done, info = env.step(actual_action)
        
        # 更新最佳结果
        if 'score' in info and info['score'] < best_score:
            best_score = info['score']
            best_binary = info.get('binary', None)
        
        # 检查是否达到目标
        if 'score' in info and info['score'] < target_score:
            return True, best_score, best_binary, step + 1
        
        state = next_state
        
        if done:
            break
    
    return False, best_score, best_binary, step + 1


def example_single_inference():
    """
    示例1: 单次推理
    """
    print("=" * 80)
    print("示例1: 单次推理")
    print("=" * 80)
    
    # 1. 加载模型
    model_path = "rl_models/ppo_model_best.pt"
    print(f"加载模型: {model_path}")
    agent = load_model(model_path, state_dim=64, device='cpu')
    print("✓ 模型加载成功")
    
    # 2. 初始化环境
    binary_path = "workdir_1/ls"
    function_name = "usage"
    save_path = "example_output"
    
    print(f"\n初始化环境:")
    print(f"  二进制: {binary_path}")
    print(f"  函数: {function_name}")
    
    env = BinaryPerturbationEnv(
        original_binary=binary_path,
        function_name=function_name,
        save_path=save_path
    )
    env.set_state_dim(64)
    print("✓ 环境初始化成功")
    
    # 3. 执行变异
    print("\n开始变异...")
    success, best_score, best_binary, steps = mutate_binary(
        agent, env, max_steps=30, target_score=0.40
    )
    
    # 4. 输出结果
    print("\n结果:")
    print(f"  成功: {success}")
    print(f"  最佳分数: {best_score:.4f}")
    print(f"  执行步数: {steps}")
    print(f"  变异结果: {best_binary}")
    
    return success, best_score


def example_batch_inference():
    """
    示例2: 批量推理
    """
    print("\n" + "=" * 80)
    print("示例2: 批量推理")
    print("=" * 80)
    
    # 1. 加载模型（复用同一个模型）
    model_path = "rl_models/ppo_model_best.pt"
    print(f"加载模型: {model_path}")
    agent = load_model(model_path, state_dim=64, device='cpu')
    print("✓ 模型加载成功")
    
    # 2. 批量任务配置
    tasks = [
        {"binary": "workdir_1/ls", "function": "usage", "output": "batch_ls"},
        {"binary": "workdir_1/pwd", "function": "usage", "output": "batch_pwd"},
        {"binary": "workdir_1/cat", "function": "main", "output": "batch_cat"},
    ]
    
    # 3. 逐个执行
    results = []
    for i, task in enumerate(tasks):
        print(f"\n任务 {i+1}/{len(tasks)}: {task['binary']} - {task['function']}")
        
        try:
            # 初始化环境
            env = BinaryPerturbationEnv(
                original_binary=task['binary'],
                function_name=task['function'],
                save_path=task['output']
            )
            env.set_state_dim(64)
            
            # 执行变异
            success, best_score, best_binary, steps = mutate_binary(
                agent, env, max_steps=30, target_score=0.40
            )
            
            results.append({
                'task': task,
                'success': success,
                'score': best_score,
                'steps': steps,
                'binary': best_binary
            })
            
            print(f"  ✓ 完成: 分数={best_score:.4f}, 步数={steps}")
            
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            results.append({
                'task': task,
                'success': False,
                'score': float('inf'),
                'steps': 0,
                'binary': None
            })
    
    # 4. 汇总结果
    print("\n批量结果汇总:")
    success_count = sum(1 for r in results if r['success'])
    print(f"  总任务数: {len(results)}")
    print(f"  成功数: {success_count}")
    print(f"  成功率: {success_count / len(results) * 100:.1f}%")
    
    return results


def example_custom_usage():
    """
    示例3: 自定义使用场景
    """
    print("\n" + "=" * 80)
    print("示例3: 自定义使用 - 早停策略")
    print("=" * 80)
    
    # 加载模型
    agent = load_model("rl_models/ppo_model_best.pt", state_dim=64)
    
    # 初始化环境
    env = BinaryPerturbationEnv(
        original_binary="workdir_1/ls",
        function_name="usage",
        save_path="custom_output"
    )
    env.set_state_dim(64)
    
    # 自定义推理逻辑：如果3步内没有改善就停止
    state = env.reset()
    best_score = float('inf')
    no_improvement_steps = 0
    
    for step in range(30):
        # 选择动作
        action_idx, actual_action, log_prob, value = agent.select_action(state, explore=False)
        
        # 执行动作
        next_state, reward, done, info = env.step(actual_action)
        
        # 检查改善
        if 'score' in info:
            if info['score'] < best_score:
                best_score = info['score']
                no_improvement_steps = 0
                print(f"步骤 {step+1}: 新的最佳分数 {best_score:.4f}")
            else:
                no_improvement_steps += 1
        
        # 早停策略
        if no_improvement_steps >= 3:
            print(f"3步内没有改善，提前停止 (步数: {step+1})")
            break
        
        # 达到目标
        if 'score' in info and info['score'] < 0.40:
            print(f"达到目标! (步数: {step+1})")
            break
        
        state = next_state
        
        if done:
            break
    
    print(f"最终最佳分数: {best_score:.4f}")


def example_get_action_probabilities():
    """
    示例4: 获取动作概率分布（用于分析）
    """
    print("\n" + "=" * 80)
    print("示例4: 分析动作概率分布")
    print("=" * 80)
    
    # 加载模型
    agent = load_model("rl_models/ppo_model_best.pt", state_dim=64)
    
    # 创建一个示例状态
    dummy_state = np.random.randn(64)
    state_tensor = torch.FloatTensor(dummy_state).unsqueeze(0).to(agent.device)
    
    # 获取动作概率和状态价值
    with torch.no_grad():
        action_probs, state_value = agent.policy(state_tensor)
    
    action_probs = action_probs.squeeze().cpu().numpy()
    state_value = state_value.item()
    
    # 输出分析
    print("状态价值估计:", f"{state_value:.4f}")
    print("\n动作概率分布:")
    action_names = ["动作1", "动作2", "动作7", "动作8", "动作9", "动作11"]
    for i, (name, prob) in enumerate(zip(action_names, action_probs)):
        print(f"  {name}: {prob:.4f} {'★' if prob > 0.2 else ''}")
    
    # 选择最优动作
    best_action_idx = np.argmax(action_probs)
    print(f"\n推荐动作: {action_names[best_action_idx]} (索引: {best_action_idx})")


def main():
    """
    主函数：运行所有示例
    """
    print("PPO 模型使用示例")
    print("=" * 80)
    print()
    
    # 检查模型文件是否存在
    if not os.path.exists("rl_models/ppo_model_best.pt"):
        print("错误: 未找到模型文件 rl_models/ppo_model_best.pt")
        print("请先运行训练脚本生成模型")
        return
    
    try:
        # 运行示例1
        example_single_inference()
        
        # 运行示例2（如果想运行批量推理，取消注释）
        # example_batch_inference()
        
        # 运行示例3
        # example_custom_usage()
        
        # 运行示例4
        # example_get_action_probabilities()
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

