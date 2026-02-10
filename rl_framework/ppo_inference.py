#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPO Inference Script - 使用训练好的模型进行二进制代码变异
"""

import os
import sys
import glob
import shutil
import json
import pickle
import numpy as np
import torch
import argparse
from loguru import logger
from tqdm import tqdm
import random   

# 导入环境和 Agent
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_wrapper import BinaryPerturbationEnv
from ppo_agent import PPOAgent
from run_utils import run_one

_INFERENCE_LOGGER_READY = False


def _setup_inference_logging():
    global _INFERENCE_LOGGER_READY
    if _INFERENCE_LOGGER_READY:
        return
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "log")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "ppo_inference.log")
    def _inference_log_filter(record):
        return record["name"] in ("ppo_inference", "__main__")
    logger.add(
        log_path,
        level="INFO",
        filter=_inference_log_filter,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )
    logger.info(f"PPO inference log file: {log_path}")
    _INFERENCE_LOGGER_READY = True


def cleanup_inference_files(save_path, keep_binary):
    """
    清理推理中间文件，仅保留最佳结果
    
    保留: inference_log.txt, 最佳变异结果目录
    删除: 所有其他 *_container 目录和临时文件
    """
    if not os.path.exists(save_path) or not keep_binary:
        return
    
    keep_path = os.path.abspath(keep_binary)
    keep_container = None
    
    # 找到需要保留的container目录
    if '_container' in keep_path:
        # keep_path 可能是 /path/xxx_container/xxx 或 /path/xxx_container
        parts = keep_path.split('_container')
        if parts:
            keep_container = parts[0] + '_container'
    
    deleted = 0
    freed = 0
    
    # 清理所有container目录（除了需要保留的）
    for container in glob.glob(os.path.join(save_path, '*_container')):
        container_abs = os.path.abspath(container)
        
        # 保留最佳结果所在的container
        if keep_container and container_abs == keep_container:
            continue
        
        try:
            size = sum(os.path.getsize(os.path.join(d, f)) 
                      for d, _, files in os.walk(container) for f in files)
            shutil.rmtree(container)
            deleted += 1
            freed += size
        except Exception as e:
            logger.warning(f"无法删除 {os.path.basename(container)}: {e}")
    
    # 清理其他临时文件（保留 inference_log.txt）
    for item in os.listdir(save_path):
        if item == 'inference_log.txt':
            continue
        
        path = os.path.join(save_path, item)
        if os.path.isfile(path):
            try:
                freed += os.path.getsize(path)
                os.remove(path)
                deleted += 1
            except Exception as e:
                logger.warning(f"无法删除 {item}: {e}")
    
    # 清理 rl_output 中的中间文件（优先使用 save_path 下的私有目录）
    rl_output = os.path.join(save_path, 'rl_output')
    if not os.path.exists(rl_output):
        rl_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rl_output')
    if os.path.exists(rl_output):
        for mutant in glob.glob(os.path.join(rl_output, 'mutant_*.bin*')):
            try:
                freed += os.path.getsize(mutant)
                os.remove(mutant)
                deleted += 1
            except Exception as e:
                logger.warning(f"无法删除 {os.path.basename(mutant)}: {e}")
    
    if deleted > 0:
        size_mb = freed / (1024 * 1024)
        logger.success(f"✓ 清理完成: 删除 {deleted} 项，释放 {size_mb:.2f} MB")
    
    if keep_container:
        logger.info(f"✓ 已保留最佳结果: {os.path.basename(keep_container)}")


def _find_sym_to_addr(binary_path):
    """Best-effort lookup for sym_to_addr.pickle near a binary."""
    base_dir = os.path.dirname(os.path.abspath(binary_path))
    candidates = [
        os.path.join(base_dir, "sym_to_addr.pickle"),
        os.path.join(os.path.dirname(base_dir), "sym_to_addr.pickle"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _load_sym_to_addr(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return {}


def _resolve_func_addr(binary_path, func_name):
    sym_path = _find_sym_to_addr(binary_path)
    sym_map = _load_sym_to_addr(sym_path)
    return sym_map.get(func_name)


def inference_ppo(args):
    """
    使用训练好的 PPO 模型进行推理
    
    参数:
        args: 命令行参数
            - model_path: 训练好的模型路径
            - binary: 原始二进制文件路径
            - function: 目标函数名
            - save_path: 变异结果保存路径
            - max_steps: 最大步数（默认30）
            - target_score: 目标分数（默认0.40）
    """
    logger.info("=" * 80)
    logger.info("PPO 推理模式")
    logger.info("=" * 80)
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"原始二进制: {args.binary}")
    logger.info(f"目标函数: {args.function}")
    logger.info(f"保存路径: {args.save_path}")
    logger.info(f"最大步数: {args.max_steps}")
    logger.info(f"目标分数: {args.target_score}")
    logger.info(f"检测方法: {args.detection_method}")
    _setup_inference_logging()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        logger.error(f"模型文件不存在: {args.model_path}")
        return
    
    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
    
    # 初始化环境
    logger.info("初始化变异环境...")
    dataset_path = os.path.join(args.save_path, "inference_dataset.json")
    dataset = [{
        "binary_path": os.path.abspath(args.binary),
        "binary_name": os.path.basename(args.binary),
        "version": "inference",
        "opt_level": "unknown",
        "func_name": args.function,
        "func_addr": 0,
        "size": 0,
        "id": "inference"
    }]
    with open(dataset_path, "w") as f:
        json.dump(dataset, f, indent=2)

    env = BinaryPerturbationEnv(
        save_path=args.save_path,
        dataset_path=dataset_path,
        sample_hold_interval=1,
        max_steps=args.max_steps
    )
    env.set_state_dim(args.state_dim)
    env.target_score = args.target_score
    logger.info("环境初始化完成 ✓")
    
    # 初始化 PPO Agent 并加载模型
    logger.info("加载训练好的模型...")
    device = 'cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'
    agent = PPOAgent(
        state_dim=args.state_dim,
        n_actions=env.n_actions,
        action_map=list(env.action_ids),
        device=device
    )
    agent.load(args.model_path)
    agent.policy.eval()  # 设置为评估模式
    logger.info(f"模型加载完成 ✓ (设备: {device})")
    
    # 执行推理
    logger.info("=" * 80)
    logger.info("开始变异过程")
    logger.info("=" * 80)
    
    state = env.reset()
    success = False
    best_score = float('inf')
    best_binary = None
    asm_work_dir = os.path.join(args.save_path, "_asm2vec_eval")
    os.makedirs(asm_work_dir, exist_ok=True)
    original_asm_cache = {}
    
    # 记录每步信息
    step_records = []
    
    for step in range(args.max_steps):
        logger.info(f"\n步骤 {step + 1}/{args.max_steps}")
        logger.info("-" * 60)
        
        # 使用模型选择动作（不探索，选择最优动作）
        joint_idx, loc_idx, act_idx, actual_action, log_prob, value = agent.select_action(state, explore=False)

        logger.info(f"选择动作: {actual_action} (位置: {loc_idx}, 动作索引: {act_idx}, 联合索引: {joint_idx})")
        logger.info(f"状态价值: {value:.4f}")
        
        # 执行动作
        next_state, reward, done, info = env.step(actual_action, loc_idx)
        
        # 可选：使用指定检测方法重新计算相似度
        eval_score = info.get('score', 1.0)
        if info.get('binary') and args.detection_method != "asm2vec":
            mutated_func_addr = None
            sym_to_addr_path = _find_sym_to_addr(info.get('binary'))
            if args.detection_method == "safe":
                mutated_func_addr = _resolve_func_addr(info.get('binary'), env.function_name)
                sym_to_addr_path = _find_sym_to_addr(env.original_binary)
            eval_score, _ = run_one(
                env.original_binary,
                info.get('binary'),
                model_original=None,
                checkdict={},
                function_name=env.function_name,
                detection_method=args.detection_method,
                asm_work_dir=asm_work_dir,
                original_asm_cache=original_asm_cache,
                simple_mode=True,
                original_func_addr=dataset[0].get("func_addr"),
                mutated_func_addr=mutated_func_addr,
                sym_to_addr_path=sym_to_addr_path,
                safe_checkpoint_dir=args.safe_checkpoint_dir,
                safe_i2v_dir=args.safe_i2v_dir,
                safe_use_gpu=args.safe_use_gpu,
            )
            if eval_score is None:
                eval_score = info.get('score', 1.0)
        logger.info(
            f"Step {step+1}: action={actual_action}, loc={loc_idx}, reward={reward:.4f}, eval_score={eval_score}"
            f"env_score={info.get('score')}, eval_score={eval_score}, done={done}, "
            f"should_reset={info.get('should_reset')}"
        )

        # 记录信息
        step_info = {
            'step': step + 1,
            'loc': loc_idx,
            'act_idx': act_idx,
            'action': actual_action,
            'score': eval_score,
            'grad': info.get('grad', 0.0),
            'binary': info.get('binary', None),
            'reward': reward,
            'value': value
        }
        step_records.append(step_info)
        
        # 输出结果
        if eval_score is not None:
            logger.info(f"相似度分数: {eval_score:.4f}")
            logger.info(f"梯度值: {info.get('grad', 0):.4f}")
            logger.info(f"奖励: {reward:.4f}")
            
            # 更新最佳结果
            if eval_score < best_score:
                best_score = eval_score
                best_binary = info.get('binary', None)
                logger.success(f"✨ 发现更好的结果! 分数: {best_score:.4f}")
            
            # 检查是否达到目标
            if eval_score < args.target_score:
                success = True
                logger.success(f"🎉 成功达到目标! 分数: {eval_score:.4f} < {args.target_score}")
                logger.success(f"变异后的二进制: {info.get('binary', 'unknown')}")
                break
        else:
            logger.warning("未能获取评估分数")
        
        state = next_state
        
        if args.detection_method == "asm2vec":
            if done:
                logger.info("回合结束")
                break
        else:
            if info.get("should_reset"):
                logger.info("回合结束 (should_reset)")
                break
    
    # 输出总结
    logger.info("")
    logger.info("=" * 80)
    logger.info("推理完成")
    logger.info("=" * 80)
    logger.info(f"执行步数: {step + 1}")
    logger.info(f"最佳分数: {best_score:.4f}")
    
    if success:
        logger.success(f"✓ 成功达到目标 (分数 < {args.target_score})")
    else:
        logger.warning(f"✗ 未达到目标 (分数 >= {args.target_score})")
    
    if best_binary:
        logger.info(f"最佳变异结果: {best_binary}")
    logger.info(
        f"RESULT: success={int(success)}, steps={step + 1}, "
        f"best_score={best_score:.6f}, target_score={args.target_score}"
    )
    
    # 保存推理日志
    log_file = os.path.join(args.save_path, 'inference_log.txt')
    with open(log_file, 'w') as f:
        f.write(f"模型: {args.model_path}\n")
        f.write(f"二进制: {args.binary}\n")
        f.write(f"函数: {args.function}\n")
        f.write(f"最佳分数: {best_score:.4f}\n")
        f.write(f"成功: {success}\n")
        f.write(f"最佳结果: {best_binary}\n\n")
        f.write("步骤详情:\n")
        f.write("step,loc,act_idx,action,score,grad,reward,value,binary\n")
        for record in step_records:
            f.write(f"{record['step']},{record['loc']},{record['act_idx']},{record['action']},{record['score']:.4f},"
                   f"{record['grad']:.4f},{record['reward']:.4f},{record['value']:.4f},"
                   f"{record['binary']}\n")
    
    logger.info(f"推理日志已保存: {log_file}")
    
    # 清理中间文件，只保留最佳结果
    logger.info("")
    cleanup_inference_files(args.save_path, best_binary)
    
    return best_score, best_binary, success


def batch_inference(args):
    """
    批量推理：对多个二进制文件或函数进行变异
    """
    logger.info("批量推理模式")
    
    # 读取批量配置文件
    if not os.path.exists(args.batch_file):
        logger.error(f"批量配置文件不存在: {args.batch_file}")
        return
    
    with open(args.batch_file, 'r') as f:
        lines = f.readlines()
    
    results = []
    
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        parts = line.split(',')
        if len(parts) < 2:
            logger.warning(f"跳过无效行: {line}")
            continue
        
        binary, function = parts[0], parts[1]
        save_path = parts[2] if len(parts) > 2 else f"{args.save_path}_{idx}"
        
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"批量任务 {idx + 1}/{len(lines)}")
        logger.info("=" * 80)
        
        # 设置参数
        args.binary = binary
        args.function = function
        args.save_path = save_path
        
        # 执行推理
        try:
            best_score, best_binary, success = inference_ppo(args)
            results.append({
                'binary': binary,
                'function': function,
                'score': best_score,
                'success': success,
                'output': best_binary
            })
        except Exception as e:
            logger.error(f"任务失败: {e}")
            results.append({
                'binary': binary,
                'function': function,
                'score': float('inf'),
                'success': False,
                'output': None
            })
    
    # 输出批量结果
    logger.info("")
    logger.info("=" * 80)
    logger.info("批量推理完成")
    logger.info("=" * 80)
    
    success_count = sum(1 for r in results if r['success'])
    logger.info(f"总任务数: {len(results)}")
    logger.info(f"成功数: {success_count}")
    logger.info(f"成功率: {success_count / len(results) * 100:.2f}%")
    
    # 保存批量结果
    batch_log = os.path.join(os.path.dirname(args.batch_file), 'batch_inference_results.txt')
    with open(batch_log, 'w') as f:
        f.write("binary,function,score,success,output\n")
        for r in results:
            f.write(f"{r['binary']},{r['function']},{r['score']:.4f},"
                   f"{r['success']},{r['output']}\n")
    
    logger.info(f"批量结果已保存: {batch_log}")


def _pin_sample(env, sample_idx, sample):
    env.current_sample_idx = sample_idx
    env.current_sample_data = sample
    env.episodes_on_current = 0
    env.original_func_addr = None
    env.original_binary = sample.get("binary_path")
    env.function_name = sample.get("func_name")


def evaluate_dataset(args):
    logger.info("=" * 80)
    logger.info("PPO 数据集评估模式")
    logger.info("=" * 80)
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"数据集: {args.dataset}")
    logger.info(f"保存路径: {args.save_path}")
    logger.info(f"最大步数: {args.max_steps}")
    logger.info(f"目标分数: {args.target_score}")
    logger.info(f"检测方法: {args.detection_method}")
    _setup_inference_logging()

    if not os.path.exists(args.model_path):
        logger.error(f"模型文件不存在: {args.model_path}")
        return
    if not os.path.exists(args.dataset):
        logger.error(f"数据集文件不存在: {args.dataset}")
        return

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        logger.error("数据集格式错误：必须是 JSON 列表")
        return

    if args.seed is not None:
        random.seed(args.seed)
    random.shuffle(dataset)

    if args.limit is not None:
        dataset = dataset[: args.limit]

    os.makedirs(args.save_path, exist_ok=True)

    env = BinaryPerturbationEnv(
        save_path=args.save_path,
        dataset_path=args.dataset,
        sample_hold_interval=10**9,
        max_steps=args.max_steps,
    )
    env.set_state_dim(args.state_dim)
    env.target_score = args.target_score

    device = 'cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'
    agent = PPOAgent(
        state_dim=args.state_dim,
        n_actions=env.n_actions,
        action_map=list(env.action_ids),
        device=device
    )
    agent.load(args.model_path)
    agent.policy.eval()

    asm_work_dir = os.path.join(args.save_path, "_asm2vec_eval")
    os.makedirs(asm_work_dir, exist_ok=True)
    original_asm_cache = {}

    success_count = 0
    total = len(dataset)

    pbar = tqdm(enumerate(dataset), total=total, desc="PPO Eval", unit="sample")
    for idx, sample in pbar:
        _pin_sample(env, idx, sample)
        state = env.reset(force_switch=False)

        best_score = 1.0
        success = False

        for step in range(args.max_steps):
            joint_idx, loc_idx, act_idx, actual_action, log_prob, value = agent.select_action(state, explore=True)
            next_state, reward, done, info = env.step(actual_action, loc_idx)
            print(f"Step {step}: action={actual_action}, loc={loc_idx}")
            eval_score = info.get('score', 1.0)
            if info.get('binary') and args.detection_method != "asm2vec":
                mutated_func_addr = None
                sym_to_addr_path = _find_sym_to_addr(info.get('binary'))
                if args.detection_method == "safe":
                    mutated_func_addr = _resolve_func_addr(info.get('binary'), env.function_name)
                    sym_to_addr_path = _find_sym_to_addr(env.original_binary)
                eval_score, _ = run_one(
                    env.original_binary,
                    info.get('binary'),
                    model_original=None,
                    checkdict={},
                    function_name=env.function_name,
                    detection_method=args.detection_method,
                    asm_work_dir=asm_work_dir,
                    original_asm_cache=original_asm_cache,
                    simple_mode=True,
                    original_func_addr=sample.get("func_addr"),
                    mutated_func_addr=mutated_func_addr,
                    sym_to_addr_path=sym_to_addr_path,
                    safe_checkpoint_dir=args.safe_checkpoint_dir,
                    safe_i2v_dir=args.safe_i2v_dir,
                    safe_use_gpu=args.safe_use_gpu,
                )
                if eval_score is None:
                    eval_score = info.get('score', 1.0)

            logger.info(
                f"Eval step {step+1}: action={actual_action}, loc={loc_idx}, reward={reward:.4f}, "
                f"env_score={info.get('score')}, eval_score={eval_score}, done={done}, "
                f"should_reset={info.get('should_reset')}"
            )
            if eval_score is not None and eval_score < best_score:
                best_score = eval_score
                if eval_score < args.target_score:
                    success = True
                    break

            state = next_state
            if args.detection_method == "asm2vec":
                if done:
                    break
            else:
                if info.get("should_reset"):
                    break

        if success:
            success_count += 1

        if (idx + 1) % 10 == 0:
            pbar.set_postfix({"success_rate": f"{success_count/max(1, idx+1):.3f}"})

    success_rate = success_count / max(1, total)
    logger.success(f"✓ 测试完成: success_rate={success_rate:.4f} ({success_count}/{total})")
    logger.info(
        f"RESULT: success_rate={success_rate:.6f}, success_count={success_count}, total={total}, "
        f"target_score={args.target_score}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO Inference for Binary Perturbation')
    
    # 模型参数
    default_model = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rl_models/ppo_model_ep200.pt')
    parser.add_argument('--model-path', default=default_model, help='训练好的模型路径')
    
    # 目标参数
    default_dataset = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets/fast_0128/dataset_test.json')
    parser.add_argument('--binary', help='原始二进制文件路径')
    parser.add_argument('--function', help='目标函数名')
    parser.add_argument('--save-path', help='变异结果保存路径')
    parser.add_argument('--dataset', default=default_dataset, help='测试数据集路径')
    parser.add_argument('--limit', type=int, default=None, help='限制评估样本数量')
    parser.add_argument('--eval-dataset', action='store_true', help='评估整个数据集的成功率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 推理参数
    parser.add_argument('--state-dim', type=int, default=256, help='状态维度（必须与训练时一致，默认 256）')
    parser.add_argument('--max-steps', type=int, default=30, help='最大步数')
    parser.add_argument('--target-score', type=float, default=0.40, help='目标相似度分数')
    parser.add_argument('--detection-method', choices=['asm2vec', 'safe'], default='asm2vec', help='相似度检测方法')
    parser.add_argument('--use-gpu', action='store_true', help='使用 GPU')
    parser.add_argument('--safe-checkpoint-dir', default=None, help='SAFE 模型 checkpoint 目录')
    parser.add_argument('--safe-i2v-dir', default=None, help='SAFE i2v 目录')
    parser.add_argument('--safe-use-gpu', action='store_true', help='SAFE 使用 GPU')
    
    # 批量模式
    parser.add_argument('--batch', action='store_true', help='批量推理模式')
    parser.add_argument('--batch-file', help='批量配置文件 (格式: binary,function,save_path)')
    
    args = parser.parse_args()
    
    # 批量模式/数据集评估/单次推理
    if args.eval_dataset:
        if not args.save_path:
            logger.error("数据集评估需要指定 --save-path")
            sys.exit(1)
        evaluate_dataset(args)
    elif args.batch:
        if not args.batch_file:
            logger.error("批量模式需要指定 --batch-file")
            sys.exit(1)
        batch_inference(args)
    else:
        if not args.binary or not args.function or not args.save_path:
            logger.error("单次推理模式需要指定 --binary, --function, --save-path")
            sys.exit(1)
        inference_ppo(args)
