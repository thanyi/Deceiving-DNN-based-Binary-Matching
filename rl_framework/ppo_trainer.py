#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPO Trainer for Binary Code Perturbation
PPO 训练器（直接调用环境）
"""
import os
import csv
import json
import numpy as np
import torch
from ppo_agent import PPOAgent
import argparse
from loguru import logger
import sys
import shutil
import glob
import random
import pickle
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict, List, Optional

# 导入环境
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_wrapper import BinaryPerturbationEnv
from run_utils import run_one


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
    
    # 清理 rl_output 中的中间文件（优先使用 save_path 下的私有目录）
    rl_output_dir = os.path.join(save_path, 'rl_output')
    if not os.path.exists(rl_output_dir):
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


def _load_trainer_state(path):
    if not path or not os.path.exists(path):
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as e:
        logger.warning(f"无法读取训练状态: {path}, {e}")
        return None
    return checkpoint.get("trainer_state")


def _find_sym_to_addr(binary_path: str) -> Optional[str]:
    base_dir = os.path.dirname(os.path.abspath(binary_path))
    candidates = [
        os.path.join(base_dir, "sym_to_addr.pickle"),
        os.path.join(os.path.dirname(base_dir), "sym_to_addr.pickle"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _load_sym_to_addr(path: Optional[str]) -> Dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return {}


def _parse_addr(value) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            return int(value, 16) if value.startswith(("0x", "0X")) else int(value)
        return int(value)
    except Exception:
        return None


def _opt_rank(opt_level: str) -> int:
    order = {"O0": 0, "O1": 1, "O2": 2, "O3": 3, "Os": 4, "Oz": 5}
    return order.get(str(opt_level), 99)


class TargetedAttackEvaluator:
    """
    定向攻击评测：
    - attacker 函数: 随机样本
    - target 身份: 随机另一函数身份的多个编译变体
    - 目标: 最大化 min(sim(attacker_adv, v_i))
    """

    def __init__(self, args):
        self.args = args
        self.dataset_path = args.targeted_eval_dataset or args.dataset
        self.eval_steps = max(1, int(args.targeted_eval_max_steps))
        self.pairs = max(1, int(args.targeted_eval_pairs))
        self.max_target_variants = max(1, int(args.targeted_eval_max_target_variants))
        self.min_target_variants = max(1, int(args.targeted_eval_min_target_variants))
        self.success_threshold = float(args.targeted_eval_threshold)
        self.rng = random.Random(args.targeted_eval_seed)

        with open(self.dataset_path, "r") as f:
            self.dataset = json.load(f)
        if not isinstance(self.dataset, list) or not self.dataset:
            raise ValueError("targeted eval dataset is empty or invalid")

        self.id_to_item = {}
        self.id_to_index = {}
        for idx, item in enumerate(self.dataset):
            sid = str(item.get("id", "")).strip()
            if not sid:
                continue
            self.id_to_item[sid] = item
            self.id_to_index[sid] = idx

        eval_save_path = os.path.join(args.save_path, "targeted_eval")
        os.makedirs(eval_save_path, exist_ok=True)
        self.env = BinaryPerturbationEnv(
            save_path=eval_save_path,
            dataset_path=self.dataset_path,
            sample_hold_interval=10**9,
            max_steps=self.eval_steps,
            detection_method=args.detection_method,
            safe_checkpoint_dir=args.safe_checkpoint_dir,
            safe_i2v_dir=args.safe_i2v_dir,
            safe_use_gpu=args.safe_use_gpu,
            safe_cache_enabled=(args.detection_method == "safe" and not args.no_safe_cache),
            jtrans_model_dir=args.jtrans_model_dir,
            jtrans_tokenizer_dir=args.jtrans_tokenizer_dir,
            jtrans_use_gpu=args.jtrans_use_gpu,
            feature_mode=args.feature_mode,
            seed=args.seed,
            stall_limit=args.stall_limit,
            progress_eps=args.progress_eps,
            progress_reward_eps=args.progress_reward_eps,
            include_schedule_feature=args.include_schedule_feature,
            strict_invalid_loc=(not args.non_strict_invalid_loc),
            hold_min=args.hold_min,
            hold_max=args.hold_max,
        )
        self.env.set_state_dim(args.state_dim)
        # 定向评测只受 max_steps 约束，不希望被 env.target_score 提前 done。
        self.env.target_score = -1.0

        self._target_original_asm_cache = {}
        self._mutated_asm_cache = {}
        self._mutated_sym_cache = {}

    def _pin_sample(self, sample_idx: int) -> None:
        sample = self.env.dataset[sample_idx]
        self.env.current_sample_idx = sample_idx
        self.env.current_sample_data = sample
        self.env.episodes_on_current = 0
        self.env.original_func_addr = None
        self.env.original_binary = sample["binary_path"]
        self.env.function_name = sample["func_name"]

    def _collect_target_variants(self, target_item: Dict) -> List[Dict]:
        target_ids = []
        anchor_id = str(target_item.get("id", "")).strip()
        if anchor_id:
            target_ids.append(anchor_id)
        for vid in target_item.get("variants", []) or []:
            svid = str(vid).strip()
            if svid:
                target_ids.append(svid)

        dedup = []
        seen = set()
        for tid in target_ids:
            if tid in seen:
                continue
            seen.add(tid)
            item = self.id_to_item.get(tid)
            if item is not None:
                dedup.append(item)

        dedup.sort(
            key=lambda x: (
                _opt_rank(x.get("opt_level")),
                str(x.get("version", "")),
                str(x.get("binary_name", "")),
            )
        )
        return dedup[: self.max_target_variants]

    def _sample_pairs(self) -> List[Dict]:
        target_candidates = []
        for item in self.dataset:
            variants = self._collect_target_variants(item)
            if len(variants) >= self.min_target_variants:
                t_ids = {str(v.get("id")) for v in variants if v.get("id")}
                target_candidates.append((item, variants, t_ids))

        if not target_candidates:
            return []

        pairs = []
        seen = set()
        max_attempts = self.pairs * 40
        attempts = 0
        while len(pairs) < self.pairs and attempts < max_attempts:
            attempts += 1
            target_item, target_variants, target_id_set = self.rng.choice(target_candidates)
            target_func = str(target_item.get("func_name", ""))
            attacker_idx = self.rng.randrange(len(self.dataset))
            attacker_item = self.dataset[attacker_idx]
            attacker_id = str(attacker_item.get("id", ""))
            attacker_func = str(attacker_item.get("func_name", ""))
            if not attacker_id or attacker_id in target_id_set:
                continue
            if attacker_func == target_func:
                continue
            key = (attacker_id, str(target_item.get("id", "")))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(
                {
                    "attacker_idx": attacker_idx,
                    "attacker": attacker_item,
                    "target": target_item,
                    "target_variants": target_variants,
                }
            )
        return pairs

    def _resolve_mutated_addr(self, mutated_binary: str, attacker_func_name: str) -> Optional[int]:
        bpath = os.path.abspath(mutated_binary)
        sym_map = self._mutated_sym_cache.get(bpath)
        if sym_map is None:
            sym_map = _load_sym_to_addr(_find_sym_to_addr(bpath))
            self._mutated_sym_cache[bpath] = sym_map
        if not sym_map:
            return None
        return _parse_addr(sym_map.get(attacker_func_name))

    def _score_binary_against_target_variants(
        self, mutated_binary: str, attacker_func_name: str, target_variants: List[Dict]
    ) -> Dict:
        attacker_addr = self._resolve_mutated_addr(mutated_binary, attacker_func_name)
        if attacker_addr is None:
            return {"valid": False, "scores": [], "min_score": -1.0, "avg_score": -1.0}

        scores = []
        for tv in target_variants:
            target_binary = tv.get("binary_path")
            target_func = str(tv.get("func_name", ""))
            target_addr = _parse_addr(tv.get("func_addr"))
            if not target_binary or not target_func or target_addr is None:
                continue

            score, _grad = run_one(
                original_binary=target_binary,
                mutated_binary=mutated_binary,
                model_original=None,
                checkdict={},
                function_name=target_func,
                detection_method=self.args.detection_method,
                asm_work_dir=self.env._asm_work_dir,
                original_asm_cache=self._target_original_asm_cache,
                simple_mode=True,
                original_func_addr=target_addr,
                mutated_func_addr=attacker_addr,
                safe_checkpoint_dir=self.args.safe_checkpoint_dir,
                safe_i2v_dir=self.args.safe_i2v_dir,
                safe_use_gpu=self.args.safe_use_gpu,
                mutated_asm_cache=self._mutated_asm_cache,
                safe_cache=self.env.safe_cache,
                jtrans_model_dir=self.args.jtrans_model_dir,
                jtrans_tokenizer_dir=self.args.jtrans_tokenizer_dir,
                jtrans_use_gpu=self.args.jtrans_use_gpu,
                jtrans_cache=self.env._jtrans_cache,
            )
            if score is None:
                continue
            scores.append(float(score))

        if len(scores) != len(target_variants):
            return {"valid": False, "scores": scores, "min_score": -1.0, "avg_score": -1.0}
        return {
            "valid": True,
            "scores": scores,
            "min_score": float(min(scores)),
            "avg_score": float(sum(scores) / len(scores)),
        }

    def _run_one_pair(self, agent: PPOAgent, pair: Dict) -> Dict:
        attacker = pair["attacker"]
        target = pair["target"]
        target_variants = pair["target_variants"]
        self._pin_sample(pair["attacker_idx"])
        state = self.env.reset(force_switch=False)

        pre_eval = self._score_binary_against_target_variants(
            self.env.current_binary,
            str(attacker.get("func_name", "")),
            target_variants,
        )
        best_eval = dict(pre_eval)
        best_binary = self.env.current_binary
        best_step = 0
        err = ""
        steps_used = 0

        for step in range(self.eval_steps):
            loc_mask = self.env.get_loc_mask(self.args.n_locs)
            (
                _joint_idx,
                loc_idx,
                _act_idx,
                actual_action,
                _log_prob,
                _value,
            ) = agent.select_action(state, explore=self.args.targeted_eval_explore, loc_mask=loc_mask)
            next_state, _reward, done, info = self.env.step(actual_action, loc_idx)
            steps_used = step + 1

            cand_binary = info.get("binary")
            if cand_binary:
                cur_eval = self._score_binary_against_target_variants(
                    cand_binary,
                    str(attacker.get("func_name", "")),
                    target_variants,
                )
                if cur_eval["valid"] and (not best_eval["valid"] or cur_eval["min_score"] > best_eval["min_score"]):
                    best_eval = cur_eval
                    best_binary = cand_binary
                    best_step = steps_used

            if info.get("should_reset"):
                err = str(info.get("error", "should_reset"))
                break

            state = next_state
            if done:
                break

        pre_min = pre_eval["min_score"] if pre_eval["valid"] else -1.0
        post_min = best_eval["min_score"] if best_eval["valid"] else -1.0
        pre_avg = pre_eval["avg_score"] if pre_eval["valid"] else -1.0
        post_avg = best_eval["avg_score"] if best_eval["valid"] else -1.0

        return {
            "attacker_id": str(attacker.get("id", "")),
            "attacker_func": str(attacker.get("func_name", "")),
            "target_id": str(target.get("id", "")),
            "target_func": str(target.get("func_name", "")),
            "target_variants": len(target_variants),
            "steps_used": steps_used,
            "best_step": best_step,
            "pre_valid": int(pre_eval["valid"]),
            "post_valid": int(best_eval["valid"]),
            "pre_min": pre_min,
            "post_min": post_min,
            "pre_avg": pre_avg,
            "post_avg": post_avg,
            "gain_min": post_min - pre_min,
            "gain_avg": post_avg - pre_avg,
            "success_pre": int(pre_min >= self.success_threshold),
            "success_post": int(post_min >= self.success_threshold),
            "improved": int(post_min > pre_min),
            "error": err,
            "best_binary": best_binary or "",
        }

    def evaluate(self, agent: PPOAgent, episode: int) -> Optional[Dict]:
        pairs = self._sample_pairs()
        if not pairs:
            logger.warning("TargetedEval: 无法采样到有效 attacker-target 配对，跳过")
            return None

        rows = []
        for pair in pairs:
            try:
                rows.append(self._run_one_pair(agent, pair))
            except Exception as e:
                rows.append(
                    {
                        "attacker_id": str(pair["attacker"].get("id", "")),
                        "attacker_func": str(pair["attacker"].get("func_name", "")),
                        "target_id": str(pair["target"].get("id", "")),
                        "target_func": str(pair["target"].get("func_name", "")),
                        "target_variants": len(pair["target_variants"]),
                        "steps_used": 0,
                        "best_step": 0,
                        "pre_valid": 0,
                        "post_valid": 0,
                        "pre_min": -1.0,
                        "post_min": -1.0,
                        "pre_avg": -1.0,
                        "post_avg": -1.0,
                        "gain_min": 0.0,
                        "gain_avg": 0.0,
                        "success_pre": 0,
                        "success_post": 0,
                        "improved": 0,
                        "error": str(e),
                        "best_binary": "",
                    }
                )

        valid_rows = [r for r in rows if r["pre_valid"] and r["post_valid"]]
        metrics = {"pairs_total": len(rows), "pairs_valid": len(valid_rows)}
        if valid_rows:
            denom = float(len(valid_rows))
            metrics.update(
                {
                    "min_pre": float(sum(r["pre_min"] for r in valid_rows) / denom),
                    "min_post": float(sum(r["post_min"] for r in valid_rows) / denom),
                    "avg_pre": float(sum(r["pre_avg"] for r in valid_rows) / denom),
                    "avg_post": float(sum(r["post_avg"] for r in valid_rows) / denom),
                    "gain_min": float(sum(r["gain_min"] for r in valid_rows) / denom),
                    "gain_avg": float(sum(r["gain_avg"] for r in valid_rows) / denom),
                    "success_pre": float(sum(r["success_pre"] for r in valid_rows) / denom),
                    "success_post": float(sum(r["success_post"] for r in valid_rows) / denom),
                    "improved_rate": float(sum(r["improved"] for r in valid_rows) / denom),
                }
            )
        else:
            metrics.update(
                {
                    "min_pre": -1.0,
                    "min_post": -1.0,
                    "avg_pre": -1.0,
                    "avg_post": -1.0,
                    "gain_min": 0.0,
                    "gain_avg": 0.0,
                    "success_pre": 0.0,
                    "success_post": 0.0,
                    "improved_rate": 0.0,
                }
            )

        out_csv = os.path.join(self.args.model_dir, f"targeted_eval_ep{episode}.csv")
        os.makedirs(self.args.model_dir, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        metrics["csv"] = out_csv

        logger.success(
            "TargetedEval: "
            f"pairs={metrics['pairs_total']} valid={metrics['pairs_valid']} "
            f"min(pre->post)={metrics['min_pre']:.4f}->{metrics['min_post']:.4f} "
            f"gain={metrics['gain_min']:.4f} "
            f"success@{self.success_threshold:.2f}={metrics['success_post']:.2%} "
            f"improved={metrics['improved_rate']:.2%}"
        )
        cleanup_intermediate_files(self.env.save_path, episode_binaries=None)
        return metrics


def train_ppo(args):
    """
    PPO 训练主函数
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_log_dir = os.path.join(project_root, 'log')
    default_log_path = os.path.join(default_log_dir, 'train.log')
    train_log_path = args.log_path or default_log_path
    log_dir = os.path.dirname(os.path.abspath(train_log_path))
    os.makedirs(log_dir, exist_ok=True)

    def _console_log_filter(record):
        level = record["level"].name
        if level in ("WARNING", "ERROR", "CRITICAL", "SUCCESS"):
            return True
        if record["name"] not in ["ppo_trainer", 'run_one']:
            return False
        msg = record["message"]
        return (
            msg.startswith("PPO 训练启动") or
            msg.startswith("数据集:") or
            msg.startswith("保存路径:") or
            msg.startswith("TensorBoard:") or
            msg.startswith("回合总结:") or
            msg.startswith("action_stats:") or
            msg.startswith("loc_valid统计:") or
            msg.startswith("TargetedEval:")
        )

    def _file_log_filter(record):
        level = record["level"].name
        if level in ("WARNING", "ERROR", "CRITICAL", "SUCCESS"):
            return True
        name = record["name"]
        msg = record["message"]
        if name == "ppo_trainer":
            return (
                msg.startswith("PPO 训练启动") or
                msg.startswith("数据集:") or
                msg.startswith("保存路径:") or
                msg.startswith("TensorBoard:") or
                msg.startswith("回合总结:") or
                msg.startswith("action_stats:") or
                msg.startswith("loc_valid统计:") or
                msg.startswith("TargetedEval:") or
                "训练结束" in msg
            )
        if name == "ppo_agent" and msg.startswith("✅ 模型已保存"):
            return True
        return False

    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        filter=_console_log_filter,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    logger.add(
        train_log_path,
        level="INFO",
        filter=_file_log_filter,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )

    logger.info("PPO 训练启动 (Multi-Sample Mode)")
    logger.info(f"数据集: {args.dataset}")
    logger.info(f"保存路径: {args.save_path}")
    logger.info(f"检测方法: {args.detection_method}")
    
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    tensorboard_dir = os.path.join(args.model_dir, 'tensorboard')
    writer = SummaryWriter(log_dir=tensorboard_dir)
    logger.info(f"TensorBoard: {tensorboard_dir}")
    
    # 初始化环境
    env = BinaryPerturbationEnv(
        save_path=args.save_path,
        dataset_path=args.dataset,
        sample_hold_interval=args.sample_hold_interval, # Hold-N 策略
        max_steps=args.max_steps,
        detection_method=args.detection_method,
        safe_checkpoint_dir=args.safe_checkpoint_dir,
        safe_i2v_dir=args.safe_i2v_dir,
        safe_use_gpu=args.safe_use_gpu,
        safe_cache_enabled=(args.detection_method == "safe" and not args.no_safe_cache),
        jtrans_model_dir=args.jtrans_model_dir,
        jtrans_tokenizer_dir=args.jtrans_tokenizer_dir,
        jtrans_use_gpu=args.jtrans_use_gpu,
        feature_mode=args.feature_mode,
        seed=args.seed,
        stall_limit=args.stall_limit,
        progress_eps=args.progress_eps,
        progress_reward_eps=args.progress_reward_eps,
        include_schedule_feature=args.include_schedule_feature,
        strict_invalid_loc=(not args.non_strict_invalid_loc),
        hold_min=args.hold_min,
        hold_max=args.hold_max,
    )
    env.set_state_dim(args.state_dim)
    if args.detection_method == "safe":
        safe_target_start = 0.9
        safe_target_end = 0.4
        safe_target_decay_episodes = 1200
        env.target_score = safe_target_start
        env.no_change_penalty = 0.05
        logger.success(
            f"[SAFE train] target_score linear decay {safe_target_start}->{safe_target_end} over {safe_target_decay_episodes} eps; "
            f"no_change_penalty={env.no_change_penalty}"
        )
    elif args.detection_method == "jtrans":
        jtrans_target_start = 0.9
        jtrans_target_end = 0.4
        jtrans_target_decay_episodes = 1200
        env.target_score = jtrans_target_start
        logger.success(
            f"[JTRANS train] target_score linear decay {jtrans_target_start}->{jtrans_target_end} over {jtrans_target_decay_episodes} eps"
        )

    agent = PPOAgent(
        state_dim=args.state_dim,
        n_actions=env.n_actions,
        action_map=list(env.action_ids),
        lr=args.lr,
        gamma=args.gamma,
        epsilon=args.epsilon,
        n_locs=args.n_locs,
        device='cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'
    )

    targeted_evaluator = None
    if args.targeted_eval_interval > 0:
        try:
            targeted_evaluator = TargetedAttackEvaluator(args)
            logger.success(
                "TargetedEval: "
                f"enabled interval={args.targeted_eval_interval} "
                f"pairs={args.targeted_eval_pairs} "
                f"max_target_variants={args.targeted_eval_max_target_variants} "
                f"eval_steps={args.targeted_eval_max_steps} "
                f"threshold={args.targeted_eval_threshold}"
            )
        except Exception as e:
            logger.warning(f"TargetedEval: 初始化失败，已禁用 ({e})")
            targeted_evaluator = None

    action_ids = list(getattr(env, "action_ids", []))
    action_stats = {aid: {'count': 0, 'reward_sum': 0.0, 'success': 0} for aid in action_ids}
    
    if args.resume and os.path.exists(args.resume):
        agent.load(args.resume)
    
    # log_file = os.path.join(args.model_dir, 'training_log.txt')
    
    episode_binaries = []
    
    # 滑动窗口统计
    success_window = deque(maxlen=50)
    similarity_drop_window = deque(maxlen=50)
    
    # 初始化统计变量
    success_count = 0
    best_score = float('inf')
    info = {}  # 初始化 info，避免作用域问题
   
    global_total_steps = 0
    global_loc_total_steps = 0
    global_loc_invalid_steps = 0
    start_episode = 0

    trainer_state = _load_trainer_state(args.resume) if args.resume else None
    if trainer_state:
        start_episode = int(trainer_state.get("episode", -1)) + 1
        global_total_steps = int(trainer_state.get("global_total_steps", 0))
        global_loc_total_steps = int(trainer_state.get("global_loc_total_steps", 0))
        global_loc_invalid_steps = int(trainer_state.get("global_loc_invalid_steps", 0))
        success_count = int(trainer_state.get("success_count", 0))
        best_score = float(trainer_state.get("best_score", float('inf')))
        success_window = deque(trainer_state.get("success_window", []), maxlen=50)
        similarity_drop_window = deque(trainer_state.get("similarity_drop_window", []), maxlen=50)
        logger.info(f"从断点恢复训练: start_episode={start_episode}")
    if start_episode >= args.episodes:
        logger.warning(f"resume 进度已达到 args.episodes ({args.episodes}), 无需继续训练")
        return
    def _make_trainer_state(episode):
        return {
            "episode": int(episode),
            "global_total_steps": int(global_total_steps),
            "global_loc_total_steps": int(global_loc_total_steps),
            "global_loc_invalid_steps": int(global_loc_invalid_steps),
            "success_count": int(success_count),
            "best_score": float(best_score),
            "success_window": list(success_window),
            "similarity_drop_window": list(similarity_drop_window),
        }

    try:
        pbar = tqdm(total=args.episodes - start_episode, desc="Training", unit="ep", dynamic_ncols=True)
        for episode in range(start_episode, args.episodes):
            logger.info("=" * 60)
            logger.info(f"回合 {episode + 1}/{args.episodes}")
            # 训练动态：基于最近成功率调整目标分数（避免单纯线性下降）
            recent_success_rate = np.mean(success_window) if success_window else 0.0
            if args.detection_method == "safe":
                progress = min(1.0, episode / max(1, safe_target_decay_episodes))
                linear_target = safe_target_start + (safe_target_end - safe_target_start) * progress
                safe_success_gate = 0.55
                step = abs(safe_target_start - safe_target_end) / max(1, safe_target_decay_episodes)
                if recent_success_rate >= safe_success_gate:
                    env.target_score = max(linear_target, env.target_score - step)
                # 否则保持当前 target_score，不继续降低
                env.target_score = max(safe_target_end, min(safe_target_start, env.target_score))
                if episode % 10 == 0:
                    logger.success(
                        f"[SAFE train] target_score={env.target_score:.4f} (ep={episode}) "
                        f"sr={recent_success_rate:.2f} gate={safe_success_gate:.2f}"
                    )
            elif args.detection_method == "jtrans":
                progress = min(1.0, episode / max(1, jtrans_target_decay_episodes))
                linear_target = jtrans_target_start + (jtrans_target_end - jtrans_target_start) * progress
                jtrans_success_gate = 0.55
                step = abs(jtrans_target_start - jtrans_target_end) / max(1, jtrans_target_decay_episodes)
                if recent_success_rate >= jtrans_success_gate:
                    env.target_score = max(linear_target, env.target_score - step)
                env.target_score = max(jtrans_target_end, min(jtrans_target_start, env.target_score))
                if episode % 10 == 0:
                    logger.success(
                        f"[JTRANS train] target_score={env.target_score:.4f} (ep={episode}) "
                        f"sr={recent_success_rate:.2f} gate={jtrans_success_gate:.2f}"
                    )
            
            state = env.reset()
            
            episode_actions = [] 
            initial_score = 1.0 # 【优化】默认初始为1.0，防止第一步没取到score导致计算错误
            
            episode_reward = 0
            episode_loc_total_steps = 0
            episode_loc_invalid_steps = 0
            last_binary_info = None
            should_skip_update = False
            episode_done = False  # 标记 episode 是否正常结束
            
            for step in range(args.max_steps):
                global_total_steps += 1 

                loc_mask = env.get_loc_mask(args.n_locs)
                joint_idx, loc_idx, act_idx, actual_action, log_prob, value = agent.select_action(
                    state, explore=True, loc_mask=loc_mask
                )
                episode_actions.append(actual_action)
                prev_state = state
                
                # 执行动作
                next_state, reward, done, info = env.step(actual_action, loc_idx)
                logger.success(f"Step {step}: Loc {loc_idx}, Action {actual_action} (JointIdx {joint_idx}), reward {reward:.4f}")
                # input("step down, press enter to continue")
                episode_reward += reward
                state = next_state

                loc_valid = info.get('loc_valid')
                if loc_valid is not None:
                    episode_loc_total_steps += 1
                    global_loc_total_steps += 1
                    if not loc_valid:
                        episode_loc_invalid_steps += 1
                        global_loc_invalid_steps += 1
                
                
                if step % 100 == 0:
                    # 记录每步指标 (当前设置：每步都记，如果太慢可改为 if step % 5 == 0)
                    writer.add_scalar('Step/Shaped_Reward', reward, global_total_steps)            # Agent 每做一步动作得到的即时反馈（包含进步分、惩罚分等）。
                    writer.add_scalar('Step/Critic_Value', value, global_total_steps)                     # Critic 网络（裁判）认为“当前这个状态，未来能拿多少分”。
                    if 'score' in info:
                        writer.add_scalar('Step/Similarity_Score', info['score'], global_total_steps)     # 每一步变异后的代码与原代码的相似度。

                # 统计动作级别的 reward/success
                stat = action_stats.setdefault(actual_action, {'count': 0, 'reward_sum': 0.0, 'success': 0})
                stat['count'] += 1
                stat['reward_sum'] += reward
                if info.get('score', 1.0) < env.target_score:
                    stat['success'] += 1

                # 存储经验
                agent.store_transition(prev_state, joint_idx, reward, log_prob, value, done, loc_mask=loc_mask)
 
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

            is_success = final_score < env.target_score
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
                pbar.update(1)
                continue
            
            # 截断回合时做 bootstrap
            next_value = 0.0
            if not episode_done:
                next_value = agent.estimate_value(state)
            
            # PPO 更新
            update_info = agent.update(next_value=next_value)
            loss = update_info['loss']

            # 打印动作分布
            agent.log_action_distribution(episode)
            
            # === Episode 级别记录 (核心) ===
            current_success_rate = np.mean(success_window) if success_window else 0.0
            avg_drop = np.mean(similarity_drop_window) if similarity_drop_window else 0.0
            episode_loc_invalid_ratio = (
                episode_loc_invalid_steps / episode_loc_total_steps
                if episode_loc_total_steps > 0 else 0.0
            )
            global_loc_invalid_ratio = (
                global_loc_invalid_steps / global_loc_total_steps
                if global_loc_total_steps > 0 else 0.0
            )

            logger.success(
                f"回合总结: 总奖={episode_reward:.2f} | 滑动成功率={current_success_rate:.2f} | 平均降分={avg_drop:.2f} "
                f"| loc_valid=False(本回合)={episode_loc_invalid_ratio:.2%} | loc_valid=False(全局)={global_loc_invalid_ratio:.2%} "
                f"| 步数={step+1} | 目标函数={target_func} | 目标二进制={last_binary_info['binary'] if last_binary_info else 'N/A'}"
            )
            logger.info(
                f"loc_valid统计: 本回合无效比例={episode_loc_invalid_ratio:.2%} | 全局无效比例={global_loc_invalid_ratio:.2%} "
                f"| 本回合统计步数={episode_loc_total_steps} | 全局统计步数={global_loc_total_steps}"
            )
            if episode % 50 == 0 and action_stats:
                parts = []
                for aid in sorted(action_stats.keys()):
                    stat = action_stats[aid]
                    if stat['count'] == 0:
                        continue
                    avg_r = stat['reward_sum'] / stat['count']
                    succ = stat['success'] / stat['count']
                    parts.append(f"a{aid}:cnt={stat['count']} avgR={avg_r:.3f} succ={succ:.1%}")
                if parts:
                    logger.info("action_stats: " + " | ".join(parts))
                for stat in action_stats.values():
                    stat['count'] = 0
                    stat['reward_sum'] = 0.0
                    stat['success'] = 0
            
            writer.add_scalar('Main/Success_Rate_MA50', current_success_rate, episode)      # 最近 50 个回合中，成功绕过检测（分数 < 0.4）的比例。
            writer.add_scalar('Main/Similarity_Drop_MA50', avg_drop, episode)               # 最近 50 个回合中，平均把相似度降低了多少（初始分 1.0 - 最终分）
            writer.add_scalar('Main/Episode_Reward', episode_reward, episode)               # Agent 在一个回合内拿到的所有奖励之和。
            writer.add_scalar('Main/Episode_Length', step + 1, episode)                      # 一个回合内总共执行了多少步。
            writer.add_scalar('Debug/Loc_Invalid_Ratio_Episode', episode_loc_invalid_ratio, episode)
            writer.add_scalar('Debug/Loc_Invalid_Ratio_Global', global_loc_invalid_ratio, episode)
            writer.add_histogram('Debug/Action_Distribution', np.array(episode_actions), episode)   # 在当前回合中，Agent 选择了哪些变异动作（Action 0-5）。
            writer.add_scalar('Debug/Policy_Loss', loss, episode)                           # PPO 算法更新时的 Loss 值。
            writer.add_scalar('Debug/Advantage_Mean_Raw', update_info['adv_mean_raw'], episode)
            writer.add_scalar('Debug/Advantage_Std_Raw', update_info['adv_std_raw'], episode)
            writer.add_scalar('Debug/Advantage_AbsMean_Raw', update_info['adv_abs_mean_raw'], episode)
            writer.add_scalar('Debug/Advantage_MaxAbs_Raw', update_info['adv_max_abs_raw'], episode)

            if targeted_evaluator is not None and (episode + 1) % args.targeted_eval_interval == 0:
                try:
                    agent.policy.eval()
                    t_metrics = targeted_evaluator.evaluate(agent, episode + 1)
                finally:
                    agent.policy.train()
                if t_metrics is not None:
                    writer.add_scalar('Targeted/Pairs_Valid', t_metrics['pairs_valid'], episode)
                    writer.add_scalar('Targeted/MinSim_Pre', t_metrics['min_pre'], episode)
                    writer.add_scalar('Targeted/MinSim_Post', t_metrics['min_post'], episode)
                    writer.add_scalar('Targeted/MinSim_Gain', t_metrics['gain_min'], episode)
                    writer.add_scalar('Targeted/AvgSim_Pre', t_metrics['avg_pre'], episode)
                    writer.add_scalar('Targeted/AvgSim_Post', t_metrics['avg_post'], episode)
                    writer.add_scalar('Targeted/AvgSim_Gain', t_metrics['gain_avg'], episode)
                    writer.add_scalar('Targeted/Success_Pre', t_metrics['success_pre'], episode)
                    writer.add_scalar('Targeted/Success_Post', t_metrics['success_post'], episode)
                    writer.add_scalar('Targeted/Improved_Rate', t_metrics['improved_rate'], episode)

            pbar.set_postfix_str(
                f"sr={current_success_rate:.2f} drop={avg_drop:.2f} "
                f"loss={loss:.2f} adv_std={update_info['adv_std_raw']:.2f}"
            )
            pbar.update(1)
            
            # 写日志文件
            # with open(log_file, 'a') as f:
            #     f.write(
            #         f"{episode},{step+1},{episode_reward:.4f},{loss:.4f},"
            #         f"{current_success_rate:.2f},{update_info['adv_std_raw']:.4f}\n"
            #     )
            
            # 保存模型
            if (episode + 1) % args.save_interval == 0:
                agent.save(
                    os.path.join(args.model_dir, f'ppo_model_ep{episode+1}.pt'),
                    extra_state=_make_trainer_state(episode),
                )
            
            if 'score' in info and info['score'] < best_score:
                best_score = info['score']
                agent.save(
                    os.path.join(args.model_dir, 'ppo_model_best.pt'),
                    extra_state=_make_trainer_state(episode),
                )

            # 定期清理
            if episode % 40 == 0:   
                cleanup_intermediate_files(args.save_path, episode_binaries)
    
    except KeyboardInterrupt:
        logger.warning("训练中断")
    
    finally:
        if 'pbar' in locals():
            pbar.close()
        last_episode = locals().get("episode", start_episode - 1)
        agent.save(
            os.path.join(args.model_dir, 'ppo_model_final.pt'),
            extra_state=_make_trainer_state(last_episode),
        )
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
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon', type=float, default=0.15)
    parser.add_argument('--n-locs', type=int, default=3)
    parser.add_argument('--episodes', type=int, default=6000)
    parser.add_argument('--max-steps', type=int, default=40)
    parser.add_argument('--save-interval', type=int, default=50)
    parser.add_argument('--sample-hold-interval', type=int, default=15)
    parser.add_argument('--stall-limit', type=int, default=8)
    parser.add_argument('--progress-eps', type=float, default=5e-4)
    parser.add_argument('--progress-reward-eps', type=float, default=2e-3)
    parser.add_argument('--include-schedule-feature', action='store_true')
    parser.add_argument('--non-strict-invalid-loc', action='store_true')
    parser.add_argument('--hold-min', type=int, default=4)
    parser.add_argument('--hold-max', type=int, default=10)
    parser.add_argument('--model-dir', default='./rl_models')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--use-gpu', action='store_true')
    parser.add_argument('--log-path', default=None, help='训练日志路径（默认 log/train.log）')
    parser.add_argument('--detection-method', choices=['asm2vec', 'safe', 'jtrans'], default='asm2vec')
    # SAFE 相关参数
    parser.add_argument('--safe-checkpoint-dir', default=None)
    parser.add_argument('--safe-i2v-dir', default=None)
    parser.add_argument('--safe-use-gpu', action='store_true')
    parser.add_argument('--no-safe-cache', action='store_true', help='Disable SAFE cache reuse during training')
    # jtrans
    parser.add_argument('--jtrans-model-dir', default=None)
    parser.add_argument('--jtrans-tokenizer-dir', default=None)
    parser.add_argument('--jtrans-use-gpu', action='store_true')
    parser.add_argument(
        '--feature-mode',
        choices=['full', 'no_progress', 'no_api', 'no_progress_api', 'no_section_c'],
        default='full'
    )
    parser.add_argument('--seed', type=int, default=None)
    # 定向攻击评测（Targeted Attack Eval）
    parser.add_argument('--targeted-eval-interval', type=int, default=0,
                        help='每隔多少个 episode 执行一次定向攻击评测（0 表示关闭）')
    parser.add_argument('--targeted-eval-dataset', default=None,
                        help='定向评测数据集路径（默认复用 --dataset）')
    parser.add_argument('--targeted-eval-pairs', type=int, default=8,
                        help='每次评测随机 attacker-target 配对数')
    parser.add_argument('--targeted-eval-max-target-variants', type=int, default=4,
                        help='每个 target identity 最多使用多少个编译变体')
    parser.add_argument('--targeted-eval-min-target-variants', type=int, default=2,
                        help='每个 target identity 至少需要多少个可用变体')
    parser.add_argument('--targeted-eval-max-steps', type=int, default=30,
                        help='定向评测每个 pair 的攻击步数预算')
    parser.add_argument('--targeted-eval-threshold', type=float, default=0.85,
                        help='定向成功阈值：min(sim_to_target_variants) >= threshold')
    parser.add_argument('--targeted-eval-seed', type=int, default=1234,
                        help='定向评测随机种子')
    parser.add_argument('--targeted-eval-explore', action='store_true',
                        help='定向评测时是否使用随机采样动作（默认贪心）')
    
    args = parser.parse_args()

    if args.detection_method == "jtrans":
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        jtrans_root = os.path.join(project_root, "detection_model", "jTrans")
        if args.jtrans_model_dir is None:
            args.jtrans_model_dir = os.path.join(jtrans_root, "models", "jTrans-finetune")
        if args.jtrans_tokenizer_dir is None:
            args.jtrans_tokenizer_dir = os.path.join(jtrans_root, "jtrans_tokenizer")
    
    # 清理旧日志
    log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'log/uroboro.log')
    if os.path.exists(log_path): os.remove(log_path)
        
    train_ppo(args)
