#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environment Wrapper for Binary Code Perturbation
二进制代码变异环境包装器（Python 3）

功能：
- 调用 uroboros (Python 2) 进行代码变异
- 使用 run_utils (Python 3) 进行相似度评估
- 提供标准 RL 环境接口
"""

import sys
import os
import json
import time
import subprocess
import numpy as np
import pickle
import hashlib
from loguru import logger
import random

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入现有模块
from run_utils import run_one
import run_objdump
from rl_framework.utils.acfg.r2_acfg_features import RadareACFGExtractor

class BinaryPerturbationEnv:
    """
    二进制代码变异环境 (Python 3)
    
    与 PPO Agent 在同一进程中运行，通过函数调用通信
    """
    
    def __init__(
        self,
        save_path,
        dataset_path,
        sample_hold_interval=3,
        max_steps=30,
        output_dir=None,
        detection_method="asm2vec",
        safe_checkpoint_dir=None,
        safe_i2v_dir=None,
        safe_use_gpu=False,
        safe_cache_enabled=False,
        safe_cache=None,
        jtrans_model_dir=None,
        jtrans_tokenizer_dir=None,
        jtrans_use_gpu=False,
        uniasm_root_dir=None,
        uniasm_model_path=None,
        uniasm_vocab_path=None,
        uniasm_use_gpu=False,
        feature_mode="full",
        seed=None,
        adaptive_hold=True,
        hold_min=None,
        hold_max=None,
        stall_limit=6,
        progress_eps=1e-4,
        progress_reward_eps=1e-3,
        include_schedule_feature=False,
        strict_invalid_loc=True,
        include_action12=True,
    ):
        """
        参数:
            original_binary: 原始二进制文件路径
            function_name: 目标函数名
            save_path: 保存变异结果的路径
        """
        self.save_path = os.path.abspath(save_path)
        # 项目根目录（uroboros 所在目录）
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # 用于存储当前二进制文件的 Top-K 关键块地址列表
        # 格式: [0x401000, 0x401050, 0x401090]
        self.current_critical_blocks = []
        # 当前函数的所有基本块地址（用于随机选非Top块）
        self.current_all_blocks = []
        # 加载数据集
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
            
        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)
        
        logger.info(f"已加载数据集: {len(self.dataset)} 个样本")
        
        # 切换策略控制
        self.sample_hold_interval = sample_hold_interval
        self.adaptive_hold = adaptive_hold
        self.hold_max = int(hold_max) if hold_max is not None else int(sample_hold_interval)
        if hold_min is None:
            self.hold_min = max(1, int(round(self.hold_max * 0.4)))
        else:
            self.hold_min = int(hold_min)
        self.hold_min = max(1, min(self.hold_min, self.hold_max))
        self.stall_limit = int(stall_limit)
        self.progress_eps = float(progress_eps)
        self.progress_reward_eps = float(progress_reward_eps)
        self.include_schedule_feature = bool(include_schedule_feature)
        self.strict_invalid_loc = bool(strict_invalid_loc)
        self.include_action12 = True
        self.current_hold_limit = self.hold_max
        self.episodes_on_current = 0
        self.current_sample_data = None # 存储当前样本的元数据
        
        # 当前环境状态变量
        self.original_binary = None # 原始二进制文件路径
        self.function_name = None # 目标函数名
        self.current_binary = None # 当前变异后的二进制文件路径
        self.original_func_addr = None
        
        # 不再需要加载模型（默认使用 asm2vec 方法）
        self.model_original = None
        self.detection_method = detection_method
        
        self.safe_checkpoint_dir = safe_checkpoint_dir
        self.safe_i2v_dir = safe_i2v_dir
        self.safe_use_gpu = safe_use_gpu
        if safe_cache is not None:
            self.safe_cache_enabled = True
            self.safe_cache = safe_cache
        else:
            self.safe_cache_enabled = bool(safe_cache_enabled)
            self.safe_cache = {} if self.safe_cache_enabled else None

        self.jtrans_model_dir = jtrans_model_dir
        self.jtrans_tokenizer_dir = jtrans_tokenizer_dir
        self.jtrans_use_gpu = jtrans_use_gpu
        self._jtrans_cache = {}
        self.uniasm_root_dir = uniasm_root_dir
        self.uniasm_model_path = uniasm_model_path
        self.uniasm_vocab_path = uniasm_vocab_path
        self.uniasm_use_gpu = uniasm_use_gpu
        if self.detection_method == "uniasm":
            default_root = os.path.join(self.project_root, "detection_model", "UniASM")
            if self.uniasm_root_dir is None:
                self.uniasm_root_dir = default_root
            if self.uniasm_model_path is None:
                self.uniasm_model_path = os.path.join(self.uniasm_root_dir, "data", "uniasm_base.h5")
            if self.uniasm_vocab_path is None:
                self.uniasm_vocab_path = os.path.join(self.uniasm_root_dir, "data", "vocab_base.txt")
        self._uniasm_cache = {}
        self.feature_mode = feature_mode
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        logger.info(f"Using {self.detection_method} detection method")
        
        # 变异历史
        self.mutation_history = []
        self.step_count = 0
        self.max_steps = max_steps
        self.target_score = 0.40
        self.state_dim = 256  # 默认状态维度（256维），可以通过参数修改
        # 动作空间（必须与 PPOAgent 的 action_map 保持一致）
        # 12/13/14/15/16 已接入；action 12 固定启用。
        self.all_action_ids = [1, 2, 4, 7, 8, 9, 11, 12, 13, 14, 15, 16]
        self.action_ids = list(self.all_action_ids)
        self.action_id_to_index = {aid: idx for idx, aid in enumerate(self.action_ids)}
        # 固定历史特征维度：保持 7 bins，避免改变 16 维历史特征布局（影响已实现网络切片）。
        # 新动作 13/14/15/16 会参与策略决策与执行，但不进入历史直方图统计。
        self.hist_action_ids = [1, 2, 4, 7, 8, 9, 11]
        self.hist_action_id_to_index = {aid: idx for idx, aid in enumerate(self.hist_action_ids)}
        self.hist_action_dim = len(self.hist_action_ids)
        self.n_actions = len(self.action_ids)
        # 奖励塑形超参：显式惩罚“无变化/无效位置”
        self.no_change_eps = 1e-4
        self.no_change_penalty = 0.1
        self.invalid_loc_penalty = 0.5
        # 分段惩罚参数（按“连续次数 + 训练阶段”调整）
        self.no_change_penalty_factors = (0.6, 1.0, 1.6, 2.2)   # 1, 2~3, 4~6, >=7
        self.invalid_loc_penalty_factors = (1.0, 1.5, 2.2)      # 1, 2~3, >=4
        self.time_penalty_schedule = (0.02, 0.05, 0.08)         # early, mid, late
        self.penalty_cap = 2.5
        # 连续失败计数（用于分段惩罚）
        self.no_change_streak = 0
        self.invalid_loc_streak = 0
        # 分数降幅奖励（越降越多）
        self.drop_bonus_scale = 3.0
        self.drop_bonus_cap = 2.0
        # 任何有效变异（分数下降）给予小正奖，避免长期负反馈
        self.effective_mutation_bonus = 0.2
        # 奖励尺度（更均衡，避免只靠终极奖励）
        self.incremental_scale = 24.0
        self.ultimate_base_reward = 20.0
        self.ultimate_quality_scale = 30.0
        self.ultimate_efficiency_scale = 0.3
        
        # 【性能优化】原始文件汇编缓存（原始文件不变，可复用）
        # 缓存键: (original_binary, function_name, ori_sym_addr)
        # 缓存值: 汇编文件路径
        self._original_asm_cache = {}
        
        # 【性能优化】复用临时目录，避免频繁创建删除
        # 在 save_path 下创建固定工作目录
        self._asm_work_dir = os.path.join(self.save_path, '_asm_work')
        os.makedirs(self._asm_work_dir, exist_ok=True)
        # 独立的变异输出目录（避免多进程/多实验互相污染）
        self.output_dir = os.path.abspath(output_dir) if output_dir else os.path.join(self.save_path, 'rl_output')
        os.makedirs(self.output_dir, exist_ok=True)

        # 奖励机制
        # 【关键】固定权重，不再动态调整
        self.reward_weights = {
            'incremental': 1.0,   # 基础权重
            'ultimate': 1.0,      # 成功奖励已经很大了
            'penalty': 1.0        # 惩罚权重
        }
        # 奖励裁剪统计（用于判断奖励是否经常触顶/触底）
        self._reward_clip_total = 0
        self._reward_clip_hi = 0
        self._reward_clip_lo = 0
        self.reward_clip_log_interval = 200

        # 记录每个样本的历史成功率
        # 格式: {sample_id: deque(maxlen=10)}
        from collections import deque
        self.sample_history = {}

        # 记录每个样本的权重 (用于采样)
        # 初始权重都为 1.0
        self.sample_weights = np.ones(len(self.dataset))

        # 建立索引映射 (index -> sample_id) 以便更新权重
        self.idx_to_id = {}
        for idx, item in enumerate(self.dataset):
            s_id = f"{item['binary_name']}::{item['func_name']}::{item['version']}"
            self.idx_to_id[idx] = s_id
            self.sample_history[s_id] = deque(maxlen=10) # 只看最近10次
        # 记录每个样本的历史最优与停滞计数（自适应切换）
        self.sample_best_score = {s_id: None for s_id in self.idx_to_id.values()}
        self.sample_no_progress = {s_id: 0 for s_id in self.idx_to_id.values()}

        self.current_sample_idx = 0 # 追踪当前样本在数据集中的索引
        self.current_difficulty = 0.5

        logger.info(
            f"Environment initialized (Hold Strategy: adaptive={self.adaptive_hold}, "
            f"hold_min={self.hold_min}, hold_max={self.hold_max}, stall_limit={self.stall_limit})"
        )

    def _find_sym_to_addr(self, binary_path):
        base_dir = os.path.dirname(os.path.abspath(binary_path))
        candidates = [
            os.path.join(base_dir, "sym_to_addr.pickle"),
            os.path.join(os.path.dirname(base_dir), "sym_to_addr.pickle"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _load_sym_to_addr(self, path):
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}

    def _resolve_original_addr(self):
        if self.current_sample_data and "func_addr" in self.current_sample_data:
            return self.current_sample_data.get("func_addr")
        return None
    
    def set_state_dim(self, state_dim):
        """
        设置状态维度（用于与 PPO Agent 保持一致）
        
        参数:
            state_dim: 状态维度
        """
        self.state_dim = state_dim
        logger.info(f"状态维度设置为: {state_dim}")


    def _resolve_mutated_address(self, binary_path):
        """
        核心辅助函数：解析变异后函数的真实地址
        解决 Strip 文件无法通过函数名定位的问题
        """
        # 1. 如果是原始文件，我们需要知道原始地址
        # 这里假设原始文件未 Strip，或者你能通过函数名找到
        # print(f"[_resolve_mutated_address]:Resolving address for binary: {binary_path}")
        if binary_path == self.original_binary:
            return None, self.function_name

        # 2. 寻找 sym_to_addr.pickle 映射文件
        mutated_dir = os.path.dirname(binary_path)
        pickle_path = os.path.join(mutated_dir, "sym_to_addr.pickle")
        
        # 有时候 uroboros 会生成在上一级
        if not os.path.exists(pickle_path):
            pickle_path = os.path.join(os.path.dirname(mutated_dir), "sym_to_addr.pickle")

        if not os.path.exists(pickle_path):
            # 如果找不到映射，只能返回 None，后续逻辑会尝试盲猜入口点
            # logger.warning(f"Map file missing for {binary_path}")
            return None, None

        try:
            with open(pickle_path, 'rb') as f:
                addr_map = pickle.load(f)

            # 尝试获取目标函数的地址
            # Uroboros 的 map key 可能是原始函数名
            if self.function_name in addr_map:
                addr_str = addr_map[self.function_name]
                if isinstance(addr_str, str):
                    return int(addr_str, 16), None
                return int(addr_str), None
            
            # 如果找不到直接匹配，尝试寻找 'func_xxxx' 格式
            # 这里简化处理，返回 None 让 r2 尝试 entry0
            return None, None
            
        except Exception as e:
            logger.error(f"Error resolving address: {e}")
            return None, None


    def extract_features_from_function(self, binary_path, function_name=None):
        """
        特征提取函数 (256维)
        组成: [历史特征(16)] + [ACFG特征(240)]
        用于特征提取时，指定函数名，而不是通过解析地址
        """
        features = []
        # ==========================================
        # Part 1: 变异历史与环境状态 (16维)
        # ==========================================
        # 1. Score History (5 dims)
        if self.mutation_history:
            scores = [m.get('score', 1.0) for m in self.mutation_history[-5:]]
            features.extend(scores)
            features.extend([1.0] * (5 - len(scores)))
        else:
            features.extend([1.0] * 5)
            
        # 2. Action Histogram (7 dims, action_id -> action_index)
        features.extend(self._action_histogram_features())
        
        # 3. Progress (2 dims)
        step_ratio = self.step_count / max(float(self.max_steps), 1.0)
        features.append(step_ratio)
        features.append(1.0 if step_ratio > 0.5 else 0.0)
        
        # 4. Global State (2 dims)
        features.append(1.0 if len(self.mutation_history) > 0 else 0.0) # Is Modified
        if self.include_schedule_feature:
            features.append(self.episodes_on_current / max(float(self.current_hold_limit), 1.0))
        else:
            features.append(0.0)

        # ==========================================
        # Part 2: 基于 Radare2 的 ACFG 特征 (核心)
        # ==========================================
        
        # 初始化默认向量 (全0) 用于失败情况
        acfg_vec = [0.0] * (self.state_dim - len(features))
        
        try:
            # 1. 检查地址
            if function_name is None:
                raise Exception("function_name is None, please specify function_name")
            target_name = function_name
            target_addr = None
            # print(f"binary_path: {binary_path}")
            # print(f"[env_wrapper.py:extract_features_from_function] target_name: {target_name}, target_addr: {target_addr}")
            # 2. 调用 R2 提取
            r2_ext = RadareACFGExtractor(binary_path)
            acfg_data = r2_ext.get_acfg_features(function_name=target_name, function_addr=target_addr)
            r2_ext.close()
            
            if acfg_data:
                # logger.debug(f"acfg_data: {acfg_data}")
                self.current_critical_blocks = self._select_top_blocks(acfg_data, k=3)
                self.current_all_blocks = list(acfg_data.get('basic_blocks', {}).keys())
                acfg_vec = self._vectorize_acfg(acfg_data)
            else:
                self.current_critical_blocks = []
                self.current_all_blocks = []
                
        except (FileNotFoundError, KeyError, ValueError, AttributeError) as e:
            logger.warning(f"Feature extraction failed for {binary_path}: {e}")
            self.current_critical_blocks = []
            self.current_all_blocks = []
            # 保持全0
        
        features.extend(acfg_vec)
        
        # 最终截断或补齐到 256 维
        if len(features) > self.state_dim:
            features = features[:self.state_dim]
        elif len(features) < self.state_dim:
            features.extend([0.0] * (self.state_dim - len(features)))

        # === 【核心修复】数据清洗 ===
        # 1. 转为 numpy 数组
        features = np.array(features, dtype=np.float32)
        
        # 2. 替换 NaN 为 0，替换 Infinity 为最大/最小有限值
        # 防止任何计算错误产生的 NaN 传入神经网络
        features = np.nan_to_num(features, nan=0.0, posinf=100.0, neginf=-100.0)

        features = self._apply_feature_mode(features)
        
        # 3. 裁剪数值范围 (Clip)
        # 防止某些特征数值过大（比如 total_instr 突然很大），导致梯度爆炸
        # 将所有特征限制在 [-10, 100] 之间通常足够了
        features = np.clip(features, -10.0, 100.0)
            
        return features

    def _action_histogram_features(self):
        """
        将历史动作(action_id)映射到稳定的索引空间(action_index)，避免把 action_id 当索引导致统计失真。
        返回长度为 7 的归一化直方图（固定布局，保证历史特征总维度不变）。
        """
        counts = np.zeros(self.hist_action_dim, dtype=np.float32)
        total = max(len(self.mutation_history), 1)

        for m in self.mutation_history:
            action_id = m.get('action')
            action_idx = self.hist_action_id_to_index.get(action_id)

            # 兼容旧数据：如果历史里存的是索引而不是 action_id，则回退为索引。
            if action_idx is None and isinstance(action_id, int) and 0 <= action_id < self.hist_action_dim:
                action_idx = action_id

            if action_idx is not None:
                counts[action_idx] += 1.0

        return (counts / total).tolist()

    def _select_top_blocks(self, acfg_data, k=3):
        """
        选择 Top-K 关键块并保持“重要性顺序”。
        - 优先使用提取器返回的 top_critical_blocks 顺序；
        - 若数量不足 K，按 critical_score/dominator/degree 兜底补齐。
        """
        try:
            k = max(1, int(k))
        except Exception:
            k = 3

        if not acfg_data:
            return []

        top_blocks = list(acfg_data.get('top_critical_blocks', []) or [])
        bbs = acfg_data.get('basic_blocks', {}) or {}

        ordered = []
        seen = set()
        for addr in top_blocks:
            if addr in seen:
                continue
            ordered.append(addr)
            seen.add(addr)
            if len(ordered) >= k:
                return ordered

        if isinstance(bbs, dict) and bbs:
            ranked = sorted(
                bbs.items(),
                key=lambda kv: (
                    -float(kv[1].get('critical_score', 0.0)),
                    -float(kv[1].get('dominator_score', 0.0)),
                    -float(kv[1].get('postdominator_score', 0.0)),
                    -float(kv[1].get('control_dependence_score', 0.0)),
                    -float(kv[1].get('loop_score', 0.0)),
                    -float(kv[1].get('centrality_degree', 0.0)),
                    kv[0],
                )
            )
            for addr, _ in ranked:
                if addr in seen:
                    continue
                ordered.append(addr)
                seen.add(addr)
                if len(ordered) >= k:
                    break

        return ordered[:k]

    def _apply_feature_mode(self, features):
        if self.feature_mode == "full":
            return features
        feats = features.copy()
        if self.state_dim < 256 or len(feats) < 16:
            return feats
        if self.feature_mode == "no_history":
            feats[0:16] = 0.0
        if self.feature_mode == "no_section_a":
            a_start = 16
            a_end = a_start + 40
            if a_end <= len(feats):
                feats[a_start:a_end] = 0.0
        if self.feature_mode == "no_section_b":
            b_start = 16 + 40
            b_end = b_start + 160
            if b_end <= len(feats):
                feats[b_start:b_end] = 0.0
        if self.feature_mode in ("no_progress", "no_progress_api"):
            feats[12:16] = 0.0
        if self.feature_mode in ("no_api", "no_progress_api"):
            c_start = 16 + 40 + 160
            api_start = c_start + 8
            api_end = c_start + 30
            if api_end <= len(feats):
                feats[api_start:api_end] = 0.0
        if self.feature_mode == "no_section_c":
            c_start = 16 + 40 + 160
            if c_start < len(feats):
                feats[c_start:] = 0.0
        return feats

    def extract_features(self, binary_path):
        """
        特征提取函数
        组成: [历史特征(16)] + [ACFG特征(240)]
        """
        features = []
        
        # ==========================================
        # Part 1: 变异历史与环境状态 (16维)
        # ==========================================
        # 1. Score History (5 dims)
        if self.mutation_history:
            scores = [m.get('score', 1.0) for m in self.mutation_history[-5:]]
            features.extend(scores)
            features.extend([1.0] * (5 - len(scores)))
        else:
            features.extend([1.0] * 5)
            
        # 2. Action Histogram (7 dims, action_id -> action_index)
        features.extend(self._action_histogram_features())
        
        # 3. Progress (2 dims)
        step_ratio = self.step_count / max(float(self.max_steps), 1.0)
        features.append(step_ratio)
        features.append(1.0 if step_ratio > 0.5 else 0.0)
        
        # 4. Global State (2 dims)
        features.append(1.0 if len(self.mutation_history) > 0 else 0.0) # Is Modified
        if self.include_schedule_feature:
            features.append(self.episodes_on_current / max(float(self.current_hold_limit), 1.0))
        else:
            features.append(0.0)

        # ==========================================
        # Part 2: 基于 Radare2 的 ACFG 特征 (核心)
        # ==========================================
        
        # 初始化默认向量 (全0) 用于失败情况
        acfg_vec = [0.0] * (self.state_dim - len(features))
        
        try:
            # 1. 解析地址
            # print("[extract_features]:binary_path:", binary_path)
            target_addr, target_name = self._resolve_mutated_address(binary_path)
            # print(f"target_addr: {target_addr}, target_name: {target_name}")
            # 2. 调用 R2 提取
            r2_ext = RadareACFGExtractor(binary_path)
            acfg_data = r2_ext.get_acfg_features(function_name=target_name, function_addr=target_addr)
            r2_ext.close()
            
            if acfg_data:
                # 保持重要性顺序，避免地址排序打乱 Top-1/2/3 的语义。
                self.current_critical_blocks = self._select_top_blocks(acfg_data, k=3)
                self.current_all_blocks = list(acfg_data.get('basic_blocks', {}).keys())
                acfg_vec = self._vectorize_acfg(acfg_data)
            else:
                self.current_critical_blocks = []
                self.current_all_blocks = []
                
                
        except (FileNotFoundError, KeyError, ValueError, AttributeError) as e:
            logger.warning(f"Feature extraction failed for {binary_path}: {e}")
            self.current_critical_blocks = []
            self.current_all_blocks = []
            # 保持全0
        
        features.extend(acfg_vec)
        
        # 最终截断或补齐
        if len(features) > self.state_dim:
            features = features[:self.state_dim]
        elif len(features) < self.state_dim:
            features.extend([0.0] * (self.state_dim - len(features)))

        # === 【核心修复】数据清洗 ===
        # 1. 转为 numpy 数组
        features = np.array(features, dtype=np.float32)
        
        # 2. 替换 NaN 为 0，替换 Infinity 为最大/最小有限值
        # 防止任何计算错误产生的 NaN 传入神经网络
        features = np.nan_to_num(features, nan=0.0, posinf=100.0, neginf=-100.0)

        features = self._apply_feature_mode(features)
        
        # 3. 裁剪数值范围 (Clip)
        # 防止某些特征数值过大（比如 total_instr 突然很大），导致梯度爆炸
        # 将所有特征限制在 [-10, 100] 之间通常足够了
        features = np.clip(features, -10.0, 100.0)
            
        return features
    

    def _vectorize_acfg(self, data, state_dim=256):
        """
        【终极修复版】
        Part 1 (16维): RL History (已在外部填充)
        Part 2 (40维): Section A - Macro Topology
        Part 3 (160维): Section B - Critical Semantics (Micro)
        Part 4 (40维): Section C - Global Semantics (Macro & Fingerprints)
        Total: 16 + 40 + 160 + 40 = 256 维
        """
        vec = []
        
        n_nodes = max(data.get('num_nodes', 0), 1.0)
        n_edges = data.get('num_edges', 0)
        complexity = data.get('cyclomatic_complexity', 0)
        bbs = list(data.get('basic_blocks', {}).values())
        # 保持提取器给出的重要性顺序（Top-1/2/3），并在缺失时做兜底补齐。
        stable_top_addrs = self._select_top_blocks(data, k=3)
        fingerprints = data.get('fingerprints', {})
        
        # === 【核心修正1】计算“有效指令总数” (Effective Total) ===
        # 我们不关心 MOV/PUSH/POP，只关心真正干活的指令
        # 这样分母在 O0/O3 之间会相对稳定
        effective_keys = ['n_arith', 'n_logic', 'n_branch', 'n_cmp', 'n_xor', 'n_shift', 'n_consts', 'n_call']
        
        total_effective_instr = 0
        for b in bbs:
            for k in effective_keys:
                total_effective_instr += b.get(k, 0)
        # 加上 API 调用 (这也是有效逻辑)
        total_effective_instr += fingerprints.get('n_calls', 0)
        
        safe_eff_total = max(total_effective_instr, 1.0)
        
        # 辅助函数：使用有效总数进行归一化
        def safe_div_eff(a): return a / safe_eff_total

        # ==========================================
        # Section A: Macro Topology (40 dims)
        # ==========================================
        # 1. Scale (8 dims)
        vec.append(np.log1p(n_nodes))
        vec.append(np.log1p(n_edges))
        vec.append(n_edges / n_nodes if n_nodes > 0 else 0) 
        vec.append(np.log1p(max(complexity, 0.0)))
        vec.append(complexity / n_nodes if n_nodes > 0 else 0)
        
        # Leaf / Branch nodes ratio
        leaf_cnt = sum(1 for b in bbs if b.get('n_branch', 0) == 0)
        branch_cnt = sum(1 for b in bbs if b.get('n_branch', 0) > 1)
        vec.append(leaf_cnt / n_nodes)
        vec.append(branch_cnt / n_nodes)

        # Effective Size (Log) -> 替代原来的 Total Size
        vec.append(np.log1p(total_effective_instr))

        # 2. Distributions (32 dims)
        # 【修正】统一使用 log1p 处理 Moments，防止数值爆炸
        def get_moments_log(values):
            if not values: return [0.0]*4
            arr = np.array(values)
            # 对原始值取 log1p 后再算矩，拉平量纲
            log_arr = np.log1p(np.maximum(arr, 0.0))
            return [np.mean(log_arr), np.max(log_arr), np.std(log_arr), np.median(log_arr)]

        dist_bet = [b.get('centrality_betweenness', 0) for b in bbs]
        dist_deg = [b.get('centrality_degree', 0) for b in bbs]
        dist_dom = [b.get('dominator_score', 0) for b in bbs]
        dist_eff_size = []

        for b in bbs:
            eff = sum(b.get(k, 0) for k in effective_keys)
            dist_eff_size.append(eff)

        for dist in [dist_bet, dist_deg, dist_dom, dist_eff_size]:
            vec.extend(get_moments_log(dist)) # 4 dims
            s_dist = sorted(dist, reverse=True)
            # Top-4 values
            top4 = s_dist[:4] + [0.0]*(4-len(s_dist))
            if dist is dist_dom or dist is dist_eff_size:
                vec.extend([np.log1p(x) for x in top4])
            else:
                vec.extend(top4)
        
        # Section A Total: 8 + 4*8 = 40. Correct.

        # ==========================================
        # Section B: Critical Semantics (160 dims)
        # ==========================================
        crit_vectors = []
        # print(f"top_critical_addrs: {top_critical_addrs}")
        # === 定义安全除法辅助函数 ===
        def safe_div(a, b):
            """
            安全除法：如果分母为0或非常小，返回0.0，否则返回 a/b。
            """
            # 使用一个极小值 epsilon (1e-9) 防止浮点数精度问题，或者直接判断 > 0
            return a / b if abs(b) > 1e-9 else 0.0
        # 定义有效指令集 (用于计算分母，剔除噪声)
        effective_keys = ['n_arith', 'n_logic', 'n_branch', 'n_cmp', 'n_xor', 'n_shift', 'n_consts']
        
        # 遍历 Top-3 关键块 (如果不足3个，循环会自动结束)
        for addr in stable_top_addrs:
            if addr not in data.get('basic_blocks', {}): continue
            bb = data['basic_blocks'][addr]
            
            # 计算该块的有效指令数 (分母)
            bb_eff = sum(bb.get(k, 0) for k in effective_keys)
            safe_bb_eff = max(bb_eff, 1.0)
            n_inst = max(bb.get('n_instructions', 0), 1.0) # 物理指令数
            
            v = []
            
            # --- [1] 规模与基础比率 (9 dims) ---
            v.append(np.log1p(bb_eff))                         # 0. Effective Size (Log)
            v.append(bb.get('n_arith', 0) / safe_bb_eff)       # 1. Arith Ratio
            v.append(bb.get('n_logic', 0) / safe_bb_eff)       # 2. Logic Ratio
            v.append(bb.get('n_branch', 0) / safe_bb_eff)      # 3. Branch Ratio
            v.append(bb.get('n_cmp', 0) / safe_bb_eff)         # 4. Cmp Ratio
            v.append(bb.get('n_xor', 0) / safe_bb_eff)         # 5. Xor Ratio (Crypto feature)
            v.append(bb.get('n_shift', 0) / safe_bb_eff)       # 6. Shift Ratio (Crypto feature)
            v.append(bb.get('n_consts', 0) / safe_bb_eff)      # 7. Constant Ratio
            v.append(bb.get('n_transfer', 0) / n_inst)         # 8. Transfer Density (搬运指令占比)

            # --- [2] 数据流与资源 (5 dims) ---
            v.append(safe_div(bb.get('n_regs_gp', 0), 16.0))   # 9. GP Reg Pressure
            v.append(safe_div(bb.get('n_regs_vec', 0), 16.0))  # 10. Vector Reg Pressure (SIMD)
            v.append(safe_div(bb.get('n_mem_write', 0), n_inst)) # 11. Mem Write Intensity
            v.append(safe_div(bb.get('n_mem_read', 0), n_inst))  # 12. Mem Read Intensity
            # 13. Compute/Mem Ratio (计算密集度)
            compute_ops = bb.get('n_arith', 0) + bb.get('n_logic', 0)
            mem_ops = bb.get('n_mem_write', 0) + bb.get('n_mem_read', 0)
            v.append(safe_div(compute_ops, mem_ops + 1.0))
            
            # --- [3] 拓扑与中心性 (6 dims) ---
            v.append(bb.get('centrality_betweenness', 0))         # 14. Betweenness
            v.append(bb.get('centrality_degree', 0))              # 15. Degree
            v.append(np.log1p(bb.get('dominator_score', 0)))      # 16. Dom Score
            v.append(np.log1p(bb.get('postdominator_score', 0)))  # 17. Post-Dom Score
            v.append(np.log1p(bb.get('control_dependence_score', 0)))  # 18. CDG Score
            v.append(bb.get('critical_score', 0))                 # 19. Aggregated Score
            
            # --- [4] 结构标志位 (3 dims) ---
            v.append(1.0 if bb.get('n_consts', 0) > 0 else 0.0)      # 20. Has Constant?
            v.append(1.0 if bb.get('n_branch', 0) > 1 else 0.0)      # 21. Is Multi-Branch?
            v.append(1.0 if compute_ops > mem_ops else 0.0)          # 22. Is Compute Heavy?
            
            # --- [5] 高级组合特征 (9 dims, 填满32) ---
            # 23. Entropy Proxy (操作码种类丰富度)
            uniq_types = sum(1 for k in effective_keys if bb.get(k, 0) > 0)
            v.append(uniq_types / 7.0)
            
            # 24. Stack Heaviness (是否主要是栈操作)
            v.append(1.0 if bb.get('n_transfer', 0) > n_inst * 0.5 else 0.0)
            
            # 25. Loop Header Heuristic (有跳转且在环中)
            is_loop = 1.0 if (
                bb.get('n_branch', 0) > 0 and (
                    bb.get('is_in_loop', 0.0) > 0.5 or bb.get('loop_score', 0) > 1
                )
            ) else 0.0
            v.append(is_loop)
            
            # 26. Logic+Xor Density (混淆常见特征)
            v.append(safe_div(bb.get('n_logic', 0) + bb.get('n_xor', 0), safe_bb_eff))
            
            # 27. Write/Read Ratio (写多读少可能是初始化)
            v.append(safe_div(bb.get('n_mem_write', 0), bb.get('n_mem_read', 0) + 1.0))
            
            # 28. Arith/Logic Ratio
            v.append(safe_div(bb.get('n_arith', 0), bb.get('n_logic', 0) + 1.0))
            
            # 29. Branch/Compute Ratio (控制流密集度)
            v.append(safe_div(bb.get('n_branch', 0), compute_ops + 1.0))
            
            # 30. Reg Diversity (通用+向量寄存器总数归一化)
            v.append(safe_div(bb.get('n_regs_gp', 0) + bb.get('n_regs_vec', 0), 16.0))
            
            # 31. Tiny Block Flag (是否极小块，如Trampoline)
            v.append(1.0 if n_inst < 5 else 0.0)

            # 确保长度为 32
            v = v[:32] 
            crit_vectors.append(v)

        # === 扁平化填充 (Top-1, Top-2, Top-3) ===
        # 占用 32 * 3 = 96 维
        for i in range(3):
            if i < len(crit_vectors):
                vec.extend(crit_vectors[i])
            else:
                # 如果没有这个块（比如函数很小），补 0
                vec.extend([0.0] * 32)
        
        # === 全局聚合上下文 (Context) ===
        # 占用 32 * 2 = 64 维
        if crit_vectors:
            mat = np.array(crit_vectors)
            vec.extend(np.mean(mat, axis=0)) # Global Mean of Critical Areas
            vec.extend(np.max(mat, axis=0))  # Global Max of Critical Areas
        else:
            vec.extend([0.0] * 64)

        # ==========================================
        # === Section C: Global Semantics (40 dims) ===
        # ==========================================
        
        # 1. Global Logic Ratios (8 dims)
        # 只统计逻辑指令，忽略数据搬运
        global_sums = {k: sum(b.get(k, 0) for b in bbs) for k in effective_keys}
        for k in effective_keys:
            vec.append(global_sums[k] / safe_eff_total)
        # 补 1 维：全局环块比例 (替代占位常量)
        loop_block_ratio = (
            sum(1 for b in bbs if float(b.get('is_in_loop', 0.0)) > 0.5) / n_nodes
            if bbs else 0.0
        )
        vec.append(loop_block_ratio)
            
        # 2. API & Strings (22 dims)
        vec.append(np.log1p(fingerprints.get('n_calls', 0)))
        vec.append(np.log1p(fingerprints.get('n_strings', 0)))
        # 相对于有效指令的密度
        vec.append(fingerprints.get('n_calls', 0) / safe_eff_total)
        vec.append(fingerprints.get('n_strings', 0) / safe_eff_total)
        
        api_cats = ['io', 'mem', 'str', 'sys', 'net', 'crypto', 'error', 'other', 'internal']
        apis = fingerprints.get('api_types', set())
        # 9 dims: API 类别 flags
        for cat in api_cats:
            vec.append(1.0 if cat in apis else 0.0)
        # 9 dims: 原本恒定 0 -> 替换为真实统计
        n_ops_imm = fingerprints.get('n_ops_imm', 0)
        n_ops_reg = fingerprints.get('n_ops_reg', 0)
        n_ops_mem = fingerprints.get('n_ops_mem', 0)
        total_ops = max(n_ops_imm + n_ops_reg + n_ops_mem, 1.0)
        op_imm_ratio = n_ops_imm / total_ops
        op_reg_ratio = n_ops_reg / total_ops
        op_mem_ratio = n_ops_mem / total_ops

        consts = fingerprints.get('consts', [])
        if consts:
            abs_consts = [abs(v) for v in consts]
            log_abs = [np.log1p(v) for v in abs_consts]
            mean_abs_log = float(np.mean(log_abs))
            max_abs_log = float(np.max(log_abs))
            pos_cnt = sum(1 for v in consts if v > 0)
            neg_cnt = sum(1 for v in consts if v < 0)
            const_pos_ratio = pos_cnt / (neg_cnt + 1.0)
            const_unique_ratio = len(set(consts)) / (len(consts) + 1.0)
            const_small_ratio = sum(1 for v in abs_consts if v <= 255) / (len(consts) + 1.0)
            n_consts_log = float(np.log1p(len(consts)))
        else:
            mean_abs_log = 0.0
            max_abs_log = 0.0
            const_pos_ratio = 0.0
            const_unique_ratio = 0.0
            const_small_ratio = 0.0
            n_consts_log = 0.0

        vec.extend([
            op_imm_ratio,
            op_reg_ratio,
            op_mem_ratio,
            n_consts_log,
            mean_abs_log,
            max_abs_log,
            const_pos_ratio,
            const_unique_ratio,
            const_small_ratio,
        ])
            
        # 3. Block Size Dist (5 dims) - 基于 Effective Size
        sizes = [sum(b.get(k, 0) for k in effective_keys) for b in bbs]
        if sizes:
            vec.append(sum(1 for s in sizes if s < 2) / len(sizes)) # Tiny logic
            vec.append(sum(1 for s in sizes if 2 <= s < 10) / len(sizes)) 
            vec.append(sum(1 for s in sizes if 10 <= s < 30) / len(sizes))
            vec.append(sum(1 for s in sizes if s >= 30) / len(sizes))
            vec.append(np.max(sizes) / safe_eff_total)
        else:
            vec.extend([0.0] * 5)
            
        # 4. Global Logic Ratios (Logic/Arith etc.) (5 dims)
        vec.append(global_sums['n_arith'] / (global_sums['n_logic'] + 1.0))
        vec.append(global_sums['n_branch'] / (global_sums['n_arith'] + 1.0))
        vec.append(global_sums['n_consts'] / safe_eff_total)
        vec.append(1.0 if global_sums['n_consts'] > 3 else 0.0)
        vec.append(1.0 if global_sums['n_shift'] > 0 else 0.0) # Crypto hint
        
        # Section C Total: 8 + 22 + 5 + 5 = 40 dims
        # ACFG Total: 40 + 160 + 40 = 240 dims
        
        return vec


    
    def apply_mutation(self, seed_binary, action, target_addr):
        """
        应用变异操作
        
        参数:
            seed_binary: 种子二进制文件路径
            action: 变异模式 (1,2,4,7,8,9,11,13,14,15,16)
            target_addr: 攻击位置地址
        返回:
            mutated_binary: 变异后的二进制文件路径
        """
        try:
            if self.function_name == 'main':
                logger.warning("Skip mutation for function 'main' (avoid unstable target)")
                return None, None
            # 记录调用时的 step_count，用于调试
            current_step = getattr(self, 'step_count', 'unknown')
            logger.info("Applying mutation {} to {} (step={})".format(action, os.path.basename(seed_binary), current_step))
            
            # 生成临时二进制文件名
            tmp_bin = os.path.join(self.output_dir, 'mutant_' + str(int(time.time() * 1000)) + '.bin')
            
            # 确定模式
            fmode = 'mutated' if seed_binary != self.original_binary else 'original'
            
            # 为每次变异创建独立的临时目录
            tmp_id = str(int(time.time() * 1000))
            tmp_dir = os.path.join(self.save_path, 'tmp_' + tmp_id)
            os.makedirs(tmp_dir, exist_ok=True)
            
            # 调用 uroboros (Python 2) 进行变异
            # 注意：这里必须使用 python2，因为 uroboros 是 Python 2 代码
            cmd = [
                'python2',
                os.path.join(self.project_root, 'uroboros_automate-func-name.py'),
                seed_binary,
                '-i', '1',  # 迭代次数
                '-o', tmp_bin,
                '-d', str(action),
                '-m', fmode,
                '-f', tmp_dir,
                '--function', self.function_name
            ]
            
            if target_addr is not None:
                # 确保转为 hex 字符串
                hex_addr = hex(target_addr) if isinstance(target_addr, int) else str(target_addr)
                cmd.extend(['--target_addr', hex_addr])
                # logger.debug(f"🎯 Targeting specific block: {hex_addr}")

            logger.debug("Command: " + " ".join(cmd))
            
            # 在项目根目录执行命令
            try:
                # 可选：限制 uroboros(Python2) 子进程内存，避免 OOM 把整次 ASR 跑挂。
                # 默认不启用（保持用户可见行为不变），需要显式设置环境变量：
                #   UROBOROS_MEM_MB=4096  (单位 MB)
                # 同时可设置超时：
                #   UROBOROS_TIMEOUT_SEC=600
                uroboros_mem_mb = None
                uroboros_timeout = None
                try:
                    uroboros_mem_mb = int(os.environ.get("UROBOROS_MEM_MB", "0"))
                except Exception:
                    uroboros_mem_mb = 0
                try:
                    uroboros_timeout = int(os.environ.get("UROBOROS_TIMEOUT_SEC", "0"))
                except Exception:
                    uroboros_timeout = 0

                def _limit_child_resources():
                    if not uroboros_mem_mb or uroboros_mem_mb <= 0:
                        return
                    try:
                        import resource
                        limit_bytes = int(uroboros_mem_mb) * 1024 * 1024
                        # RLIMIT_AS: 虚拟内存上限（比 RSS 更早拦住暴涨）
                        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
                    except Exception:
                        # 限制失败就算了，别因为防护逻辑反而打断正常流程
                        return

                output = subprocess.check_output(
                    cmd, 
                    stderr=subprocess.STDOUT,
                    cwd=self.project_root,
                    universal_newlines=True,  # 返回字符串而不是字节
                    timeout=(uroboros_timeout if uroboros_timeout and uroboros_timeout > 0 else None),
                    preexec_fn=_limit_child_resources,
                )
                logger.debug("Uroboros output: {}".format(output))
            except subprocess.TimeoutExpired as e:
                logger.error(
                    "Uroboros timeout after {}s (action={}, step={})".format(
                        uroboros_timeout, action, current_step
                    )
                )
                raise Exception("Uroboros mutation timeout: {}".format(e))
            except subprocess.CalledProcessError as e:
                # 捕获详细的错误信息
                error_output = e.output if hasattr(e, 'output') else str(e)
                logger.error("Uroboros command failed with exit code {}: {}".format(e.returncode, error_output))
                logger.error("Command was: {}".format(" ".join(cmd)))
                raise Exception("Uroboros mutation failed: {}\nOutput: {}".format(e, error_output))
            
            # 验证输出文件是否存在
            if not os.path.exists(tmp_bin):
                raise FileNotFoundError("Mutation output not found: {}".format(tmp_bin))
            
            # 计算 hash 并移动文件
            h = hashlib.md5(open(tmp_bin, 'rb').read()).hexdigest()
            container_path = os.path.join(self.save_path, h + '_container')
            
            if not os.path.exists(container_path):
                # 移动 tmp 目录到 container
                import shutil
                shutil.move(tmp_dir, container_path)
                # 移动二进制文件
                shutil.move(tmp_bin, os.path.join(container_path, h))
            
            mutated_binary = os.path.join(container_path, h)
            logger.info("Mutation successful: {}".format(mutated_binary))
            
            return mutated_binary, h
            
        except Exception as e:
            logger.error("Mutation failed: {}".format(e))
            return None, None
    
    def evaluate(self, mutated_binary, checkdict):
        """
        评估变异后的二进制文件
        
        返回:
            score: 相似度分数
            grad: 梯度值
        """
        try:
            # 【性能优化】传递工作目录和缓存，避免每次创建临时目录
            mutated_func_addr = None
            if self.detection_method == "safe":
                if self.original_func_addr is None:
                    self.original_func_addr = self._resolve_original_addr()
                # print(f"[evaluate]:mutated_binary: {mutated_binary}")
                mutated_func_addr, _ = self._resolve_mutated_address(mutated_binary)
                # print(f"[evaluate]:mutated_func_addr: {mutated_func_addr}")
            score, grad = run_one(
                self.original_binary,
                mutated_binary,
                self.model_original,
                checkdict,
                self.function_name,
                detection_method=self.detection_method,
                asm_work_dir=self._asm_work_dir,
                original_asm_cache=self._original_asm_cache,
                original_func_addr=self.original_func_addr,
                mutated_func_addr=mutated_func_addr,
                safe_checkpoint_dir=self.safe_checkpoint_dir,
                safe_i2v_dir=self.safe_i2v_dir,
                safe_use_gpu=self.safe_use_gpu,
                safe_cache=self.safe_cache,
                jtrans_model_dir=self.jtrans_model_dir,
                jtrans_tokenizer_dir=self.jtrans_tokenizer_dir,
                jtrans_use_gpu=self.jtrans_use_gpu,
                jtrans_cache=self._jtrans_cache,
                uniasm_root_dir=self.uniasm_root_dir,
                uniasm_model_path=self.uniasm_model_path,
                uniasm_vocab_path=self.uniasm_vocab_path,
                uniasm_use_gpu=self.uniasm_use_gpu,
                uniasm_cache=self._uniasm_cache,
            )
            if self.detection_method == "safe":
                logger.success(f"[SAFE] eval_score={score}")
            elif self.detection_method == "jtrans":
                logger.success(f"[JTRANS] eval_score={score}")
            elif self.detection_method == "uniasm":
                logger.success(f"[UNIASM] eval_score={score}")
            else:
                logger.debug(f"[ASM2VEC] eval_score={score}")
            if score is None or grad is None:
                logger.warning("Evaluation returned None")
                return 1.0, 0.0  # 默认最差值
            
            return abs(score), abs(grad)
            
        except Exception as e:
            logger.error("Evaluation failed: {}".format(e))
            return 1.0, 0.0
    
    def step(self, action, loc_idx):
        """
        执行一步环境交互
        
        参数:
            action: 变异模式
            loc_idx: 攻击位置索引
        
        返回:
            state: 新状态特征
            reward: 奖励
            done: 是否完成
            info: 额外信息
        """
        # input("Press Enter to continue step...")
        self.step_count += 1
        # 记录上一步分数，用于计算差分奖励
        prev_score = self.mutation_history[-1]['score'] if self.mutation_history else 1.0
        # === 【核心逻辑】解析攻击位置 ===
        target_addr = None

        # 位置选择：
        # - 有关键块时，loc_idx 必须命中 Top-N
        # - 无关键块时，退回任意块避免环境僵死
        loc_valid = False
        if self.current_critical_blocks:
            if 0 <= loc_idx < len(self.current_critical_blocks):
                target_addr = self.current_critical_blocks[loc_idx]
                loc_valid = True
            elif not self.strict_invalid_loc and self.current_all_blocks:
                # 可选兼容模式：无效 loc_idx 时随机兜底
                target_addr = random.choice(self.current_all_blocks)
                loc_valid = True
        else:
            if self.current_all_blocks:
                target_addr = random.choice(self.current_all_blocks)
                loc_valid = True

        if not loc_valid and self.current_critical_blocks and self.strict_invalid_loc:
            # 严格模式：位置越界直接惩罚并返回，不再用随机兜底掩盖错误动作。
            logger.debug(
                f"Invalid loc_idx={loc_idx} for top_blocks={len(self.current_critical_blocks)}; strict penalty."
            )
            self.invalid_loc_streak += 1
            self.no_change_streak += 1
            state = self.extract_features(self.current_binary)
            reward = self.compute_reward_v2(
                prev_score,
                prev_score,
                self.step_count,
                invalid_loc=True,
                no_change=True,
                no_change_streak=self.no_change_streak,
                invalid_streak=self.invalid_loc_streak,
            )
            done = self.step_count >= self.max_steps
            info = {
                'score': prev_score,
                'grad': 0.0,
                'step': self.step_count,
                'binary': self.current_binary,
                'target_func': self.function_name,
                'loc_valid': False,
                'no_change': True,
                'no_change_streak': self.no_change_streak,
                'invalid_loc_streak': self.invalid_loc_streak,
                'score_delta': 0.0,
                'error': 'invalid_loc',
            }
            return state, reward, done, info

        # 应用变异
        mutated_binary, hash_val = self.apply_mutation(self.current_binary, action, target_addr)
        logger.info(f"[step] apply_mutation returned, step={self.step_count}")
        # input("press enter to continue")
        if mutated_binary is None:
            # 变异失败：标记需要重置环境并切换文件
            logger.warning("Mutation failed, will reset environment and switch to new file")
            state = self.extract_features(self.current_binary)
            return state, -10.0, True, {
                'error': 'mutation_failed',
                'should_reset': True,  # 标志：需要重置并切换文件
                'score': 1.0,  # 默认最差分数
                'grad': 0.0
            }
        
        # 评估
        # TODO: 获取正确的 checkdict
        checkdict = {}  # 需要从实际文件中加载
        score, grad = self.evaluate(mutated_binary, checkdict)
        
        # 更新状态
        self.current_binary = mutated_binary
        self.mutation_history.append({
            'step': self.step_count,
            'action': action,
            'binary': mutated_binary,
            'hash': hash_val,
            'score': score,
            'grad': grad
        })
        
        # 提取新状态特征
        state = self.extract_features(mutated_binary)
        
        # 计算奖励
        score_delta = prev_score - score
        no_change = abs(score_delta) <= self.no_change_eps
        if no_change:
            self.no_change_streak += 1
        else:
            self.no_change_streak = 0
        if loc_valid:
            self.invalid_loc_streak = 0
        else:
            self.invalid_loc_streak += 1
        reward = self.compute_reward_v2(
            prev_score,
            score,
            self.step_count,
            invalid_loc=not loc_valid,
            no_change=no_change,
            no_change_streak=self.no_change_streak,
            invalid_streak=self.invalid_loc_streak,
        )
        # reward = self.compute_reward(score, grad)
        
        # 判断是否完成
        done = score < self.target_score or self.step_count >= self.max_steps

        info = {
            'score': score,
            'grad': grad,
            'step': self.step_count,
            'binary': mutated_binary,
            'target_func': self.function_name, # 记录当前目标函数名
            'loc_valid': loc_valid,
            'no_change': no_change,
            'no_change_streak': self.no_change_streak,
            'invalid_loc_streak': self.invalid_loc_streak,
            'score_delta': score_delta,
        }
        
        # === 核心修改：更新难度权重 ===
        if done:
            s_id = self.idx_to_id[self.current_sample_idx]
            final_score = info.get('score', 1.0)
            is_success = 1 if final_score < self.target_score else 0
            
            # 1. 更新历史记录
            self.sample_history[s_id].append(is_success)
            
            # 2. 动态更新采样权重
            # 如果失败了(0)，难度增加，权重增加
            # 如果成功了(1)，难度降低，权重降低
            # 公式：Weight = 1.0 + (1.0 - SuccessRate) * 2.0
            # 最难样本权重 = 3.0，最易样本权重 = 1.0
            
            history = self.sample_history[s_id]
            current_rate = sum(history) / len(history)
            new_weight = 1.0 + (1.0 - current_rate) * 2.0
            
            # 更新到全局权重数组
            self.sample_weights[self.current_sample_idx] = new_weight

            # 3. 记录进步/停滞（用于自适应切换）
            best_score = self.sample_best_score.get(s_id)
            if is_success or best_score is None or final_score < (best_score - self.progress_eps):
                self.sample_best_score[s_id] = final_score
                self.sample_no_progress[s_id] = 0
            else:
                self.sample_no_progress[s_id] = self.sample_no_progress.get(s_id, 0) + 1

        
        return state, reward, done, info




    def compute_reward_v2(self, prev_score, current_score, step_count,
                          invalid_loc=False, no_change=False,
                          no_change_streak=0, invalid_streak=0):
        """计算奖励"""
        # 基础奖励
        total_reward = 0.0
        # === 1. 基础进步奖励 ===
        diff = prev_score - current_score
       
        if diff > self.progress_reward_eps:  # 显著进步
            # 固定权重：简化奖励，避免分段重复放大
            incremental = diff * self.incremental_scale
            # 额外降幅奖励：按“相对降幅”给 bonus，保证降得越多奖励越多
            rel_drop = diff / max(prev_score, 1e-6)
            drop_bonus = min(self.drop_bonus_cap, rel_drop * self.drop_bonus_scale)
            total_reward += drop_bonus
            total_reward += self.effective_mutation_bonus
        elif diff < -self.progress_reward_eps:  # 显著退步
            # 对称处理，避免过度惩罚导致探索受限
            incremental = diff * self.incremental_scale
        else:
            # 变化幅度在噪声带内，当作无进展处理，避免学到“抖动投机”。
            incremental = 0.0
            no_change = True

        # 关键决策：难度已经用于“采样更频繁”，这里不再二次放大奖励，
        # 避免对“难样本”重复加成导致训练不稳定。
        total_reward += incremental * self.reward_weights['incremental']

        # === 2. 终极成功奖励 ===
        if current_score < self.target_score:
            # 基础 10 + 质量加成 + 效率加成
            base_reward = self.ultimate_base_reward
            quality_bonus = (self.target_score - current_score) * self.ultimate_quality_scale
            efficiency_bonus = max(0, (self.max_steps - step_count) * self.ultimate_efficiency_scale)      # max_step - step_count
            
            ultimate = base_reward + quality_bonus + efficiency_bonus

            total_reward += ultimate * self.reward_weights['ultimate']

        # === 3. 显式惩罚（分段式）===
        penalty = 0.0
        step_ratio = step_count / max(float(self.max_steps), 1.0)
        if step_ratio < 0.33:
            time_penalty = self.time_penalty_schedule[0]
        elif step_ratio < 0.66:
            time_penalty = self.time_penalty_schedule[1]
        else:
            time_penalty = self.time_penalty_schedule[2]

        if no_change:
            s = max(int(no_change_streak), 1)
            if s <= 1:
                factor = self.no_change_penalty_factors[0]
            elif s <= 3:
                factor = self.no_change_penalty_factors[1]
            elif s <= 6:
                factor = self.no_change_penalty_factors[2]
            else:
                factor = self.no_change_penalty_factors[3]
            penalty += self.no_change_penalty * factor

        if invalid_loc:
            s = max(int(invalid_streak), 1)
            if s <= 1:
                factor = self.invalid_loc_penalty_factors[0]
            elif s <= 3:
                factor = self.invalid_loc_penalty_factors[1]
            else:
                factor = self.invalid_loc_penalty_factors[2]
            penalty += self.invalid_loc_penalty * factor

        penalty += time_penalty
        penalty = min(penalty, self.penalty_cap)


        total_reward -= penalty * self.reward_weights['penalty']
        
        # === 5. 硬裁剪（不是归一化！）===
        # 只是防止极端值，不改变奖励的相对大小关系
        clipped = float(np.clip(total_reward, -50.0, 100.0))
        self._record_reward_clip(total_reward, clipped)
        return clipped

    def _record_reward_clip(self, raw_reward, clipped_reward):
        self._reward_clip_total += 1
        if raw_reward > 80.0:
            self._reward_clip_hi += 1
        elif raw_reward < -40.0:
            self._reward_clip_lo += 1

        if self.reward_clip_log_interval > 0 and self._reward_clip_total % self.reward_clip_log_interval == 0:
            total = self._reward_clip_total
            hi_ratio = self._reward_clip_hi / total
            lo_ratio = self._reward_clip_lo / total
            logger.info(
                f"[reward_clip] total={total} hi={self._reward_clip_hi} ({hi_ratio:.1%}) "
                f"lo={self._reward_clip_lo} ({lo_ratio:.1%})"
            )

    def _refresh_current_difficulty(self):
        if self.current_sample_data is None:
            self.current_difficulty = 0.5
            return
        s_id = self.idx_to_id[self.current_sample_idx]
        history = self.sample_history.get(s_id, [])
        if len(history) > 0:
            success_rate = sum(history) / len(history)
            self.current_difficulty = 1.0 - success_rate
            self.current_difficulty = max(0.1, self.current_difficulty)
        else:
            self.current_difficulty = 0.5

    def _adaptive_hold_limit(self):
        if not self.adaptive_hold:
            return max(1, int(self.sample_hold_interval))
        hold = int(round(self.hold_min + self.current_difficulty * (self.hold_max - self.hold_min)))
        return max(self.hold_min, min(self.hold_max, hold))

    def _switch_next_target(self):
        """
        [辅助函数] 根据难度权重采样下一个目标
        """
        # 权重归一化成概率
        probs = self.sample_weights / np.sum(self.sample_weights)
        # 按概率抽取索引
        self.current_sample_idx = np.random.choice(len(self.dataset), p=probs)
        self.current_sample_data = self.dataset[self.current_sample_idx]

        self.episodes_on_current = 0
        # Reset cached function address when switching targets.
        self.original_func_addr = None

        # 计算当前难度 (1.0 - 成功率)
        self._refresh_current_difficulty()
        self.current_hold_limit = self._adaptive_hold_limit()

        self.original_binary = self.current_sample_data['binary_path']
        self.function_name = self.current_sample_data['func_name']

    def reset(self, force_switch=False):
        """
        重置环境：实现自动切换目标 (Hold-N Strategy)
        
        参数:
            force_switch: 如果为 True，强制切换目标（用于错误恢复）
        """
        # 强制切换（错误恢复）：忽略 Hold-N 策略，直接切换目标
        if force_switch:
            self._switch_next_target()
            logger.warning(f"🔄 FORCE SWITCH (Error Recovery) -> {os.path.basename(self.original_binary)}::{self.function_name}")
            logger.warning(f"   Version: {self.current_sample_data.get('version')} | Opt: {self.current_sample_data.get('opt_level')}")
        # 正常切换：检查是否需要切换目标
        elif self.current_sample_data is None:
            self._switch_next_target()
            logger.success(f"🔄 SWITCH TARGET -> {os.path.basename(self.original_binary)}::{self.function_name}")
            logger.success(f"   Version: {self.current_sample_data.get('version')} | Opt: {self.current_sample_data.get('opt_level')}")
        else:
            # 更新当前难度与自适应 hold 限制
            self._refresh_current_difficulty()
            self.current_hold_limit = self._adaptive_hold_limit()
            s_id = self.idx_to_id[self.current_sample_idx]
            stall_count = self.sample_no_progress.get(s_id, 0)
            should_switch = (
                self.episodes_on_current >= self.current_hold_limit or
                stall_count >= self.stall_limit
            )
            if should_switch:
                self._switch_next_target()
                reason = "stall" if stall_count >= self.stall_limit else "hold"
                logger.success(
                    f"🔄 SWITCH TARGET ({reason}) -> {os.path.basename(self.original_binary)}::{self.function_name} "
                    f"(hold={self.current_hold_limit}, stall={stall_count})"
                )
                logger.success(f"   Version: {self.current_sample_data.get('version')} | Opt: {self.current_sample_data.get('opt_level')}")
            else:
                # 保持当前目标，增加计数
                self.episodes_on_current += 1
                logger.info(
                    f"🔄 KEEP TARGET ({self.episodes_on_current}/{self.current_hold_limit}) -> {self.function_name} "
                    f"(stall={stall_count})"
                )

        # 重置环境状态
        self.current_binary = self.original_binary
        self.mutation_history = []
        self.step_count = 0
        self.no_change_streak = 0
        self.invalid_loc_streak = 0
        
        # 提取初始特征
        state = self.extract_features(self.original_binary)
        return state

    def get_loc_mask(self, n_locs=3):
        """
        返回位置掩码（长度为 n_locs）。
        1 表示该位置可用（有对应关键块），0 表示不可用。
        若当前关键块为空，返回全 1（不做硬屏蔽，避免全零导致无法采样）。
        """
        try:
            n_locs = int(n_locs)
        except Exception:
            n_locs = 3
        if n_locs <= 0:
            return []
        if not self.current_critical_blocks:
            return [1] * n_locs
        valid = min(len(self.current_critical_blocks), n_locs)
        return [1] * valid + [0] * (n_locs - valid)
    
    def clear_acfg_cache(self):
        """
        清理 ACFG 特征缓存
        
        用于释放内存，通常在切换大量不同目标时调用
        """
        logger.info("ACFG 缓存已禁用，无需清理")
    
    def get_cache_stats(self):
        """
        获取缓存统计信息
        
        返回:
            dict: 包含命中率、命中数、未命中数等统计信息
        """
        return {
            'cache_size': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'hit_rate': 0.0
        }

if __name__ == '__main__':
    bin_path = '/home/ycy/ours/Deceiving-DNN-based-Binary-Matching/rl_framework/datasets/coreutils/bin/coreutils-8.15-O0/sort'
    DATASET_PATH = '/home/ycy/ours/Deceiving-DNN-based-Binary-Matching/rl_framework/utils/dataset_test.json'
    env = BinaryPerturbationEnv(save_path="/tmp/test_env", dataset_path=DATASET_PATH)
    state = env.extract_features_from_function(bin_path,'xstrtoumax')
    # print(len(state))
