#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASR@K test runner with decoupled retrieval models.
"""

import argparse
import csv
import hashlib
import json
import os
import pickle
import random
import shutil
import threading
import time
import tempfile
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from tqdm import tqdm

from env_wrapper import BinaryPerturbationEnv
from ppo_agent import PPOAgent
from run_utils import run_one


def load_dataset(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Dataset must be a list, got: {type(data)}")
    return data


def _is_valid_dir(path: Optional[str]) -> bool:
    return bool(path) and os.path.isdir(path)


def validate_safe_config(checkpoint_dir: Optional[str], i2v_dir: Optional[str]) -> None:
    if not _is_valid_dir(checkpoint_dir):
        raise ValueError(f"SAFE checkpoint dir invalid: {checkpoint_dir}")
    if not _is_valid_dir(i2v_dir):
        raise ValueError(f"SAFE i2v dir invalid: {i2v_dir}")


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


def _parse_addr(value: Optional[object]) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            return int(value, 16) if value.startswith(("0x", "0X")) else int(value)
        return int(value)
    except Exception:
        return None


def _resolve_mutated_addr(binary_path: str, func_name: Optional[str]) -> Optional[int]:
    if not func_name:
        return None
    sym_path = _find_sym_to_addr(binary_path)
    sym_map = _load_sym_to_addr(sym_path)
    if not sym_map:
        return None
    return _parse_addr(sym_map.get(func_name))


def choose_target_id(sample: Dict, sample_id: str, topk_before: List[Dict]) -> Optional[str]:
    for key in ("gt_id", "func_id", "target_id"):
        if sample.get(key):
            return str(sample[key])
    if topk_before:
        for item in topk_before:
            cand_id = str(item.get("id"))
            if cand_id and cand_id != sample_id:
                return cand_id
        return str(topk_before[0]["id"])
    return None


def get_rank(topk: List[Dict], target_id: Optional[str]) -> Optional[int]:
    if not target_id:
        return None
    for i, item in enumerate(topk, start=1):
        if str(item.get("id")) == str(target_id):
            return i
    return None


def format_topk(topk: List[Dict], max_items: Optional[int] = None) -> str:
    if not topk:
        return "empty"
    parts: List[str] = []
    for i, item in enumerate(topk, start=1):
        if max_items is not None and i > max_items:
            break
        item_id = str(item.get("id"))
        score = item.get("score")
        score_str = f"{float(score):.4f}" if score is not None else "NA"
        parts.append(f"{i}:{item_id}({score_str})")
    return ", ".join(parts)


def format_id_list(ids: List[str], max_items: int = 20) -> str:
    if not ids:
        return "empty"
    if max_items <= 0 or len(ids) <= max_items:
        return ", ".join(ids)
    shown = ids[:max_items]
    return f"{', '.join(shown)}, ...(+{len(ids) - max_items})"


def format_variant_scores(variant_ids: List[str], scores: List[Dict]) -> str:
    if not variant_ids:
        return "empty"
    score_map = {str(item.get("id")): item.get("score") for item in scores}
    parts: List[str] = []
    for vid in variant_ids:
        score = score_map.get(str(vid))
        score_str = f"{float(score):.4f}" if score is not None else "NA"
        parts.append(f"{vid}({score_str})")
    return ", ".join(parts)


def make_mutated_query(sample: Dict, mutated_binary: Optional[str]) -> Dict:
    mutated = dict(sample)
    if mutated_binary:
        mutated["binary_path"] = mutated_binary
    base = f"{mutated.get('binary_path', '')}::{mutated.get('func_name', '')}"
    mutated["id"] = hashlib.md5(base.encode("utf-8")).hexdigest()[:8]
    return mutated


def pick_after_binary(attack_info: Dict) -> Tuple[Optional[str], str]:
    """
    选择 after 检索使用的二进制：
    优先使用变异过程中最低相似度分数对应的 best_binary，
    如果不可用则回退到 final_binary。
    """
    best_binary = attack_info.get("best_binary")
    if best_binary:
        return best_binary, "best_binary"
    return attack_info.get("final_binary"), "final_binary"


def _get_variant_set(sample: Dict) -> Optional[set]:
    variants = sample.get("variants")
    if not variants or not isinstance(variants, list):
        return None
    cleaned = [str(v) for v in variants if v]
    if not cleaned:
        return None
    return set(cleaned)


ACTION_IDS_WITH12 = [1, 2, 4, 7, 8, 9, 11, 12, 13, 14, 15, 16]


def _remove_path_quiet(path: str) -> bool:
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        return True
    except Exception:
        return False


def cleanup_save_path_garbage(
    save_path: str,
    max_age_seconds: int,
    include_containers: bool = False,
) -> Dict[str, int]:
    now = time.time()
    max_age = max(0, int(max_age_seconds))
    stats = {"removed": 0, "failed": 0}

    def is_old(path: str) -> bool:
        try:
            return (now - os.path.getmtime(path)) >= max_age
        except Exception:
            return False

    # 1) Top-level tmp_*
    try:
        for name in os.listdir(save_path):
            p = os.path.join(save_path, name)
            if not os.path.isdir(p):
                continue
            if name.startswith("tmp_") and is_old(p):
                if _remove_path_quiet(p):
                    stats["removed"] += 1
                else:
                    stats["failed"] += 1
            elif include_containers and name.endswith("_container") and is_old(p):
                if _remove_path_quiet(p):
                    stats["removed"] += 1
                else:
                    stats["failed"] += 1
    except Exception:
        pass

    # 2) Retrieval worker dirs: _retrieval_work/cand_*
    retrieval_dir = os.path.join(save_path, "_retrieval_work")
    if os.path.isdir(retrieval_dir):
        try:
            for name in os.listdir(retrieval_dir):
                p = os.path.join(retrieval_dir, name)
                if os.path.isdir(p) and name.startswith("cand_") and is_old(p):
                    if _remove_path_quiet(p):
                        stats["removed"] += 1
                    else:
                        stats["failed"] += 1
        except Exception:
            pass

    # 3) Leftover mutant binaries: rl_output/mutant_*.bin
    rl_output_dir = os.path.join(save_path, "rl_output")
    if os.path.isdir(rl_output_dir):
        try:
            for name in os.listdir(rl_output_dir):
                p = os.path.join(rl_output_dir, name)
                if (
                    os.path.isfile(p)
                    and name.startswith("mutant_")
                    and name.endswith(".bin")
                    and is_old(p)
                ):
                    if _remove_path_quiet(p):
                        stats["removed"] += 1
                    else:
                        stats["failed"] += 1
        except Exception:
            pass

    return stats


class SavePathCleanupManager:
    def __init__(
        self,
        save_path: str,
        interval_seconds: int,
        max_age_seconds: int,
        include_containers: bool = False,
    ) -> None:
        self.save_path = save_path
        self.interval_seconds = max(0, int(interval_seconds or 0))
        self.max_age_seconds = max(0, int(max_age_seconds or 0))
        self.include_containers = bool(include_containers)
        self._last_ts = 0.0

    @property
    def enabled(self) -> bool:
        return self.interval_seconds > 0

    def maybe_cleanup(self, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.time()
        if (not force) and (now - self._last_ts < self.interval_seconds):
            return
        stats = cleanup_save_path_garbage(
            save_path=self.save_path,
            max_age_seconds=self.max_age_seconds,
            include_containers=self.include_containers,
        )
        self._last_ts = now
        removed = int(stats.get("removed", 0))
        failed = int(stats.get("failed", 0))
        if removed > 0 or failed > 0:
            tqdm.write(
                f"[ASR-CLEANUP] removed={removed} failed={failed} "
                f"max_age={self.max_age_seconds}s include_containers={self.include_containers}"
            )


class RetrievalBase:
    name = "base"

    def __init__(self, log_every: int = 0) -> None:
        self.log_every = max(0, int(log_every or 0))

    def _log_start(self, total: int, query_kind: str) -> None:
        if self.log_every > 0:
            tqdm.write(
                f"[ASR] Scoring candidates ({self.name}) query={query_kind} total={total} log_every={self.log_every}"
            )

    def _log_candidate(
        self,
        idx: int,
        total: int,
        cand_id: str,
        score: float,
        query_kind: str,
    ) -> None:
        if self.log_every <= 0:
            return
        if idx == 1 or idx == total or idx % self.log_every == 0:
            tqdm.write(
                f"[ASR] Scored ({self.name}) {query_kind} cand={idx}/{total} id={cand_id} score={score:.4f}"
            )

    def topk(self, query: Dict, dataset: List[Dict], k: int, query_kind: str) -> List[Dict]:
        raise NotImplementedError

    def score_candidates(
        self,
        query: Dict,
        candidates: List[Dict],
        query_kind: str,
    ) -> List[Dict]:
        raise NotImplementedError


class Asm2VecRetriever(RetrievalBase):
    name = "asm2vec"

    def __init__(
        self,
        asm_work_dir: Optional[str],
        retrieval_workers: int = 1,
        log_every: int = 0,
    ) -> None:
        super().__init__(log_every=log_every)
        self.asm_work_dir = asm_work_dir
        self.retrieval_workers = int(retrieval_workers or 1)
        self.original_asm_cache: Dict = {}
        self.mutated_asm_cache: Dict = {}

    def _get_worker_dir(self, cand_idx: int) -> Optional[str]:
        if not self.asm_work_dir:
            return None
        if self.retrieval_workers <= 1:
            return self.asm_work_dir
        worker_dir = os.path.join(self.asm_work_dir, f"cand_{cand_idx}")
        os.makedirs(worker_dir, exist_ok=True)
        return worker_dir

    def _get_query_addr(self, query: Dict, query_kind: str) -> Optional[int]:
        if query_kind == "original":
            addr = _parse_addr(query.get("func_addr"))
            if addr is None:
                raise ValueError("original query missing func_addr in dataset json")
            return addr
        if query_kind == "mutated":
            return _resolve_mutated_addr(query.get("binary_path", ""), query.get("func_name"))
        raise ValueError(f"query_kind must be original/mutated, got {query_kind}")

    def _score_candidate(
        self,
        query: Dict,
        cand: Dict,
        query_kind: str,
        worker_dir: Optional[str],
    ) -> Optional[Dict]:
        query_addr = self._get_query_addr(query, query_kind)
        cand_addr = _parse_addr(cand.get("func_addr"))
        if cand_addr is None:
            return None

        score, _grad = run_one(
            original_binary=query.get("binary_path"),
            mutated_binary=cand.get("binary_path"),
            model_original=None,
            checkdict={},
            function_name=str(query.get("func_name")),
            detection_method="asm2vec",
            asm_work_dir=worker_dir,
            original_asm_cache=self.original_asm_cache,
            simple_mode=True,
            original_func_addr=query_addr,
            mutated_func_addr=cand_addr,
            sym_to_addr_path=None,
            sym_to_addr_map=None,
            safe_checkpoint_dir=None,
            safe_i2v_dir=None,
            safe_use_gpu=False,
            original_asm_path=None,
            mutated_asm_cache=self.mutated_asm_cache,
        )
        if score is None:
            return None
        cand_id = str(cand.get("id") or cand.get("func_name") or "unknown_cand")
        return {"id": cand_id, "score": float(score)}

    def topk(self, query: Dict, dataset: List[Dict], k: int, query_kind: str) -> List[Dict]:
        scored: List[Dict] = []
        total = len(dataset)
        self._log_start(total, query_kind)
        if self.retrieval_workers > 1 and total > 0:
            max_workers = min(self.retrieval_workers, total)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._score_candidate,
                        query,
                        cand,
                        query_kind,
                        self._get_worker_dir(idx),
                    ): idx
                    for idx, cand in enumerate(dataset, start=1)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    result = future.result()
                    if result is not None:
                        scored.append(result)
                        self._log_candidate(
                            idx=idx,
                            total=total,
                            cand_id=str(result.get("id")),
                            score=float(result.get("score", 0.0)),
                            query_kind=query_kind,
                        )
        else:
            for idx, cand in enumerate(dataset, start=1):
                result = self._score_candidate(
                    query, cand, query_kind, self._get_worker_dir(idx)
                )
                if result is not None:
                    scored.append(result)
                    self._log_candidate(
                        idx=idx,
                        total=total,
                        cand_id=str(result.get("id")),
                        score=float(result.get("score", 0.0)),
                        query_kind=query_kind,
                    )
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]

    def score_candidates(
        self,
        query: Dict,
        candidates: List[Dict],
        query_kind: str,
    ) -> List[Dict]:
        scored: List[Dict] = []
        for idx, cand in enumerate(candidates, start=1):
            result = self._score_candidate(query, cand, query_kind, self._get_worker_dir(idx))
            if result is not None:
                scored.append(result)
        return scored


class SafeRetriever(RetrievalBase):
    name = "safe"

    def __init__(
        self,
        checkpoint_dir: str,
        i2v_dir: str,
        use_gpu: bool,
        asm_work_dir: Optional[str],
        retrieval_workers: int = 1,
        log_every: int = 0,
    ) -> None:
        super().__init__(log_every=log_every)
        validate_safe_config(checkpoint_dir, i2v_dir)
        self.checkpoint_dir = checkpoint_dir
        self.i2v_dir = i2v_dir
        self.use_gpu = bool(use_gpu)
        self.asm_work_dir = asm_work_dir
        self.retrieval_workers = int(retrieval_workers or 1)
        self.safe_cache: Dict = {}

    def _get_worker_dir(self, cand_idx: int) -> Optional[str]:
        if not self.asm_work_dir:
            return None
        if self.retrieval_workers <= 1:
            return self.asm_work_dir
        worker_dir = os.path.join(self.asm_work_dir, f"cand_{cand_idx}")
        os.makedirs(worker_dir, exist_ok=True)
        return worker_dir

    def _get_safe_cache(self) -> Dict:
        # SAFE session/cache must be shared across workers to avoid repeated
        # model loading and memory blow-up under multi-thread retrieval.
        return self.safe_cache

    def _get_query_addr(self, query: Dict, query_kind: str) -> int:
        if query_kind == "original":
            addr = _parse_addr(query.get("func_addr"))
            if addr is None:
                raise ValueError("original query missing func_addr in dataset json")
            return addr
        if query_kind == "mutated":
            addr = _resolve_mutated_addr(query.get("binary_path", ""), query.get("func_name"))
            if addr is None:
                raise ValueError("mutated query missing addr in sym_to_addr")
            return addr
        raise ValueError(f"query_kind must be original/mutated, got {query_kind}")

    def _score_candidate(
        self,
        query: Dict,
        cand: Dict,
        query_kind: str,
        worker_dir: Optional[str],
    ) -> Optional[Dict]:
        query_addr = self._get_query_addr(query, query_kind)
        cand_addr = _parse_addr(cand.get("func_addr"))
        if cand_addr is None:
            return None

        score, _grad = run_one(
            original_binary=query.get("binary_path"),
            mutated_binary=cand.get("binary_path"),
            model_original=None,
            checkdict={},
            function_name=str(query.get("func_name")),
            detection_method="safe",
            asm_work_dir=worker_dir,
            original_asm_cache=None,
            simple_mode=True,
            original_func_addr=query_addr,
            mutated_func_addr=cand_addr,
            sym_to_addr_path=None,
            sym_to_addr_map=None,
            safe_checkpoint_dir=self.checkpoint_dir,
            safe_i2v_dir=self.i2v_dir,
            safe_use_gpu=self.use_gpu,
            original_asm_path=None,
            safe_cache=self._get_safe_cache(),
        )
        if score is None:
            return None
        cand_id = str(cand.get("id") or cand.get("func_name") or "unknown_cand")
        return {"id": cand_id, "score": float(score)}

    def topk(self, query: Dict, dataset: List[Dict], k: int, query_kind: str) -> List[Dict]:
        scored: List[Dict] = []
        total = len(dataset)
        self._log_start(total, query_kind)
        if self.retrieval_workers > 1 and total > 0:
            max_workers = min(self.retrieval_workers, total)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._score_candidate,
                        query,
                        cand,
                        query_kind,
                        self._get_worker_dir(idx),
                    ): idx
                    for idx, cand in enumerate(dataset, start=1)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    result = future.result()
                    if result is not None:
                        scored.append(result)
                        self._log_candidate(
                            idx=idx,
                            total=total,
                            cand_id=str(result.get("id")),
                            score=float(result.get("score", 0.0)),
                            query_kind=query_kind,
                        )
        else:
            for idx, cand in enumerate(dataset, start=1):
                result = self._score_candidate(
                    query, cand, query_kind, self._get_worker_dir(idx)
                )
                if result is not None:
                    scored.append(result)
                    self._log_candidate(
                        idx=idx,
                        total=total,
                        cand_id=str(result.get("id")),
                        score=float(result.get("score", 0.0)),
                        query_kind=query_kind,
                    )
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]

    def score_candidates(
        self,
        query: Dict,
        candidates: List[Dict],
        query_kind: str,
    ) -> List[Dict]:
        scored: List[Dict] = []
        for idx, cand in enumerate(candidates, start=1):
            result = self._score_candidate(query, cand, query_kind, self._get_worker_dir(idx))
            if result is not None:
                scored.append(result)
        return scored


class JTransRetriever(RetrievalBase):
    name = "jtrans"

    def __init__(
        self,
        model_dir: Optional[str],
        tokenizer_dir: Optional[str],
        use_gpu: bool,
        asm_work_dir: Optional[str],
        retrieval_workers: int = 1,
        log_every: int = 0,
    ) -> None:
        super().__init__(log_every=log_every)
        if model_dir and not _is_valid_dir(model_dir):
            raise ValueError(f"jTrans model dir invalid: {model_dir}")
        if tokenizer_dir and not _is_valid_dir(tokenizer_dir):
            raise ValueError(f"jTrans tokenizer dir invalid: {tokenizer_dir}")
        self.model_dir = model_dir
        self.tokenizer_dir = tokenizer_dir
        self.use_gpu = bool(use_gpu)
        self.asm_work_dir = asm_work_dir
        self.retrieval_workers = int(retrieval_workers or 1)
        self.original_asm_cache: Dict = {}
        # 共享缓存：避免 retrieval_workers>1 时每线程重复加载 jTrans 模型导致内存膨胀。
        try:
            emb_cache_max = int(os.environ.get("JTRANS_EMB_CACHE_MAX", "512"))
        except Exception:
            emb_cache_max = 512
        self.jtrans_cache: Dict = {
            "emb_cache_max": max(0, emb_cache_max),
        }

    def _get_worker_dir(self, cand_idx: int) -> Optional[str]:
        if not self.asm_work_dir:
            return None
        if self.retrieval_workers <= 1:
            return self.asm_work_dir
        worker_dir = os.path.join(self.asm_work_dir, f"cand_{cand_idx}")
        os.makedirs(worker_dir, exist_ok=True)
        return worker_dir

    def _get_jtrans_cache(self) -> Dict:
        # 使用共享缓存，防止线程本地缓存在频繁创建线程池时反复占用大内存。
        return self.jtrans_cache

    def _get_query_addr(self, query: Dict, query_kind: str) -> Optional[int]:
        if query_kind == "original":
            addr = _parse_addr(query.get("func_addr"))
            if addr is None:
                raise ValueError("original query missing func_addr in dataset json")
            return addr
        if query_kind == "mutated":
            return _resolve_mutated_addr(query.get("binary_path", ""), query.get("func_name"))
        raise ValueError(f"query_kind must be original/mutated, got {query_kind}")

    def _score_candidate(
        self,
        query: Dict,
        cand: Dict,
        query_kind: str,
        worker_dir: Optional[str],
    ) -> Optional[Dict]:
        query_addr = self._get_query_addr(query, query_kind)
        cand_addr = _parse_addr(cand.get("func_addr"))
        if cand_addr is None:
            return None

        score, _grad = run_one(
            original_binary=query.get("binary_path"),
            mutated_binary=cand.get("binary_path"),
            model_original=None,
            checkdict={},
            function_name=str(query.get("func_name")),
            detection_method="jtrans",
            asm_work_dir=worker_dir,
            original_asm_cache=self.original_asm_cache,
            simple_mode=True,
            original_func_addr=query_addr,
            mutated_func_addr=cand_addr,
            sym_to_addr_path=None,
            sym_to_addr_map=None,
            safe_checkpoint_dir=None,
            safe_i2v_dir=None,
            safe_use_gpu=False,
            original_asm_path=None,
            mutated_asm_cache=None,
            safe_cache=None,
            jtrans_model_dir=self.model_dir,
            jtrans_tokenizer_dir=self.tokenizer_dir,
            jtrans_use_gpu=self.use_gpu,
            jtrans_cache=self._get_jtrans_cache(),
        )
        if score is None:
            return None
        cand_id = str(cand.get("id") or cand.get("func_name") or "unknown_cand")
        return {"id": cand_id, "score": float(score)}

    def topk(self, query: Dict, dataset: List[Dict], k: int, query_kind: str) -> List[Dict]:
        scored: List[Dict] = []
        total = len(dataset)
        self._log_start(total, query_kind)
        if self.retrieval_workers > 1 and total > 0:
            max_workers = min(self.retrieval_workers, total)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._score_candidate,
                        query,
                        cand,
                        query_kind,
                        self._get_worker_dir(idx),
                    ): idx
                    for idx, cand in enumerate(dataset, start=1)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    result = future.result()
                    if result is not None:
                        scored.append(result)
                        self._log_candidate(
                            idx=idx,
                            total=total,
                            cand_id=str(result.get("id")),
                            score=float(result.get("score", 0.0)),
                            query_kind=query_kind,
                        )
        else:
            for idx, cand in enumerate(dataset, start=1):
                result = self._score_candidate(
                    query, cand, query_kind, self._get_worker_dir(idx)
                )
                if result is not None:
                    scored.append(result)
                    self._log_candidate(
                        idx=idx,
                        total=total,
                        cand_id=str(result.get("id")),
                        score=float(result.get("score", 0.0)),
                        query_kind=query_kind,
                    )
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]

    def score_candidates(
        self,
        query: Dict,
        candidates: List[Dict],
        query_kind: str,
    ) -> List[Dict]:
        scored: List[Dict] = []
        for idx, cand in enumerate(candidates, start=1):
            result = self._score_candidate(query, cand, query_kind, self._get_worker_dir(idx))
            if result is not None:
                scored.append(result)
        return scored


class GMNRetriever(RetrievalBase):
    name = "gmn"

    def __init__(
        self,
        checkpoint_dir: str,
        features_json: str,
        dataset: str = "one",
        features_type: str = "opc",
        batch_size: int = 20,
        output_dir: Optional[str] = None,
        idb_root: Optional[str] = None,
        idb_prefix: Optional[str] = None,
        opcodes_json: Optional[str] = None,
        log_every: int = 0,
    ) -> None:
        super().__init__(log_every=log_every)
        self.checkpoint_dir = checkpoint_dir
        self.features_json = features_json
        self.dataset = dataset
        self.features_type = features_type
        self.batch_size = batch_size
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="gmn_out_")
        self.idb_root = idb_root
        self.idb_prefix = idb_prefix
        self.opcodes_json = opcodes_json
        self._engine = None
        self._feature_idb_keys: Optional[set] = None
        self._base_feature_dict: Optional[Dict] = None
        self._opcodes_dict: Optional[Dict[str, int]] = None
        self._runtime_feature_cache: Dict[Tuple[str, str, str], Dict] = {}

    def _ensure_engine(self):
        if self._engine is not None:
            return self._engine

        if not os.path.isdir(self.checkpoint_dir):
            raise ValueError(f"GMN checkpoint dir invalid: {self.checkpoint_dir}")
        if not os.path.isfile(self.features_json):
            raise ValueError(f"GMN features json invalid: {self.features_json}")

        try:
            import tensorflow as tf  # noqa: F401
        except Exception as e:
            raise RuntimeError(f"GMN requires tensorflow==1.x. Import failed: {e}")

        gmn_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "detection_model",
            "binary_function_similarity",
            "Models",
            "GGSNN-GMN",
            "NeuralNetwork",
        )
        gmn_root = os.path.abspath(gmn_root)
        if gmn_root not in os.sys.path:
            os.sys.path.insert(0, gmn_root)

        from core.config import get_config
        from core.gnn_model import GNNModel
        from core.build_dataset import build_testing_generator
        from core.model_evaluation import evaluate_sim

        args = SimpleNamespace(
            model_type="matching",
            training_mode="pair",
            features_type=self.features_type,
            dataset=self.dataset,
            num_epochs=1,
            checkpointdir=self.checkpoint_dir,
            outputdir=self.output_dir,
            featuresdir=os.path.dirname(os.path.dirname(self.features_json)),
        )
        config = get_config(args)
        config["checkpoint_dir"] = self.checkpoint_dir
        config["batch_size"] = self.batch_size
        config["testing"]["features_testing_path"] = self.features_json

        self._engine = {
            "config": config,
            "model": GNNModel(config),
            "build_testing_generator": build_testing_generator,
            "evaluate_sim": evaluate_sim,
            "initialized": False,
        }
        return self._engine

    def _format_fva(self, addr: Optional[object]) -> Optional[str]:
        val = _parse_addr(addr)
        if val is None:
            return None
        return hex(val)

    def _get_feature_idb_keys(self) -> set:
        if self._feature_idb_keys is not None:
            return self._feature_idb_keys
        try:
            with open(self.features_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._feature_idb_keys = set(str(k) for k in data.keys())
            else:
                self._feature_idb_keys = set()
        except Exception:
            self._feature_idb_keys = set()
        return self._feature_idb_keys

    def _get_base_feature_dict(self) -> Dict:
        if self._base_feature_dict is not None:
            return self._base_feature_dict
        with open(self.features_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"GMN features json must be dict: {self.features_json}")
        self._base_feature_dict = data
        return self._base_feature_dict

    def _get_opcodes_dict(self) -> Dict[str, int]:
        if self.features_type != "opc":
            return {}
        if self._opcodes_dict is not None:
            return self._opcodes_dict

        path = self.opcodes_json
        if not path:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(
                project_root,
                "detection_model",
                "binary_function_similarity",
                "Models",
                "GGSNN-GMN",
                "Preprocessing",
                "Dataset-1_training",
                "opcodes_dict.json",
            )
        if not os.path.isfile(path):
            raise ValueError(
                "GMN opc features require opcodes json. "
                f"Provide --gmn-opcodes-json. Missing: {path}"
            )

        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError(f"Invalid opcodes json: {path}")
        self._opcodes_dict = {str(k).lower(): int(v) for k, v in loaded.items()}
        self.opcodes_json = path
        return self._opcodes_dict

    @staticmethod
    def _np_to_sparse_string(mat: np.ndarray) -> str:
        rows, cols = np.nonzero(mat)
        data = mat[rows, cols]
        row_str = ";".join(str(int(x)) for x in rows.tolist())
        col_str = ";".join(str(int(x)) for x in cols.tolist())
        data_str = ";".join(str(int(x)) for x in data.tolist())
        n_row = str(int(mat.shape[0]))
        n_col = str(int(mat.shape[1]))
        return "::".join([row_str, col_str, data_str, n_row, n_col])

    @staticmethod
    def _normalize_mnemonic(ins: Dict) -> Optional[str]:
        mnem = ins.get("mnemonic")
        if mnem:
            return str(mnem).strip().lower()
        opcode = str(ins.get("opcode") or "").strip()
        if not opcode:
            return None
        head = opcode.split(" ", 1)[0].strip().lower()
        return head or None

    def _extract_runtime_feature_for_query(
        self,
        query: Dict,
        query_idb: str,
        query_fva: str,
    ) -> Dict:
        binary_path = str(query.get("binary_path") or "")
        func_name = str(query.get("func_name") or "")
        cache_key = (binary_path, query_idb, query_fva)
        cached = self._runtime_feature_cache.get(cache_key)
        if cached is not None:
            return cached

        query_addr = _parse_addr(query_fva)
        if query_addr is None:
            raise ValueError("GMN mutated query missing valid function addr")
        if not os.path.isfile(binary_path):
            raise ValueError(f"mutated binary not found: {binary_path}")

        try:
            import r2pipe
        except Exception as e:
            raise RuntimeError(f"GMN mutated retrieval requires r2pipe: {e}")

        r2 = r2pipe.open(binary_path, flags=["-2"])
        try:
            r2.cmd("aaa")
            blocks = r2.cmdj(f"afbj @ {query_addr}") or []
            if not blocks:
                r2.cmd(f"af @ {query_addr}")
                blocks = r2.cmdj(f"afbj @ {query_addr}") or []
            if not blocks:
                raise ValueError(
                    f"cannot extract CFG for mutated function: {func_name}@{hex(query_addr)}"
                )

            node_addrs = sorted(
                int(bb.get("addr"))
                for bb in blocks
                if bb.get("addr") is not None
            )
            if not node_addrs:
                raise ValueError(
                    f"mutated CFG has no basic blocks: {func_name}@{hex(query_addr)}"
                )
            node_to_idx = {addr: idx for idx, addr in enumerate(node_addrs)}

            graph_mat = np.zeros((len(node_addrs), len(node_addrs)), dtype=np.int8)
            bb_map = {int(bb.get("addr")): bb for bb in blocks if bb.get("addr") is not None}
            for src in node_addrs:
                bb = bb_map.get(src) or {}
                for edge_key in ("jump", "fail"):
                    dst = bb.get(edge_key)
                    if dst is None:
                        continue
                    dst_int = int(dst)
                    if dst_int in node_to_idx:
                        graph_mat[node_to_idx[src], node_to_idx[dst_int]] = 1

            opcodes = self._get_opcodes_dict()
            opc_dim = len(opcodes)
            opc_mat = np.zeros((len(node_addrs), opc_dim), dtype=np.int8)
            if opc_dim > 0:
                for row_idx, bb_addr in enumerate(node_addrs):
                    bb = bb_map.get(bb_addr) or {}
                    size = int(bb.get("size") or 0)
                    if size <= 0:
                        continue
                    instrs = r2.cmdj(f"pDj {size} @ {bb_addr}") or []
                    for ins in instrs:
                        mnem = self._normalize_mnemonic(ins)
                        if not mnem:
                            continue
                        opc_idx = opcodes.get(mnem)
                        if opc_idx is None or opc_idx < 0 or opc_idx >= opc_dim:
                            continue
                        if opc_mat[row_idx, opc_idx] < np.iinfo(np.int8).max:
                            opc_mat[row_idx, opc_idx] += 1

            runtime = {
                query_idb: {
                    query_fva: {
                        "graph": self._np_to_sparse_string(graph_mat),
                        "opc": self._np_to_sparse_string(opc_mat),
                    }
                }
            }
            self._runtime_feature_cache[cache_key] = runtime
            return runtime
        finally:
            try:
                r2.quit()
            except Exception:
                pass

    def _build_features_for_query(self, query: Dict, query_kind: str) -> Tuple[str, str, Optional[Dict]]:
        query_idb = self._map_idb_path(query.get("binary_path"))
        if not query_idb:
            raise ValueError("GMN requires query binary_path")

        if query_kind == "original":
            query_fva = self._format_fva(query.get("func_addr"))
            if not query_fva:
                raise ValueError("GMN requires original query func_addr in dataset")
            return query_idb, query_fva, None

        if query_kind == "mutated":
            query_addr = _resolve_mutated_addr(
                str(query.get("binary_path") or ""),
                query.get("func_name"),
            )
            if query_addr is None:
                query_addr = _parse_addr(query.get("func_addr"))
            query_fva = self._format_fva(query_addr)
            if not query_fva:
                raise ValueError(
                    "GMN mutated query missing addr (sym_to_addr and func_addr are both unavailable)"
                )

            runtime = self._extract_runtime_feature_for_query(
                query=query,
                query_idb=query_idb,
                query_fva=query_fva,
            )
            merged = dict(self._get_base_feature_dict())
            for idb_path, funcs in runtime.items():
                existing = dict(merged.get(idb_path, {}))
                existing.update(funcs)
                merged[idb_path] = existing
            return query_idb, query_fva, merged

        raise ValueError(f"query_kind must be original/mutated, got {query_kind}")

    def _map_idb_path(self, binary_path: Optional[str]) -> Optional[str]:
        if not binary_path:
            return None
        mapped = binary_path
        if self.idb_root:
            rel = os.path.relpath(binary_path, self.idb_root)
            if self.idb_prefix:
                mapped = os.path.join(self.idb_prefix, rel)
            else:
                mapped = rel

        mapped = mapped.replace(os.sep, "/")
        candidates: List[str] = []

        def add_cand(path: Optional[str]) -> None:
            if not path:
                return
            path = path.replace(os.sep, "/")
            if path not in candidates:
                candidates.append(path)

        add_cand(mapped)
        if mapped.endswith(".i64"):
            add_cand(mapped[:-4])
        else:
            add_cand(f"{mapped}.i64")

        for p in list(candidates):
            add_cand(p.replace("IDBs/coreutils/coreutils-", "IDBs/coreutils-"))

        for p in list(candidates):
            if p.startswith("IDBs/"):
                add_cand(f"../../rl_framework/datasets/coreutils/{p}")
            if p.startswith("rl_framework/"):
                add_cand(f"../../{p}")

        feature_keys = self._get_feature_idb_keys()
        for p in candidates:
            if p in feature_keys:
                return p

        # Fall back to a deterministic default even when we cannot pre-match.
        for p in candidates:
            if p.endswith(".i64"):
                return p
        return candidates[0] if candidates else None

    def _score_pairs(self, pairs: List[Dict], features_override: Optional[Dict] = None) -> List[float]:
        engine = self._ensure_engine()
        config = engine["config"]
        model = engine["model"]
        build_testing_generator = engine["build_testing_generator"]
        evaluate_sim = engine["evaluate_sim"]

        if not pairs:
            return []

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as tmp:
            writer = csv.DictWriter(
                tmp,
                fieldnames=[
                    "idb_path_1",
                    "fva_1",
                    "func_name_1",
                    "idb_path_2",
                    "fva_2",
                    "func_name_2",
                    "db_type",
                ],
            )
            writer.writeheader()
            writer.writerows(pairs)
            csv_path = tmp.name

        prev_features = config["testing"].get("features_testing_path")
        if features_override is not None:
            config["testing"]["features_testing_path"] = features_override

        try:
            if not engine["initialized"]:
                init_gen = build_testing_generator(config, csv_path)
                model._model_initialize(init_gen, is_training=False)
                model._create_tfsaver()
                model._restore_model()
                engine["initialized"] = True

            batch_gen = build_testing_generator(config, csv_path)
            sims = evaluate_sim(
                model._session,
                model._tensors["metrics"]["evaluation"],
                model._placeholders,
                batch_gen,
            )
            return sims.tolist()
        finally:
            try:
                os.remove(csv_path)
            except Exception:
                pass
            config["testing"]["features_testing_path"] = prev_features

    def _score_dataset(
        self,
        query: Dict,
        dataset: List[Dict],
        query_kind: str,
    ) -> List[Dict]:
        query_idb, query_fva, features_override = self._build_features_for_query(
            query, query_kind
        )

        pairs = []
        cand_ids = []
        for cand in dataset:
            cand_idb = self._map_idb_path(cand.get("binary_path"))
            cand_fva = self._format_fva(cand.get("func_addr"))
            if not cand_idb or not cand_fva:
                continue
            pairs.append(
                {
                    "idb_path_1": query_idb,
                    "fva_1": query_fva,
                    "func_name_1": str(query.get("func_name") or ""),
                    "idb_path_2": cand_idb,
                    "fva_2": cand_fva,
                    "func_name_2": str(cand.get("func_name") or ""),
                    "db_type": "QQ",
                }
            )
            cand_ids.append(str(cand.get("id") or cand.get("func_name") or "unknown_cand"))

        sims = self._score_pairs(pairs, features_override=features_override)
        scored = []
        for cand_id, score in zip(cand_ids, sims):
            scored.append({"id": cand_id, "score": float(score)})
        return scored

    def topk(self, query: Dict, dataset: List[Dict], k: int, query_kind: str) -> List[Dict]:
        scored = self._score_dataset(query, dataset, query_kind)
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]

    def score_candidates(
        self,
        query: Dict,
        candidates: List[Dict],
        query_kind: str,
    ) -> List[Dict]:
        return self._score_dataset(query, candidates, query_kind)


class RLAgentRunner:
    def __init__(
        self,
        dataset_path: str,
        model_path: str,
        save_path: str,
        max_steps: int,
        state_dim: int = 256,
        use_gpu: bool = False,
        detection_method: str = "asm2vec",
        safe_checkpoint_dir: Optional[str] = None,
        safe_i2v_dir: Optional[str] = None,
        safe_use_gpu: bool = False,
        safe_cache: Optional[Dict] = None,
        jtrans_model_dir: Optional[str] = None,
        jtrans_tokenizer_dir: Optional[str] = None,
        jtrans_use_gpu: bool = False,
    ) -> None:
        self.dataset_path = dataset_path
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        print(f"[ASR] Loading model: {model_path} (device={device})")
        load_start = time.time()
        include_action12 = True
        action_ids = list(ACTION_IDS_WITH12)
        infer_src = "fixed_with_action12"

        print(
            f"[ASR] Action layout: include_action12={include_action12} "
            f"n_actions={len(action_ids)} source={infer_src}"
        )
        self.agent = PPOAgent(
            state_dim=state_dim,
            device=device,
            action_map=list(action_ids),
            n_actions=len(action_ids),
        )
        self.agent.load(model_path)
        self.agent.policy.eval()
        print(f"[ASR] Model loaded in {time.time() - load_start:.2f}s")

        self.env = BinaryPerturbationEnv(
            save_path=self.save_path,
            dataset_path=self.dataset_path,
            sample_hold_interval=10**9,
            max_steps=max_steps,
            detection_method=detection_method,
            safe_checkpoint_dir=safe_checkpoint_dir,
            safe_i2v_dir=safe_i2v_dir,
            safe_use_gpu=safe_use_gpu,
            safe_cache_enabled=(detection_method == "safe"),
            safe_cache=safe_cache,
            jtrans_model_dir=jtrans_model_dir,
            jtrans_tokenizer_dir=jtrans_tokenizer_dir,
            jtrans_use_gpu=jtrans_use_gpu,
            include_action12=True,
        )
        self.env.target_score = 0.4
        self.env.set_state_dim(state_dim)

    def _pin_sample(self, sample_idx: int) -> None:
        sample = self.env.dataset[sample_idx]
        self.env.current_sample_idx = sample_idx
        self.env.current_sample_data = sample
        self.env.episodes_on_current = 0
        self.env.original_func_addr = None
        self.env.original_binary = sample["binary_path"]
        self.env.function_name = sample["func_name"]

    def attack_sample(self, sample_idx: int, max_steps: int) -> Dict:
        self._pin_sample(sample_idx)
        state = self.env.reset(force_switch=False)

        steps_used = 0
        final_score = 1.0
        final_binary = None
        best_score = float("inf")
        best_binary = None
        best_step = 0
        error = ""

        print(f"[ASR] Attack sample={sample_idx} func={self.env.function_name}")
        for step in range(max_steps):
            (
                _joint_idx,
                loc_idx,
                _act_idx,
                actual_action,
                _log_prob,
                _value,
            ) = self.agent.select_action(state, explore=True)

            next_state, _reward, done, info = self.env.step(actual_action, loc_idx)
            score = info.get("score", 1.0)
            print(
                f"[ASR] Step {step + 1}/{max_steps} action={actual_action} "
                f"loc={loc_idx} reward={_reward:.4f} score({self.env.detection_method})={score} "
                f"done={done} should_reset={info.get('should_reset', False)}"
            )
            steps_used = step + 1

            if info.get("binary"):
                final_binary = info["binary"]
            if "score" in info:
                final_score = float(info["score"])
                loc_valid = info.get("loc_valid", True)
                if (
                    final_binary
                    and loc_valid
                    and final_score < best_score
                ):
                    best_score = final_score
                    best_binary = final_binary
                    best_step = step + 1

            if info.get("should_reset"):
                error = str(info.get("error", "should_reset"))
                break

            state = next_state
            if done:
                break

        print(
            f"[ASR] Finished sample={sample_idx} steps={steps_used} "
            f"final_score={final_score} error={error}"
        )
        if best_binary is not None:
            print(
                f"[ASR] Best mutation during attack: step={best_step} "
                f"best_score={best_score:.6f}"
            )
        return {
            "steps_used": steps_used,
            "final_score": final_score,
            "final_binary": final_binary,
            "best_score": float(best_score) if best_binary is not None else float(final_score),
            "best_binary": best_binary if best_binary is not None else final_binary,
            "best_step": best_step,
            "error": error,
        }


class GAAgentRunner:
    def __init__(
        self,
        dataset_path: str,
        save_path: str,
        max_steps: int,
        state_dim: int = 256,
        detection_method: str = "asm2vec",
        safe_checkpoint_dir: Optional[str] = None,
        safe_i2v_dir: Optional[str] = None,
        safe_use_gpu: bool = False,
        safe_cache: Optional[Dict] = None,
        jtrans_model_dir: Optional[str] = None,
        jtrans_tokenizer_dir: Optional[str] = None,
        jtrans_use_gpu: bool = False,
        population_size: int = 6,
        generations: int = 4,
        elite_size: int = 2,
        mutation_rate: float = 0.25,
        crossover_rate: float = 0.70,
        seq_len: int = 8,
        loc_slots: int = 3,
        seed: Optional[int] = None,
    ) -> None:
        self.dataset_path = dataset_path
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.population_size = max(2, int(population_size))
        self.generations = max(1, int(generations))
        self.elite_size = max(1, int(elite_size))
        self.mutation_rate = min(max(float(mutation_rate), 0.0), 1.0)
        self.crossover_rate = min(max(float(crossover_rate), 0.0), 1.0)
        self.seq_len = max(1, int(seq_len))
        self.loc_slots = max(1, int(loc_slots))
        self.rng = random.Random(seed)

        self.env = BinaryPerturbationEnv(
            save_path=self.save_path,
            dataset_path=self.dataset_path,
            sample_hold_interval=10**9,
            max_steps=max_steps,
            detection_method=detection_method,
            safe_checkpoint_dir=safe_checkpoint_dir,
            safe_i2v_dir=safe_i2v_dir,
            safe_use_gpu=safe_use_gpu,
            safe_cache_enabled=(detection_method == "safe"),
            safe_cache=safe_cache,
            jtrans_model_dir=jtrans_model_dir,
            jtrans_tokenizer_dir=jtrans_tokenizer_dir,
            jtrans_use_gpu=jtrans_use_gpu,
            include_action12=True,
        )
        self.env.target_score = 0.4
        self.env.set_state_dim(state_dim)
        print(
            "[ASR] GA baseline loaded "
            f"(pop={self.population_size}, gen={self.generations}, "
            f"elite={self.elite_size}, mut={self.mutation_rate}, cross={self.crossover_rate}, "
            f"seq_len={self.seq_len}, loc_slots={self.loc_slots})"
        )

    def _pin_sample(self, sample_idx: int) -> None:
        sample = self.env.dataset[sample_idx]
        self.env.current_sample_idx = sample_idx
        self.env.current_sample_data = sample
        self.env.episodes_on_current = 0
        self.env.original_func_addr = None
        self.env.original_binary = sample["binary_path"]
        self.env.function_name = sample["func_name"]

    def _reset_pinned_env(self, sample_idx: int) -> None:
        self._pin_sample(sample_idx)
        sample_id = self.env.idx_to_id.get(self.env.current_sample_idx)
        if sample_id in self.env.sample_no_progress:
            self.env.sample_no_progress[sample_id] = 0
        self.env.episodes_on_current = 0
        self.env.reset(force_switch=False)

    def _valid_locs(self) -> List[int]:
        mask = self.env.get_loc_mask(self.loc_slots)
        valid = [i for i, m in enumerate(mask) if m]
        return valid if valid else [0]

    def _random_genome(self, genome_len: int, valid_locs: List[int]) -> List[Tuple[int, int]]:
        genes: List[Tuple[int, int]] = []
        for _ in range(genome_len):
            genes.append(
                (
                    self.rng.choice(self.env.action_ids),
                    self.rng.choice(valid_locs),
                )
            )
        return genes

    def _mutate_genome(self, genome: List[Tuple[int, int]], valid_locs: List[int]) -> List[Tuple[int, int]]:
        mutated: List[Tuple[int, int]] = []
        for action, loc_idx in genome:
            if self.rng.random() < self.mutation_rate:
                if self.rng.random() < 0.5:
                    action = self.rng.choice(self.env.action_ids)
                else:
                    loc_idx = self.rng.choice(valid_locs)
            mutated.append((action, loc_idx))
        return mutated

    def _crossover(
        self, p1: List[Tuple[int, int]], p2: List[Tuple[int, int]]
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        if len(p1) <= 1 or self.rng.random() >= self.crossover_rate:
            return list(p1), list(p2)
        point = self.rng.randint(1, len(p1) - 1)
        c1 = p1[:point] + p2[point:]
        c2 = p2[:point] + p1[point:]
        return c1, c2

    def _select_parent(self, scored: List[Dict]) -> List[Tuple[int, int]]:
        if len(scored) == 1:
            return list(scored[0]["genome"])
        a, b = self.rng.sample(scored, 2)
        winner = a if a["fitness"] >= b["fitness"] else b
        return list(winner["genome"])

    def _evaluate_genome(self, sample_idx: int, genome: List[Tuple[int, int]], step_budget: int) -> Dict:
        self._reset_pinned_env(sample_idx)
        final_score = 1.0
        final_binary = None
        best_score = float("inf")
        best_binary = None
        best_step = 0
        error = ""
        steps_used = 0
        done = False

        for action, loc_idx in genome:
            if steps_used >= step_budget:
                break
            _state, _reward, done, info = self.env.step(action, loc_idx)
            steps_used += 1
            if info.get("binary"):
                final_binary = info["binary"]
            if "score" in info:
                final_score = float(info["score"])
                loc_valid = info.get("loc_valid", True)
                if (
                    final_binary
                    and loc_valid
                    and final_score < best_score
                ):
                    best_score = final_score
                    best_binary = final_binary
                    best_step = steps_used
            if info.get("should_reset"):
                error = str(info.get("error", "should_reset"))
                break
            if done:
                break

        success = final_score < self.env.target_score
        fitness = 1.0 - final_score
        if success:
            fitness += 1.0
        if error:
            fitness -= 1.0
        return {
            "steps_used": steps_used,
            "final_score": final_score,
            "final_binary": final_binary,
            "best_score": float(best_score) if best_binary is not None else float(final_score),
            "best_binary": best_binary if best_binary is not None else final_binary,
            "best_step": best_step,
            "error": error,
            "fitness": fitness,
            "success": success,
            "done": done,
        }

    def attack_sample(self, sample_idx: int, max_steps: int) -> Dict:
        self._reset_pinned_env(sample_idx)
        valid_locs = self._valid_locs()
        genome_len = max(1, min(self.seq_len, int(max_steps)))
        elite_size = min(self.elite_size, self.population_size - 1)
        population = [
            self._random_genome(genome_len, valid_locs) for _ in range(self.population_size)
        ]
        remaining_steps = max(1, int(max_steps))
        total_steps = 0
        best = {
            "steps_used": 0,
            "final_score": 1.0,
            "final_binary": None,
            "best_score": 1.0,
            "best_binary": None,
            "best_step": 0,
            "error": "ga_no_eval",
            "fitness": -1.0,
            "success": False,
        }

        print(f"[ASR-GA] Attack sample={sample_idx} func={self.env.function_name}")
        for gen in range(self.generations):
            if remaining_steps <= 0:
                break
            scored: List[Dict] = []
            for genome in population:
                if remaining_steps <= 0:
                    break
                eval_budget = min(genome_len, remaining_steps)
                result = self._evaluate_genome(sample_idx, genome, eval_budget)
                result["genome"] = genome
                scored.append(result)
                total_steps += result["steps_used"]
                remaining_steps -= result["steps_used"]
                if result["fitness"] > best["fitness"]:
                    best = dict(result)
                if result["success"]:
                    break

            if not scored:
                break
            scored.sort(key=lambda x: x["fitness"], reverse=True)
            gen_best = scored[0]
            print(
                f"[ASR-GA] Gen {gen + 1}/{self.generations} "
                f"best_score={gen_best['final_score']:.4f} "
                f"fitness={gen_best['fitness']:.4f} "
                f"remaining_steps={remaining_steps}"
            )
            if gen_best["success"]:
                break

            elites = [list(item["genome"]) for item in scored[:elite_size]]
            next_population: List[List[Tuple[int, int]]] = list(elites)
            while len(next_population) < self.population_size:
                p1 = self._select_parent(scored)
                p2 = self._select_parent(scored)
                c1, c2 = self._crossover(p1, p2)
                next_population.append(self._mutate_genome(c1, valid_locs))
                if len(next_population) < self.population_size:
                    next_population.append(self._mutate_genome(c2, valid_locs))
            population = next_population[: self.population_size]

        print(
            f"[ASR-GA] Finished sample={sample_idx} steps={total_steps} "
            f"final_score={best['final_score']} error={best['error']}"
        )
        return {
            "steps_used": total_steps,
            "final_score": float(best["final_score"]),
            "final_binary": best.get("final_binary"),
            "best_score": float(best.get("best_score", best["final_score"])),
            "best_binary": best.get("best_binary") or best.get("final_binary"),
            "best_step": int(best.get("best_step", 0)),
            "error": str(best.get("error", "")),
        }


def build_retriever(args: argparse.Namespace, save_path: str) -> RetrievalBase:
    asm_work_dir = os.path.join(save_path, "_retrieval_work")
    os.makedirs(asm_work_dir, exist_ok=True)
    if args.retrieval_model == "asm2vec":
        return Asm2VecRetriever(
            asm_work_dir=asm_work_dir,
            retrieval_workers=args.retrieval_workers,
            log_every=args.retrieval_log_every,
        )
    if args.retrieval_model == "safe":
        return SafeRetriever(
            checkpoint_dir=args.safe_checkpoint_dir,
            i2v_dir=args.safe_i2v_dir,
            use_gpu=args.safe_use_gpu,
            asm_work_dir=asm_work_dir,
            retrieval_workers=args.retrieval_workers,
            log_every=args.retrieval_log_every,
        )
    if args.retrieval_model == "jtrans":
        return JTransRetriever(
            model_dir=args.jtrans_model_dir,
            tokenizer_dir=args.jtrans_tokenizer_dir,
            use_gpu=args.jtrans_use_gpu,
            asm_work_dir=asm_work_dir,
            retrieval_workers=args.retrieval_workers,
            log_every=args.retrieval_log_every,
        )
    if args.retrieval_model == "gmn":
        return GMNRetriever(
            checkpoint_dir=args.gmn_checkpoint_dir,
            features_json=args.gmn_features_json,
            dataset=args.gmn_dataset,
            features_type=args.gmn_features_type,
            batch_size=args.gmn_batch_size,
            output_dir=args.gmn_output_dir or asm_work_dir,
            idb_root=args.gmn_idb_root,
            idb_prefix=args.gmn_idb_prefix,
            opcodes_json=args.gmn_opcodes_json,
            log_every=args.retrieval_log_every,
        )
    raise ValueError(f"Unknown retrieval model: {args.retrieval_model}")


def build_attack_runner(
    args: argparse.Namespace,
    dataset_path: str,
    save_path: str,
    max_steps: int,
    retriever: RetrievalBase,
):
    safe_checkpoint_dir = args.safe_checkpoint_dir
    safe_i2v_dir = args.safe_i2v_dir
    safe_use_gpu = args.safe_use_gpu
    safe_cache = None
    if isinstance(retriever, SafeRetriever):
        if not safe_checkpoint_dir:
            safe_checkpoint_dir = retriever.checkpoint_dir
        if not safe_i2v_dir:
            safe_i2v_dir = retriever.i2v_dir
        safe_use_gpu = bool(safe_use_gpu or retriever.use_gpu)
        if args.env_detection == "safe":
            safe_cache = retriever.safe_cache

    common_kwargs = {
        "dataset_path": dataset_path,
        "save_path": save_path,
        "max_steps": max_steps,
        "detection_method": args.env_detection,
        "safe_checkpoint_dir": safe_checkpoint_dir,
        "safe_i2v_dir": safe_i2v_dir,
        "safe_use_gpu": safe_use_gpu,
        "safe_cache": safe_cache,
        "jtrans_model_dir": args.jtrans_model_dir,
        "jtrans_tokenizer_dir": args.jtrans_tokenizer_dir,
        "jtrans_use_gpu": args.jtrans_use_gpu,
    }
    if args.attack_agent == "ppo":
        return RLAgentRunner(
            model_path=args.model_path,
            use_gpu=args.use_gpu,
            **common_kwargs,
        )
    if args.attack_agent == "ga":
        return GAAgentRunner(
            population_size=args.ga_population_size,
            generations=args.ga_generations,
            elite_size=args.ga_elite_size,
            mutation_rate=args.ga_mutation_rate,
            crossover_rate=args.ga_crossover_rate,
            seq_len=args.ga_seq_len,
            loc_slots=args.ga_loc_slots,
            seed=args.ga_seed,
            **common_kwargs,
        )
    raise ValueError(f"Unknown attack agent: {args.attack_agent}")


def _apply_limit(
    dataset: List[Dict],
    limit: Optional[int],
    limit_mode: str,
    limit_seed: Optional[int],
) -> List[Dict]:
    if limit is None:
        return dataset
    total = len(dataset)
    if total == 0:
        return dataset
    limit = max(0, min(int(limit), total))
    if limit_mode == "head":
        return dataset[:limit]
    rng = random.Random(limit_seed)
    if limit_mode == "random":
        return rng.sample(dataset, limit)
    if limit_mode == "variants":
        id_to_item = {str(d.get("id")): d for d in dataset if d.get("id") is not None}
        indices = list(range(total))
        rng.shuffle(indices)
        selected_ids = []
        selected_set = set()
        anchors = 0
        for idx in indices:
            if anchors >= limit:
                break
            item = dataset[idx]
            item_id = item.get("id")
            if item_id is None or item_id in selected_set:
                continue
            group_ids = [item_id] + [
                vid for vid in item.get("variants", []) if vid in id_to_item
            ]
            for gid in group_ids:
                if gid in selected_set:
                    continue
                selected_set.add(gid)
                selected_ids.append(gid)
            anchors += 1
        return [id_to_item[i] for i in selected_ids if i in id_to_item]
    if limit_mode == "variants_strict":
        id_to_item = {str(d.get("id")): d for d in dataset if d.get("id") is not None}
        indices = list(range(total))
        rng.shuffle(indices)
        selected_ids = []
        selected_set = set()
        for idx in indices:
            if len(selected_ids) >= limit:
                break
            item = dataset[idx]
            item_id = item.get("id")
            if item_id is None or item_id in selected_set:
                continue
            group_ids = [item_id] + [
                vid for vid in item.get("variants", []) if vid in id_to_item
            ]
            for gid in group_ids:
                if len(selected_ids) >= limit:
                    break
                if gid in selected_set:
                    continue
                selected_set.add(gid)
                selected_ids.append(gid)
        return [id_to_item[i] for i in selected_ids if i in id_to_item]
    raise ValueError(f"Unknown limit_mode: {limit_mode}")


def run_asr_simple(
    dataset_path: str,
    k: int,
    budget: int,
    limit: Optional[int],
    limit_mode: str,
    limit_seed: Optional[int],
    save_path: str,
    csv_path: str,
    retriever: RetrievalBase,
    runner,
    cleanup_mgr: Optional[SavePathCleanupManager] = None,
) -> Tuple[float, List[Dict]]:
    dataset = load_dataset(dataset_path)
    dataset = _apply_limit(dataset, limit, limit_mode, limit_seed)

    print(
        f"[ASR] Dataset loaded: {dataset_path} samples={len(dataset)} "
        f"topk={k} budget={budget} retrieval={retriever.name} "
        f"limit={limit} mode={limit_mode}"
    )
    print("[ASR] Environment ready. Start mutation.")

    rows: List[Dict] = []
    success = 0

    pbar = tqdm(enumerate(dataset), total=len(dataset), desc=f"ASR@{k}", unit="sample")
    for idx, sample in pbar:
        if cleanup_mgr is not None:
            cleanup_mgr.maybe_cleanup()
        sample_id = str(sample.get("id") or f"sample_{idx}")
        tqdm.write(
            f"[ASR] Start sample={idx} id={sample_id} func={sample.get('func_name')}"
        )

        try:
            topk_before = retriever.topk(sample, dataset, k, query_kind="original")
            if not topk_before:
                raise ValueError("retrieval_before_empty")
        except Exception as e:
            rows.append(
                {
                    "sample_id": sample_id,
                    "target_id": "",
                    "rank_before": "",
                    "rank_after": "",
                    "steps_used": 0,
                    "final_score": "",
                    "success": 0,
                    "error": f"retrieval_before_failed:{e}",
                }
            )
            continue

        target_id = choose_target_id(sample, sample_id, topk_before)
        rank_before = get_rank(topk_before, target_id)
        tqdm.write(f"[ASR] Pre-rank target={target_id} rank_before={rank_before}")
        tqdm.write(f"[ASR] TopK before ({retriever.name}): {format_topk(topk_before)}")

        attack_info = runner.attack_sample(sample_idx=idx, max_steps=budget)
        after_binary, after_binary_source = pick_after_binary(attack_info)
        mutated_query = make_mutated_query(sample, after_binary)
        tqdm.write(
            f"[ASR] After-query source={after_binary_source} "
            f"best_score={attack_info.get('best_score', attack_info.get('final_score', 1.0)):.6f} "
            f"final_score={attack_info.get('final_score', 1.0):.6f}"
        )

        if attack_info.get("error"):
            rows.append(
                {
                    "sample_id": sample_id,
                    "target_id": target_id or "",
                    "rank_before": rank_before if rank_before is not None else "",
                    "rank_after": "",
                    "steps_used": attack_info["steps_used"],
                    "final_score": attack_info["final_score"],
                    "success": 0,
                    "error": f"mutation_failed_skip_after:{attack_info['error']}",
                }
            )
            continue

        try:
            topk_after = retriever.topk(mutated_query, dataset, k, query_kind="mutated")
            if not topk_after:
                raise ValueError("retrieval_after_empty")
        except Exception as e:
            rows.append(
                {
                    "sample_id": sample_id,
                    "target_id": target_id or "",
                    "rank_before": rank_before if rank_before is not None else "",
                    "rank_after": "",
                    "steps_used": attack_info["steps_used"],
                    "final_score": attack_info["final_score"],
                    "success": 0,
                    "error": f"retrieval_after_failed:{e}",
                }
            )
            continue

        rank_after = get_rank(topk_after, target_id)
        tqdm.write(f"[ASR] Post-rank target={target_id} rank_after={rank_after}")
        is_success = int(target_id is not None and rank_after is None)
        success += is_success
        tqdm.write(f"[ASR] TopK after ({retriever.name}): {format_topk(topk_after)}")
        if rank_after is None:
            tqdm.write(f"[ASR] Target missing after TopK: target={target_id} k={k}")
        tqdm.write(
            f"[ASR] sample={idx} target={target_id} rank_before={rank_before} "
            f"rank_after={rank_after} success={is_success} "
            f"steps={attack_info['steps_used']} final_score={attack_info['final_score']}"
        )

        rows.append(
            {
                "sample_id": sample_id,
                "target_id": target_id or "",
                "rank_before": rank_before if rank_before is not None else "",
                "rank_after": rank_after if rank_after is not None else "",
                "steps_used": attack_info["steps_used"],
                "final_score": attack_info["final_score"],
                "success": is_success,
                "error": attack_info["error"],
            }
        )

        if (idx + 1) % 10 == 0:
            current_asr = success / (idx + 1)
            pbar.set_postfix({"asr": f"{current_asr:.3f}"})

    asr = success / max(len(dataset), 1)
    save_csv(
        rows,
        csv_path,
        fieldnames=[
            "sample_id",
            "target_id",
            "rank_before",
            "rank_after",
            "steps_used",
            "final_score",
            "success",
            "error",
        ],
    )
    if cleanup_mgr is not None:
        cleanup_mgr.maybe_cleanup(force=True)
    return asr, rows


def run_asr_multi_variant(
    dataset_path: str,
    k: int,
    budget: int,
    limit: Optional[int],
    limit_mode: str,
    limit_seed: Optional[int],
    save_path: str,
    csv_path: str,
    retriever: RetrievalBase,
    runner,
    cleanup_mgr: Optional[SavePathCleanupManager] = None,
) -> Dict[str, float]:
    dataset = load_dataset(dataset_path)
    dataset = _apply_limit(dataset, limit, limit_mode, limit_seed)

    print(
        f"[ASR-MV] Dataset loaded: {dataset_path} samples={len(dataset)} "
        f"topk={k} budget={budget} retrieval={retriever.name} "
        f"limit={limit} mode={limit_mode}"
    )
    print("[ASR-MV] Environment ready. Start mutation.")

    sampled_ids = {str(item.get("id")) for item in dataset if item.get("id") is not None}
    id_to_item = {
        str(item.get("id")): item for item in dataset if item.get("id") is not None
    }
    total = 0
    skipped = 0
    skipped_low_pre = 0
    errors = 0
    sum_asr1 = 0.0
    sum_asr2 = 0.0
    sum_asr3 = 0.0
    sum_asr4 = 0.0
    sum_wasr = 0.0
    sum_recall_pre = 0.0
    sum_recall_post = 0.0
    checked = 0
    rows: List[Dict] = []

    pbar = tqdm(enumerate(dataset), total=len(dataset), desc=f"ASR-MV@{k}", unit="sample")
    for idx, sample in pbar:
        if cleanup_mgr is not None:
            cleanup_mgr.maybe_cleanup()
        variants_raw = _get_variant_set(sample)
        if not variants_raw:
            skipped += 1
            rows.append(
                {
                    "sample_id": str(sample.get("id") or f"sample_{idx}"),
                    "sample_idx": idx,
                    "func_name": sample.get("func_name", ""),
                    "variant_count_raw": 0,
                    "variant_count": 0,
                    "variant_ids": "",
                    "status": "skipped",
                    "error": "no_variants",
                    "topk_before": "",
                    "topk_after": "",
                    "hits_pre": "",
                    "hits_post": "",
                    "pushed_out": "",
                    "asr1_hit": "",
                    "asr2_hit": "",
                    "asr3_hit": "",
                    "asr4_hit": "",
                    "wasr": "",
                    "recall_pre": "",
                    "recall_post": "",
                    "steps_used": "",
                    "final_score": "",
                }
            )
            continue
        variants = set(v for v in variants_raw if v in sampled_ids)
        sample_id = str(sample.get("id") or f"sample_{idx}")
        if sample_id in sampled_ids:
            variants.add(sample_id)
        if not variants:
            skipped += 1
            rows.append(
                {
                    "sample_id": str(sample.get("id") or f"sample_{idx}"),
                    "sample_idx": idx,
                    "func_name": sample.get("func_name", ""),
                    "variant_count_raw": len(variants_raw),
                    "variant_count": 0,
                    "variant_ids": "",
                    "status": "skipped",
                    "error": "variants_not_in_dataset",
                    "topk_before": "",
                    "topk_after": "",
                    "hits_pre": "",
                    "hits_post": "",
                    "pushed_out": "",
                    "asr1_hit": "",
                    "asr2_hit": "",
                    "asr3_hit": "",
                    "asr4_hit": "",
                    "wasr": "",
                    "recall_pre": "",
                    "recall_post": "",
                    "steps_used": "",
                    "final_score": "",
                }
            )
            continue

        variant_list = sorted(variants)
        tqdm.write(
            f"[ASR-MV] Start sample={idx} id={sample_id} func={sample.get('func_name')} variants={len(variants)}"
        )

        try:
            topk_before = retriever.topk(sample, dataset, k, query_kind="original")
            if not topk_before:
                raise ValueError("retrieval_before_empty")
        except Exception as e:
            errors += 1
            checked += 1
            tqdm.write(f"[ASR-MV] retrieval_before_failed:{e}")
            rows.append(
                {
                    "sample_id": sample_id,
                    "sample_idx": idx,
                    "func_name": sample.get("func_name", ""),
                    "variant_count_raw": len(variants_raw),
                    "variant_count": len(variant_list),
                    "variant_ids": format_id_list(variant_list),
                    "status": "error",
                    "error": f"retrieval_before_failed:{e}",
                    "topk_before": "",
                    "topk_after": "",
                    "hits_pre": "",
                    "hits_post": "",
                    "pushed_out": "",
                    "asr1_hit": "",
                    "asr2_hit": "",
                    "asr3_hit": "",
                    "asr4_hit": "",
                    "wasr": "",
                    "recall_pre": "",
                    "recall_post": "",
                    "steps_used": "",
                    "final_score": "",
                }
            )
            if checked % 6 == 0:
                denom = max(total, 1)
                tqdm.write(
                    "[ASR-MV] "
                    f"ASR@1={sum_asr1 / denom:.4f} (n={total}) "
                    f"ASR@2={sum_asr2 / denom:.4f} (n={total}) "
                    f"ASR@3={sum_asr3 / denom:.4f} (n={total}) "
                    f"ASR@4={sum_asr4 / denom:.4f} (n={total}) "
                    f"wASR={sum_wasr / denom:.4f} "
                    f"recall_pre={sum_recall_pre / denom:.4f} "
                    f"recall_post={sum_recall_post / denom:.4f}"
                )
            continue

        tqdm.write(f"[ASR-MV] TopK before ({retriever.name}): {format_topk(topk_before)}")
        variant_candidates = [id_to_item[v] for v in variant_list if v in id_to_item]
        variant_scores_pre = retriever.score_candidates(
            sample, variant_candidates, query_kind="original"
        )
        tqdm.write(
            f"[ASR-MV] Variant scores pre: {format_variant_scores(variant_list, variant_scores_pre)}"
        )

        topk_pre_ids = {
            str(item.get("id")) for item in topk_before if item.get("id") is not None
        }
        pre_topk_variants = topk_pre_ids & variants
        hits_pre = len(pre_topk_variants)
        if hits_pre < 4:
            skipped += 1
            skipped_low_pre += 1
            checked += 1
            tqdm.write(
                f"[ASR-MV] skipped_pre_hits<{4}: hits_pre={hits_pre} sample={sample_id}"
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "sample_idx": idx,
                    "func_name": sample.get("func_name", ""),
                    "variant_count_raw": len(variants_raw),
                    "variant_count": len(variant_list),
                    "variant_ids": format_id_list(variant_list),
                    "status": "skipped",
                    "error": f"insufficient_pre_hits:{hits_pre}<4",
                    "topk_before": format_topk(topk_before),
                    "topk_after": "",
                    "hits_pre": hits_pre,
                    "hits_post": "",
                    "pushed_out": "",
                    "asr1_hit": "",
                    "asr2_hit": "",
                    "asr3_hit": "",
                    "asr4_hit": "",
                    "wasr": "",
                    "recall_pre": "",
                    "recall_post": "",
                    "steps_used": "",
                    "final_score": "",
                }
            )
            if checked % 6 == 0:
                denom = max(total, 1)
                tqdm.write(
                    "[ASR-MV] "
                    f"ASR@1={sum_asr1 / denom:.4f} (n={total}) "
                    f"ASR@2={sum_asr2 / denom:.4f} (n={total}) "
                    f"ASR@3={sum_asr3 / denom:.4f} (n={total}) "
                    f"ASR@4={sum_asr4 / denom:.4f} (n={total}) "
                    f"wASR={sum_wasr / denom:.4f} "
                    f"recall_pre={sum_recall_pre / denom:.4f} "
                    f"recall_post={sum_recall_post / denom:.4f}"
                )
            continue

        attack_info = runner.attack_sample(sample_idx=idx, max_steps=budget)
        after_binary, after_binary_source = pick_after_binary(attack_info)
        mutated_query = make_mutated_query(sample, after_binary)
        tqdm.write(
            f"[ASR-MV] After-query source={after_binary_source} "
            f"best_score={attack_info.get('best_score', attack_info.get('final_score', 1.0)):.6f} "
            f"final_score={attack_info.get('final_score', 1.0):.6f}"
        )

        if attack_info.get("error"):
            errors += 1
            checked += 1
            tqdm.write(f"[ASR-MV] mutation_failed_skip_after:{attack_info['error']}")
            rows.append(
                {
                    "sample_id": sample_id,
                    "sample_idx": idx,
                    "func_name": sample.get("func_name", ""),
                    "variant_count_raw": len(variants_raw),
                    "variant_count": len(variant_list),
                    "variant_ids": format_id_list(variant_list),
                    "status": "error",
                    "error": f"mutation_failed_skip_after:{attack_info['error']}",
                    "topk_before": format_topk(topk_before),
                    "topk_after": "",
                    "hits_pre": hits_pre,
                    "hits_post": "",
                    "pushed_out": "",
                    "asr1_hit": "",
                    "asr2_hit": "",
                    "asr3_hit": "",
                    "asr4_hit": "",
                    "wasr": "",
                    "recall_pre": "",
                    "recall_post": "",
                    "steps_used": attack_info["steps_used"],
                    "final_score": attack_info["final_score"],
                }
            )
            if checked % 6 == 0:
                denom = max(total, 1)
                tqdm.write(
                    "[ASR-MV] "
                    f"ASR@1={sum_asr1 / denom:.4f} (n={total}) "
                    f"ASR@2={sum_asr2 / denom:.4f} (n={total}) "
                    f"ASR@3={sum_asr3 / denom:.4f} (n={total}) "
                    f"ASR@4={sum_asr4 / denom:.4f} (n={total}) "
                    f"wASR={sum_wasr / denom:.4f} "
                    f"recall_pre={sum_recall_pre / denom:.4f} "
                    f"recall_post={sum_recall_post / denom:.4f}"
                )
            continue

        try:
            topk_after = retriever.topk(mutated_query, dataset, k, query_kind="mutated")
            if not topk_after:
                raise ValueError("retrieval_after_empty")
        except Exception as e:
            errors += 1
            checked += 1
            tqdm.write(f"[ASR-MV] retrieval_after_failed:{e}")
            rows.append(
                {
                    "sample_id": sample_id,
                    "sample_idx": idx,
                    "func_name": sample.get("func_name", ""),
                    "variant_count_raw": len(variants_raw),
                    "variant_count": len(variant_list),
                    "variant_ids": format_id_list(variant_list),
                    "status": "error",
                    "error": f"retrieval_after_failed:{e}",
                    "topk_before": format_topk(topk_before),
                    "topk_after": "",
                    "hits_pre": hits_pre,
                    "hits_post": "",
                    "pushed_out": "",
                    "asr1_hit": "",
                    "asr2_hit": "",
                    "asr3_hit": "",
                    "asr4_hit": "",
                    "wasr": "",
                    "recall_pre": "",
                    "recall_post": "",
                    "steps_used": attack_info["steps_used"],
                    "final_score": attack_info["final_score"],
                }
            )
            if checked % 6 == 0:
                denom = max(total, 1)
                tqdm.write(
                    "[ASR-MV] "
                    f"ASR@1={sum_asr1 / denom:.4f} (n={total}) "
                    f"ASR@2={sum_asr2 / denom:.4f} (n={total}) "
                    f"ASR@3={sum_asr3 / denom:.4f} (n={total}) "
                    f"ASR@4={sum_asr4 / denom:.4f} (n={total}) "
                    f"wASR={sum_wasr / denom:.4f} "
                    f"recall_pre={sum_recall_pre / denom:.4f} "
                    f"recall_post={sum_recall_post / denom:.4f}"
                )
            continue

        tqdm.write(f"[ASR-MV] TopK after ({retriever.name}): {format_topk(topk_after)}")
        variant_scores_post = retriever.score_candidates(
            mutated_query, variant_candidates, query_kind="mutated"
        )
        tqdm.write(
            f"[ASR-MV] Variant scores post: {format_variant_scores(variant_list, variant_scores_post)}"
        )

        topk_post_ids = {
            str(item.get("id")) for item in topk_after if item.get("id") is not None
        }
        hits_post = len(topk_post_ids & pre_topk_variants)
        pushed_out = max(hits_pre - hits_post, 0)

        total += 1
        sum_asr1 += 1.0 if pushed_out >= 1 else 0.0
        sum_asr2 += 1.0 if pushed_out >= 2 else 0.0
        sum_asr3 += 1.0 if pushed_out >= 3 else 0.0
        sum_asr4 += 1.0 if pushed_out >= 4 else 0.0
        wasr = min(max(pushed_out, 0), 4) / 4.0
        sum_wasr += wasr
        sum_recall_pre += hits_pre / max(len(variants), 1)
        sum_recall_post += hits_post / max(len(variants), 1)
        checked += 1

        tqdm.write(
            f"[ASR-MV] hits_pre={hits_pre} hits_post={hits_post} pushed_out={pushed_out}"
        )
        recall_pre = hits_pre / max(len(variants), 1)
        recall_post = hits_post / max(len(variants), 1)
        rows.append(
            {
                "sample_id": sample_id,
                "sample_idx": idx,
                "func_name": sample.get("func_name", ""),
                "variant_count_raw": len(variants_raw),
                "variant_count": len(variant_list),
                "variant_ids": format_id_list(variant_list),
                "status": "ok",
                "error": "",
                "topk_before": format_topk(topk_before),
                "topk_after": format_topk(topk_after),
                "hits_pre": hits_pre,
                "hits_post": hits_post,
                "pushed_out": pushed_out,
                "asr1_hit": 1 if pushed_out >= 1 else 0,
                "asr2_hit": 1 if pushed_out >= 2 else 0,
                "asr3_hit": 1 if pushed_out >= 3 else 0,
                "asr4_hit": 1 if pushed_out >= 4 else 0,
                "wasr": wasr,
                "recall_pre": recall_pre,
                "recall_post": recall_post,
                "steps_used": attack_info["steps_used"],
                "final_score": attack_info["final_score"],
            }
        )
        if checked % 6 == 0:
            denom = max(total, 1)
            tqdm.write(
                "[ASR-MV] "
                f"ASR@1={sum_asr1 / denom:.4f} (n={total}) "
                f"ASR@2={sum_asr2 / denom:.4f} (n={total}) "
                f"ASR@3={sum_asr3 / denom:.4f} (n={total}) "
                f"ASR@4={sum_asr4 / denom:.4f} (n={total}) "
                f"wASR={sum_wasr / denom:.4f} "
                f"recall_pre={sum_recall_pre / denom:.4f} "
                f"recall_post={sum_recall_post / denom:.4f}"
            )

    denom = max(total, 1)
    metrics = {
        "ASR@1": sum_asr1 / denom,
        "ASR@2": sum_asr2 / denom,
        "ASR@3": sum_asr3 / denom,
        "ASR@4": sum_asr4 / denom,
        "wASR": sum_wasr / denom,
        "recall_pre": sum_recall_pre / denom,
        "recall_post": sum_recall_post / denom,
    }
    print(
        f"[ASR-MV] samples={total} skipped={skipped} "
        f"skipped_low_pre={skipped_low_pre} errors={errors}"
    )
    print(
        "[ASR-MV] "
        f"ASR@1={metrics['ASR@1']:.4f} (n={total}) "
        f"ASR@2={metrics['ASR@2']:.4f} (n={total}) "
        f"ASR@3={metrics['ASR@3']:.4f} (n={total}) "
        f"ASR@4={metrics['ASR@4']:.4f} (n={total}) "
        f"wASR={metrics['wASR']:.4f} "
        f"recall_pre={metrics['recall_pre']:.4f} "
        f"recall_post={metrics['recall_post']:.4f}"
    )
    save_csv(
        rows,
        csv_path,
        fieldnames=[
            "sample_id",
            "sample_idx",
            "func_name",
            "variant_count_raw",
            "variant_count",
            "variant_ids",
            "status",
            "error",
            "topk_before",
            "topk_after",
            "hits_pre",
            "hits_post",
            "pushed_out",
            "asr1_hit",
            "asr2_hit",
            "asr3_hit",
            "asr4_hit",
            "wasr",
            "recall_pre",
            "recall_post",
            "steps_used",
            "final_score",
        ],
    )
    if cleanup_mgr is not None:
        cleanup_mgr.maybe_cleanup(force=True)
    print(f"[ASR-MV] CSV saved to: {csv_path}")
    return metrics


def save_csv(
    rows: List[Dict],
    path: str,
    fieldnames: Optional[List[str]] = None,
) -> None:
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    default_dataset = os.path.join(repo_root, "utils/fast/dataset_test.json")
    default_model = os.path.join(
        os.path.dirname(repo_root), "rl_models/ppo_model_ep150.pt"
    )
    default_save = os.path.join(repo_root, "asr_workdir")

    parser = argparse.ArgumentParser(description="ASR@K experiment (clean)")
    parser.add_argument("--dataset", default=default_dataset)
    parser.add_argument("--model-path", default=default_model)
    parser.add_argument("-k", "--topk", type=int, default=10)
    parser.add_argument("-b", "--budget", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--limit-mode",
        choices=["head", "random", "variants", "variants_strict"],
        default="head",
        help="How to apply --limit: head (slice), random, variants-first, or variants_strict (cap to limit).",
    )
    parser.add_argument("--limit-seed", type=int, default=None)
    parser.add_argument("--save-path", default=default_save)
    parser.add_argument("--csv", default="asr_test.csv")
    parser.add_argument(
        "--cleanup-interval-seconds",
        type=int,
        default=0,
        help="Periodic cleanup interval for --save-path. 0 disables cleanup.",
    )
    parser.add_argument(
        "--cleanup-max-age-seconds",
        type=int,
        default=1800,
        help="Only remove temp folders/files older than this age.",
    )
    parser.add_argument(
        "--cleanup-include-containers",
        action="store_true",
        help="Also cleanup *_container folders under --save-path when they are old enough.",
    )
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["single", "multi_variant"],
        default="single",
        help="single uses target id; multi_variant uses variants list for robustness.",
    )
    parser.add_argument(
        "--retrieval-model",
        choices=["asm2vec", "safe", "jtrans", "gmn"],
        default="asm2vec",
        help="Retrieval model for Top-K ranking.",
    )
    parser.add_argument("--safe-checkpoint-dir", default=None)
    parser.add_argument("--safe-i2v-dir", default=None)
    parser.add_argument("--safe-use-gpu", action="store_true")
    parser.add_argument(
        "--safe-emb-cache-max",
        type=int,
        default=256,
        help="SAFE embedding cache size limit (LRU, 0 disables cache).",
    )
    parser.add_argument(
        "--safe-instr-cache-max",
        type=int,
        default=512,
        help="SAFE instruction cache size limit (LRU, 0 disables cache).",
    )
    parser.add_argument(
        "--uroboros-mem-mb",
        type=int,
        default=12288,
        help="Memory limit for python2 uroboros subprocess (MB, 0 disables limit).",
    )
    parser.add_argument(
        "--uroboros-timeout-sec",
        type=int,
        default=300,
        help="Timeout for python2 uroboros subprocess (seconds, 0 disables timeout).",
    )
    parser.add_argument("--jtrans-model-dir", default=None)
    parser.add_argument("--jtrans-tokenizer-dir", default=None)
    parser.add_argument("--jtrans-use-gpu", action="store_true")
    parser.add_argument(
        "--env-detection",
        choices=["asm2vec", "safe", "jtrans"],
        default="asm2vec",
        help="Detection method for env.step() scoring.",
    )
    parser.add_argument(
        "--retrieval-workers",
        type=int,
        default=1,
        help="Parallel workers for candidate scoring (1 disables parallelism).",
    )
    parser.add_argument(
        "--retrieval-log-every",
        type=int,
        default=0,
        help="Log candidate scoring every N items (0 disables).",
    )
    parser.add_argument(
        "--attack-agent",
        choices=["ppo", "ga"],
        default="ppo",
        help="Mutation attacker: PPO policy or GA baseline.",
    )
    parser.add_argument("--ga-population-size", type=int, default=6)
    parser.add_argument("--ga-generations", type=int, default=4)
    parser.add_argument("--ga-elite-size", type=int, default=2)
    parser.add_argument("--ga-mutation-rate", type=float, default=0.25)
    parser.add_argument("--ga-crossover-rate", type=float, default=0.70)
    parser.add_argument("--ga-seq-len", type=int, default=8)
    parser.add_argument("--ga-loc-slots", type=int, default=3)
    parser.add_argument("--ga-seed", type=int, default=None)
    parser.add_argument("--gmn-checkpoint-dir", default=None)
    parser.add_argument("--gmn-features-json", default=None)
    parser.add_argument("--gmn-dataset", default="one", choices=["one", "two", "vuln"])
    parser.add_argument("--gmn-features-type", default="opc", choices=["opc", "nofeatures"])
    parser.add_argument("--gmn-batch-size", type=int, default=20)
    parser.add_argument("--gmn-output-dir", default=None)
    parser.add_argument("--gmn-idb-root", default=None)
    parser.add_argument("--gmn-idb-prefix", default=None)
    parser.add_argument("--gmn-opcodes-json", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    os.environ["SAFE_EMB_CACHE_MAX"] = str(max(0, int(args.safe_emb_cache_max)))
    os.environ["SAFE_INSTR_CACHE_MAX"] = str(max(0, int(args.safe_instr_cache_max)))
    os.environ["UROBOROS_MEM_MB"] = str(max(0, int(args.uroboros_mem_mb)))
    os.environ["UROBOROS_TIMEOUT_SEC"] = str(max(0, int(args.uroboros_timeout_sec)))

    os.makedirs(args.save_path, exist_ok=True)
    csv_path = args.csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(args.save_path, csv_path)

    if args.env_detection == "jtrans" or args.retrieval_model == "jtrans":
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        jtrans_root = os.path.join(project_root, "detection_model", "jTrans")
        if args.jtrans_model_dir is None:
            args.jtrans_model_dir = os.path.join(jtrans_root, "models", "jTrans-finetune")
        if args.jtrans_tokenizer_dir is None:
            args.jtrans_tokenizer_dir = os.path.join(jtrans_root, "jtrans_tokenizer")

    retriever = build_retriever(args, args.save_path)
    cleanup_mgr = SavePathCleanupManager(
        save_path=args.save_path,
        interval_seconds=args.cleanup_interval_seconds,
        max_age_seconds=args.cleanup_max_age_seconds,
        include_containers=args.cleanup_include_containers,
    )
    if cleanup_mgr.enabled:
        print(
            "[ASR] Cleanup enabled: "
            f"interval={args.cleanup_interval_seconds}s "
            f"max_age={args.cleanup_max_age_seconds}s "
            f"include_containers={bool(args.cleanup_include_containers)}"
        )
    runner = build_attack_runner(
        args=args,
        dataset_path=args.dataset,
        save_path=args.save_path,
        max_steps=args.budget,
        retriever=retriever,
    )
    print(f"[ASR] Retrieval model: {retriever.name}")
    print(f"[ASR] Attack agent: {args.attack_agent}")
    if args.mode == "single":
        asr, rows = run_asr_simple(
            dataset_path=args.dataset,
            k=args.topk,
            budget=args.budget,
            limit=args.limit,
            limit_mode=args.limit_mode,
            limit_seed=args.limit_seed,
            save_path=args.save_path,
            csv_path=csv_path,
            retriever=retriever,
            runner=runner,
            cleanup_mgr=cleanup_mgr,
        )

        print(f"ASR@{args.topk}: {asr:.4f} ({sum(r['success'] for r in rows)}/{len(rows)})")
        print(f"CSV saved to: {csv_path}")
    else:
        run_asr_multi_variant(
            dataset_path=args.dataset,
            k=args.topk,
            budget=args.budget,
            limit=args.limit,
            limit_mode=args.limit_mode,
            limit_seed=args.limit_seed,
            save_path=args.save_path,
            csv_path=csv_path,
            retriever=retriever,
            runner=runner,
            cleanup_mgr=cleanup_mgr,
        )


if __name__ == "__main__":
    main()
