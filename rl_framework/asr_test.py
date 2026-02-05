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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
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


def make_mutated_query(sample: Dict, mutated_binary: Optional[str]) -> Dict:
    mutated = dict(sample)
    if mutated_binary:
        mutated["binary_path"] = mutated_binary
    mutated.pop("func_addr", None)
    base = f"{mutated.get('binary_path', '')}::{mutated.get('func_name', '')}"
    mutated["id"] = hashlib.md5(base.encode("utf-8")).hexdigest()[:8]
    return mutated


def _get_variant_set(sample: Dict) -> Optional[set]:
    variants = sample.get("variants")
    if not variants or not isinstance(variants, list):
        return None
    cleaned = [str(v) for v in variants if v]
    if not cleaned:
        return None
    return set(cleaned)


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
        self._thread_local = threading.local()

    def _get_worker_dir(self, cand_idx: int) -> Optional[str]:
        if not self.asm_work_dir:
            return None
        if self.retrieval_workers <= 1:
            return self.asm_work_dir
        worker_dir = os.path.join(self.asm_work_dir, f"cand_{cand_idx}")
        os.makedirs(worker_dir, exist_ok=True)
        return worker_dir

    def _get_safe_cache(self) -> Dict:
        if self.retrieval_workers > 1:
            cache = getattr(self._thread_local, "safe_cache", None)
            if cache is None:
                cache = {}
                self._thread_local.safe_cache = cache
            return cache
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
    ) -> None:
        self.dataset_path = dataset_path
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        print(f"[ASR] Loading model: {model_path} (device={device})")
        load_start = time.time()
        self.agent = PPOAgent(state_dim=state_dim, device=device)
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
        return {
            "steps_used": steps_used,
            "final_score": final_score,
            "final_binary": final_binary,
            "error": error,
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
    raise ValueError(f"Unknown retrieval model: {args.retrieval_model}")


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
    model_path: str,
    k: int,
    budget: int,
    limit: Optional[int],
    limit_mode: str,
    limit_seed: Optional[int],
    save_path: str,
    csv_path: str,
    use_gpu: bool,
    env_detection: str,
    retriever: RetrievalBase,
) -> Tuple[float, List[Dict]]:
    dataset = load_dataset(dataset_path)
    dataset = _apply_limit(dataset, limit, limit_mode, limit_seed)

    print(
        f"[ASR] Dataset loaded: {dataset_path} samples={len(dataset)} "
        f"topk={k} budget={budget} retrieval={retriever.name} "
        f"limit={limit} mode={limit_mode}"
    )
    runner = RLAgentRunner(
        dataset_path=dataset_path,
        model_path=model_path,
        save_path=save_path,
        max_steps=budget,
        use_gpu=use_gpu,
        detection_method=env_detection,
        safe_checkpoint_dir=getattr(retriever, "checkpoint_dir", None),
        safe_i2v_dir=getattr(retriever, "i2v_dir", None),
        safe_use_gpu=getattr(retriever, "use_gpu", False),
    )
    print("[ASR] Environment ready. Start mutation.")

    rows: List[Dict] = []
    success = 0

    pbar = tqdm(enumerate(dataset), total=len(dataset), desc=f"ASR@{k}", unit="sample")
    for idx, sample in pbar:
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
        mutated_query = make_mutated_query(sample, attack_info["final_binary"])

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
    save_csv(rows, csv_path)
    return asr, rows


def run_asr_multi_variant(
    dataset_path: str,
    model_path: str,
    k: int,
    budget: int,
    limit: Optional[int],
    limit_mode: str,
    limit_seed: Optional[int],
    save_path: str,
    use_gpu: bool,
    env_detection: str,
    retriever: RetrievalBase,
) -> Dict[str, float]:
    dataset = load_dataset(dataset_path)
    dataset = _apply_limit(dataset, limit, limit_mode, limit_seed)

    print(
        f"[ASR-MV] Dataset loaded: {dataset_path} samples={len(dataset)} "
        f"topk={k} budget={budget} retrieval={retriever.name} "
        f"limit={limit} mode={limit_mode}"
    )
    runner = RLAgentRunner(
        dataset_path=dataset_path,
        model_path=model_path,
        save_path=save_path,
        max_steps=budget,
        use_gpu=use_gpu,
        detection_method=env_detection,
        safe_checkpoint_dir=getattr(retriever, "checkpoint_dir", None),
        safe_i2v_dir=getattr(retriever, "i2v_dir", None),
        safe_use_gpu=getattr(retriever, "use_gpu", False),
    )
    print("[ASR-MV] Environment ready. Start mutation.")

    sampled_ids = {str(item.get("id")) for item in dataset if item.get("id") is not None}
    total = 0
    skipped = 0
    errors = 0
    sum_asr1 = 0.0
    sum_asr3 = 0.0
    sum_asr5 = 0.0
    sum_wasr = 0.0
    sum_recall_pre = 0.0
    sum_recall_post = 0.0
    checked = 0

    pbar = tqdm(enumerate(dataset), total=len(dataset), desc=f"ASR-MV@{k}", unit="sample")
    for idx, sample in pbar:
        variants = _get_variant_set(sample)
        if not variants:
            skipped += 1
            continue
        variants = set(v for v in variants if v in sampled_ids)
        if not variants:
            skipped += 1
            continue

        sample_id = str(sample.get("id") or f"sample_{idx}")
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
            if checked % 6 == 0:
                denom = max(total, 1)
                tqdm.write(
                    "[ASR-MV] "
                    f"ASR@1={sum_asr1 / denom:.4f} "
                    f"ASR@3={sum_asr3 / denom:.4f} "
                    f"ASR@5={sum_asr5 / denom:.4f} "
                    f"wASR={sum_wasr / denom:.4f} "
                    f"recall_pre={sum_recall_pre / denom:.4f} "
                    f"recall_post={sum_recall_post / denom:.4f}"
                )
            continue

        attack_info = runner.attack_sample(sample_idx=idx, max_steps=budget)
        mutated_query = make_mutated_query(sample, attack_info["final_binary"])

        if attack_info.get("error"):
            errors += 1
            checked += 1
            tqdm.write(f"[ASR-MV] mutation_failed_skip_after:{attack_info['error']}")
            if checked % 6 == 0:
                denom = max(total, 1)
                tqdm.write(
                    "[ASR-MV] "
                    f"ASR@1={sum_asr1 / denom:.4f} "
                    f"ASR@3={sum_asr3 / denom:.4f} "
                    f"ASR@5={sum_asr5 / denom:.4f} "
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
            if checked % 6 == 0:
                denom = max(total, 1)
                tqdm.write(
                    "[ASR-MV] "
                    f"ASR@1={sum_asr1 / denom:.4f} "
                    f"ASR@3={sum_asr3 / denom:.4f} "
                    f"ASR@5={sum_asr5 / denom:.4f} "
                    f"wASR={sum_wasr / denom:.4f} "
                    f"recall_pre={sum_recall_pre / denom:.4f} "
                    f"recall_post={sum_recall_post / denom:.4f}"
                )
            continue

        topk_pre_ids = {str(item.get("id")) for item in topk_before if item.get("id") is not None}
        topk_post_ids = {str(item.get("id")) for item in topk_after if item.get("id") is not None}

        hits_pre = len(topk_pre_ids & variants)
        hits_post = len(topk_post_ids & variants)
        pushed_out = max(hits_pre - hits_post, 0)

        total += 1
        sum_asr1 += 1.0 if pushed_out >= 1 else 0.0
        sum_asr3 += 1.0 if pushed_out >= 3 else 0.0
        sum_asr5 += 1.0 if pushed_out >= 5 else 0.0
        sum_wasr += min(max(pushed_out, 0), 5) / 5.0
        sum_recall_pre += hits_pre / max(len(variants), 1)
        sum_recall_post += hits_post / max(len(variants), 1)
        checked += 1

        tqdm.write(
            f"[ASR-MV] hits_pre={hits_pre} hits_post={hits_post} pushed_out={pushed_out}"
        )
        if checked % 6 == 0:
            denom = max(total, 1)
            tqdm.write(
                "[ASR-MV] "
                f"ASR@1={sum_asr1 / denom:.4f} "
                f"ASR@3={sum_asr3 / denom:.4f} "
                f"ASR@5={sum_asr5 / denom:.4f} "
                f"wASR={sum_wasr / denom:.4f} "
                f"recall_pre={sum_recall_pre / denom:.4f} "
                f"recall_post={sum_recall_post / denom:.4f}"
            )

    denom = max(total, 1)
    metrics = {
        "ASR@1": sum_asr1 / denom,
        "ASR@3": sum_asr3 / denom,
        "ASR@5": sum_asr5 / denom,
        "wASR": sum_wasr / denom,
        "recall_pre": sum_recall_pre / denom,
        "recall_post": sum_recall_post / denom,
    }
    print(f"[ASR-MV] samples={total} skipped={skipped} errors={errors}")
    print(
        "[ASR-MV] "
        f"ASR@1={metrics['ASR@1']:.4f} "
        f"ASR@3={metrics['ASR@3']:.4f} "
        f"ASR@5={metrics['ASR@5']:.4f} "
        f"wASR={metrics['wASR']:.4f} "
        f"recall_pre={metrics['recall_pre']:.4f} "
        f"recall_post={metrics['recall_post']:.4f}"
    )
    return metrics


def save_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    fieldnames = [
        "sample_id",
        "target_id",
        "rank_before",
        "rank_after",
        "steps_used",
        "final_score",
        "success",
        "error",
    ]
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
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["single", "multi_variant"],
        default="single",
        help="single uses target id; multi_variant uses variants list for robustness.",
    )
    parser.add_argument(
        "--retrieval-model",
        choices=["asm2vec", "safe"],
        default="asm2vec",
        help="Retrieval model for Top-K ranking.",
    )
    parser.add_argument("--safe-checkpoint-dir", default=None)
    parser.add_argument("--safe-i2v-dir", default=None)
    parser.add_argument("--safe-use-gpu", action="store_true")
    parser.add_argument(
        "--env-detection",
        choices=["asm2vec", "safe"],
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    csv_path = args.csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(args.save_path, csv_path)

    retriever = build_retriever(args, args.save_path)
    print(f"[ASR] Retrieval model: {retriever.name}")
    if args.mode == "single":
        asr, rows = run_asr_simple(
            dataset_path=args.dataset,
            model_path=args.model_path,
            k=args.topk,
            budget=args.budget,
            limit=args.limit,
            limit_mode=args.limit_mode,
            limit_seed=args.limit_seed,
            save_path=args.save_path,
            csv_path=csv_path,
            use_gpu=args.use_gpu,
            env_detection=args.env_detection,
            retriever=retriever,
        )

        print(f"ASR@{args.topk}: {asr:.4f} ({sum(r['success'] for r in rows)}/{len(rows)})")
        print(f"CSV saved to: {csv_path}")
    else:
        run_asr_multi_variant(
            dataset_path=args.dataset,
            model_path=args.model_path,
            k=args.topk,
            budget=args.budget,
            limit=args.limit,
            limit_mode=args.limit_mode,
            limit_seed=args.limit_seed,
            save_path=args.save_path,
            use_gpu=args.use_gpu,
            env_detection=args.env_detection,
            retriever=retriever,
        )


if __name__ == "__main__":
    main()
