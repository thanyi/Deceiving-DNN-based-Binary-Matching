#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple ASR@K experiment runner.

This script is intentionally modular:
- asm2vec_topk(...) is a placeholder interface you can replace later.
- RLAgentRunner wraps the PPO model and the environment.
"""

import argparse
import csv
import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Local imports
from env_wrapper import BinaryPerturbationEnv
from ppo_agent import PPOAgent

try:
    # Uses the project's asm2vec pipeline (bin2asm + compare_functions).
    from run_utils import run_one
except Exception:
    run_one = None


def load_dataset(path: str) -> List[Dict]:
    """Load a JSON list dataset and fail fast if the shape is wrong."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Dataset must be a list, got: {type(data)}")
    return data


def _stable_similarity(query_id: str, cand_id: str) -> float:
    """
    A deterministic similarity stub.
    Replace this with your real asm2vec retrieval later.
    """
    q = hashlib.md5(query_id.encode("utf-8")).hexdigest()
    c = hashlib.md5(cand_id.encode("utf-8")).hexdigest()
    # Count matching hex chars for a stable pseudo-score.
    matches = sum(1 for a, b in zip(q, c) if a == b)
    return matches / len(q)


def asm2vec_topk(query: Dict, dataset: List[Dict], k: int) -> List[Dict]:
    """
    Placeholder asm2vec retrieval interface.

    Contract:
        asm2vec_topk(query, dataset, k) -> List[{"id": str, "score": float}]
    """
    query_id = str(query.get("id") or query.get("func_name") or "unknown_query")
    scored = []
    for cand in dataset:
        cand_id = str(cand.get("id") or cand.get("func_name") or "unknown_cand")
        score = _stable_similarity(query_id, cand_id)
        scored.append({"id": cand_id, "score": float(score)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]


def choose_target_id(sample: Dict, sample_id: str, topk_before: List[Dict]) -> Optional[str]:
    """
    Decide the "correct match" we try to knock out of Top-K.

    Priority:
    1) Use ground-truth style ids if present.
    2) Otherwise use the best non-self item from the initial Top-K.
    """
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


def make_mutated_query(sample: Dict, mutated_binary: Optional[str]) -> Dict:
    """
    Build a synthetic query descriptor for post-attack retrieval.

    We mint a new id so the mutated query does not collide with the original.
    """
    mutated = dict(sample)
    if mutated_binary:
        mutated["binary_path"] = mutated_binary
    base = f"{mutated.get('binary_path', '')}::{mutated.get('func_name', '')}"
    mutated["id"] = hashlib.md5(base.encode("utf-8")).hexdigest()[:8]
    return mutated


def _find_sym_to_addr(binary_path: str) -> Optional[str]:
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


class RLAgentRunner:
    """
    Thin wrapper that:
    - Loads the PPO model.
    - Forces the env to run a specific dataset sample.
    """

    def __init__(
        self,
        dataset_path: str,
        model_path: str,
        save_path: str,
        max_steps: int,
        state_dim: int = 256,
        use_gpu: bool = False,
    ) -> None:
        self.dataset_path = dataset_path
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        self.agent = PPOAgent(state_dim=state_dim, device=device)
        self.agent.load(model_path)
        self.agent.policy.eval()

        # Use a very large hold interval; we manually pin the sample anyway.
        self.env = BinaryPerturbationEnv(
            save_path=self.save_path,
            dataset_path=self.dataset_path,
            sample_hold_interval=10**9,
            max_steps=max_steps,
        )
        self.env.set_state_dim(state_dim)

    def _pin_sample(self, sample_idx: int) -> None:
        # The environment normally samples targets; pinning removes randomness.
        sample = self.env.dataset[sample_idx]
        self.env.current_sample_idx = sample_idx
        self.env.current_sample_data = sample
        # Reset the hold counter so reset() will "keep target".
        self.env.episodes_on_current = 0
        # Pin the actual target fields used by step()/evaluate().
        self.env.original_binary = sample["binary_path"]
        self.env.function_name = sample["func_name"]

    def attack_sample(self, sample_idx: int, max_steps: int) -> Dict:
        """
        Run a greedy PPO attack for at most max_steps.

        Greedy = explore=False, so the policy picks its current best guess.
        """
        self._pin_sample(sample_idx)
        state = self.env.reset(force_switch=False)

        steps_used = 0
        final_score = 1.0
        final_binary = None
        error = ""

        for step in range(max_steps):
            (
                _joint_idx,
                loc_idx,
                _act_idx,
                actual_action,
                _log_prob,
                _value,
            ) = self.agent.select_action(state, explore=False)

            next_state, _reward, done, info = self.env.step(actual_action, loc_idx)
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

        return {
            "steps_used": steps_used,
            "final_score": final_score,
            "final_binary": final_binary,
            "error": error,
        }


def run_asr_simple(
    dataset_path: str,
    model_path: str,
    k: int,
    budget: int,
    limit: Optional[int],
    save_path: str,
    csv_path: str,
    use_gpu: bool,
    retrieval_mode: str,
) -> Tuple[float, List[Dict]]:
    dataset = load_dataset(dataset_path)
    if limit is not None:
        dataset = dataset[:limit]

    runner = RLAgentRunner(
        dataset_path=dataset_path,
        model_path=model_path,
        save_path=save_path,
        max_steps=budget,
        use_gpu=use_gpu,
    )

    # Shared asm2vec work dir and caches reduce repeated bin->asm extraction.
    retrieval_ctx = {
        "asm_work_dir": os.path.join(save_path, "_asm2vec_retrieval"),
        "original_asm_cache": {},
        "sym_to_addr_cache": {},
    }
    os.makedirs(retrieval_ctx["asm_work_dir"], exist_ok=True)

    rows: List[Dict] = []
    success = 0

    for idx, sample in enumerate(dataset):
        sample_id = str(sample.get("id") or f"sample_{idx}")

        try:
            topk_before = asm2vec_topk_with_mode(
                sample,
                dataset,
                k,
                retrieval_mode=retrieval_mode,
                retrieval_ctx=retrieval_ctx,
            )
        except Exception as e:  # pragma: no cover - defensive
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

        # Attack the pinned sample with the trained PPO policy.
        attack_info = runner.attack_sample(sample_idx=idx, max_steps=budget)
        mutated_query = make_mutated_query(sample, attack_info["final_binary"])

        try:
            topk_after = asm2vec_topk_with_mode(
                mutated_query,
                dataset,
                k,
                retrieval_mode=retrieval_mode,
                retrieval_ctx=retrieval_ctx,
            )
        except Exception as e:  # pragma: no cover - defensive
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
        # Success = the chosen target is no longer present in post-attack Top-K.
        is_success = int(target_id is not None and rank_after is None)
        success += is_success

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
            print(f"[progress] {idx+1}/{len(dataset)} | ASR@{k}={current_asr:.3f}")

    asr = success / max(len(dataset), 1)
    save_csv(rows, csv_path)
    return asr, rows


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
        os.path.dirname(repo_root), "rl_models/ppo_model_ep750.pt"
    )
    default_save = os.path.join(repo_root, "asr_workdir")

    parser = argparse.ArgumentParser(description="Simple ASR@K experiment")
    parser.add_argument("--dataset", default=default_dataset)
    parser.add_argument("--model-path", default=default_model)
    parser.add_argument("-k", "--topk", type=int, default=10)
    parser.add_argument("-b", "--budget", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-path", default=default_save)
    parser.add_argument("--csv", default="asr_simple.csv")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument(
        "--retrieval-mode",
        choices=["run_one", "stub"],
        default="run_one",
        help="run_one uses run_utils.py asm2vec pipeline; stub is a deterministic placeholder.",
    )
    return parser


def asm2vec_topk_with_mode(
    query: Dict,
    dataset: List[Dict],
    k: int,
    retrieval_mode: str,
    retrieval_ctx: Optional[Dict],
) -> List[Dict]:
    """
    Retrieval wrapper:
    - run_one: uses run_utils.run_one(simple_mode=True) for real scores.
    - stub: stable hash-based placeholder.
    """
    if retrieval_mode != "run_one":
        return asm2vec_topk(query, dataset, k)

    if run_one is None:
        return asm2vec_topk(query, dataset, k)

    func_name = query.get("func_name")
    original_binary = query.get("binary_path")
    if not func_name or not original_binary:
        return asm2vec_topk(query, dataset, k)

    asm_work_dir = None
    original_asm_cache = None
    sym_to_addr_cache = None
    if retrieval_ctx:
        asm_work_dir = retrieval_ctx.get("asm_work_dir")
        original_asm_cache = retrieval_ctx.get("original_asm_cache")
        sym_to_addr_cache = retrieval_ctx.get("sym_to_addr_cache")

    scored: List[Dict] = []
    sym_to_addr_path = None
    if sym_to_addr_cache is not None:
        sym_to_addr_path = sym_to_addr_cache.get(original_binary)
        if sym_to_addr_path is None:
            sym_to_addr_path = _find_sym_to_addr(original_binary)
            sym_to_addr_cache[original_binary] = sym_to_addr_path
    else:
        sym_to_addr_path = _find_sym_to_addr(original_binary)

    for cand in dataset:
        cand_binary = cand.get("binary_path")
        if not cand_binary or not os.path.exists(cand_binary):
            continue

        # run_one is heavy but accurate: it extracts asm and compares functions.
        score, _grad = run_one(
            original_binary,
            cand_binary,
            model_original=None,
            checkdict={},
            function_name=str(func_name),
            detection_method="asm2vec",
            asm_work_dir=asm_work_dir,
            original_asm_cache=original_asm_cache,
            simple_mode=True,
            mutated_func_addr=cand.get("func_addr"),
            sym_to_addr_path=sym_to_addr_path,
        )
        if score is None:
            continue

        cand_id = str(cand.get("id") or cand.get("func_name") or "unknown_cand")
        scored.append({"id": cand_id, "score": float(score)})

    if not scored:
        return asm2vec_topk(query, dataset, k)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    csv_path = args.csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(args.save_path, csv_path)

    asr, rows = run_asr_simple(
        dataset_path=args.dataset,
        model_path=args.model_path,
        k=args.topk,
        budget=args.budget,
        limit=args.limit,
        save_path=args.save_path,
        csv_path=csv_path,
        use_gpu=args.use_gpu,
        retrieval_mode=args.retrieval_mode,
    )

    print(f"ASR@{args.topk}: {asr:.4f} ({sum(r['success'] for r in rows)}/{len(rows)})")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
