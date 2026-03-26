#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import random
import shutil
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import r2pipe
import seaborn as sns
from scipy import stats
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl_framework.utils.acfg.r2_acfg_features import RadareACFGExtractor

DATASET_PATH = os.path.join(SCRIPT_DIR, "dataset_train.json")
SAVE_DIR = os.path.join(SCRIPT_DIR, "chapter3_results_final")
TEMP_DIR = os.path.join(SCRIPT_DIR, "tmp", "binary_patch_test_exp3_2")

BAR_ORDER = ["Random", "Edge", "Critical"]
BAR_COLORS = {"Random": "#95a5a6", "Edge": "#3498db", "Critical": "#e74c3c"}


def parse_topk_list(text: str) -> List[int]:
    items = []
    for seg in str(text).split(","):
        seg = seg.strip()
        if not seg:
            continue
        try:
            k = int(seg)
        except Exception:
            continue
        if k <= 0:
            continue
        items.append(k)
    dedup = sorted(set(items))
    if not dedup:
        raise ValueError(f"Invalid --topk-list: {text}")
    return dedup


def apply_nop_patch(binary_path: str, target_addr: int, num_bytes: int = 4) -> Optional[str]:
    filename = os.path.basename(binary_path)
    temp_path = os.path.join(TEMP_DIR, f"patched_{random.randint(1000, 9999)}_{filename}")
    shutil.copy(binary_path, temp_path)
    try:
        r = r2pipe.open(temp_path, flags=["-w", "-2"])
        r.cmd("e asm.arch=x86")
        r.cmd("e asm.bits=64")
        hex_str = "90" * int(num_bytes)
        addr_str = hex(target_addr) if isinstance(target_addr, int) else str(target_addr)
        r.cmd(f"wx {hex_str} @ {addr_str}")
        r.quit()
        return temp_path
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return None


def extract_acfg(binary_path: str, func_name: str) -> Optional[Dict]:
    extractor = RadareACFGExtractor(binary_path)
    try:
        return extractor.get_acfg_features(function_name=func_name)
    finally:
        extractor.close()


def rank_block_addrs(acfg_data: Dict) -> List[int]:
    bbs = acfg_data.get("basic_blocks", {}) if acfg_data else {}
    if not bbs:
        return []
    ranked = sorted(
        bbs.items(),
        key=lambda kv: (
            -float(kv[1].get("critical_score", 0.0)),
            -float(kv[1].get("dominator_score", 0.0)),
            -float(kv[1].get("centrality_degree", 0.0)),
            kv[0],
        ),
    )
    return [addr for addr, _ in ranked]


def get_patch_targets(acfg_data: Dict) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    addrs = rank_block_addrs(acfg_data)
    if not addrs:
        return None, None, None
    return addrs[0], addrs[-1], random.choice(addrs)


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if abs(float(b)) > 1e-9 else 0.0


def block_to_vec32(bb: Dict) -> np.ndarray:
    n_inst = max(float(bb.get("n_instructions", 0.0)), 1.0)
    n_arith = float(bb.get("n_arith", 0.0))
    n_logic = float(bb.get("n_logic", 0.0))
    n_branch = float(bb.get("n_branch", 0.0))
    n_cmp = float(bb.get("n_cmp", 0.0))
    n_xor = float(bb.get("n_xor", 0.0))
    n_shift = float(bb.get("n_shift", 0.0))
    n_call = float(bb.get("n_call", 0.0))
    n_transfer = float(bb.get("n_transfer", 0.0))
    n_mem_read = float(bb.get("n_mem_read", 0.0))
    n_mem_write = float(bb.get("n_mem_write", 0.0))
    n_regs_gp = float(bb.get("n_regs_gp", 0.0))
    n_regs_vec = float(bb.get("n_regs_vec", 0.0))
    n_consts = float(bb.get("n_consts", 0.0))

    compute_ops = n_arith + n_logic
    mem_ops = n_mem_read + n_mem_write

    vec = [
        np.log1p(n_inst),
        safe_div(n_arith, n_inst),
        safe_div(n_logic, n_inst),
        safe_div(n_branch, n_inst),
        safe_div(n_cmp, n_inst),
        safe_div(n_xor, n_inst),
        safe_div(n_shift, n_inst),
        safe_div(n_call, n_inst),
        safe_div(n_transfer, n_inst),
        safe_div(n_mem_read, n_inst),
        safe_div(n_mem_write, n_inst),
        safe_div(n_consts, n_inst),
        safe_div(n_regs_gp, 16.0),
        safe_div(n_regs_vec, 16.0),
        float(bb.get("centrality_betweenness", 0.0)),
        float(bb.get("centrality_degree", 0.0)),
        np.log1p(float(bb.get("dominator_score", 0.0))),
        np.log1p(float(bb.get("postdominator_score", 0.0))),
        np.log1p(float(bb.get("loop_score", 0.0))),
        np.log1p(float(bb.get("control_dependence_score", 0.0))),
        float(bb.get("critical_score", 0.0)),
        float(bb.get("is_in_loop", 0.0)),
        safe_div(compute_ops, mem_ops + 1.0),
        safe_div(n_branch, compute_ops + 1.0),
        safe_div(n_mem_write, n_mem_read + 1.0),
        safe_div(n_arith, n_logic + 1.0),
        safe_div(n_xor + n_shift, n_inst),
        safe_div(n_consts, n_branch + 1.0),
        1.0 if n_branch > 1 else 0.0,
        1.0 if n_call > 0 else 0.0,
        1.0 if n_consts > 0 else 0.0,
        1.0 if n_inst < 5 else 0.0,
    ]
    return np.asarray(vec, dtype=np.float32)


def build_section_b_topk_vector(acfg_data: Dict, topk: int) -> Optional[np.ndarray]:
    if not acfg_data:
        return None
    bbs = acfg_data.get("basic_blocks", {})
    ranked = rank_block_addrs(acfg_data)
    if not ranked or not bbs:
        return None

    selected = ranked[: int(topk)]
    real_vectors = []
    for addr in selected:
        bb = bbs.get(addr)
        if bb is None:
            continue
        real_vectors.append(block_to_vec32(bb))
    if not real_vectors:
        return None

    vectors = [v.copy() for v in real_vectors]
    while len(vectors) < int(topk):
        vectors.append(np.zeros(32, dtype=np.float32))

    flat = np.concatenate(vectors, axis=0)
    mat = np.stack(real_vectors, axis=0)
    ctx_mean = np.mean(mat, axis=0)
    ctx_max = np.max(mat, axis=0)
    return np.concatenate([flat, ctx_mean, ctx_max], axis=0).astype(np.float32)


def build_topk_std(samples: List[Dict], topk_list: List[int]) -> Dict[int, np.ndarray]:
    collectors = {k: [] for k in topk_list}
    for sample in samples[:50]:
        try:
            acfg_data = extract_acfg(sample["binary_path"], sample["func_name"])
        except Exception:
            continue
        if not acfg_data:
            continue
        for k in topk_list:
            vec = build_section_b_topk_vector(acfg_data, k)
            if vec is not None:
                collectors[k].append(vec)

    std_map = {}
    for k, vecs in collectors.items():
        if not vecs:
            continue
        arr = np.stack(vecs, axis=0)
        std = np.std(arr, axis=0)
        std[std == 0] = 1.0
        std_map[k] = std.astype(np.float32)
    return std_map


def calc_normalized_rms_drift(vec_orig: np.ndarray, vec_new: np.ndarray, vec_std: np.ndarray) -> float:
    diff = (vec_orig - vec_new) / vec_std
    return float(np.sqrt(np.mean(np.square(diff))))


def collect_drifts(samples: List[Dict], topk_list: List[int], patch_bytes: int) -> Dict[int, Dict[str, List[float]]]:
    drifts = {k: {label: [] for label in BAR_ORDER} for k in topk_list}
    if not samples:
        return drifts

    std_map = build_topk_std(samples, topk_list)
    if not std_map:
        return drifts

    for sample in tqdm(samples, desc="Exp3.2-TopK"):
        orig_path = sample["binary_path"]
        func_name = sample["func_name"]
        try:
            acfg_orig = extract_acfg(orig_path, func_name)
        except Exception:
            continue
        if not acfg_orig:
            continue

        crit_addr, edge_addr, rand_addr = get_patch_targets(acfg_orig)
        if crit_addr is None:
            continue

        vec_orig_map = {}
        for k in topk_list:
            if k not in std_map:
                continue
            vec_o = build_section_b_topk_vector(acfg_orig, k)
            if vec_o is None or vec_o.shape != std_map[k].shape:
                continue
            vec_orig_map[k] = vec_o
        if not vec_orig_map:
            continue

        for label, target_addr in [("Critical", crit_addr), ("Edge", edge_addr), ("Random", rand_addr)]:
            patched_path = apply_nop_patch(orig_path, target_addr, num_bytes=patch_bytes)
            if not patched_path:
                continue
            try:
                acfg_new = extract_acfg(patched_path, func_name)
                if not acfg_new:
                    continue
                for k, vec_orig in vec_orig_map.items():
                    vec_new = build_section_b_topk_vector(acfg_new, k)
                    if vec_new is None or vec_new.shape != vec_orig.shape:
                        continue
                    dist = calc_normalized_rms_drift(vec_orig, vec_new, std_map[k])
                    if np.isfinite(dist) and dist > 0:
                        drifts[k][label].append(float(dist))
            except Exception:
                pass
            finally:
                if os.path.exists(patched_path):
                    os.remove(patched_path)
    return drifts


def ttest_greater(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    try:
        _, p = stats.ttest_ind(a, b, alternative="greater")
        return float(p)
    except TypeError:
        _, p2 = stats.ttest_ind(a, b)
        if float(np.mean(a)) > float(np.mean(b)):
            return float(p2 / 2.0)
        return float(1.0 - p2 / 2.0)


def summarize(drifts: Dict[int, Dict[str, List[float]]], topk_list: List[int]) -> Dict:
    out = {}
    for k in topk_list:
        out[k] = {"bars": {}, "p_val_critical_gt_edge": float("nan")}
        for label in BAR_ORDER:
            arr = np.asarray(drifts[k][label], dtype=np.float32)
            arr = arr[np.isfinite(arr) & (arr > 0)]
            if arr.size == 0:
                out[k]["bars"][label] = {"mean": 0.0, "sem": 0.0, "n": 0}
                continue
            sem = float(np.std(arr) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
            out[k]["bars"][label] = {"mean": float(np.mean(arr)), "sem": sem, "n": int(len(arr))}
        a = np.asarray(drifts[k]["Critical"], dtype=np.float32)
        b = np.asarray(drifts[k]["Edge"], dtype=np.float32)
        a = a[np.isfinite(a) & (a > 0)]
        b = b[np.isfinite(b) & (b > 0)]
        out[k]["p_val_critical_gt_edge"] = ttest_greater(a, b)
    return out


def significance_text(p_val: float) -> str:
    if not np.isfinite(p_val):
        return "n/a"
    if p_val < 0.001:
        return "***"
    if p_val < 0.01:
        return "**"
    if p_val < 0.05:
        return "*"
    return "ns"


def plot_topk_grid(stats_map: Dict, topk_list: List[int], save_path: str) -> None:
    try:
        plt.style.use("seaborn-v0_8-paper")
    except Exception:
        sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 14

    n = len(topk_list)
    ncols = 2 if n > 1 else 1
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.0 * ncols, 6.0 * nrows), squeeze=False)

    global_max = 0.0
    for idx, k in enumerate(topk_list):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]
        bars_info = stats_map[k]["bars"]
        means = [bars_info[label]["mean"] for label in BAR_ORDER]
        errs = [bars_info[label]["sem"] for label in BAR_ORDER]
        colors = [BAR_COLORS[label] for label in BAR_ORDER]
        x = np.arange(len(BAR_ORDER))
        bars = ax.bar(
            x,
            means,
            yerr=errs,
            capsize=6,
            color=colors,
            edgecolor="white",
            linewidth=1.2,
            alpha=0.9,
        )
        local_max = max(means) + max(errs) if means else 0.0
        global_max = max(global_max, local_max)

        anchor = max(errs) if max(errs) > 0 else (local_max * 0.05 if local_max > 0 else 0.02)
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + anchor * 0.65,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
            )

        p_val = stats_map[k]["p_val_critical_gt_edge"]
        ax.text(
            0.98,
            0.98,
            f"{significance_text(p_val)} | p={p_val:.2e}" if np.isfinite(p_val) else "p=n/a",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=11,
            color="#c0392b",
            fontweight="bold",
        )
        ax.set_title(f"Section B Sensitivity (Top-{k})", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(BAR_ORDER, fontsize=12)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if c == 0:
            ax.set_ylabel("Normalized RMS Drift", fontsize=12)

    if global_max <= 0:
        global_max = 1.0
    for idx, _ in enumerate(topk_list):
        r = idx // ncols
        c = idx % ncols
        axes[r][c].set_ylim(0, global_max * 1.4)

    total_slots = nrows * ncols
    for idx in range(n, total_slots):
        r = idx // ncols
        c = idx % ncols
        axes[r][c].axis("off")

    fig.suptitle("Exp3.2: Section-B Top-K Sensitivity (No RL)", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(save_path, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def run_experiment(args: argparse.Namespace) -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    if not os.path.exists(args.dataset_path):
        print(f"Dataset not found: {args.dataset_path}")
        return

    with open(args.dataset_path, "r") as f:
        data = json.load(f)

    candidates = [
        d
        for d in data
        if d.get("func_name") != "main"
        and args.min_size < d.get("size", 0) < args.max_size
        and d.get("binary_path")
        and d.get("func_name")
    ]
    if not candidates:
        print("No candidate samples.")
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        return

    sample_size = min(int(args.sample_size), len(candidates))
    samples = random.sample(candidates, sample_size)
    print(f"[*] Selected {len(samples)} samples.")
    print(f"[*] Top-K list: {args.topk_list}")

    try:
        drifts = collect_drifts(samples, args.topk_list, patch_bytes=int(args.patch_bytes))
        stats_map = summarize(drifts, args.topk_list)

        for k in args.topk_list:
            print(f"\n[Top-{k}]")
            for label in BAR_ORDER:
                item = stats_map[k]["bars"][label]
                print(f"{label:>8}: mean={item['mean']:.4f} sem={item['sem']:.4f} n={item['n']}")
            p_val = stats_map[k]["p_val_critical_gt_edge"]
            if np.isfinite(p_val):
                print(f"Critical > Edge p-value: {p_val:.4e} ({significance_text(p_val)})")
            else:
                print("Critical > Edge p-value: n/a")

        fig_path = os.path.join(SAVE_DIR, "exp3_2_sectionB_topk_sensitivity.png")
        plot_topk_grid(stats_map, args.topk_list, fig_path)
        print(f"\nSaved figure: {fig_path}")

        json_path = os.path.join(SAVE_DIR, "exp3_2_sectionB_topk_sensitivity_stats.json")
        with open(json_path, "w") as f:
            json.dump(
                {
                    "topk_list": args.topk_list,
                    "sample_size": len(samples),
                    "patch_bytes": int(args.patch_bytes),
                    "stats": stats_map,
                },
                f,
                indent=2,
            )
        print(f"Saved stats: {json_path}")
    finally:
        shutil.rmtree(TEMP_DIR, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp3.2 Section-B Top-K sensitivity analysis (No RL)")
    parser.add_argument("--dataset-path", default=DATASET_PATH)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--topk-list", default="1,3,5,7")
    parser.add_argument("--patch-bytes", type=int, default=4)
    parser.add_argument("--min-size", type=int, default=200)
    parser.add_argument("--max-size", type=int, default=5000)
    args = parser.parse_args()
    args.topk_list = parse_topk_list(args.topk_list)
    return args


if __name__ == "__main__":
    random.seed(2024)
    np.random.seed(2024)
    run_experiment(parse_args())
