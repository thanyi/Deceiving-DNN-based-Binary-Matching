#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import random
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import r2pipe
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, "/home/ycy/ours/Deceiving-DNN-based-Binary-Matching")
from rl_framework.env_wrapper import BinaryPerturbationEnv
from rl_framework.utils.acfg.r2_acfg_features import RadareACFGExtractor


DATASET_PATH = "dataset_train.json"
SAVE_DIR = "chapter3_results_final"
TEMP_DIR = "tmp/binary_patch_test_exp3_2"

SECTION_SLICES = {
    "A": slice(0, 40),
    "B": slice(40, 200),
    "C": slice(200, 240),
}
BAR_ORDER = ["Random", "Edge", "Critical"]
BAR_COLORS = {"Random": "#95a5a6", "Edge": "#3498db", "Critical": "#e74c3c"}


def apply_nop_patch(binary_path, target_addr, num_bytes=4):
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


def get_target_blocks(binary_path, func_name):
    extractor = RadareACFGExtractor(binary_path)
    data = extractor.get_acfg_features(function_name=func_name)
    extractor.close()
    if not data or not data.get("basic_blocks"):
        return None, None, None
    scored_blocks = [
        (addr, info.get("critical_score", 0.0))
        for addr, info in data["basic_blocks"].items()
    ]
    if not scored_blocks:
        return None, None, None
    scored_blocks.sort(key=lambda x: x[1], reverse=True)
    addrs = [addr for addr, _ in scored_blocks]
    return scored_blocks[0][0], scored_blocks[-1][0], random.choice(addrs)


def get_vec240(env, binary_path, func_name):
    vec = np.asarray(env.extract_features_from_function(binary_path, func_name)[16:], dtype=np.float32)
    if vec.shape[0] < 240:
        return None
    return vec[:240]


def build_section_std(base_vecs):
    sec_std = {}
    for sec, sec_slice in SECTION_SLICES.items():
        arr = np.stack([v[sec_slice] for v in base_vecs], axis=0)
        std = np.std(arr, axis=0)
        std[std == 0] = 1.0
        sec_std[sec] = std
    return sec_std


def calc_section_dist(vec_orig, vec_new, sec_std):
    out = {}
    for sec, sec_slice in SECTION_SLICES.items():
        out[sec] = float(np.linalg.norm((vec_orig[sec_slice] - vec_new[sec_slice]) / sec_std[sec]))
    return out


def summarize(drifts):
    stats = {}
    for sec in SECTION_SLICES.keys():
        stats[sec] = {}
        for label in BAR_ORDER:
            arr = np.asarray(drifts[sec][label], dtype=np.float32)
            if arr.size == 0:
                stats[sec][label] = {"mean": 0.0, "sem": 0.0, "n": 0}
                continue
            sem = float(np.std(arr) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
            stats[sec][label] = {"mean": float(np.mean(arr)), "sem": sem, "n": int(len(arr))}
    return stats


def plot_three_subplots(stats, save_path):
    try:
        plt.style.use("seaborn-v0_8-paper")
    except Exception:
        sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 11

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, sec in zip(axes, ["A", "B", "C"]):
        means = [stats[sec][label]["mean"] for label in BAR_ORDER]
        errs = [stats[sec][label]["sem"] for label in BAR_ORDER]
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
        for idx, bar in enumerate(bars):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + (max(errs) if errs else 0.0) * 0.8 + 1e-6,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        ax.set_title(f"Section {sec}")
        ax.set_xticks(x)
        ax.set_xticklabels(BAR_ORDER)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    axes[0].set_ylabel("Normalized L2 Drift")
    fig.suptitle("Exp3.2: Section-wise Sensitivity (A/B/C)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")


def run_experiment():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found: {DATASET_PATH}")
        return

    with open(DATASET_PATH, "r") as f:
        data = json.load(f)
    candidates = [d for d in data if d.get("func_name") != "main" and 200 < d.get("size", 0) < 5000]
    samples = random.sample(candidates, min(50, len(candidates)))
    if not samples:
        print("No candidate samples.")
        return

    env = BinaryPerturbationEnv(save_path=TEMP_DIR, dataset_path=DATASET_PATH)
    drifts = {sec: {label: [] for label in BAR_ORDER} for sec in SECTION_SLICES.keys()}

    base_vecs = []
    for s in samples:
        try:
            vec = get_vec240(env, s["binary_path"], s["func_name"])
            if vec is not None:
                base_vecs.append(vec)
        except Exception:
            pass
    if not base_vecs:
        print("Failed to build base vectors.")
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        return
    sec_std = build_section_std(base_vecs)

    for sample in tqdm(samples, desc="Exp3.2"):
        orig_path = sample["binary_path"]
        fname = sample["func_name"]
        try:
            vec_orig = get_vec240(env, orig_path, fname)
            if vec_orig is None:
                continue
            crit_addr, edge_addr, rand_addr = get_target_blocks(orig_path, fname)
            if crit_addr is None:
                continue
            tasks = [("Critical", crit_addr), ("Edge", edge_addr), ("Random", rand_addr)]
            for label, target_addr in tasks:
                patched_path = apply_nop_patch(orig_path, target_addr, num_bytes=4)
                if not patched_path:
                    continue
                try:
                    vec_new = get_vec240(env, patched_path, fname)
                    if vec_new is None:
                        continue
                    dist_map = calc_section_dist(vec_orig, vec_new, sec_std)
                    for sec, dist in dist_map.items():
                        drifts[sec][label].append(dist)
                except Exception:
                    pass
                finally:
                    if os.path.exists(patched_path):
                        os.remove(patched_path)
        except Exception:
            pass

    stats = summarize(drifts)
    for sec in ["A", "B", "C"]:
        print(f"\n[Section {sec}]")
        for label in BAR_ORDER:
            item = stats[sec][label]
            print(f"{label:>8}: mean={item['mean']:.4f} sem={item['sem']:.4f} n={item['n']}")

    fig_path = os.path.join(SAVE_DIR, "exp3_2_section_sensitivity.png")
    plot_three_subplots(stats, fig_path)
    print(f"\nSaved figure: {fig_path}")

    json_path = os.path.join(SAVE_DIR, "exp3_2_section_sensitivity_stats.json")
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats: {json_path}")

    shutil.rmtree(TEMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    random.seed(2024)
    np.random.seed(2024)
    run_experiment()
