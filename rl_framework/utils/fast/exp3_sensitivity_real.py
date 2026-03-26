#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import random
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ================= 配置 =================
DATASET_PATH = os.path.join(SCRIPT_DIR, "dataset_train.json")
SAVE_DIR = os.path.join(SCRIPT_DIR, "chapter3_0228")
TEMP_DIR = os.path.join(SCRIPT_DIR, "tmp", "binary_patch_test")

LABELS = ["Random", "Edge", "Critical"]
BAR_LABELS = ["Random\nPatching", "Edge Block\nPatching", "Critical Block\nPatching"]
BAR_COLORS = ["#95a5a6", "#3498db", "#e74c3c"]

# 绘图风格设置
try:
    plt.style.use("seaborn-v0_8-paper")
except Exception:
    sns.set_style("whitegrid")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 18
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def ensure_runtime_dirs():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)


def cleanup_temp_dir():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)


def create_env(dataset_path):
    from rl_framework.env_wrapper import BinaryPerturbationEnv

    return BinaryPerturbationEnv(save_path=TEMP_DIR, dataset_path=dataset_path)


def create_acfg_extractor(binary_path):
    from rl_framework.utils.acfg.r2_acfg_features import RadareACFGExtractor

    return RadareACFGExtractor(binary_path)


def load_dataset(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    with open(dataset_path, "r") as f:
        return json.load(f)


def filter_candidates(data):
    return [
        d
        for d in data
        if d.get("func_name") != "main"
        and 200 < d.get("size", 0) < 5000
        and d.get("binary_path")
        and d.get("func_name")
    ]


def select_drift_subvector(vec, mode="no_section_b"):
    """
    从 240 维 ACFG 向量（去掉前 16 维 RL 历史后）选择漂移计算子向量。

    布局:
    - Section A: [0:40]
    - Section B: [40:200]
    - Section C: [200:240]
    """
    arr = np.asarray(vec)
    if arr.shape[0] < 240:
        return arr
    if mode == "full":
        return arr
    if mode == "no_section_b":
        return np.concatenate([arr[:40], arr[200:240]])
    if mode == "section_c_only":
        return arr[200:240]
    raise ValueError(f"Unsupported drift mode: {mode}")


def apply_nop_patch(binary_path, target_addr, num_bytes=4):
    """
    【物理攻击】在二进制文件的指定虚拟地址处写入 NOP 指令 (0x90)
    """
    import r2pipe

    filename = os.path.basename(binary_path)
    temp_path = os.path.join(TEMP_DIR, f"patched_{random.randint(1000, 9999)}_{filename}")
    shutil.copy(binary_path, temp_path)

    try:
        r = r2pipe.open(temp_path, flags=["-w", "-2"])
        r.cmd("e asm.arch=x86")
        r.cmd("e asm.bits=64")
        hex_str = "90" * num_bytes
        r.cmd(f"wx {hex_str} @ {target_addr}")
        r.quit()
        return temp_path
    except Exception as e:
        print(f"[!] Patch failed: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return None


def get_target_blocks(binary_path, func_name):
    """
    找到 Top-1 关键块和 Bottom-1 边缘块
    """
    extractor = create_acfg_extractor(binary_path)
    data = extractor.get_acfg_features(function_name=func_name)
    extractor.close()

    if not data or not data.get("basic_blocks"):
        return None, None, None

    scored_blocks = []
    all_addrs = []
    for addr, info in data["basic_blocks"].items():
        bet = info.get("centrality_betweenness", 0)
        dom = info.get("dominator_score", 0)
        deg = info.get("centrality_degree", 0)
        score = 0.4 * bet + 0.3 * deg + 0.3 * math.log1p(dom)
        all_addrs.append(addr)
        scored_blocks.append((addr, score))

    if not scored_blocks:
        return None, None, None

    scored_blocks.sort(key=lambda x: x[1], reverse=True)
    crit_addr = scored_blocks[0][0]
    edge_addr = scored_blocks[-1][0]
    rand_addr = random.choice(all_addrs)
    return crit_addr, edge_addr, rand_addr


def build_norm_std(env, samples, drift_mode):
    base_vecs = []
    for s in samples[:50]:
        try:
            raw_v = np.array(env.extract_features_from_function(s["binary_path"], s["func_name"])[16:])
            v = select_drift_subvector(raw_v, mode=drift_mode)
            base_vecs.append(v)
        except Exception as e:
            print(
                f"[!] Failed to build std vector for {s['func_name']} "
                f"({os.path.basename(s['binary_path'])}): {e}"
            )
    if not base_vecs:
        return None
    vec_std = np.std(base_vecs, axis=0)
    vec_std[vec_std == 0] = 1.0
    return vec_std


def collect_drifts(env, samples, drift_mode="no_section_b", progress_desc=None):
    drifts = {"Critical": [], "Edge": [], "Random": []}
    if not samples:
        return drifts

    vec_std = build_norm_std(env, samples, drift_mode)
    if vec_std is None:
        print("[!] Failed to build normalization std, skipping this sample group.")
        return drifts

    iterator = tqdm(samples, desc=progress_desc) if progress_desc else tqdm(samples)
    for sample in iterator:
        orig_path = sample["binary_path"]
        fname = sample["func_name"]

        try:
            env.original_binary = orig_path
            env.function_name = fname
            raw_orig = np.array(env.extract_features_from_function(orig_path, fname)[16:])
            vec_orig = select_drift_subvector(raw_orig, mode=drift_mode)
        except Exception as e:
            print(f"[!] Failed to extract original feature {fname}: {e}")
            continue

        crit_addr, edge_addr, rand_addr = get_target_blocks(orig_path, fname)
        if crit_addr is None:
            continue

        for label, target_addr in [("Critical", crit_addr), ("Edge", edge_addr), ("Random", rand_addr)]:
            patched_path = apply_nop_patch(orig_path, target_addr, num_bytes=4)
            if not patched_path:
                continue

            try:
                raw_new = np.array(env.extract_features_from_function(patched_path, fname)[16:])
                vec_new = select_drift_subvector(raw_new, mode=drift_mode)
                if vec_new.shape != vec_orig.shape:
                    continue
                dist = np.linalg.norm((vec_orig - vec_new) / vec_std)
                if np.isfinite(dist) and dist > 0:
                    drifts[label].append(float(dist))
            except Exception as e:
                print(f"[!] Failed to extract features for {label} @ {hex(target_addr)}: {e}")
            finally:
                if os.path.exists(patched_path):
                    os.remove(patched_path)
    return drifts


def sanitize_drifts(drifts):
    clean = {}
    for label in LABELS:
        arr = np.asarray(drifts.get(label, []), dtype=float)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        clean[label] = arr
    return clean


def ttest_greater(a, b):
    if len(a) < 2 or len(b) < 2:
        return np.nan
    try:
        _, p_val = stats.ttest_ind(a, b, alternative="greater")
        return float(p_val)
    except TypeError:
        _, p_two_sided = stats.ttest_ind(a, b)
        if np.mean(a) > np.mean(b):
            return float(p_two_sided / 2.0)
        return float(1.0 - p_two_sided / 2.0)


def summarize_drifts(drifts):
    clean = sanitize_drifts(drifts)
    if any(clean[label].size == 0 for label in LABELS):
        return None

    means = [float(np.mean(clean["Random"])), float(np.mean(clean["Edge"])), float(np.mean(clean["Critical"]))]
    errors = [
        float(np.std(clean["Random"]) / np.sqrt(len(clean["Random"]))),
        float(np.std(clean["Edge"]) / np.sqrt(len(clean["Edge"]))),
        float(np.std(clean["Critical"]) / np.sqrt(len(clean["Critical"]))),
    ]
    p_val = ttest_greater(clean["Critical"], clean["Edge"])

    return {
        "means": means,
        "errors": errors,
        "p_val": p_val,
        "counts": {
            "Random": int(len(clean["Random"])),
            "Edge": int(len(clean["Edge"])),
            "Critical": int(len(clean["Critical"])),
        },
        "raw_drifts": {label: clean[label].tolist() for label in LABELS},
    }


def significance_text(p_val):
    if not np.isfinite(p_val):
        return "n/a"
    if p_val < 0.001:
        return "***"
    if p_val < 0.01:
        return "**"
    if p_val < 0.05:
        return "*"
    return "ns"


def draw_single_plot(summary, save_path):
    means = summary["means"]
    errors = summary["errors"]
    p_val = summary["p_val"]

    plt.figure(figsize=(13, 8.6))
    x_pos = np.arange(len(BAR_LABELS))
    bars = plt.bar(
        x_pos,
        means,
        yerr=errors,
        align="center",
        alpha=0.85,
        ecolor="black",
        capsize=8,
        color=BAR_COLORS,
        width=0.65,
        edgecolor="white",
        linewidth=1.5,
    )

    max_err = max(errors) if errors else 0.0
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max_err * 1.3,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=22,
            fontweight="bold",
            color="#2c3e50",
        )

    max_h = max(means) + max_err
    if np.isfinite(p_val):
        sig_y_start = max_h + max_err * 2.5
        sig_y_end = sig_y_start + max_err * 0.8
        plt.plot([1, 1, 2, 2], [sig_y_start, sig_y_end, sig_y_end, sig_y_start], lw=1.8, c="black")
        plt.text(
            1.5,
            sig_y_end + max_err * 0.3,
            f"{significance_text(p_val)}\n(p = {p_val:.4e})",
            ha="center",
            va="bottom",
            color="#e74c3c",
            fontsize=18,
            fontweight="bold",
        )

    plt.ylabel("Feature Vector Drift (Normalized L2 Distance)", fontsize=22, fontweight="bold")
    plt.title("Sensitivity Analysis of Critical Region Perception", fontsize=26, fontweight="bold", pad=24)
    plt.xticks(x_pos, BAR_LABELS, fontsize=20)
    plt.yticks(fontsize=18)
    plt.ylim(0, max_h * 1.8 if max_h > 0 else 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close()


def draw_multi_binary_plot(results, save_path):
    if not results:
        print("[!] No per-binary summary to plot.")
        return

    n = len(results)
    ncols = 1 if n == 1 else 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.4 * ncols, 6.2 * nrows), squeeze=False)

    global_max = 0.0
    for i, res in enumerate(results):
        r = i // ncols
        c = i % ncols
        ax = axes[r][c]

        means = res["means"]
        errors = res["errors"]
        p_val = res["p_val"]

        x_pos = np.arange(len(BAR_LABELS))
        bars = ax.bar(
            x_pos,
            means,
            yerr=errors,
            align="center",
            alpha=0.85,
            ecolor="black",
            capsize=5,
            color=BAR_COLORS,
            width=0.62,
            edgecolor="white",
            linewidth=1.2,
        )

        local_max = max(means) + max(errors)
        global_max = max(global_max, local_max)
        err_anchor = max(errors) if max(errors) > 0 else (local_max * 0.05 if local_max > 0 else 0.03)
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + err_anchor * 0.55,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=14,
            )

        short_name = os.path.basename(res["binary_path"])
        ax.set_title(f"{short_name} (n={res['n_funcs']})", fontsize=18, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(["R", "E", "C"], fontsize=16)
        ax.tick_params(axis="y", labelsize=15)
        if c == 0:
            ax.set_ylabel("Drift", fontsize=17)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.text(
            0.98,
            0.98,
            f"p={res['p_val']:.2e}" if np.isfinite(p_val) else "p=n/a",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=15,
            color="#c0392b",
        )

    if global_max <= 0:
        global_max = 1.0
    for i in range(n):
        r = i // ncols
        c = i % ncols
        axes[r][c].set_ylim(0, global_max * 1.35)

    total_slots = nrows * ncols
    for i in range(n, total_slots):
        r = i // ncols
        c = i % ncols
        axes[r][c].axis("off")

    fig.suptitle("Real Binary Sensitivity (Multiple Binaries)", fontsize=24, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(save_path, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def run_experiment(drift_mode="no_section_b", dataset_path=DATASET_PATH, sample_size=50):
    print("\n[Experiment 3] Real Binary Patching Sensitivity Test (single figure)...")
    print(f"[*] Drift mode: {drift_mode}")
    print(f"[*] Dataset path: {dataset_path}")

    ensure_runtime_dirs()
    try:
        data = load_dataset(dataset_path)
        candidates = filter_candidates(data)
        if not candidates:
            print("[!] No valid candidate functions after filtering.")
            return

        sample_size = min(sample_size, len(candidates))
        samples = random.sample(candidates, sample_size)
        print(f"[*] Selected {len(samples)} samples for physical patching.")

        env = create_env(dataset_path)
        drifts = collect_drifts(env, samples, drift_mode=drift_mode, progress_desc="single")
        summary = summarize_drifts(drifts)
        if summary is None:
            print("[!] No valid data collected! All feature extractions failed.")
            return

        print("\n[Results]")
        print(f"Random   Mean Drift: {summary['means'][0]:.4f} (n={summary['counts']['Random']})")
        print(f"Edge     Mean Drift: {summary['means'][1]:.4f} (n={summary['counts']['Edge']})")
        print(f"Critical Mean Drift: {summary['means'][2]:.4f} (n={summary['counts']['Critical']})")
        if np.isfinite(summary["p_val"]):
            print(f"T-Test (Critical > Edge): p-value = {summary['p_val']:.4e}")
        else:
            print("T-Test (Critical > Edge): p-value = n/a (samples too few)")

        save_path = os.path.join(SAVE_DIR, "exp3_real_sensitivity.png")
        draw_single_plot(summary, save_path)
        print(f"[+] Plot saved to {save_path}")
    finally:
        cleanup_temp_dir()


def run_multi_binary_experiment(
    drift_mode="no_section_b",
    dataset_path=DATASET_PATH,
    max_binaries=9,
    samples_per_binary=8,
    min_funcs_per_binary=5,
    num_figures=1,
    binaries_per_figure=4,
):
    print("\n[Experiment 3] Real Binary Patching Sensitivity Test (multi-binary figure)...")
    print(f"[*] Drift mode: {drift_mode}")
    print(f"[*] Dataset path: {dataset_path}")

    ensure_runtime_dirs()
    try:
        data = load_dataset(dataset_path)
        candidates = filter_candidates(data)
        if not candidates:
            print("[!] No valid candidate functions after filtering.")
            return

        grouped = {}
        for item in candidates:
            grouped.setdefault(item["binary_path"], []).append(item)

        ranked = sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)
        ranked = [x for x in ranked if len(x[1]) >= min_funcs_per_binary]

        # 去重：同名二进制（如不同 coreutils 版本下的 sort）只保留一个，
        # 优先保留候选函数数量更多的那个（ranked 已按数量降序）。
        selected = []
        seen_names = set()
        if num_figures > 1:
            total_needed = max(1, int(num_figures)) * max(1, int(binaries_per_figure))
            target_count = min(max_binaries, total_needed)
        else:
            target_count = max_binaries
        for binary_path, funcs in ranked:
            binary_name = os.path.basename(binary_path)
            if binary_name in seen_names:
                continue
            selected.append((binary_path, funcs))
            seen_names.add(binary_name)
            if len(selected) >= target_count:
                break
        if not selected:
            print("[!] No binary has enough candidate functions.")
            return

        print(f"[*] Selected {len(selected)} binaries for plotting.")
        env = create_env(dataset_path)

        results = []
        for binary_path, funcs in selected:
            k = min(samples_per_binary, len(funcs))
            samples = random.sample(funcs, k)
            print(f"[-] Processing {os.path.basename(binary_path)} with {k} sampled functions...")
            drifts = collect_drifts(
                env,
                samples,
                drift_mode=drift_mode,
                progress_desc=f"binary:{os.path.basename(binary_path)}",
            )
            summary = summarize_drifts(drifts)
            if summary is None:
                print(f"[!] Skip {binary_path}: not enough valid drift data.")
                continue

            summary["binary_path"] = binary_path
            summary["n_funcs"] = k
            results.append(summary)

        if not results:
            print("[!] No valid per-binary results were collected.")
            return

        groups = []
        if num_figures > 1:
            chunk_size = max(1, int(binaries_per_figure))
            for i in range(max(1, int(num_figures))):
                start = i * chunk_size
                end = start + chunk_size
                chunk = results[start:end]
                if not chunk:
                    break
                groups.append(chunk)
        else:
            groups = [results]

        for i, chunk in enumerate(groups, start=1):
            if len(groups) == 1:
                save_name = "exp3_real_sensitivity_multi_binary.png"
            else:
                save_name = f"exp3_real_sensitivity_multi_binary_part{i}.png"
            save_path = os.path.join(SAVE_DIR, save_name)
            draw_multi_binary_plot(chunk, save_path)
            print(f"[+] Multi-binary plot saved to {save_path}")

        summary_path = os.path.join(SAVE_DIR, "exp3_real_sensitivity_multi_binary_summary.json")
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "drift_mode": drift_mode,
                    "num_figures": num_figures,
                    "binaries_per_figure": binaries_per_figure,
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"[+] Summary JSON saved to {summary_path}")
    finally:
        cleanup_temp_dir()


def parse_args():
    parser = argparse.ArgumentParser(description="Real binary patching sensitivity analysis")
    parser.add_argument("--mode", choices=["single", "multi"], default="single")
    parser.add_argument("--dataset-path", default=DATASET_PATH, help="Path to dataset_train.json")
    parser.add_argument(
        "--drift-mode",
        choices=["full", "no_section_b", "section_c_only"],
        default="no_section_b",
        help="Feature subvector mode for drift computation",
    )
    parser.add_argument("--sample-size", type=int, default=50, help="Function samples for single-figure mode")
    parser.add_argument("--max-binaries", type=int, default=9, help="Max binaries in multi-figure mode")
    parser.add_argument("--samples-per-binary", type=int, default=8, help="Sampled functions per binary")
    parser.add_argument("--min-funcs-per-binary", type=int, default=5, help="Min candidate functions to include binary")
    parser.add_argument("--num-figures", type=int, default=1, help="Number of output figures in multi mode")
    parser.add_argument("--binaries-per-figure", type=int, default=4, help="Subplots per output figure in multi mode")
    return parser.parse_args()


if __name__ == "__main__":
    random.seed(2024)
    np.random.seed(2024)
    args = parse_args()

    if args.mode == "multi":
        run_multi_binary_experiment(
            drift_mode=args.drift_mode,
            dataset_path=args.dataset_path,
            max_binaries=args.max_binaries,
            samples_per_binary=args.samples_per_binary,
            min_funcs_per_binary=args.min_funcs_per_binary,
            num_figures=args.num_figures,
            binaries_per_figure=args.binaries_per_figure,
        )
    else:
        run_experiment(
            drift_mode=args.drift_mode,
            dataset_path=args.dataset_path,
            sample_size=args.sample_size,
        )
