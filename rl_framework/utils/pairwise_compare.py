#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
import time


def _repo_root():
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))


def _load_dataset(path):
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset json must be a list of items.")
    return data


def _pick_items(data, limit, seed):
    items = list(data)
    rnd = random.Random(seed)
    rnd.shuffle(items)
    if limit is not None and limit > 0:
        items = items[: min(limit, len(items))]
    return items


def _pair_items(items, seed):
    rnd = random.Random(seed)
    items = list(items)
    rnd.shuffle(items)
    pairs = []
    for i in range(0, len(items) - 1, 2):
        pairs.append((items[i], items[i + 1]))
    return pairs


def _get_addr(item):
    addr = item.get("func_addr")
    if addr is None:
        return None
    return addr


def _make_progress_printer(total):
    width = 30
    start = time.time()
    last_print = 0

    def _maybe_print(i):
        nonlocal last_print
        if total <= 0:
            return
        if not sys.stdout.isatty():
            step = max(1, total // 50)
            if i == total or i - last_print >= step:
                last_print = i
                print(f"Progress: {i}/{total}", flush=True)
            return
        ratio = i / total
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        elapsed = time.time() - start
        print(f"\r[{bar}] {i}/{total} {ratio * 100:5.1f}% {elapsed:6.1f}s", end="", flush=True)
        if i == total:
            print("")

    return _maybe_print


def main():
    parser = argparse.ArgumentParser(description="Pairwise function similarity evaluation.")
    parser.add_argument("--dataset", required=True, help="Path to dataset json")
    parser.add_argument("--limit", type=int, default=None, help="Randomly sample N items")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--detection-method",
        choices=["asm2vec", "safe"],
        default="asm2vec",
        help="Similarity method",
    )
    parser.add_argument("--safe-checkpoint-dir", default=None, help="SAFE checkpoint dir")
    parser.add_argument("--safe-i2v-dir", default=None, help="SAFE i2v dir")
    parser.add_argument("--safe-use-gpu", action="store_true", help="Use GPU for SAFE")
    parser.add_argument(
        "--asm-work-dir",
        default=None,
        help="Shared asm2vec work dir (optional, for caching)",
    )
    parser.add_argument(
        "--output",
        default="pairwise_results.json",
        help="Output json path",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from run_utils import run_one

    data = _load_dataset(args.dataset)
    items = _pick_items(data, args.limit, args.seed)
    pairs = _pair_items(items, args.seed + 1)

    if args.detection_method == "asm2vec" and args.asm_work_dir:
        os.makedirs(args.asm_work_dir, exist_ok=True)

    original_asm_cache = {} if args.detection_method == "asm2vec" else None

    valid_scores = []
    skipped = 0
    errors = 0
    start = time.time()
    progress = _make_progress_printer(len(pairs))
    progress(0)

    for idx, (a, b) in enumerate(pairs, 1):
        func_name = a.get("func_name")
        if not func_name:
            skipped += 1
            progress(idx)
            continue

        addr_a = _get_addr(a)
        addr_b = _get_addr(b)
        if addr_a is None or addr_b is None:
            skipped += 1
            progress(idx)
            continue

        try:
            score, _grad = run_one(
                a.get("binary_path"),
                b.get("binary_path"),
                None,
                None,
                func_name,
                detection_method=args.detection_method,
                asm_work_dir=args.asm_work_dir,
                original_asm_cache=original_asm_cache,
                simple_mode=True,
                original_func_addr=addr_a,
                mutated_func_addr=addr_b,
                safe_checkpoint_dir=args.safe_checkpoint_dir,
                safe_i2v_dir=args.safe_i2v_dir,
                safe_use_gpu=args.safe_use_gpu,
            )
        except Exception:
            errors += 1
            progress(idx)
            continue

        if score is None:
            skipped += 1
            progress(idx)
            continue

        valid_scores.append(float(score))
        progress(idx)

    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
    elapsed = time.time() - start

    result = {
        "dataset": args.dataset,
        "limit": args.limit,
        "seed": args.seed,
        "detection_method": args.detection_method,
        "pair_count": len(pairs),
        "valid_count": len(valid_scores),
        "skipped": skipped,
        "errors": errors,
        "avg_score": avg_score,
        "elapsed_sec": round(elapsed, 3),
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
