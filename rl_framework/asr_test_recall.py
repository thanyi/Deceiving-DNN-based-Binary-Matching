import argparse
import json
import os
import random
from tqdm import tqdm
from rl_framework.asr_test import Asm2VecRetriever, _get_variant_set


def _load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sample_dataset(dataset, limit, seed):
    if limit is None:
        return dataset
    limit = max(1, int(limit))
    rng = random.Random(seed)
    if limit >= len(dataset):
        return dataset
    return rng.sample(dataset, limit)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="rl_framework/asr_workdir_run1/dataset_test_sample128.json",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--workdir", default="rl_framework/asr_workdir_run1/_recall_only")
    args = parser.parse_args()

    dataset = _load_dataset(args.dataset)
    dataset = _sample_dataset(dataset, args.limit, args.seed)

    workdir = args.workdir
    os.makedirs(workdir, exist_ok=True)
    retriever = Asm2VecRetriever(asm_work_dir=workdir, retrieval_workers=1, log_every=0)

    topk = args.topk
    total = 0
    recall_sum = 0.0

    sampled_ids = {str(item.get("id")) for item in dataset if item.get("id") is not None}
    for i, sample in tqdm(enumerate(dataset), total=len(dataset), desc=f"Recall@{topk}", unit="sample"):
        variants = _get_variant_set(sample)
        if not variants:
            continue
        variants = set(v for v in variants if v in sampled_ids)
        if not variants:
            continue
        topk_pre = retriever.topk(sample, dataset, topk, query_kind="original")
        topk_ids = {str(x.get("id")) for x in topk_pre if x.get("id") is not None}
        hits = len(topk_ids & variants)
        recall_sum += hits / max(len(variants), 1)
        total += 1

    print("samples_used", total)
    print("recall_pre", recall_sum / max(total, 1))


if __name__ == "__main__":
    main()
