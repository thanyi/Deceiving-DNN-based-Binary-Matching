#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sample a subset where each item has at least one in-subset variant."""

import argparse
import json
import random
from collections import defaultdict


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("dataset must be a list")
    return data


def build_groups(data, key_fields):
    groups = defaultdict(list)
    for idx, item in enumerate(data):
        key = tuple(item.get(k) for k in key_fields)
        groups[key].append(idx)
    return list(groups.values())


def sample_with_min_per_group(groups, size, min_per_group, seed):
    eligible = [g for g in groups if len(g) >= min_per_group]
    if not eligible:
        raise ValueError("no eligible groups with enough members")

    if len(eligible) * min_per_group < size:
        raise ValueError("not enough eligible groups to fill target size")

    random.seed(seed)
    random.shuffle(eligible)

    selected_groups = []
    total_capacity = 0
    for g in eligible:
        if (len(selected_groups) + 1) * min_per_group > size:
            continue
        selected_groups.append(g)
        total_capacity += len(g)
        if total_capacity >= size:
            break

    if total_capacity < size:
        raise ValueError("selected groups do not have enough total capacity")

    selected = set()
    for g in selected_groups:
        selected.update(random.sample(g, min_per_group))

    remaining = [i for g in selected_groups for i in g if i not in selected]
    need = size - len(selected)
    if need > 0:
        if len(remaining) < need:
            raise ValueError("not enough remaining items to fill target size")
        selected.update(random.sample(remaining, need))

    return sorted(selected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--min-per-group", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--key",
        default="binary_name,func_name",
        help="Comma-separated group key fields",
    )
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    key_fields = [k.strip() for k in args.key.split(",") if k.strip()]
    groups = build_groups(data, key_fields)
    idxs = sample_with_min_per_group(groups, args.size, args.min_per_group, args.seed)
    subset = [data[i] for i in idxs]

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)

    print(f"wrote {len(subset)} samples to {args.out}")


if __name__ == "__main__":
    main()
