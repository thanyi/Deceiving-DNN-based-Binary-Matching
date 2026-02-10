#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Add/refresh `variants` field in a dataset JSON by grouping samples. "
            "Default grouping matches current project convention: "
            "(binary_name, func_name, opt_level)."
        )
    )
    p.add_argument("--input", required=True, help="Input dataset JSON path")
    p.add_argument(
        "--output",
        required=True,
        help="Output dataset JSON path (can be the same as --input)",
    )
    p.add_argument(
        "--group-keys",
        default="binary_name,func_name,opt_level",
        help="Comma-separated fields used to define one variant group",
    )
    p.add_argument(
        "--id-key",
        default="id",
        help="Sample ID field used inside variants list",
    )
    p.add_argument(
        "--variants-key",
        default="variants",
        help="Field name to write variant ID list",
    )
    p.add_argument(
        "--exclude-self",
        action="store_true",
        help="If set, remove current sample id from its own variants list",
    )
    p.add_argument(
        "--sort-ids-by",
        default="version,opt_level,id",
        help=(
            "Comma-separated sort keys used to produce stable variants order. "
            "Keys missing in sample are treated as empty string."
        ),
    )
    return p.parse_args()


def normalize_keys(raw):
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        raise ValueError("No valid keys parsed")
    return keys


def main():
    args = parse_args()
    group_keys = normalize_keys(args.group_keys)
    sort_keys = normalize_keys(args.sort_ids_by)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input dataset must be a JSON list")

    groups = defaultdict(list)
    missing_id = 0
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        sid = item.get(args.id_key)
        if not sid:
            missing_id += 1
            continue
        gk = tuple(item.get(k) for k in group_keys)
        groups[gk].append((idx, item))

    # Pre-compute deterministic variant ids for each group.
    group_variant_ids = {}
    for gk, members in groups.items():
        sorted_members = sorted(
            members,
            key=lambda x: tuple(str(x[1].get(k, "")) for k in sort_keys),
        )
        ids = [str(m[1][args.id_key]) for m in sorted_members]
        group_variant_ids[gk] = ids

    touched = 0
    for item in data:
        if not isinstance(item, dict):
            continue
        sid = item.get(args.id_key)
        if not sid:
            continue
        gk = tuple(item.get(k) for k in group_keys)
        variants = list(group_variant_ids.get(gk, []))
        if args.exclude_self:
            variants = [v for v in variants if v != str(sid)]
        item[args.variants_key] = variants
        touched += 1

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    group_size_stats = defaultdict(int)
    for ids in group_variant_ids.values():
        group_size_stats[len(ids)] += 1
    top_sizes = sorted(group_size_stats.items(), key=lambda x: x[0])[:10]

    print("[add-variants] done")
    print("[add-variants] input:", args.input)
    print("[add-variants] output:", args.output)
    print("[add-variants] group_keys:", group_keys)
    print("[add-variants] exclude_self:", args.exclude_self)
    print("[add-variants] samples_total:", len(data))
    print("[add-variants] samples_touched:", touched)
    print("[add-variants] groups_total:", len(group_variant_ids))
    print("[add-variants] missing_id_skipped:", missing_id)
    print("[add-variants] group_size_stats(first10):", dict(top_sizes))


if __name__ == "__main__":
    main()
