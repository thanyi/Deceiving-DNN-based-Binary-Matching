#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import r2pipe
import hashlib
from tqdm import tqdm
import random
import re
import glob

# ================= 配置区域 =================
# 数据集根目录 (二进制)
DATA_ROOT = "/home/ycy/ours/Deceiving-DNN-based-Binary-Matching/rl_framework/datasets/coreutils/bin"

# 源码目录根路径
SOURCE_ROOT = "/home/ycy/ours/Deceiving-DNN-based-Binary-Matching/rl_framework/datasets/coreutils/src"

# 测试集版本
TEST_VERSIONS = ["8.30", "8.32"]
VAL_RATIO = 0.1

MIN_INSTR = 15
MAX_INSTR = 800
BLACKLIST_FUNCS = [
    "_init", "_start", "_fini", "__libc_csu_init", 
    "entry0", "usage", "emit_bug_reporting_address",
    "version_etc", "version_etc_va", "deregister_tm_clones",
    "register_tm_clones", "frame_dummy", "atexit", "set_char_quoting"
]

# ===========================================

class SourceIndexer:
    """
    源码索引器：负责快速建立版本对应的函数白名单
    """
    def __init__(self, source_root):
        self.source_root = source_root
        self.cache = {} # version -> set(func_names)

    def get_user_functions(self, version):
        """
        获取指定版本的用户函数集合（带缓存）
        只扫描 src/ 目录，自动忽略 lib/ (gnulib)
        """
        if version in self.cache:
            return self.cache[version]
        
        # 假设源码目录结构是: source_root/coreutils-8.32/src/*.c
        # 或者是 source_root/coreutils-8.32/*.c (取决于你解压的方式)
        # 我们优先找 src/ 目录，因为那是核心逻辑所在
        
        # src_dir = os.path.join(self.source_root, f"coreutils-{version}", "src")
        src_dir = os.path.join(self.source_root, f"coreutils-{version}")
        # if not os.path.isdir(src_dir):
        #     # 回退尝试根目录
        #     src_dir = os.path.join(self.source_root, f"coreutils-{version}")
        #     if not os.path.isdir(src_dir):
        #         print(f"[!] Warning: Source directory not found for version {version}: {src_dir}")
        #         return set()

        func_set = set()
        
        # 简单的 C 函数定义正则
        # 匹配行首的单词，或者返回类型后的单词，紧接着是 '('
        # 这是一个 heuristic，不完美但够快
        # 排除 static 函数可能更好，但这里先保留
        func_pattern = re.compile(
                        r'^\s*(?:static\s+|inline\s+)?'  # 可选的 static/inline
                        r'(?:const\s+|struct\s+\w+\s*)?'  # 可选的 const/struct
                        r'\w+(?:\s*\*+\s*|\s+)'           # 返回类型（可能包含指针）
                        r'(\w+)\s*\(',                    # 函数名 + (
                        re.MULTILINE
                    )
        
        # 扫描 src 下的所有 .c 文件
        c_files = glob.glob(os.path.join(src_dir, "*.c"))
        # print(f"C files: {c_files}")
        if not c_files:
             print(f"[!] Warning: No .c files found in {src_dir}")
        
        for c_file in c_files:
            try:
                with open(c_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    # 提取所有匹配的函数名
                    matches = func_pattern.findall(content)
                    for m in matches:
                        # 过滤掉显然不是函数的关键字
                        if m not in ['if', 'while', 'for', 'switch', 'return', 'sizeof']:
                            func_set.add(m)
            except Exception:
                pass
        
        print(f"[*] Indexed {len(func_set)} user functions for version {version}")
        self.cache[version] = func_set
        return func_set

# 全局索引器实例
indexer = SourceIndexer(SOURCE_ROOT)

def get_binary_metadata(folder_name):
    try:
        parts = folder_name.split('-')
        version = parts[1]
        opt = parts[2]
        return version, opt
    except:
        return "unknown", "unknown"

def parse_opt_level(name):
    """从文件名中尽量解析优化等级（如 O0/O1/O2/O3/Os）"""
    m = re.search(r'-(O[0-3sS])(?:-|$)', name)
    if not m:
        return "unknown"
    opt = m.group(1)
    return opt.upper().replace("OS", "Os")

def extract_valid_functions(binary_path):
    valid_funcs = []
    try:
        r2 = r2pipe.open(binary_path, flags=['-2'])
        # r2.cmd('e asm.arch=x86') # 很多时候不需要显式设置，除非是异构
        # r2.cmd('e asm.bits=64')
        r2.cmd('aa') # 基础分析
        
        funcs = r2.cmdj('aflj') or []
        r2.quit()
        
        if not funcs: return []
            
        for f in funcs:
            raw_name = f.get('name', '')
            f_size = f.get('size', 0)
            # 兼容不同 r2 版本的地址字段
            f_addr = f.get('offset') or f.get('addr') or f.get('vaddr') or 0
            
            # 1. 基础清洗
            if not raw_name or f_addr == 0: continue
            if raw_name.startswith("sym.imp.") or raw_name.startswith("imp."): continue
            
            # 去除 sym. 前缀
            f_name = raw_name[4:] if raw_name.startswith("sym.") else raw_name
            
            # 2. 黑名单过滤
            if any(blk == f_name for blk in BLACKLIST_FUNCS): continue # 精确匹配黑名单
            if f_name.startswith("_"): continue # 过滤下划线开头的通常是系统函数
            
            # 3. 尺寸过滤
            if f_size < MIN_INSTR * 3 or f_size > MAX_INSTR * 5: continue
            
            valid_funcs.append({
                "func_name": f_name,
                "func_addr": f_addr,
                "func_size": f_size
            })
            
        return valid_funcs
    except Exception as e:
        print(f"Error processing {binary_path}: {e}")
        return []

def main():
    train_pool = []
    test_pool = []
    
    dirs = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    dirs.sort()
    flat_mode = not bool(dirs)

    # 模式1：版本子目录结构（原始设计）
    if dirs:
        print(f"Total versions: {dirs}")
        pbar = tqdm(dirs)
        for folder in pbar:
            version, opt = get_binary_metadata(folder)
            folder_path = os.path.join(DATA_ROOT, folder)
            pbar.set_description(f"Ver: {version}")

            # === 核心优化：先获取该版本的源码函数白名单 ===
            user_funcs_whitelist = indexer.get_user_functions(version)
            if not user_funcs_whitelist:
                # 没有源码就降级为仅靠黑名单/尺寸过滤
                pass

            is_test_version = version in TEST_VERSIONS

            for bin_file in os.listdir(folder_path):
                bin_path = os.path.join(folder_path, bin_file)
                if not os.path.isfile(bin_path) or bin_file.endswith(".o") or bin_file.endswith(".so"):
                    continue

                funcs = extract_valid_functions(bin_path)
                in_src_cnt = 0

                for f in funcs:
                    func_name = f['func_name']
                    if user_funcs_whitelist and func_name not in user_funcs_whitelist:
                        continue

                    in_src_cnt += 1
                    sample = {
                        "binary_path": os.path.abspath(bin_path),
                        "binary_name": bin_file,
                        "version": version,
                        "opt_level": opt,
                        "func_name": func_name,
                        "func_addr": f['func_addr'],
                        "size": f['func_size'],
                        "id": hashlib.md5(f"{bin_path}_{func_name}".encode()).hexdigest()[:8]
                    }

                    if is_test_version:
                        test_pool.append(sample)
                    else:
                        train_pool.append(sample)
    else:
        # 模式2：单层二进制文件夹（如 BinaryCorp-3M/small_train）
        bin_files = [f for f in os.listdir(DATA_ROOT) if os.path.isfile(os.path.join(DATA_ROOT, f))]
        bin_files.sort()
        print(f"Flat binary mode: {len(bin_files)} files")
        pbar = tqdm(bin_files)
        for bin_file in pbar:
            bin_path = os.path.join(DATA_ROOT, bin_file)
            # 单层模式下不过滤 .so（很多数据集就是 .so）
            if bin_file.endswith(".o"):
                continue

            opt = parse_opt_level(bin_file)
            version = "unknown"
            is_test_version = version in TEST_VERSIONS
            pbar.set_description(f"Flat | Opt: {opt}")

            funcs = extract_valid_functions(bin_path)
            for f in funcs:
                func_name = f['func_name']
                sample = {
                    "binary_path": os.path.abspath(bin_path),
                    "binary_name": bin_file,
                    "version": version,
                    "opt_level": opt,
                    "func_name": func_name,
                    "func_addr": f['func_addr'],
                    "size": f['func_size'],
                    "id": hashlib.md5(f"{bin_path}_{func_name}".encode()).hexdigest()[:8]
                }

                if is_test_version:
                    test_pool.append(sample)
                else:
                    train_pool.append(sample)

    # 划分验证集
    if flat_mode:
        # 单层模式按 binary 切分，避免同一 binary 的函数泄漏到 val
        by_binary = {}
        for sample in train_pool:
            by_binary.setdefault(sample["binary_path"], []).append(sample)

        binary_keys = list(by_binary.keys())
        random.shuffle(binary_keys)

        if len(binary_keys) <= 1:
            val_binary_keys = set()
        else:
            val_binary_size = int(len(binary_keys) * VAL_RATIO)
            val_binary_size = max(1, val_binary_size)
            val_binary_keys = set(binary_keys[:val_binary_size])

        val_pool = []
        final_train_pool = []
        for bkey in binary_keys:
            bucket = by_binary[bkey]
            if bkey in val_binary_keys:
                val_pool.extend(bucket)
            else:
                final_train_pool.extend(bucket)
    else:
        random.shuffle(train_pool)
        val_size = int(len(train_pool) * VAL_RATIO)
        val_pool = train_pool[:val_size]
        final_train_pool = train_pool[val_size:]
    
    print("\n" + "="*40)
    print(f"Summary:")
    print(f"  Training Samples:   {len(final_train_pool)}")
    print(f"  Validation Samples: {len(val_pool)}")
    print(f"  Testing Samples:    {len(test_pool)}")
    print("="*40)
    
    os.makedirs("fast", exist_ok=True)
    with open("fast/dataset_train.json", "w") as f: json.dump(final_train_pool, f, indent=2)
    with open("fast/dataset_val.json", "w") as f: json.dump(val_pool, f, indent=2)
    with open("fast/dataset_test.json", "w") as f: json.dump(test_pool, f, indent=2)
    
    print(f"[+] Done.")

if __name__ == "__main__":
    main()
