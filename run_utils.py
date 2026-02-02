#!/usr/bin/env python3
"""
二进制匹配系统的运行时工具函数
包含用于评估二进制文件相似度的核心功能
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import ast
import subprocess
from loguru import logger
import random
import hashlib
import time
import tempfile
import shutil

# # 仅输出 WARNING/ERROR 级别日志（避免信息噪音）
logger.remove()
logger.add(sys.stderr, level="WARNING", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


# def only_success(record):
#     return record["level"].name == "SUCCESS"

# logger.add(
#     sink=lambda msg: print(msg, end=""),  # 输出到 stdout（可替换为文件等）
#     filter=only_success,
#     format="{time} | {level} | {message}"
# )

# 添加 asm2vec-pytorch 路径到 sys.path
_asm2vec_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             'detection_model', 'asm2vec-pytorch')
if _asm2vec_path not in sys.path:
    sys.path.insert(0, _asm2vec_path)

# 导入 compare_functions
from scripts.compare_util import *
from scripts.bin2asm_util import *


# === Shared helpers ===
def _find_sym_to_addr(binary_path):
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


def _load_sym_to_addr(path):
    try:
        return pickle.load(open(path, "rb"))
    except Exception as e:
        logger.warning(f"加载 sym_to_addr 失败: {path}, {e}")
        return {}


def _normalize_addr(addr):
    """Return a hex string like 0x401000, or None."""
    if addr is None:
        return None
    try:
        if isinstance(addr, str):
            if addr.startswith("0x") or addr.startswith("0X"):
                return addr
            return hex(int(addr, 0))
        return hex(int(addr))
    except Exception:
        return None

def _normalize_addr_int(addr):
    """Return integer address, or None."""
    if addr is None:
        return None
    try:
        if isinstance(addr, str):
            return int(addr, 16) if addr.startswith("0x") or addr.startswith("0X") else int(addr, 0)
        return int(addr)
    except Exception:
        return None


def _resolve_function_addrs(
    original_binary,
    mutated_binary,
    function_name,
    simple_mode,
    original_func_addr=None,
    mutated_func_addr=None,
    sym_to_addr_path=None,
    sym_to_addr_map=None,
):
    """
    Resolve original/mutated function addresses using:
    - explicit addr params
    - sym_to_addr.pickle (maps original symbol -> current binary addr)

    NOTE: sym_to_addr is typically generated for mutated binaries, so it should
    be used to resolve mutated_addr, not original_addr. Original can fall back
    to name-based lookup when asm2vec extracts by function name.
    """
    addr_map = sym_to_addr_map
    if addr_map is None:
        if sym_to_addr_path and os.path.exists(sym_to_addr_path):
            addr_map = _load_sym_to_addr(sym_to_addr_path)
        else:
            auto_path = _find_sym_to_addr(original_binary)
            addr_map = _load_sym_to_addr(auto_path) if auto_path else {}

    original_addr = original_func_addr
    mutated_addr = mutated_func_addr
    if not simple_mode and function_name in addr_map and mutated_addr is None:
        mutated_addr = addr_map.get(function_name)

    # Fallback: try mapping near mutated binary for mutated_addr only.
    if not simple_mode and not addr_map:
        mutated_folder = os.path.dirname(mutated_binary)
        map_path = os.path.join(mutated_folder, "sym_to_addr.pickle")
        if os.path.exists(map_path):
            addr_map = _load_sym_to_addr(map_path)
            if function_name in addr_map and mutated_addr is None:
                mutated_addr = addr_map.get(function_name)

    return _normalize_addr(original_addr), _normalize_addr(mutated_addr), addr_map or {}


def prepare_original_asm(
    original_binary,
    function_name,
    asm_work_dir=None,
    original_func_addr=None,
    sym_to_addr_path=None,
    sym_to_addr_map=None,
    simple_mode=False,
    original_asm_cache=None,
):
    """Extract and cache original function asm once for reuse."""
    original_addr_hex, _mutated_addr_hex, _addr_map = _resolve_function_addrs(
        original_binary=original_binary,
        mutated_binary=original_binary,
        function_name=function_name,
        simple_mode=simple_mode,
        original_func_addr=original_func_addr,
        mutated_func_addr=None,
        sym_to_addr_path=sym_to_addr_path,
        sym_to_addr_map=sym_to_addr_map,
    )

    if asm_work_dir is None:
        asm_work_dir = tempfile.mkdtemp(prefix="asm2vec_", suffix="_tmp")
        need_cleanup = True
    else:
        need_cleanup = False
        os.makedirs(os.path.join(asm_work_dir, "original"), exist_ok=True)

    cache_key = (os.path.abspath(original_binary), function_name, original_addr_hex)
    if original_asm_cache is not None and cache_key in original_asm_cache:
        return original_asm_cache[cache_key], original_addr_hex, need_cleanup

    original_output_file = binfunc2asm(
        ipath=original_binary,
        target_func_name=function_name,
        opath=os.path.join(asm_work_dir, "original"),
        verbose=bool(os.environ.get("ASM2VEC_VERBOSE")),
        current_sym_addr=original_addr_hex,
    )
    if not original_output_file:
        logger.error(
            f"无法提取原始函数 {function_name} 的汇编文件 (addr={original_addr_hex})"
        )
        return None, original_addr_hex, need_cleanup

    if original_asm_cache is not None:
        original_asm_cache[cache_key] = original_output_file
    return original_output_file, original_addr_hex, need_cleanup


# === SAFE helpers (lazy import) ===
_SAFE_SESSION = None
_SAFE_SESSION_KEY = None
_SAFE_GET_FUNCTIONS = None
_SAFE_EMB_CACHE = {}
_SAFE_INSTR_CACHE = {}


def _get_safe_paths(safe_root=None, checkpoint_dir=None, i2v_dir=None):
    base = safe_root or os.environ.get(
        "SAFE_ROOT",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection_model", "SAFE"),
    )
    ckpt = checkpoint_dir or os.environ.get("SAFE_CHECKPOINT_DIR", os.path.join(base, "experiments", "output", "last_run"))
    i2v = i2v_dir or os.environ.get("SAFE_I2V_DIR", os.path.join(base, "data", "i2v"))
    return base, ckpt, i2v


def _get_safe_session(checkpoint_dir, i2v_dir, use_gpu=True):
    global _SAFE_SESSION, _SAFE_SESSION_KEY, _SAFE_GET_FUNCTIONS
    key = (checkpoint_dir, i2v_dir, bool(use_gpu))
    if _SAFE_SESSION is not None and _SAFE_SESSION_KEY == key and _SAFE_GET_FUNCTIONS is not None:
        return _SAFE_SESSION, _SAFE_GET_FUNCTIONS

    # Close old session if any
    try:
        if _SAFE_SESSION is not None:
            _SAFE_SESSION.close()
    except Exception:
        pass

    safe_root, _, _ = _get_safe_paths(checkpoint_dir=checkpoint_dir, i2v_dir=i2v_dir)
    if safe_root not in sys.path:
        sys.path.insert(0, safe_root)

    # Lazy import to avoid TF load when not needed.
    from inference_example import SAFEInferenceSession, get_functions_from_binary

    sess = SAFEInferenceSession(checkpoint_dir, i2v_dir, use_gpu=use_gpu)
    _SAFE_SESSION = sess
    _SAFE_SESSION_KEY = key
    _SAFE_GET_FUNCTIONS = get_functions_from_binary
    return sess, get_functions_from_binary


def _safe_get_embedding(binary_path, addr_int, checkpoint_dir, i2v_dir, use_gpu=True):
    cache_key = (os.path.abspath(binary_path), addr_int)
    if cache_key in _SAFE_EMB_CACHE:
        return _SAFE_EMB_CACHE[cache_key]

    sess, get_functions_from_binary = _get_safe_session(checkpoint_dir, i2v_dir, use_gpu=use_gpu)
    # Reuse instruction extraction cache.
    instr = _SAFE_INSTR_CACHE.get(cache_key)
    if instr is None:
        instr_list = get_functions_from_binary(binary_path, addr_int)
        instr = instr_list[0] if instr_list else None
        _SAFE_INSTR_CACHE[cache_key] = instr
    if instr is None:
        return None

    emb = sess.get_embedding(instr, verbose=False)
    _SAFE_EMB_CACHE[cache_key] = emb
    return emb

def run_one(
    original_binary,
    mutated_binary,
    model_original,
    checkdict,
    function_name,
    detection_method="asm2vec",
    asm_work_dir=None,
    original_asm_cache=None,
    simple_mode=False,
    original_func_addr=None,
    mutated_func_addr=None,
    sym_to_addr_path=None,
    sym_to_addr_map=None,
    safe_checkpoint_dir=None,
    safe_i2v_dir=None,
    safe_use_gpu=True,
    original_asm_path=None,
):
    """
    评估原始二进制文件和变异二进制文件之间的相似度
    
    参数:
        original_binary: 原始二进制文件路径
        mutated_binary: 变异二进制文件路径  
        model_original: 预训练的模型对象（未使用，保留接口兼容性）
        checkdict: 函数映射字典（simple_mode=True 时不使用）
        function_name: 目标函数名
        simple_mode: 简单模式，直接用函数名提取汇编，不依赖 pickle 文件
        detection_method: "asm2vec" 或 "safe"
        *_func_addr: 可选的函数地址（优先于符号名）
        sym_to_addr_*: 可选映射文件/字典，用于解析变异后地址
        safe_*: SAFE 模型所需的 checkpoint 与 i2v 目录
        original_asm_path: 可选的原始汇编文件路径（用于复用）
    
    返回:
        score: 余弦相似度 (float) - 范围[0,1]，越低表示越不相似
        grad: 梯度值 (float) - 固定返回 0.0
    """
    try:
        logger.debug(f'run_one开始: 原始={original_binary}, 变异={mutated_binary}, 函数={function_name}, simple_mode={simple_mode}')
        
        # 检查输入文件
        if not os.path.exists(original_binary):
            logger.error(f"原始二进制文件不存在: {original_binary}")
            return None, None
        
        if not os.path.exists(mutated_binary):
            logger.error(f"变异二进制文件不存在: {mutated_binary}")
            return None, None
        
        # Resolve addresses once (shared by asm2vec/safe).
        original_addr_hex, mutated_addr_hex, _addr_map = _resolve_function_addrs(
            original_binary=original_binary,
            mutated_binary=mutated_binary,
            function_name=function_name,
            simple_mode=simple_mode,
            original_func_addr=original_func_addr,
            mutated_func_addr=mutated_func_addr,
            sym_to_addr_path=sym_to_addr_path,
            sym_to_addr_map=sym_to_addr_map,
        )

        if detection_method == "asm2vec":
            logger.info(f"[*] detection_method: {detection_method}")
            
            if simple_mode and original_addr_hex:
                logger.info(f"[*] simple_mode: using mapped addr {original_addr_hex}")
            elif simple_mode:
                logger.info(f"[*] simple_mode: using function name directly")
            logger.info(f"[*] original_addr: {original_addr_hex}")
            logger.info(f"[*] mutated_addr: {mutated_addr_hex}")
            
            # 使用固定工作目录，避免频繁创建删除
            if asm_work_dir is None:
                asm_work_dir = tempfile.mkdtemp(prefix='asm2vec_', suffix='_tmp')
                need_cleanup = True
            else:
                need_cleanup = False
                os.makedirs(os.path.join(asm_work_dir, 'mutated'), exist_ok=True)
                if original_asm_path is None:
                    os.makedirs(os.path.join(asm_work_dir, 'original'), exist_ok=True)
            
            try:
                # 缓存原始文件汇编
                if original_asm_path:
                    if not os.path.exists(original_asm_path):
                        logger.error(f"原始汇编文件不存在: {original_asm_path}")
                        return None, None
                    original_asm = original_asm_path
                else:
                    cache_key = (os.path.abspath(original_binary), function_name, original_addr_hex)
                    if original_asm_cache is not None and cache_key in original_asm_cache:
                        original_asm = original_asm_cache[cache_key]
                        logger.debug(f"使用缓存的原始文件汇编: {original_asm}")
                    else:
                        original_output_file = binfunc2asm(
                            ipath=original_binary,
                            target_func_name=function_name,
                            opath=os.path.join(asm_work_dir, 'original'), 
                            verbose=bool(os.environ.get("ASM2VEC_VERBOSE")),
                            current_sym_addr=original_addr_hex
                        )
                        if not original_output_file:
                            logger.error(
                                f"无法提取原始函数 {function_name} 的汇编文件 "
                                f"(addr={original_addr_hex})"
                            )
                            logger.error(
                                f"提取失败上下文: binary={original_binary} exists={os.path.exists(original_binary)} sym_to_addr={sym_to_addr_path} simple_mode={simple_mode}"
                            )
                            return None, None
                        original_asm = original_output_file
                        if original_asm_cache is not None:
                            original_asm_cache[cache_key] = original_asm
                
                # 提取变异文件的汇编
                mutate_output_file = binfunc2asm(
                    ipath=mutated_binary,
                    target_func_name=function_name,
                    opath=os.path.join(asm_work_dir, 'mutated'), 
                    verbose=bool(os.environ.get("ASM2VEC_VERBOSE")),
                    current_sym_addr=mutated_addr_hex
                )
                if not mutate_output_file:
                    logger.error(
                        f"无法提取变异函数 {function_name} 的汇编文件 "
                        f"(addr={mutated_addr_hex})"
                    )
                    logger.error(
                        "提取失败上下文: mutated_binary=%s exists=%s sym_to_addr=%s simple_mode=%s",
                        mutated_binary,
                        os.path.exists(mutated_binary),
                        sym_to_addr_path,
                        simple_mode,
                    )
                    return None, None
                logger.info(f"提取到变异函数汇编: {mutate_output_file}")
                # 计算相似度
                score = compare_functions(original_asm, mutate_output_file)
                grad = 0.0
                
                # 清理变异文件
                try:
                    os.remove(mutate_output_file)
                except Exception as e:
                    logger.debug(f"清理变异汇编文件失败: {mutate_output_file}, {e}")
                
                return abs(score), abs(grad)
            finally:
                # 只在需要时清理临时目录（固定工作目录不清理）
                if need_cleanup:
                    try:
                        shutil.rmtree(asm_work_dir)
                    except Exception as e:
                        logger.warning(f"清理临时目录失败: {asm_work_dir}, {e}")
        elif detection_method == "safe":
            logger.info(f"[*] detection_method: {detection_method}")
            if original_addr_hex is None or mutated_addr_hex is None:
                logger.error(
                    f"SAFE需要函数地址，但未解析到地址: "
                    f"original={original_addr_hex}, mutated={mutated_addr_hex}"
                )
                return None, None

            safe_root, ckpt_dir, i2v_dir = _get_safe_paths(
                checkpoint_dir=safe_checkpoint_dir,
                i2v_dir=safe_i2v_dir,
            )
            if not os.path.isdir(ckpt_dir) or not os.path.isdir(i2v_dir):
                logger.error(f"SAFE模型路径无效: ckpt={ckpt_dir}, i2v={i2v_dir}")
                return None, None

            addr1 = _normalize_addr_int(original_addr_hex)
            addr2 = _normalize_addr_int(mutated_addr_hex)
            if addr1 is None or addr2 is None:
                logger.error(f"SAFE地址解析失败: {original_addr_hex}, {mutated_addr_hex}")
                return None, None
            # print(f"SAFE function addrs: original={addr1}, mutated={addr2}")
            # input("[run_one]:Press Enter to continue...")
            emb1 = _safe_get_embedding(original_binary, addr1, ckpt_dir, i2v_dir, use_gpu=safe_use_gpu)
            if emb1 is None:
                logger.error(f"SAFE无法提取原始函数嵌入: {function_name} @ {original_addr_hex}")
                return None, None
            emb2 = _safe_get_embedding(mutated_binary, addr2, ckpt_dir, i2v_dir, use_gpu=safe_use_gpu)
            if emb2 is None:
                logger.error(f"SAFE无法提取变异函数嵌入: {function_name} @ {mutated_addr_hex}")
                return None, None

            score = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
            return abs(score), 0.0
        else:
            logger.error(f"Unsupported detection_method: {detection_method}")
            return None, None
            
    except Exception as e:
        logger.error(f"run_one函数出错: {e}")
        return None, None


# def train_pickle(asm_file):
    # TODO


# if __name__ == '__main__':
#     # compare_functions(ipath1="/home/ycy/ours/Deceiving-DNN-based-Binary-Matching/bin_bk/pwd_asm/changed_asm/0092b83bfab97f48a5e10cb3830436481e33baa977c175dbe0c63dbe9b5575fc",
#                 # ipath2='/home/ycy/ours/Deceiving-DNN-based-Binary-Matching/bin_bk/pwd_asm/original_asm/1c359ba4755040359222334bb638b769f9f58a6fd6ca9a312914f67ea70fb5b6')
#     run_one(original_binary="/home/ycy/ours/Deceiving-DNN-based-Binary-Matching/bin_bk/pwd",
#             mutated_binary="/home/ycy/ours/Deceiving-DNN-based-Binary-Matching/function_container_usage_pwd/f906ab2da901dbb8d4ecabd3f95409b1_container/f906ab2da901dbb8d4ecabd3f95409b1",
#             model_original=None,
#             checkdict=None,
#             function_name="usage",
#             detection_method="asm2vec")
