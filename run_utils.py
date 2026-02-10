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
import threading
from collections import OrderedDict

# Force transformers to skip TensorFlow; also drop any broken placeholder module.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
if "tensorflow" in sys.modules and getattr(sys.modules["tensorflow"], "__spec__", None) is None:
    del sys.modules["tensorflow"]

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


def _get_safe_paths(safe_root=None, checkpoint_dir=None, i2v_dir=None):
    base = safe_root or os.environ.get(
        "SAFE_ROOT",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection_model", "SAFE"),
    )
    ckpt = checkpoint_dir or os.environ.get("SAFE_CHECKPOINT_DIR", os.path.join(base, "experiments", "output", "last_run"))
    i2v = i2v_dir or os.environ.get("SAFE_I2V_DIR", os.path.join(base, "data", "i2v"))
    return base, ckpt, i2v


def _get_safe_session(checkpoint_dir, i2v_dir, use_gpu=True, safe_cache=None):
    if safe_cache is not None:
        key = (checkpoint_dir, i2v_dir, bool(use_gpu))
        sess = safe_cache.get("session")
        get_functions = safe_cache.get("get_functions")
        if sess is not None and safe_cache.get("session_key") == key and get_functions is not None:
            return sess, get_functions, False

        try:
            if sess is not None:
                sess.close()
        except Exception:
            pass

    safe_root, _, _ = _get_safe_paths(checkpoint_dir=checkpoint_dir, i2v_dir=i2v_dir)
    if safe_root not in sys.path:
        sys.path.insert(0, safe_root)

    # Lazy import to avoid TF load when not needed.
    from inference_example import SAFEInferenceSession, get_functions_from_binary

    sess = SAFEInferenceSession(checkpoint_dir, i2v_dir, use_gpu=use_gpu)
    if safe_cache is not None:
        safe_cache["session"] = sess
        safe_cache["session_key"] = (checkpoint_dir, i2v_dir, bool(use_gpu))
        safe_cache["get_functions"] = get_functions_from_binary
        safe_cache.setdefault("emb_cache", {})
        safe_cache.setdefault("instr_cache", {})
        return sess, get_functions_from_binary, False

    return sess, get_functions_from_binary, True


# === jTrans helpers (lazy import) ===


def _get_jtrans_paths(model_dir=None, tokenizer_dir=None):
    base = os.environ.get(
        "JTRANS_ROOT",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection_model", "jTrans"),
    )
    model = model_dir or os.environ.get(
        "JTRANS_MODEL_DIR", os.path.join(base, "models", "jTrans-finetune")
    )
    tokenizer = tokenizer_dir or os.environ.get(
        "JTRANS_TOKENIZER_DIR", os.path.join(base, "jtrans_tokenizer")
    )
    return base, model, tokenizer


def _get_jtrans_session(model_dir, tokenizer_dir, use_gpu=True, jtrans_cache=None):
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    if "tensorflow" in sys.modules and getattr(sys.modules["tensorflow"], "__spec__", None) is None:
        del sys.modules["tensorflow"]
    if jtrans_cache is not None:
        key = (model_dir, tokenizer_dir, bool(use_gpu))
        session_lock = jtrans_cache.setdefault("_session_lock", threading.Lock())
        with session_lock:
            model = jtrans_cache.get("model")
            tokenizer = jtrans_cache.get("tokenizer")
            device = jtrans_cache.get("device")
            if model is not None and tokenizer is not None and jtrans_cache.get("session_key") == key:
                return model, tokenizer, device

    try:
        import torch
        try:
            from transformers import BertTokenizer, BertModel
        except Exception:
            # 兼容部分 transformers 版本未在包根导出 Bert* 的情况
            from transformers.models.bert.tokenization_bert import BertTokenizer
            from transformers.models.bert.modeling_bert import BertModel
    except Exception as e:
        logger.error(f"jTrans依赖缺失: {e}")
        return None, None, None

    class BinBertModel(BertModel):
        def __init__(self, config, add_pooling_layer=True):
            super().__init__(config)
            self.config = config
            self.embeddings.position_embeddings = self.embeddings.word_embeddings

    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    model = BinBertModel.from_pretrained(model_dir)
    model.eval()
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
    if jtrans_cache is not None:
        session_lock = jtrans_cache.setdefault("_session_lock", threading.Lock())
        with session_lock:
            jtrans_cache["model"] = model
            jtrans_cache["tokenizer"] = tokenizer
            jtrans_cache["device"] = device
            jtrans_cache["session_key"] = (model_dir, tokenizer_dir, bool(use_gpu))
            emb_cache = jtrans_cache.get("emb_cache")
            if not isinstance(emb_cache, OrderedDict):
                emb_cache = OrderedDict(emb_cache or {})
                jtrans_cache["emb_cache"] = emb_cache
            try:
                emb_cache_max = int(
                    jtrans_cache.get("emb_cache_max", os.environ.get("JTRANS_EMB_CACHE_MAX", "512"))
                )
            except Exception:
                emb_cache_max = 512
            jtrans_cache["emb_cache_max"] = max(0, emb_cache_max)
    return model, tokenizer, device


def _jtrans_tokens_from_asm(asm_path):
    try:
        from readidadata import parse_asm
    except Exception as e:
        logger.error(f"jTrans解析模块加载失败: {e}")
        return []

    tokens = []
    try:
        with open(asm_path, "r", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.endswith(":"):
                    continue
                operator, op1, op2, op3, _ann = parse_asm(line)
                if operator:
                    tokens.append(operator)
                if op1:
                    tokens.append(op1)
                if op2:
                    tokens.append(op2)
                if op3:
                    tokens.append(op3)
    except Exception as e:
        logger.error(f"jTrans读取汇编失败: {asm_path}, {e}")
    return tokens


def _jtrans_embed_from_asm(asm_path, model, tokenizer, device, jtrans_cache=None):
    try:
        import torch
    except Exception as e:
        logger.error(f"jTrans依赖缺失: {e}")
        return None

    cache_key = None
    if jtrans_cache is not None:
        try:
            cache_key = (os.path.abspath(asm_path), os.path.getmtime(asm_path))
            cache_lock = jtrans_cache.setdefault("_cache_lock", threading.Lock())
            with cache_lock:
                emb_cache = jtrans_cache.get("emb_cache")
                if not isinstance(emb_cache, OrderedDict):
                    emb_cache = OrderedDict(emb_cache or {})
                    jtrans_cache["emb_cache"] = emb_cache
                if cache_key in emb_cache:
                    emb = emb_cache.pop(cache_key)
                    emb_cache[cache_key] = emb
                    return emb
        except Exception:
            cache_key = None

    tokens = _jtrans_tokens_from_asm(asm_path)
    if not tokens:
        return None
    text = " ".join(tokens)
    encoded = tokenizer(
        [text],
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        emb = output.pooler_output[0].detach().cpu().numpy()

    if jtrans_cache is not None and cache_key is not None:
        cache_lock = jtrans_cache.setdefault("_cache_lock", threading.Lock())
        with cache_lock:
            emb_cache = jtrans_cache.get("emb_cache")
            if not isinstance(emb_cache, OrderedDict):
                emb_cache = OrderedDict(emb_cache or {})
                jtrans_cache["emb_cache"] = emb_cache
            emb_cache[cache_key] = emb
            try:
                emb_cache_max = int(
                    jtrans_cache.get("emb_cache_max", os.environ.get("JTRANS_EMB_CACHE_MAX", "512"))
                )
            except Exception:
                emb_cache_max = 512
            if emb_cache_max <= 0:
                emb_cache.clear()
            else:
                while len(emb_cache) > emb_cache_max:
                    emb_cache.popitem(last=False)
    return emb


def _safe_get_embedding(
    binary_path,
    addr_int,
    checkpoint_dir,
    i2v_dir,
    use_gpu=True,
    safe_cache=None,
    session=None,
    get_functions=None,
):
    cache_key = (os.path.abspath(binary_path), addr_int)
    if safe_cache is not None:
        emb_cache = safe_cache.setdefault("emb_cache", {})
        if cache_key in emb_cache:
            return emb_cache[cache_key]

    if session is None or get_functions is None:
        session, get_functions, _ephemeral = _get_safe_session(
            checkpoint_dir, i2v_dir, use_gpu=use_gpu, safe_cache=safe_cache
        )

    if safe_cache is not None:
        instr_cache = safe_cache.setdefault("instr_cache", {})
        instr = instr_cache.get(cache_key)
    else:
        instr = None

    if instr is None:
        instr_list = get_functions(binary_path, addr_int)
        instr = instr_list[0] if instr_list else None
        if safe_cache is not None:
            instr_cache[cache_key] = instr

    if instr is None:
        return None

    emb = session.get_embedding(instr, verbose=False)
    if safe_cache is not None:
        emb_cache[cache_key] = emb
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
    mutated_asm_cache=None,
    safe_cache=None,
    jtrans_model_dir=None,
    jtrans_tokenizer_dir=None,
    jtrans_use_gpu=True,
    jtrans_cache=None,
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
        detection_method: "asm2vec" / "safe" / "jtrans"
        *_func_addr: 可选的函数地址（优先于符号名）
        sym_to_addr_*: 可选映射文件/字典，用于解析变异后地址
        safe_*: SAFE 模型所需的 checkpoint 与 i2v 目录
        original_asm_path: 可选的原始汇编文件路径（用于复用）
        mutated_asm_cache: 可选的变异函数汇编缓存 dict
        safe_cache: 可选的 SAFE 缓存 dict（session/embedding/instr）
        jtrans_*: jTrans 模型与 tokenizer 路径/开关
        jtrans_cache: 可选的 jTrans 缓存 dict（model/tokenizer/emb）
    
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
                
                # 提取/复用变异文件的汇编
                mutate_output_file = None
                mutate_cache_key = None
                if mutated_asm_cache is not None:
                    mutate_cache_key = (
                        os.path.abspath(mutated_binary),
                        function_name,
                        mutated_addr_hex,
                    )
                    mutate_output_file = mutated_asm_cache.get(mutate_cache_key)
                    if mutate_output_file and not os.path.exists(mutate_output_file):
                        mutate_output_file = None
                        mutated_asm_cache.pop(mutate_cache_key, None)

                if mutate_output_file is None:
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
                    if mutated_asm_cache is not None and mutate_cache_key is not None:
                        mutated_asm_cache[mutate_cache_key] = mutate_output_file
                logger.info(f"提取到变异函数汇编: {mutate_output_file}")
                # 计算相似度
                score = compare_functions(original_asm, mutate_output_file)
                grad = 0.0
                
                # 清理变异文件
                if mutated_asm_cache is None:
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
            sess = None
            get_functions = None
            ephemeral = False
            if safe_cache is None:
                sess, get_functions, ephemeral = _get_safe_session(
                    ckpt_dir, i2v_dir, use_gpu=safe_use_gpu, safe_cache=None
                )

            emb1 = _safe_get_embedding(
                original_binary,
                addr1,
                ckpt_dir,
                i2v_dir,
                use_gpu=safe_use_gpu,
                safe_cache=safe_cache,
                session=sess,
                get_functions=get_functions,
            )
            if emb1 is None:
                logger.error(f"SAFE无法提取原始函数嵌入: {function_name} @ {original_addr_hex}")
                if ephemeral and sess is not None:
                    try:
                        sess.close()
                    except Exception:
                        pass
                return None, None
            emb2 = _safe_get_embedding(
                mutated_binary,
                addr2,
                ckpt_dir,
                i2v_dir,
                use_gpu=safe_use_gpu,
                safe_cache=safe_cache,
                session=sess,
                get_functions=get_functions,
            )
            if emb2 is None:
                logger.error(f"SAFE无法提取变异函数嵌入: {function_name} @ {mutated_addr_hex}")
                if ephemeral and sess is not None:
                    try:
                        sess.close()
                    except Exception:
                        pass
                return None, None
            if ephemeral and sess is not None:
                try:
                    sess.close()
                except Exception:
                    pass

            score = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
            return abs(score), 0.0
        elif detection_method == "jtrans":
            logger.info(f"[*] detection_method: {detection_method}")

            jtrans_root, model_dir, tokenizer_dir = _get_jtrans_paths(
                model_dir=jtrans_model_dir, tokenizer_dir=jtrans_tokenizer_dir
            )
            if jtrans_root not in sys.path:
                sys.path.insert(0, jtrans_root)
            if not os.path.isdir(model_dir) or not os.path.isdir(tokenizer_dir):
                logger.error(f"jTrans模型路径无效: model={model_dir}, tokenizer={tokenizer_dir}")
                return None, None

            model, tokenizer, device = _get_jtrans_session(
                model_dir, tokenizer_dir, use_gpu=jtrans_use_gpu, jtrans_cache=jtrans_cache
            )
            if model is None or tokenizer is None:
                return None, None

            if asm_work_dir is None:
                jtrans_work_dir = tempfile.mkdtemp(prefix="jtrans_", suffix="_tmp")
                need_cleanup = True
            else:
                jtrans_work_dir = os.path.join(asm_work_dir, "jtrans")
                os.makedirs(os.path.join(jtrans_work_dir, "original"), exist_ok=True)
                os.makedirs(os.path.join(jtrans_work_dir, "mutated"), exist_ok=True)
                need_cleanup = False

            try:
                # 原始汇编（可复用 asm2vec 的缓存）
                if original_asm_path:
                    if not os.path.exists(original_asm_path):
                        logger.error(f"原始汇编文件不存在: {original_asm_path}")
                        return None, None
                    original_asm = original_asm_path
                else:
                    original_asm, _ori_addr, _ = prepare_original_asm(
                        original_binary=original_binary,
                        function_name=function_name,
                        asm_work_dir=jtrans_work_dir,
                        original_func_addr=original_func_addr,
                        sym_to_addr_path=sym_to_addr_path,
                        sym_to_addr_map=sym_to_addr_map,
                        simple_mode=simple_mode,
                        original_asm_cache=original_asm_cache,
                    )
                    if not original_asm:
                        return None, None

                mutate_output_file = binfunc2asm(
                    ipath=mutated_binary,
                    target_func_name=function_name,
                    opath=os.path.join(jtrans_work_dir, "mutated"),
                    verbose=bool(os.environ.get("ASM2VEC_VERBOSE")),
                    current_sym_addr=mutated_addr_hex,
                )
                if not mutate_output_file:
                    logger.error(f"无法提取变异函数 {function_name} 的汇编文件")
                    return None, None

                emb1 = _jtrans_embed_from_asm(
                    original_asm, model, tokenizer, device, jtrans_cache=jtrans_cache
                )
                emb2 = _jtrans_embed_from_asm(
                    mutate_output_file, model, tokenizer, device, jtrans_cache=jtrans_cache
                )
                if emb1 is None or emb2 is None:
                    return None, None
                score = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
                if not need_cleanup:
                    try:
                        os.remove(mutate_output_file)
                    except Exception:
                        pass
                return abs(score), 0.0
            finally:
                if need_cleanup:
                    try:
                        shutil.rmtree(jtrans_work_dir)
                    except Exception as e:
                        logger.warning(f"清理jTrans临时目录失败: {jtrans_work_dir}, {e}")
        else:
            logger.error(f"Unsupported detection_method: {detection_method}")
            return None, None
            
    except Exception as e:
        logger.error(f"run_one函数出错: {e}")
        return None, None
