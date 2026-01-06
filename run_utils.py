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

# 添加 asm2vec-pytorch 路径到 sys.path
_asm2vec_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             'detection_model', 'asm2vec-pytorch')
if _asm2vec_path not in sys.path:
    sys.path.insert(0, _asm2vec_path)

# 导入 compare_functions
from scripts.compare_util import *
from scripts.bin2asm_util import *
 
def run_one(original_binary, mutated_binary, model_original, checkdict, function_name, detection_method = "asm2vec", asm_work_dir=None, original_asm_cache=None):
    """
    评估原始二进制文件和变异二进制文件之间的相似度
    
    这是一个关键函数，用于计算变异二进制文件与原始文件的相似度分数和梯度值。
    分数越低表示相似度越高，梯度值用于指导优化方向。
    
    参数:
        original_binary: 原始二进制文件路径
        mutated_binary: 变异二进制文件路径  
        model_original: 预训练的模型对象
        checkdict: 函数映射字典，包含符号到地址的映射关系
        function_name: 目标函数名
    
    返回:
        score: 相似度分数 (float) - 越低表示越相似，目标是< 0.40
        grad: 梯度值 (float) - 用于指导变异方向的优化指标
    """
    try:
        logger.debug(f'run_one开始: 原始={original_binary}, 变异={mutated_binary}, 函数={function_name}')
        
        # 1. 检查输入文件是否存在
        if not os.path.exists(original_binary):
            logger.error(f"原始二进制文件不存在: {original_binary}")
            return None, None
        
        if not os.path.exists(mutated_binary):
            logger.error(f"变异二进制文件不存在: {mutated_binary}")
            return None, None
        
        # 2. 获取汇编文件路径
        if detection_method == "asm2vec":
            logger.info(f"[*] detection_method: {detection_method}")
            mutated_folder = os.path.dirname(mutated_binary)
            checkdict = pickle.load(open(os.path.join(mutated_folder, "sym_to_addr.pickle"), "rb")) # 原始函数的符号到地址的映射
            ori_sym_addr = checkdict[function_name]  # 原始函数的符号在现在的地址
            
            logger.info(f"[*] ori_sym_addr: {ori_sym_addr}")
            
            # 【性能优化】使用固定工作目录，避免频繁创建删除
            if asm_work_dir is None:
                asm_work_dir = tempfile.mkdtemp(prefix='asm2vec_', suffix='_tmp')
                need_cleanup = True
            else:
                need_cleanup = False
                os.makedirs(os.path.join(asm_work_dir, 'mutated'), exist_ok=True)
                os.makedirs(os.path.join(asm_work_dir, 'original'), exist_ok=True)
            
            try:
                # 【性能优化】缓存原始文件汇编（原始文件不变）
                cache_key = (os.path.abspath(original_binary), function_name, ori_sym_addr)
                if original_asm_cache is not None and cache_key in original_asm_cache:
                    original_asm = original_asm_cache[cache_key]
                    logger.debug(f"使用缓存的原始文件汇编: {original_asm}")
                else:
                    original_output_file = binfunc2asm(ipath = original_binary,
                                            target_func_name = function_name,    # 原始文件的函数名
                                            opath=os.path.join(asm_work_dir, 'original'), 
                                            verbose= False,
                                            current_sym_addr = ori_sym_addr)
                    if not original_output_file:
                        logger.error(f"无法提取原始函数 {function_name} 的汇编文件")
                        return None, None
                    original_asm = original_output_file
                    # 存入缓存
                    if original_asm_cache is not None:
                        original_asm_cache[cache_key] = original_asm
                
                # 生成变异文件的汇编文件（每次都需要重新提取）
                mutate_output_file = binfunc2asm(ipath = mutated_binary,
                                        target_func_name = function_name,    # 变异文件的函数名
                                        opath=os.path.join(asm_work_dir, 'mutated'), 
                                        verbose= False, 
                                        current_sym_addr = ori_sym_addr)
                if not mutate_output_file:
                    logger.error(f"无法提取变异函数 {function_name} 的汇编文件")
                    return None, None
                
                mutated_asm = mutate_output_file
                score = compare_functions(original_asm, mutated_asm)
                # asm2vec 方法不提供 grad，使用 score 的变化作为近似
                grad = 0.0  # 或者可以计算 score 的变化率
                
                # 清理变异文件（原始文件保留在缓存中）
                try:
                    os.remove(mutated_asm)
                except Exception as e:
                    logger.debug(f"清理变异汇编文件失败: {mutated_asm}, {e}")
                
                return abs(score), abs(grad)
            finally:
                # 只在需要时清理临时目录（固定工作目录不清理）
                if need_cleanup:
                    try:
                        shutil.rmtree(asm_work_dir)
                    except Exception as e:
                        logger.warning(f"清理临时目录失败: {asm_work_dir}, {e}")
            
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