# -*- coding: utf-8 -*-
"""
地址比较工具函数
用于统一处理整数地址和字符串地址的比较
"""

def normalize_addr(addr):
    """
    将地址标准化为整数
    
    参数:
        addr: 地址，可以是整数或字符串（如 '0x4169d7'、'4286935'）
    
    返回:
        int: 整数地址，失败返回 None
    """
    if addr is None:
        return None
    
    if isinstance(addr, int):
        return addr
    
    if isinstance(addr, (str, unicode)):
        addr_str = addr.strip()
        try:
            # 尝试按 16 进制解析（如 '0x4169d7'）
            if addr_str.startswith('0x') or addr_str.startswith('0X'):
                return int(addr_str, 16)
            # 尝试按 10 进制解析（如 '4286935'）
            else:
                return int(addr_str, 10)
        except ValueError:
            return None
    
    return None


def addr_equals(addr1, addr2):
    """
    比较两个地址是否相等（自动处理类型转换）
    
    参数:
        addr1: 第一个地址（int 或 str）
        addr2: 第二个地址（int 或 str）
    
    返回:
        bool: 是否相等
    """
    norm1 = normalize_addr(addr1)
    norm2 = normalize_addr(addr2)
    
    if norm1 is None or norm2 is None:
        return False
    
    return norm1 == norm2


def addr_in_range(addr, start_addr, end_addr):
    """
    判断 addr 是否落在 [start_addr, end_addr] 区间内（含端点）
    """
    norm = normalize_addr(addr)
    norm_start = normalize_addr(start_addr)
    norm_end = normalize_addr(end_addr)
    
    if norm is None or norm_start is None or norm_end is None:
        return False
    
    if norm_start > norm_end:
        norm_start, norm_end = norm_end, norm_start
    
    return norm_start <= norm <= norm_end


def select_block_by_addr(blocks, target_addr):
    """
    在基本块列表中选择匹配 target_addr 的块
    
    规则：
    1) 优先匹配 bblock_begin_loc 的精确地址
    2) 若无精确匹配，再尝试 addr 落在 [begin, end] 的范围匹配
    
    返回:
        (block, is_exact) 或 (None, False)
    """
    if target_addr is None:
        return None, False
    
    # 1) 精确匹配 block begin
    for b in blocks:
        try:
            if addr_equals(b.bblock_begin_loc.loc_addr, target_addr):
                return b, True
        except Exception:
            continue
    
    # 2) 范围匹配
    for b in blocks:
        try:
            begin_addr = b.bblock_begin_loc.loc_addr
            end_addr = b.bblock_end_loc.loc_addr
            if addr_in_range(target_addr, begin_addr, end_addr):
                return b, False
        except Exception:
            continue
    
    return None, False

