# -*- coding: utf-8 -*-
"""
Code post processing
"""
import os
import re
import config
import inline_update
from utils.ail_utils import ELF_utils
import logging
logger = logging.getLogger(__name__)

_CONST_IMM_OP_RE = re.compile(
    r'^\s*(?:[.$@A-Za-z0-9_]+\s*:\s*)?(and|or|xor|test|cmp|add|sub)(?:[bwlq])?\b',
    re.I
)

def _is_valid_main_symbol(sym):
    return re.match(r'^S_0x[0-9A-Fa-f]+$', sym or '') is not None

def _has_main_export(lines):
    for l in lines:
        s = l.strip()
        if s.startswith('.globl main') or s.startswith('.global main') \
           or s.startswith('main :') or s.startswith('main:') \
           or '.set main,' in s:
            return True
    return False

def _find_main_candidate_from_start(lines):
    # Prefer the startup convention: _start loads main into %rdi/%stack and then calls libc entry.
    limit = min(len(lines), 240)
    label_pat = re.compile(r'^\s*S_0x[0-9A-Fa-f]+\s*:')
    call_pat = re.compile(r'\bcallq?\b')
    mov_main_pat = re.compile(r'\bmov[a-z]*\s+\$(S_0x[0-9A-Fa-f]+),%rdi\b')
    push_main_pat = re.compile(r'\bpush[l]?\s+\$(S_0x[0-9A-Fa-f]+)\b')

    first_label_seen = False
    search_end = limit
    for i in range(limit):
        if label_pat.search(lines[i]):
            if not first_label_seen:
                first_label_seen = True
            else:
                search_end = i
                break

    for i in range(search_end):
        m = mov_main_pat.search(lines[i])
        if not m:
            m = push_main_pat.search(lines[i])
        if not m:
            continue
        for j in range(i + 1, min(i + 8, search_end)):
            if call_pat.search(lines[j]):
                return m.group(1)

    # Last fallback: first visible function-like label.
    for i in range(limit):
        m = label_pat.search(lines[i])
        if m:
            return m.group(0).split(':')[0].strip()
    return ''

def _inject_main_alias(lines, target_symbol):
    if not target_symbol:
        return lines, False
    alias_lines = ['.globl main\n', '.set main, ' + target_symbol + '\n']
    insert_at = 0
    for i, l in enumerate(lines):
        if l.strip() == '.section .text':
            insert_at = i + 1
            break
    new_lines = list(lines)
    new_lines[insert_at:insert_at] = alias_lines
    return new_lines, True

def _rewrite_symbolic_immediate_to_const(line):
    """
    x86_64: 某些算术/逻辑/比较指令里的 $S_0xXXXX 实际上是常量立即数，
    不应走符号重定位。否则可能触发 R_X86_64_16 / R_X86_64_32 截断错误。
    """
    if '$S_0x' not in line or not ELF_utils.elf_64():
        return line
    if not _CONST_IMM_OP_RE.match(line):
        return line
    return re.sub(r'\$S_0x([0-9A-Fa-f]+)', r'$0x\1', line)

def main(instrument=False):
    """
    Transform malformed code and add main symbol
    :param instrument: True to insert instrumentation code
    """

    with open("final.s") as f:
        lines = f.readlines()

    if ELF_utils.elf_exe():
        main_symbol1 = ''
        if os.path.isfile('main.info'):
            with open('main.info') as f:
                main_symbol1 = f.readline().strip()
        main_symbol_valid = _is_valid_main_symbol(main_symbol1)

        def helpf(l):
            if '__gmon_start__' in l:
                l = ''
            elif 'lea 0x7FFFFFFC(,%ebx,0x4),%edi' in l:
                l = l.replace('0x7FFFFFFC', '0x7FFFFFFFFFFFFFFC')
            elif 'movzbl $S_' in l:
                l = l.replace('movzbl $S_', 'movzbl S_')
            elif ELF_utils.elf_64() and 'movw $S_0x' in l:
                # x86_64 下 movw 只能接 16-bit 立即数。
                # 若写成 $S_0xXXXX 会被当成符号地址重定位，链接时可能出现
                # "relocation truncated to fit: R_X86_64_16"。
                # 对这种模式安全地回退为纯 16-bit 常量。
                l = re.sub(r'\$S_0x([0-9A-Fa-f]+)', r'$0x\1', l)
            elif '$S_0x' in l:
                # 保留符号立即数，避免 64 位下绝对地址常量导致错误重定位
                # 仅在 32 位下保留旧行为以兼容历史输出
                if ELF_utils.elf_32() and 'branch_des' not in l:
                    l = re.sub(r'\$S_0x([0-9A-Fa-f]+)', r'$0x\1', l)
            elif 'jmpq ' in l and '*' not in l:
                l = l.replace('jmpq ', 'jmp ')
            elif 'repz retq' in l:
                # to solve the error of 'expecting string instruction after `repz'
                l = l.replace('repz retq', 'retq')
            elif 'repz ret' in l:
                l = l.replace('repz ret', 'ret')
            elif 'nop ' in l:   # 处理nop指令
                l = l.replace('nop', ' ')
            l = _rewrite_symbolic_immediate_to_const(l)
            if main_symbol_valid and (main_symbol1 + ' : ') in l:
                rep = '.globl main\nmain : '
                if instrument:
                    rep += '\n'.join(map(lambda e: e['plain'].beforemain, config.instrumentors)) + '\n'
                l = l.replace(main_symbol1 + ' : ', rep)
            elif main_symbol_valid and main_symbol1 in l:
                l = l.replace(main_symbol1, 'main')
            return l

        # ret = os.system("python3 /home/ycy/ours/Deceiving-DNN-based-Binary-Matching/postprocess/fix_missing_symbols.py")
        # logger.info("[post_process.py:main]: fix_missing_symbols.py done, ret = {}".format(ret))
        lines = list(map(helpf, lines))

        # 额外检查：如果发现main :但前面没有.globl main，则添加
        main_globl_added = False
        for i, line in enumerate(lines):
            if 'main :' in line and not main_globl_added:
                # 检查前一行是否已经有.globl main
                if i == 0 or '.globl main' not in lines[i-1]:
                    lines[i] = '.globl main\n' + line
                main_globl_added = True
                break

        # 框架兜底：若 main 未被正确导出，基于启动代码自动补 main 别名。
        if not _has_main_export(lines):
            fallback_symbol = ''
            if main_symbol_valid:
                fallback_symbol = main_symbol1
            if not fallback_symbol:
                fallback_symbol = _find_main_candidate_from_start(lines)
            lines, alias_added = _inject_main_alias(lines, fallback_symbol)
            if alias_added:
                logger.warning('[post_process.py:main]: injected fallback main alias to %s', fallback_symbol)

    # # 清理行尾的无效字符（修复 "junk at end of line" 错误）
    # # 移除行尾的 "空格/制表符 + 0" 模式（但保留有效的数字如 0x10, $0 等）
    # def clean_line_endings(l):
    #     if not l:
    #         return l
    #     # 使用正则表达式移除行尾的无效 '0' 字符
    #     # 匹配：行尾有空格/制表符 + '0'，且前面不是有效数字的一部分
    #     # 但保留如 0x10, $0, 10, %rax0 等有效格式
    #     l = re.sub(r'[ \t]+0$', '', l.rstrip())
    #     # 确保行以换行符结尾
    #     if l and not l.endswith('\n'):
    #         l = l + '\n'
    #     return l
    
    # # 对所有行进行清理
    # lines = [clean_line_endings(l) for l in lines]

    with open("final.s", 'w') as f:
        f.writelines(lines)
        if instrument: f.write('\n'.join(map(lambda e: e['plain'].aftercode, config.instrumentors)) + '\n')

    if os.path.isfile('inline_symbols.txt'):
        inline_update.main()
