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

def main(instrument=False):
    """
    Transform malformed code and add main symbol
    :param instrument: True to insert instrumentation code
    """

    with open("final.s") as f:
        lines = f.readlines()

    if ELF_utils.elf_exe():
        main_symbol1 = ''

        with open('main.info') as f:
            main_symbol1 = f.readline().strip()

        if main_symbol1 != '':
            def helpf(l):
                if '__gmon_start__' in l:
                    l = ''
                elif 'lea 0x7FFFFFFC(,%ebx,0x4),%edi' in l:
                    l = l.replace('0x7FFFFFFC', '0x7FFFFFFFFFFFFFFC')
                elif 'movzbl $S_' in l:
                    l = l.replace('movzbl $S_', 'movzbl S_')
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
                if main_symbol1 + ' :' in l:
                    rep = '.globl main\nmain : '
                    if instrument:
                        rep += '\n'.join(map(lambda e: e['plain'].beforemain, config.instrumentors)) + '\n'
                    l = l.replace(main_symbol1 + ' : ', rep)
                elif main_symbol1 in l and main_symbol1 != 'S_0x':
                    l = l.replace(main_symbol1, 'main')
                return l
            
            # 额外检查：如果发现main :但前面没有.globl main，则添加
            main_globl_added = False
            for i, line in enumerate(lines):
                if 'main :' in line and not main_globl_added:
                    # 检查前一行是否已经有.globl main
                    if i == 0 or '.globl main' not in lines[i-1]:
                        lines[i] = '.globl main\n' + line
                    main_globl_added = True
                    break

            # ret = os.system("python3 /home/ycy/ours/Deceiving-DNN-based-Binary-Matching/postprocess/fix_missing_symbols.py")
            # logger.info("[post_process.py:main]: fix_missing_symbols.py done, ret = {}".format(ret))
            lines = map(helpf, lines)
            # 转换为列表（兼容 Python 2/3）
            # lines = list(lines)

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
