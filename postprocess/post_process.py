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
                    # 修复立即数符号引用：$S_0x495E -> $0x495E
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

    with open("final.s", 'w') as f:
        f.writelines(lines)
        if instrument: f.write('\n'.join(map(lambda e: e['plain'].aftercode, config.instrumentors)) + '\n')

    if os.path.isfile('inline_symbols.txt'):
        inline_update.main()
