# -*- coding: utf-8 -*-
"""
Process external symbols from shared libraries
"""

import re
import os
import config
from subprocess import check_output
import logging
logger = logging.getLogger(__name__)

def globalvar(filepath):
    """
    This code aims at solving glibc global variables issue some of the code contains comments like this:
        401599:       48 8b 3d 70 6c 20 00    mov    0x206c70(%rip),%rdi        # 608210 <stdout>
    instructions like this should be translated into:
        mov stdout,%rdi
    :param filepath: path to target executable
    """

    with open(filepath + '.temp') as f:
        lines = f.readlines()

    pat_d = re.compile(r'0x[0-9a-f]+\(%rip\)')
    pat_s = re.compile(r'<([^@]+)(@@(?!Base).*)?>')

    for i in range(len(lines)):
        l = lines[i]
        if "#" in l and not "+" in l:
            m_s = pat_s.search(l)
            m_d = pat_d.search(l)
            if m_s and m_d:
                src = m_s.group(1)
                des = m_d.group(0)
                l = l.split('#')[0]
                l = l.replace(des, src)
                lines[i] = l + '\n'

    with open(filepath + '.temp', 'w') as f:
        f.writelines(lines)


def pltgot(filepath):
    """
    Handle library functions linked through .plt.got and substitute them with correct symbols
    :param filepath: path to target executable
    """
    with open('plts.info') as f:
        f.seek(-2, os.SEEK_END)
        while f.read(1) != '\n': f.seek(-2, os.SEEK_CUR)
        lastplt = f.readline().split()
    
    lastplt = (int(lastplt[0],16), re.escape(lastplt[1].rstrip('>:')))

    pltgotsym_cmd = 'readelf -r ' + filepath + ' | awk \'/_GLOB_DAT/ {print $1,$5}\' | grep -v __gmon_start__ | cat'
    logger.debug("[extern_symbol_process.py:pltgot]: pltgotsym_cmd = {}".format(pltgotsym_cmd))
    pltgotsym = check_output(pltgotsym_cmd, shell=True).strip()
    if len(pltgotsym) == 0: return
    def pltsymmapper(l):
        items = l.strip().split()
        return (int(items[0], 16), items[1].split('@')[0])
    pltgotsym = dict(map(pltsymmapper, pltgotsym.split('\n')))  # 结果如：{0x601018: 'malloc', 0x601020: 'free'}
    logger.debug("[extern_symbol_process.py:pltgot]: pltgotsym = {}".format(pltgotsym))
    pltgottargets_cmd = config.objdump + ' -Dr -j .plt.got ' + filepath + ' | grep jmp | cut -f1,3'
    logger.debug("[extern_symbol_process.py:pltgot]: pltgottargets_cmd = {}".format(pltgottargets_cmd)) 
    pltgottargets = check_output(pltgottargets_cmd, shell=True)   # 结果如：['0x401018 <malloc@plt>', '0x401020 <free@plt>']
    def pltgotmapper(l):            # 需添加如下判断，目的是判断objdump的输出格式问题
        items = l.strip().split()
        if len(items) == 0:
            return None
        try:
            dest = int(items[items.index('#') + 1] if '#' in items else items[2].lstrip('*'), 16)
            return (int(items[0].rstrip(':'), 16), dest)
        except (ValueError, IndexError):
            return None
    pltgottargets = dict(filter(None, map(pltgotmapper, pltgottargets.strip().split('\n'))))
    pltgottargets = {e[0]: '<' + pltgotsym[e[1]] + '@plt>' for e in pltgottargets.iteritems() if e[1] in pltgotsym}  # 结果如：{0x401018: '<malloc@plt>', 0x401020: '<free@plt>'}
    logger.debug("[extern_symbol_process.py:pltgot]: pltgottargets = {}".format(pltgottargets))
    logger.debug("[extern_symbol_process.py:pltgot]: Found {} PLT.GOT mappings".format(len(pltgottargets)))
    if len(pltgottargets) == 0: 
        logger.info("[extern_symbol_process.py:pltgot]: No PLT.GOT mappings found for {}, skipping symbol replacement".format(filepath))
        return

    pltgotre = re.compile(lastplt[1] + '\+(0x[0-9a-f]+)\>', re.I)
    logger.debug("[extern_symbol_process.py:pltgot]: Looking for pattern: {}".format(pltgotre.pattern))
    logger.info("[extern_symbol_process.py:pltgot]: Proceeding with {} PLT.GOT mappings for symbol replacement".format(len(pltgottargets)))
    def calldesmapper(l):
        m = pltgotre.search(l)
        if m:
            dest = lastplt[0] + int(m.group(1), 16)
            if dest in pltgottargets: return pltgotre.sub(pltgottargets[dest], l)
        return l
    with open(filepath + '.temp') as f:
        lines = map(calldesmapper, f)

    with open(filepath + '.temp', 'w') as f:
        f.writelines(lines)
