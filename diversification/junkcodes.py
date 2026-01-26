# -*- coding: utf-8 -*-
"""
【辅助工具模块】垃圾代码生成器 (Junk Code Generator)
【框架序号】N/A（辅助工具，非独立变异操作）
【功能说明】为其他变异操作（如 instr_garbage_diversify）提供垃圾代码生成功能，
           支持多种级别的垃圾代码（NOP、寄存器操作、条件移动等），
           用于增加代码复杂度和干扰特征提取。
"""
from disasm.Types import *
from utils.ail_utils import *
from utils.pp_print import *
from copy import deepcopy
import random
from config import junk_code_level
from utils.ail_utils import ELF_utils


class JunkBase:

    def __init__(self):
        pass

    def get_codes(self):
        return []

class Junk4(JunkBase):
    def __init__(self, iloc):
        JunkBase.__init__(self)
        self._loc = deepcopy(iloc)
        self._loc.loc_label = ''
        self._regs = _safe_regs()

    def get_codes(self):
        return _safe_noop_ops(self._loc, count=4)


class Junk3(JunkBase):
    def __init__(self, iloc):
        JunkBase.__init__(self)
        self._loc = deepcopy(iloc)
        self._loc.loc_label = ''
        self._regs = _safe_regs()

    def get_codes(self):
        return _safe_noop_ops(self._loc, count=3)

class Junk1(JunkBase):

    def __init__(self, iloc):
        JunkBase.__init__(self)
        self._loc = deepcopy(iloc)
        self._loc.loc_label = ''
        self._regs = _safe_regs()

    def get_codes(self):
        return _safe_noop_ops(self._loc, count=1)

class Junk0(JunkBase):
    def __init__(self, iloc):
        pass

    def get_codes(self):
        return []


class Junk2(JunkBase):

    def __init__(self, iloc):
        JunkBase.__init__(self)
        self._loc = deepcopy(iloc)
        self._loc.loc_label = ''
        self._regs = _safe_regs()

    def get_codes(self):
        return _safe_noop_ops(self._loc, count=2)


def _safe_regs():
    if ELF_utils.elf_64():
        reg_names = ['RAX', 'RCX', 'RDX']
    else:
        reg_names = ['EAX', 'ECX', 'EDX']
    return [RegClass(r) for r in reg_names]


def _safe_noop_ops(loc, count=None):
    iloc = deepcopy(loc)
    iloc.loc_label = ''
    regs = _safe_regs()
    pool = [SingleInstr(('nop', iloc, None))]
    for reg in regs:
        pool.append(TripleInstr(('mov', reg, reg, iloc, None)))
        pool.append(TripleInstr(('xchg', reg, reg, iloc, None)))
    if count is None:
        count = random.randint(1, min(4, len(pool)))
    return [random.choice(pool) for _ in range(count)]


junk_codes = [
    Junk0,
    Junk1,
    Junk2,
    Junk3,
    Junk4,
    Junk4,
    Junk4,
]


def get_junk_codes(loc, level=None):
    if level is None:
        if junk_code_level is None:
            level = random.randint(1, len(junk_codes)-1)
        else:
            level = junk_code_level
    junk_class = junk_codes[level]
    junk_instance = junk_class(loc)
    return junk_instance.get_codes()
