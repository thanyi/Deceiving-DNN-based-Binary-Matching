# -*- coding: utf-8 -*-
"""
【变异操作类型】插入垃圾代码 (Insert Garbage Instructions)
【框架序号】Action ID: 7
【功能说明】在基本块中插入无用的指令（如 NOP、寄存器自赋值等），增加代码体积和复杂度，
           干扰基于指令序列的特征提取，但不改变程序的实际执行逻辑。
"""
from analysis.visit import *
from disasm.Types import *
from utils.ail_utils import *
from utils.pp_print import *
from junkcodes import get_junk_codes
from addr_utils import addr_equals, select_block_by_addr


class instr_garbage_diversify(ailVisitor):

    def __init__(self, funcs, fb_tbl, cfg_tbl):
        ailVisitor.__init__(self)
        self.fb_tbl = fb_tbl

    def _nop_garbage_instrs(self, loc):
        # use insert or append to add these instructions
        # no label
        iloc = copy.deepcopy(loc)
        iloc.loc_label = ''
        res = [
            SingleInstr((self._ops['nop'], iloc, None)),
            TripleInstr((self._ops['mov'], self._stack_regs['bp'], self._stack_regs['bp'], iloc, None)),
            TripleInstr((self._ops['mov'], self._stack_regs['sp'], self._stack_regs['sp'], iloc, None)),
            TripleInstr((self._ops['xchg'], self._stack_regs['bp'], self._stack_regs['bp'], iloc, None)),
            TripleInstr((self._ops['xchg'], self._stack_regs['sp'], self._stack_regs['sp'], iloc, None))
        ]
        res.extend([
            TripleInstr((self._ops['mov'], reg, reg, iloc, None)) for reg in self._regs
        ])
        return res

    def _get_garbage(self, loc, mode=1):
        if mode == 1:
            nops = self._nop_garbage_instrs(loc)
            num_instrs = len(nops)  # random.randint(1, len(nops) - 4)
            res = []
            for i in range(num_instrs):
                res.append(random.choice(nops))
            return res
        elif mode == 2:
            return get_junk_codes(loc, None)
        else:
            return []

    def _insert_garbage(self, loc, mode=1):
        garbage = self._get_garbage(loc, mode)
        for i in garbage:
            self.insert_instrs(i, loc)

    def insert_garbage(self, target_addr=None):
        """
        插入垃圾代码
        
        如果指定 target_addr，只在该地址的基本块中插入垃圾代码
        如果未指定 target_addr，在每个函数的随机基本块中插入垃圾代码
        """
        # 如果指定了 target_addr，只处理该地址的基本块
        if target_addr is not None:
            found = False
            for f in self.fb_tbl.keys():
                bl = self.fb_tbl[f]
                b, exact = select_block_by_addr(bl, target_addr)
                if b is None:
                    continue
                found = True
                if exact:
                    print '[instr_garbage_diversify.py:insert_garbage] Found target_addr: %s (matched with 0x%X)' % (target_addr, b.bblock_begin_loc.loc_addr)
                else:
                    print '[instr_garbage_diversify.py:insert_garbage] Found target_addr: %s inside block (begin=0x%X)' % (target_addr, b.bblock_begin_loc.loc_addr)
                bil = self.bb_instrs(b)
                if len(bil) > 0:
                    loc = get_loc(random.choice(bil))
                    self._insert_garbage(loc, mode=random.randint(1, 2))
                    self.update_process()
                    return
                print '[instr_garbage_diversify.py:insert_garbage] Warning: target block empty, fallback to random'
                break
            if not found:
                print '[instr_garbage_diversify.py:insert_garbage] Warning: target_addr %s not found, fallback to random' % target_addr
        
        # 未指定 target_addr，在每个函数的随机基本块中插入垃圾代码
        for f in self.fb_tbl.keys():
            # select block to insert garbage
            candidates = []
            for b in self.fb_tbl[f]:
                try:
                    bil = self.bb_instrs(b)
                except Exception:
                    continue
                if len(bil) > 0:
                    candidates.append(b)
            if not candidates:
                continue
            b = random.choice(candidates)
            bil = self.bb_instrs(b)
            loc = get_loc(random.choice(bil))
            self._insert_garbage(loc, mode=random.randint(1, 2))
        self.update_process()

    def visit(self, instrs, target_addr = None):
        print 'start garbage insertion ...'
        self.instrs = copy.deepcopy(instrs)
        self.insert_garbage(target_addr)
        return self.instrs
