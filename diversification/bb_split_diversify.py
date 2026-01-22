# -*- coding: utf-8 -*-
"""
【变异操作类型】基本块分割 (Basic Block Splitting)
【框架序号】Action ID: 4
【功能说明】将较大的基本块分割成多个较小的基本块，通过插入无条件跳转和标签来
           改变基本块粒度，影响基于基本块大小的特征提取，但不在 RL 框架的动作空间中。
"""
from analysis.visit import *
from disasm.Types import *
from utils.ail_utils import *
from utils.pp_print import *
from junkcodes import get_junk_codes
from addr_utils import addr_equals, select_block_by_addr

# Note: no less than 3
split_threshold = 10
obfs_proportion = 1.0


class bb_split_diversify(ailVisitor):

    def __init__(self, funcs, fb_tbl, cfg_tbl):
        ailVisitor.__init__(self)
        self.fb_tbl = fb_tbl

    def split_bb(self, b_name, b_instrs, split_pos):
        instr = b_instrs[split_pos]
        split_symbol = b_name + '_split'

        loc = get_loc(instr)
        loc_without_label = self._get_loc(instr)
        loc_without_label.loc_label = ''

        loc_with_label = self._get_loc(instr)
        loc_with_label.loc_label = split_symbol + ':'

        i1 = DoubleInstr((self._ops['jmp'], Label(split_symbol), loc_without_label, None))
        # i2 should be replaced with a list of garbage code
        # i2 = SingleInstr((self._ops['nop'], loc_with_label, None))

        self.insert_instrs(i1, loc)
        # self.insert_instrs(i2, loc)
        # we insert some codes after the jmp, which will never execute
        useless = []
        useless.append(DoubleInstr((self._ops['push'], self._regs[1], loc_without_label, None)))
        useless.append(TripleInstr((self._ops['mov'], self._regs[0], self._regs[1], loc_without_label, None)))
        useless.append(TripleInstr(('xor', self._regs[0], self._regs[0], loc_without_label, None)))
        useless.append(DoubleInstr((self._ops['jmp'], Label('printf'), loc_without_label, None)))
        for u in useless:
            self.insert_instrs(u, loc)

        junk = get_junk_codes(loc_with_label)
        if len(junk) == 0:
            junk.append(SingleInstr((self._ops['nop'], loc_with_label, None)))
            junk.append(SingleInstr((self._ops['nop'], loc_with_label, None)))
            junk.append(SingleInstr((self._ops['nop'], loc_with_label, None)))
            junk.append(SingleInstr((self._ops['nop'], loc_with_label, None)))
        junk[0][-2].loc_label = loc_with_label.loc_label
        for j in junk:
            self.insert_instrs(j, loc)

    def update_bb(self, b):
        instrs = self.bb_instrs(b)
        if len(instrs) > split_threshold:
            split_pos = random.randint(1, len(instrs) - 2)
            self.split_bb(b.bblock_name, instrs, split_pos)
            # self.update_process()
            return True
        else:
            return False

    def bb_div_split(self, target_addr=None):
        """
        对基本块执行分割变异
        
        如果指定 target_addr，只分割该地址的基本块
        如果未指定 target_addr，对所有基本块进行分割
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
                    print '[bb_split_diversify.py:bb_div_split] Found target_addr: %s (matched with 0x%X)' % (target_addr, b.bblock_begin_loc.loc_addr)
                else:
                    print '[bb_split_diversify.py:bb_div_split] Found target_addr: %s inside block (begin=0x%X)' % (target_addr, b.bblock_begin_loc.loc_addr)
                do_split = self.update_bb(b)
                if do_split:
                    self.update_process()
                    return
                print '[bb_split_diversify.py:bb_div_split] Warning: target_addr %s too small to split (< %d instrs), fallback to random' % (target_addr, split_threshold)
                break
            if not found:
                print '[bb_split_diversify.py:bb_div_split] Warning: target_addr %s not found, fallback to random' % target_addr
        
        # 未指定 target_addr，对所有基本块进行分割
        for f in self.fb_tbl.keys():
            bl = self.fb_tbl[f]
            for b in bl:
                do_split = self.update_bb(b)
        self.update_process()

    def visit(self, instrs, target_addr = None):
        print 'start basic block split diversification'
        self.instrs = copy.deepcopy(instrs)
        self.bb_div_split(target_addr)
        return self.instrs
