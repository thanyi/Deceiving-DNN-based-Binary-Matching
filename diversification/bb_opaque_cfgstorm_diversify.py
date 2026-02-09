# -*- coding: utf-8 -*-
"""
【变异操作类型】不透明菱形 CFG 强扰动 (Opaque Diamond CFG Storm)
【框架序号】Action ID: 16
【功能说明】在基本块入口插入一个“恒等谓词驱动的菱形分支”：
           1) 保存寄存器与标志位（push/pushf）
           2) 构造恒真比较，形成条件分支与显式 true/false 两条边
           3) 在两条边中都恢复现场（popf/pop）并汇合到 join
           4) 在 join 处继续执行原始指令
           该变换与 Action 11 类似，能显著扰动函数内 CFG，同时保持语义不变。
"""
from analysis.visit import *
from disasm.Types import *
from utils.ail_utils import *
from utils.pp_print import *
from addr_utils import select_block_by_addr
from junkcodes import get_junk_codes


class bb_opaque_cfgstorm_diversify(ailVisitor):

    def __init__(self, funcs, fb_tbl, cfg_tbl):
        ailVisitor.__init__(self)
        self.fb_tbl = fb_tbl
        self._counter = 0

    def _build_labels(self, bname):
        self._counter += 1
        base = '%s_cfgstorm_%d' % (bname, self._counter)
        return base + '_true', base + '_false', base + '_join'

    def _rewrite_block(self, b, spt_pos=0):
        bil = self.bb_instrs(b)
        if not bil or spt_pos >= len(bil):
            return False

        i = bil[spt_pos]
        loc_entry = self._get_loc(i)
        loc_no = copy.deepcopy(loc_entry)
        loc_no.loc_label = ''

        true_label, false_label, join_label = self._build_labels(b.bblock_name)
        loc_true = copy.deepcopy(loc_no)
        loc_true.loc_label = true_label + ': '
        loc_false = copy.deepcopy(loc_no)
        loc_false.loc_label = false_label + ': '
        loc_join = copy.deepcopy(loc_no)
        loc_join.loc_label = join_label + ': '

        reg_a = self._regs[0]
        reg_b = self._regs[1]

        # 为了不破坏入口语义：让原入口 label 落在第一条 push 指令上。
        seq = [
            DoubleInstr((self._ops['push'], reg_a, loc_entry, None)),
            DoubleInstr((self._ops['push'], reg_b, loc_no, None)),
            SingleInstr((self._ops['pushf'], loc_no, None)),
            TripleInstr((self._ops['mov'], reg_a, Types.Normal(0x13579BDF), loc_no, None)),
            TripleInstr((self._ops['mov'], reg_b, Types.Normal(0x13579BDF), loc_no, None)),
            TripleInstr(('cmp', reg_a, reg_b, loc_no, None)),
            DoubleInstr(('jne', Types.Label(false_label), loc_no, None)),
            DoubleInstr((self._ops['jmp'], Types.Label(true_label), loc_no, None)),
            # false 分支（理论上不会走到，但要保证语义自洽）
            SingleInstr((self._ops['nop'], loc_false, None)),
        ]

        # false 分支加少量垃圾，增强扰动
        for j in get_junk_codes(copy.deepcopy(loc_no), 0):
            seq.append(j)

        seq.extend([
            SingleInstr((self._ops['popf'], loc_no, None)),
            DoubleInstr((self._ops['pop'], reg_b, loc_no, None)),
            DoubleInstr((self._ops['pop'], reg_a, loc_no, None)),
            DoubleInstr((self._ops['jmp'], Types.Label(join_label), loc_no, None)),
            # true 分支（实际执行路径）
            SingleInstr((self._ops['nop'], loc_true, None)),
            SingleInstr((self._ops['popf'], loc_no, None)),
            DoubleInstr((self._ops['pop'], reg_b, loc_no, None)),
            DoubleInstr((self._ops['pop'], reg_a, loc_no, None)),
            DoubleInstr((self._ops['jmp'], Types.Label(join_label), loc_no, None)),
            # join + 原始首指令
            SingleInstr((self._ops['nop'], loc_join, None)),
            set_loc(i, loc_no),
        ])

        self.replace_instrs(seq[0], loc_entry, i)
        for ni in seq[1:]:
            self.append_instrs(ni, loc_entry)
        self.update_process()
        return True

    def bb_div_cfgstorm(self, target_addr=None):
        # target 模式：只改目标块
        if target_addr is not None:
            for f in self.fb_tbl.keys():
                bl = self.fb_tbl[f]
                b, exact = select_block_by_addr(bl, target_addr)
                if b is None:
                    continue
                if exact:
                    print '[bb_opaque_cfgstorm_diversify.py] Found target_addr: %s (matched with 0x%X)' % (
                        target_addr, b.bblock_begin_loc.loc_addr
                    )
                else:
                    print '[bb_opaque_cfgstorm_diversify.py] Found target_addr: %s inside block (begin=0x%X)' % (
                        target_addr, b.bblock_begin_loc.loc_addr
                    )
                if self._rewrite_block(b, 0):
                    return True
                print '[bb_opaque_cfgstorm_diversify.py] Warning: target block empty, skip'
                return False
            print '[bb_opaque_cfgstorm_diversify.py] Warning: target_addr %s not found, skip' % target_addr
            return False

        # 随机模式：每个函数随机挑一个非空块做一次
        changed = False
        for f in self.fb_tbl.keys():
            bl = self.fb_tbl[f]
            candidates = []
            for b in bl:
                try:
                    if len(self.bb_instrs(b)) > 0:
                        candidates.append(b)
                except Exception:
                    continue
            if len(candidates) == 0:
                continue
            b = random.choice(candidates)
            if self._rewrite_block(b, 0):
                changed = True
        if not changed:
            print '[bb_opaque_cfgstorm_diversify.py] Warning: no candidate block, skip'
        return changed

    def visit(self, instrs, target_addr=None):
        print 'start opaque diamond cfg-storm diversification'
        self.instrs = copy.deepcopy(instrs)

        if isinstance(target_addr, str) and target_addr.strip() == '':
            target_addr = None

        self.bb_div_cfgstorm(target_addr)
        return self.instrs
