# -*- coding: utf-8 -*-
"""
【变异操作类型】基本块不透明谓词混淆（第二代）(Basic Block Opaque Predicate Gen2)
【框架序号】Action ID: 11
【功能说明】bb_opaque_diversify 的改进版本，使用更复杂的不透明谓词生成策略，
           提供更强的混淆效果和更稳定的控制流伪装。
"""
from analysis.visit import *
from disasm.Types import *
from utils.ail_utils import *
from utils.pp_print import *
from junkcodes import get_junk_codes
from addr_utils import addr_equals, select_block_by_addr
import random

obfs_proportion = 0.2


class bb_opaque_diversify_gen2(ailVisitor):

    def __init__(self, funcs, fb_tbl, cfg_tbl):
        ailVisitor.__init__(self)
        self.fb_tbl = fb_tbl
        self.routine_constant = random.randint(-1000, 1000)

    def _change_instrs_with_changelist(self, changelist):
        for c in changelist:
            op = c[0]
            if op == instr_update.INSERT:
                self.insert_instrs(c[1], c[2])
            elif op == instr_update.REPLACE:
                self.replace_instrs(c[1], c[2], c[3])
            else:
                assert False, 'unknown operation to do.'

    def get_opaque_header2(self, b, spt_pos=0):
        opaque_symbol = b.bblock_name + '_opaque_next'
        bil = self.bb_instrs(b)
        if not bil or spt_pos >= len(bil):
            return []
        i = bil[spt_pos]

        # print 'basic block opaque transformation: ' + b.bblock_name + ' ' + dec_hex(b.bblock_begin_loc.loc_addr) \
        #       + '->' + dec_hex(b.bblock_end_loc.loc_addr)
        # print 'instruction be added opaque block is: %s' % pp_print_instr(i)

        iloc = self._get_loc(i)
        tmp_iloc = copy.deepcopy(iloc)
        tmp_iloc.loc_label = ''
        tmp_iloc2 = copy.deepcopy(iloc)
        tmp_iloc2.loc_label = opaque_symbol + ': '

        '''
        .text:1000D5B2 loc_1000D5B2: ; CODE XREF: DllMain(x,x,x)
        .text:1000D5B2                 mov     eax, dword_10020CC8
        .text:1000D5B7                 mov     ecx, dword_10020CC4
        .text:1000D5BD                 mov     ebx, 8824E2F2h
        .text:1000D5C2                 lea     edx, [eax-1]
        .text:1000D5C5                 imul    edx, eax
        .text:1000D5C8                 not     edx
        .text:1000D5CA                 or      edx, 0FFFFFFFEh
        .text:1000D5CD                 cmp     edx, esi
        .text:1000D5CF                 setz    al
        .text:1000D5D2                 cmp     ecx, 0Ah
        .text:1000D5D5                 setl    ah
        .text:1000D5D8                 xor     ah, al
        .text:1000D5DA                 mov     eax, 8824E2F2h
        .text:1000D5DF                 jz      short loc_1000D593
        .text:1000D5E1                 jmp     short loc_1000D598
        '''


        # store general regs to stack (pusha/popa are invalid on x86_64)
        save_regs = []
        for idx, reg in enumerate(self._regs):
            loc = iloc if idx == 0 else tmp_iloc
            save_regs.append(DoubleInstr((self._ops['push'], reg, loc, None)))
        save_flag = SingleInstr((self._ops['pushf'], tmp_iloc, None))
        i2 = TripleInstr(('mov', RegClass('eax'), Types.Normal(0x41414141),  tmp_iloc, None))
        i3 = TripleInstr(('mov', RegClass('ecx'), Types.Normal(0x42424242),  tmp_iloc, None))
        i4 = TripleInstr(('mov', RegClass('ebx'), Types.Normal(0x43434343),  tmp_iloc, None))
        i5 = TripleInstr(('mov', RegClass('edx'), Types.Normal(0x41414141-1),  tmp_iloc, None))
        i6 = TripleInstr(('imul', RegClass('edx'), RegClass('eax'),  tmp_iloc, None))
        i7 = DoubleInstr(('not', RegClass('edx'), tmp_iloc, None))
        i8 = TripleInstr(('or', RegClass('edx'), Types.Normal(0x0FFFFFFFE),  tmp_iloc, None))
        i9 = TripleInstr(('cmp', RegClass('edx'), RegClass('esi'),  tmp_iloc, None))
        i10 = DoubleInstr(('setz', RegClass('al'), tmp_iloc, None))
        i11 = TripleInstr(('cmp', RegClass('ecx'), Types.Normal(0x0a), tmp_iloc, None))
        i12 = DoubleInstr(('setl', RegClass('ah'), tmp_iloc, None))
        i13 = TripleInstr(('xor', RegClass('ah'), RegClass('al'), tmp_iloc, None))
        recover_flag = SingleInstr((self._ops['popf'], tmp_iloc, None))
        restore_regs = []
        for reg in reversed(self._regs):
            restore_regs.append(DoubleInstr((self._ops['pop'], reg, tmp_iloc, None)))
        i0 = set_loc(i, tmp_iloc)

        res = list()
        for ins in save_regs:
            res.append((instr_update.INSERT, ins, iloc))
        res.append((instr_update.INSERT, save_flag, iloc))
        res.append((instr_update.INSERT, i2, iloc))
        res.append((instr_update.INSERT, i3, iloc))
        res.append((instr_update.INSERT, i4, iloc))
        res.append((instr_update.INSERT, i5, iloc))
        res.append((instr_update.INSERT, i6, iloc))
        res.append((instr_update.INSERT, i7, iloc))
        res.append((instr_update.INSERT, i8, iloc))
        res.append((instr_update.INSERT, i9, iloc))
        res.append((instr_update.INSERT, i10, iloc))
        res.append((instr_update.INSERT, i11, iloc))
        res.append((instr_update.INSERT, i12, iloc))
        res.append((instr_update.INSERT, i13, iloc))
        res.append((instr_update.INSERT, recover_flag, iloc))
        for ins in restore_regs:
            res.append((instr_update.INSERT, ins, iloc))
        res.append((instr_update.REPLACE, i0, iloc, i))
        return res

    def get_opaque_header1(self, b, spt_pos=0):
        """
        get the list of instructions which work as the opaque block
        the instructions works as 'if (y < 10 || x*(x-1) % 2 == 0)'
        It is clear the the statement is always true (if something wrong and it run into false branch, halt the program)
        :param b: the opaque block will be inserted before the block
        :param spt_pos: the opaque block will be inserted before the instruction(which is b[spt_pos])
        :return: the instructions list of the opaque_block
        """
        opaque_symbol = b.bblock_name + '_opaque_next'
        bil = self.bb_instrs(b)
        if not bil or spt_pos >= len(bil):
            return []
        i = bil[spt_pos]
        iloc_with_block_label = self._get_loc(i)
        iloc_without_label = copy.deepcopy(iloc_with_block_label)
        iloc_without_label.loc_label = ''
        iloc_with_true_branch_label = copy.deepcopy(iloc_with_block_label)
        iloc_with_true_branch_label.loc_label = opaque_symbol + ': '
        # false branch will call halt_func directly

        res = []
        res.append((instr_update.INSERT, DoubleInstr(('push', self._regs[0], iloc_with_block_label, None)),
                    iloc_with_block_label))  # use this reg as x
        res.append((instr_update.INSERT, SingleInstr((self._ops['pushf'], iloc_without_label, None)),
                    iloc_with_block_label))  # save flag
        res.append((instr_update.INSERT, DoubleInstr(('push', self._regs[1], iloc_without_label, None)),
                    iloc_with_block_label))  # use this reg as y
        res.append((instr_update.INSERT, DoubleInstr(('push', self._regs[2], iloc_without_label, None)),
                    iloc_with_block_label))  # use this reg as dummy1
        res.append((instr_update.INSERT, DoubleInstr(('push', self._regs[3], iloc_without_label, None)),
                    iloc_with_block_label))  # use this reg as dummy2
        # y < 10
        res.append((
            instr_update.INSERT, TripleInstr(('cmp', self._regs[1], Types.Normal(10), iloc_without_label, None)),
            iloc_with_block_label))
        res.append((instr_update.INSERT, TripleInstr(('cmovz', self._regs[2], self._regs[3], iloc_without_label, None)),
                    iloc_with_block_label))
        res.append((instr_update.INSERT, DoubleInstr(('jl', Types.Label(opaque_symbol), iloc_without_label, None)),
                    iloc_with_block_label))


        res.append((instr_update.INSERT, TripleInstr(('xor', self._regs[2], self._regs[3], iloc_without_label, None)),
                    iloc_with_block_label))

        res.append((instr_update.INSERT, SingleInstr(('nop', iloc_without_label, None)),
                    iloc_with_block_label))  
        res.append((instr_update.INSERT, SingleInstr(('nop', iloc_without_label, None)),
                    iloc_with_block_label))  
        res.append((instr_update.INSERT, SingleInstr(('nop', iloc_without_label, None)),
                    iloc_with_block_label))  
        res.append((instr_update.INSERT, SingleInstr(('nop', iloc_without_label, None)),
                    iloc_with_block_label))  

        res.append((instr_update.INSERT, TripleInstr(('xor', self._regs[3], self._regs[3], iloc_without_label, None)),
                    iloc_with_block_label))
        res.append((instr_update.INSERT, TripleInstr(('xor', self._regs[3], self._regs[3], iloc_without_label, None)),
                    iloc_with_block_label))

        res.append((instr_update.INSERT, TripleInstr(('mov', self._regs[3], Types.Normal(0x41414141), iloc_without_label, None)),
                    iloc_with_block_label))
        res.append((
            instr_update.INSERT, TripleInstr(('cmp', self._regs[0], Types.Normal(20), iloc_without_label, None)),
            iloc_with_block_label))
        res.append((instr_update.INSERT, TripleInstr(('mov', self._regs[2], Types.Normal(0x32323232), iloc_without_label, None)),
                    iloc_with_block_label))
        res.append((instr_update.INSERT, DoubleInstr(('jl', Types.Label(opaque_symbol), iloc_without_label, None)),
                    iloc_with_block_label))

        res.append((instr_update.INSERT, DoubleInstr(('not', self._regs[2], iloc_without_label, None)),
                    iloc_with_block_label))
        res.append((
            instr_update.INSERT, TripleInstr(('cmp', self._regs[1], Types.Normal(30), iloc_without_label, None)),
            iloc_with_block_label))
        res.append((instr_update.INSERT, TripleInstr(('cmovz', self._regs[3], self._regs[2], iloc_without_label, None)),
                    iloc_with_block_label))
        res.append((instr_update.INSERT, DoubleInstr(('jz', Types.Label(opaque_symbol), iloc_without_label, None)),
                    iloc_with_block_label))

        res.append((
            instr_update.INSERT, TripleInstr(('cmp', self._regs[1], Types.Normal(10), iloc_without_label, None)),
            iloc_with_block_label))
        res.append((instr_update.INSERT, DoubleInstr(('jl', Types.Label(opaque_symbol), iloc_without_label, None)),
                    iloc_with_block_label))

        # x*(x-1) % 2 == 0, use y to store value of (x-1)
        res.append((instr_update.INSERT, TripleInstr(('mov', self._regs[0], self._regs[1], iloc_without_label, None)),
                    iloc_with_block_label))
        # junk code
        res.extend([(instr_update.INSERT, j, iloc_without_label) for j in get_junk_codes(iloc_without_label)])
        res.append((instr_update.INSERT, TripleInstr(('sub', self._regs[1], Types.Normal(1), iloc_without_label, None)),
                    iloc_with_block_label))
        res.append((instr_update.INSERT, TripleInstr(('imul', self._regs[0], self._regs[1], iloc_without_label, None)),
                    iloc_with_block_label))
        res.append((instr_update.INSERT, TripleInstr(('and', self._regs[0], Types.Normal(1), iloc_without_label, None)),
                    iloc_with_block_label))

        res.append((instr_update.INSERT, TripleInstr(('test', self._regs[0], self._regs[0], iloc_without_label, None)),
                    iloc_with_block_label))
        res.append((instr_update.INSERT, DoubleInstr(('je', Types.Label(opaque_symbol), iloc_without_label, None)),
                    iloc_with_block_label))
        # false branch
        res.append((instr_update.INSERT, DoubleInstr(('call', Types.Label('halt_func'), iloc_without_label, None)),
                    iloc_with_block_label))
        # true branch
        res.append((instr_update.INSERT, TripleInstr(('mov', self._regs[2], Types.Normal(0x41414141), iloc_with_true_branch_label, None)),
                    iloc_with_block_label))
        res.append((instr_update.INSERT, TripleInstr(('mov', self._regs[3], Types.Normal(0x42424242), iloc_with_true_branch_label, None)),
                    iloc_with_block_label))
        res.append((instr_update.INSERT, TripleInstr(('cmovz', self._regs[2], self._regs[3], iloc_with_true_branch_label, None)),
                    iloc_with_block_label))
        res.append((instr_update.INSERT, DoubleInstr(('pop', self._regs[3], iloc_with_true_branch_label, None)),
                    iloc_with_block_label))
        res.append((instr_update.INSERT, DoubleInstr(('pop', self._regs[2], iloc_with_true_branch_label, None)),
                    iloc_with_block_label))
        res.append((instr_update.INSERT, DoubleInstr(('pop', self._regs[1], iloc_with_true_branch_label, None)),
                    iloc_with_block_label))
        res.append(
            (instr_update.INSERT, SingleInstr((self._ops['popf'], iloc_without_label, None)), iloc_with_block_label))
        res.append(
            (instr_update.INSERT, DoubleInstr(('pop', self._regs[0], iloc_without_label, None)), iloc_with_block_label))
        # remove the label of original block
        new_line = set_loc(i, iloc_without_label)
        res.append((instr_update.REPLACE, new_line, iloc_with_block_label, i))
        return res

    def bb_div_opaque(self, target_addr=None):
        """
        对基本块执行不透明谓词混淆 (Gen2版本)
        
        如果指定 target_addr，只对该地址的基本块进行变异
        如果未指定 target_addr，对每个函数随机选择一个基本块进行变异
        """
        # print 'do basic block transformation on: %d functions' % len(self.fb_tbl)
        # add new opaque header method here
        modes = [self.get_opaque_header1, self.get_opaque_header2]
        # header1 may cause errors in disassembling when compiler is gcc-4.8 (x64? I am not sure)
        modes = [self.get_opaque_header1]
        
        # 如果指定了 target_addr，只处理该地址的基本块
        if target_addr is not None:
            found = False
            for f in self.fb_tbl.keys():
                bl = self.fb_tbl[f]
                b, exact = select_block_by_addr(bl, target_addr)
                if b is not None:
                    found = True
                    if exact:
                        print '[bb_opaque_diversify_gen2.py:bb_div_opaque] Found target_addr: %s (matched with 0x%X)' % (target_addr, b.bblock_begin_loc.loc_addr)
                    else:
                        print '[bb_opaque_diversify_gen2.py:bb_div_opaque] Found target_addr: %s inside block (begin=0x%X)' % (target_addr, b.bblock_begin_loc.loc_addr)
                    n_mode = random.randint(0, len(modes) - 1)
                    changelist = modes[n_mode](b, 0)
                    if not changelist:
                        print '[bb_opaque_diversify_gen2.py:bb_div_opaque] Warning: empty block at target_addr, skip'
                        return
                    self._change_instrs_with_changelist(changelist)
                    self.update_process()
                    return
            if not found:
                print '[bb_opaque_diversify_gen2.py:bb_div_opaque] Warning: target_addr %s not found, fallback to random' % target_addr
        
        # 未指定 target_addr，随机选择基本块进行变异
        for f in self.fb_tbl.keys():
            bl = self.fb_tbl[f]
            if len(bl) > 0:
                candidates = [b for b in bl if self.bb_instrs(b)]
                if not candidates:
                    continue
                n = random.randint(0, len(candidates) - 1)
                n_mode = random.randint(0, len(modes) - 1)
                changelist = modes[n_mode](candidates[n], 0)
                if changelist:
                    self._change_instrs_with_changelist(changelist)
        self.update_process()

    def get_opaque_routines(self, iloc):
        iloc1 = copy.deepcopy(iloc)
        iloc1.loc_label = 'opaque_func: '

        tmp_loc = copy.deepcopy(iloc)
        tmp_loc.loc_label = ''

        iloc6 = copy.deepcopy(iloc)
        iloc6.loc_label = 'halt_func: '

        i1 = DoubleInstr(('push', self._stack_regs['bp'], iloc1, None))
        i2 = TripleInstr(('mov', self._stack_regs['bp'], self._stack_regs['sp'], tmp_loc, None))
        junk = get_junk_codes(tmp_loc, 0)
        # set the value of %eax to be 0
        i3 = TripleInstr(('mov', self._regs[0], Types.Normal(self.routine_constant), tmp_loc, None))
        i4 = DoubleInstr(('pop', self._stack_regs['bp'], tmp_loc, None))
        i5 = SingleInstr(('ret', tmp_loc, None))
        i6 = SingleInstr(('hlt', iloc6, None))
        res = [i1, i2]
        res.extend(junk)
        res.extend([i3, i4, i5, i6])
        return res

    def attach_opaque_routines(self):
        """
        now we just append the opaque routines code at the end of all instructions
        It is supposed to
            1. find the begin of a basic block (the former block must end, use 'ret' to select)
            2. then insert the opaque_routines at that position
        :return:
        """
        bb_starts = []
        for i in range(len(self.instrs)):
            if get_op(self.instrs[i]) in ControlOp and 'ret' in p_op(self.instrs[i]):
                # Note: do not use 'jmp', because it may result in collision with bb_branchfunc_diversify
                bb_starts.append(i)
        if not self.instrs:
            return
        if bb_starts:
            selected_i = self.instrs[random.choice(bb_starts)]
        else:
            selected_i = self.instrs[-1]
        # to avoid possible error in disassembling
        # selected_i = self.instrs[-1]
        # the location of opaque routines should be carefully selected
        selected_loc = get_loc(selected_i)
        opaque_routines = self.get_opaque_routines(selected_loc)
        # should be changed to call insert_instr, then call update_process
        for ins in opaque_routines:
            self.append_instrs(ins, selected_loc)
        self.update_process()

    def bb_opaque_process(self, target_addr=None):
        self.bb_div_opaque(target_addr)
        # remove header1 mode when compiler is gcc-4.8
        self.attach_opaque_routines()

    def visit(self, instrs, target_addr = None):
        print 'start basic block opaque diversification'
        self.instrs = self._clone_instrs_for_edit(instrs)
        self.bb_opaque_process(target_addr)
        return self.instrs
