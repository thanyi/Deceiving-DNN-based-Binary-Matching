# -*- coding: utf-8 -*-
"""
【变异操作类型】等价指令替换 (Equivalent Instruction Transformation)
【框架序号】Action ID: 8
【功能说明】将指令替换为语义等价的指令序列（如 XOR 替换、寄存器交换等），
           改变指令模式但保持功能不变，用于对抗基于指令特征的模式匹配。
"""
from analysis.visit import *
from disasm.Types import *
from utils.ail_utils import *
from utils.pp_print import *
from addr_utils import addr_equals, select_block_by_addr

obfs_proportion = 1.0


class instr_replace_diversify(ailVisitor):

    def __init__(self, funcs, fb_tbl, cfg_tbl):
        ailVisitor.__init__(self)
        self.fb_tbl = fb_tbl

    @staticmethod
    def bits_of_operation(op):
        op_s = p_op(op)
        if op_s.endswith('b'):
            return 8
        elif op_s.endswith('w'):
            return 16
        elif op_s.endswith('l'):
            return 32
        elif op_s.endswith('q'):
            return 64
        else:
            return None

    @staticmethod
    def is_xor_reg(op, e1, e2):
        op_s = p_op(op)
        if (ELF_utils.elf_64() and op_s == 'xorq') or (ELF_utils.elf_32() and (op_s == 'xor' or op_s == 'xorl')):
            return Exp_utils.is_reg(e1) and Exp_utils.is_reg(e2) and p_exp(e1) == p_exp(e2)
        return False

    @staticmethod
    def is_shl_reg(op, e1, e2):
        op_s = p_op(op)
        return (op_s == 'shl' or op_s == 'shlb' or op_s == 'shlw' or op_s == 'shll' or op_s == 'shlq') \
               and Exp_utils.is_reg(e1) and Exp_utils.is_const(e2)

    @staticmethod
    def is_increment_reg(op, e1, e2):
        op_s = p_op(op)
        return op_s in {'add', 'addb', 'addw', 'addl', 'addq'} and Exp_utils.is_reg(e1) and Exp_utils.is_const(e2) \
               and e2 == 1

    @staticmethod
    def is_mov_0(op, e1, e2):
        op_s = p_op(op)
        return op_s in {'mov', 'movb', 'movw', 'movl', 'movq'} and Exp_utils.is_reg(e1) and not isinstance(e1, UnOP) \
               and Exp_utils.is_const(e2) and e2 == 0

    def update_ret(self, i):
        if isinstance(i, SingleInstr) and Opcode_utils.is_ret(get_op(i)):
            loc1 = get_loc(i)
            loc2 = self._get_loc(i)
            loc2.loc_label = ''
            i1 = DoubleInstr((self._ops['pop'], self._regs[2], loc1, None))
            i2 = DoubleInstr((self._ops['jmp'], StarDes(self._regs[2]), loc2, None))
            self.replace_instrs(i1, loc1, i)
            self.append_instrs(i2, loc1)
            return True
        return False

    def update_call(self, i):
        def helper(i, e, l):
            jmp_label = 'S_' + dec_hex(l.loc_addr) + '_next'
            l1 = copy.deepcopy(l)
            l2 = copy.deepcopy(l)
            l3 = copy.deepcopy(l)
            l2.loc_label = ''
            l3.loc_label = jmp_label + ': '
            i1 = DoubleInstr((self._ops['push'], Label('$' + jmp_label), l1, None))
            i2 = DoubleInstr((self._ops['jmp'], e, l2, None))
            i3 = SingleInstr((self._ops['nop'], l3, None))

            self.replace_instrs(i1, l, i)
            self.append_instrs(i2, l)
            self.append_instrs(i3, l)

        if isinstance(i, DoubleInstr) and Opcode_utils.is_call(get_op(i)):
            e = i[1]
            l = get_loc(i)
            if isinstance(e, StarDes):
                # StarDes is subClass of Symbol
                return False
            elif isinstance(e, Symbol):
                if isinstance(e, CallDes) and e.is_lib:
                    return False
                helper(i, e, l)
                return True
            elif isinstance(e, Label):
                helper(i, e, l)
                return True
            return False

    def update_xor(self, i):
        if isinstance(i, TripleInstr) and self.is_xor_reg(get_op(i), i[1], i[2]):
            #if random.random() < obfs_proportion:
            if True:
                loc = get_loc(i)
                new_i = TripleInstr((self._ops['mov'], i[1], Normal(0), loc, None))
                # change the flag
                il = [DoubleInstr((self._ops['push'], self._regs[0], loc, None)),
                      SingleInstr((self._ops['pushf'], loc, None)),
                      DoubleInstr((self._ops['pop'], self._regs[0], loc, None)),
                      TripleInstr(('and', self._regs[0], Normal(0xfffff77e), loc, None)),
                      TripleInstr(('or', self._regs[0], Normal(0x00000046), loc, None)),
                      DoubleInstr((self._ops['push'], self._regs[0], loc, None)),
                      SingleInstr((self._ops['popf'], loc, None)),
                      DoubleInstr((self._ops['pop'], self._regs[0], loc, None))]
                self.replace_instrs(new_i, loc, i)
                for tmp_i in il:
                    self.append_instrs(tmp_i, loc)
            return True
        return False

    def update_shl(self, i):
        if isinstance(i, TripleInstr) and self.is_shl_reg(get_op(i), i[1], i[2]):
            op, e1, e2, l = get_op(i), i[1], i[2], get_loc(i)
            if isinstance(e2, Normal):
                v = 2 ** e2
                #if random.random() < obfs_proportion:
                if True: 
                    if v == 2 and Exp_utils.is_reg(e1):
                        new_i = TripleInstr((self._ops['add'], e1, e1, l, None))
                    else:
                        new_i = FourInstr((self._ops['imul'], e1, e1, Normal(v), l, None))
                    self.replace_instrs(new_i, l, i)
                return True
        return False

    def update_add(self, i):
        if isinstance(i, TripleInstr) and self.is_increment_reg(get_op(i), i[1], i[2]):
            op, e1, e2, l = get_op(i), i[1], i[2], get_loc(i)
            if isinstance(e2, Normal):
                #if random.random() < obfs_proportion:
                if True: 
                    new_i1 = DoubleInstr(('not', e1, l, None))
                    new_i2 = DoubleInstr(('neg', e1, l, None))
                    self.replace_instrs(new_i1, l, i)
                    self.append_instrs(new_i2, l)
                return True
        return False

    def update_mov(self, i):
        if isinstance(i, TripleInstr) and self.is_mov_0(get_op(i), i[1], i[2]):
            op, e1, e2, l = get_op(i), i[1], i[2], get_loc(i)
            if isinstance(e2, Normal):
                #if random.random() < obfs_proportion:
                if True: 
                    push_flag = SingleInstr((self._ops['pushf'], l, None))
                    new_i = TripleInstr(('xor', e1, e1, l, None))
                    pop_flag = SingleInstr((self._ops['popf'], l, None))
                    self.replace_instrs(push_flag, l, i)
                    self.append_instrs(new_i, l)
                    self.append_instrs(pop_flag, l)
                return True
        return False

    def instr_replace(self, target_addr=None):
        """
        执行等价指令替换
        
        如果指定 target_addr，优先在包含该地址的基本块内替换
        如果未指定或未找到目标基本块，对所有指令进行等价替换
        """
        # we currently ignore the call and return replacement, they can hurt disassembling process
        # update `add 1, %eax` can meet unknown errors
        fs = [
            # self.update_call,
            # self.update_ret,
            self.update_mov,
            # self.update_add,
            self.update_shl,
            #self.update_xor
        ]
        
        # 如果指定了 target_addr，优先在对应基本块内替换
        if target_addr is not None:
            target_block = None
            for f in self.fb_tbl.keys():
                bl = self.fb_tbl[f]
                b, exact = select_block_by_addr(bl, target_addr)
                if b is None:
                    continue
                target_block = b
                if exact:
                    print '[instr_replace_diversify.py:instr_replace] Found target_addr: %s (matched with 0x%X)' % (target_addr, b.bblock_begin_loc.loc_addr)
                else:
                    print '[instr_replace_diversify.py:instr_replace] Found target_addr: %s inside block (begin=0x%X)' % (target_addr, b.bblock_begin_loc.loc_addr)
                break

            if target_block is not None:
                bil = self.bb_instrs(target_block)
                if len(bil) == 0:
                    print '[instr_replace_diversify.py:instr_replace] Warning: target block empty, fallback to random'
                else:
                    # 优先尝试匹配 target_addr 的指令，否则在该基本块内选择可替换指令
                    exact_instrs = []
                    for i in bil:
                        loc = get_loc(i)
                        if loc and addr_equals(loc.loc_addr, target_addr):
                            exact_instrs.append(i)
                    candidates = exact_instrs if len(exact_instrs) > 0 else bil
                    found = False
                    for i in candidates:
                        for f in fs:
                            res = f(i)
                            if res:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        self.update_process()
                        return
                    print '[instr_replace_diversify.py:instr_replace] Warning: target_addr %s found but no replaceable instr in block, fallback to random' % target_addr
            else:
                print '[instr_replace_diversify.py:instr_replace] Warning: target_addr %s not found, fallback to random' % target_addr
        
        # 未指定 target_addr，对所有指令进行等价替换
        for i in self.instrs:
            for f in fs:
                res = f(i)
                if res:
                    break
        self.update_process()

    def visit(self, instrs, target_addr = None):
        print 'start instruction replacement ...'
        self.instrs = copy.deepcopy(instrs)
        self.instr_replace(target_addr)
        return self.instrs
