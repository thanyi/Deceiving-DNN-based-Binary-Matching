# -*- coding: utf-8 -*-
"""
【变异操作类型】条件分支取反与显式落空分支 (Branch Inversion + Explicit Fallthrough)
【框架序号】Action ID: 14
【功能说明】对一个条件跳转进行等价改写：将条件取反，并显式补出落空分支标签。
           原始形式:
             jcc target
           改写后:
             j!cc INV_FALL_n
             jmp target
             INV_FALL_n:
               nop
           该变换保持行为一致，同时改变局部 CFG 结构。
"""
from analysis.visit import *
from disasm.Types import *
from utils.ail_utils import *
from utils.pp_print import *
from addr_utils import addr_equals


class bb_branch_invert_diversify(ailVisitor):

    def __init__(self, funcs, fb_tbl, cfg_tbl):
        ailVisitor.__init__(self)
        self.funcs = funcs
        self._counter = 0

    def _sanitize_label_name(self, name):
        s = str(name)
        out = []
        for ch in s:
            if ch.isalnum() or ch == '_':
                out.append(ch)
            else:
                out.append('_')
        return ''.join(out)

    def _inverse_jcc(self, op_name):
        table = {
            'je': 'jne',
            'jz': 'jnz',
            'jne': 'je',
            'jnz': 'jz',
            'jg': 'jle',
            'jnle': 'jle',
            'jge': 'jl',
            'jnl': 'jl',
            'jl': 'jge',
            'jnge': 'jge',
            'jle': 'jg',
            'jng': 'jg',
            'ja': 'jbe',
            'jnbe': 'jbe',
            'jae': 'jb',
            'jnb': 'jb',
            'jb': 'jae',
            'jnae': 'jae',
            'jbe': 'ja',
            'jna': 'ja',
            'jo': 'jno',
            'jno': 'jo',
            'js': 'jns',
            'jns': 'js',
            'jp': 'jnp',
            'jnp': 'jp',
        }
        return table.get(op_name)

    def _collect_candidates(self, target_addr=None):
        candidates = []
        for f in self.funcs:
            try:
                fil = self.func_instrs(f)
            except Exception:
                continue
            for ins in fil:
                if not isinstance(ins, DoubleInstr):
                    continue
                loc = get_loc(ins)
                if loc is None:
                    continue
                if target_addr is not None and (not addr_equals(loc.loc_addr, target_addr)):
                    continue
                op = get_op(ins)
                if not Opcode_utils.is_cond_jmp(op):
                    continue
                des = get_cf_des(ins)
                if not isinstance(des, Label):
                    continue
                inv = self._inverse_jcc(p_op(op))
                if inv is None:
                    continue
                candidates.append((f, ins, inv, des))
        return candidates

    def _rewrite_one(self, finfo, instr, inv_op, des):
        self._counter += 1
        prefix = self._sanitize_label_name(finfo.func_name)
        fall_label = 'INV_%s_FALL_%d' % (prefix, self._counter)

        loc = get_loc(instr)
        loc_no = self._get_loc(instr)
        loc_no.loc_label = ''
        loc_fall = copy.deepcopy(loc_no)
        loc_fall.loc_label = fall_label + ': '

        i0 = DoubleInstr((inv_op, Label(fall_label), loc, None))
        i1 = DoubleInstr((self._ops['jmp'], des, loc_no, None))
        i2 = SingleInstr((self._ops['nop'], loc_fall, None))

        self.replace_instrs(i0, loc, instr)
        self.append_instrs(i1, loc)
        self.append_instrs(i2, loc)
        self.update_process()
        try:
            print '[bb_branch_invert_diversify.py] inverted 0x%X in %s: %s -> %s' % (
                loc.loc_addr, finfo.func_name, p_op(get_op(instr)), inv_op
            )
        except Exception:
            print '[bb_branch_invert_diversify.py] inverted one conditional jump'

    def visit(self, instrs, target_addr=None):
        print 'start branch inversion diversification'
        self.instrs = copy.deepcopy(instrs)

        if isinstance(target_addr, str) and target_addr.strip() == '':
            target_addr = None

        candidates = self._collect_candidates(target_addr)
        if len(candidates) == 0:
            if target_addr is not None:
                print '[bb_branch_invert_diversify.py] Warning: target_addr %s not found, skip' % target_addr
            else:
                print '[bb_branch_invert_diversify.py] Warning: no invertible conditional jump, skip'
            return self.instrs

        chosen = random.choice(candidates)
        self._rewrite_one(chosen[0], chosen[1], chosen[2], chosen[3])
        return self.instrs
