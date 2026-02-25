# -*- coding: utf-8 -*-
"""
【变异操作类型】无条件边跳板分裂 (Unconditional-Edge Trampoline Split)
【框架序号】Action ID: 13
【功能说明】在函数内选择一条无条件跳转边，并插入一个中转块（trampoline）。
           原始形式:
             jmp target
           改写后:
             jmp UTRAMP_n
             ...
             UTRAMP_n:
               jmp target
           该变换会新增 CFG 节点，但保持程序语义不变。
"""
from analysis.visit import *
from disasm.Types import *
from utils.ail_utils import *
from utils.pp_print import *
from addr_utils import addr_equals


class cfg_uncond_trampoline_diversify(ailVisitor):

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

    def _collect_candidates(self, target_addr=None):
        cands = []
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
                if not Opcode_utils.is_jmp(op):
                    continue
                des = get_cf_des(ins)
                if not isinstance(des, Label):
                    continue
                cands.append((f, ins, des))
        return cands

    def _rewrite_one(self, finfo, instr, des):
        self._counter += 1
        prefix = self._sanitize_label_name(finfo.func_name)
        tramp_label = 'UTRAMP_%s_%d' % (prefix, self._counter)

        loc = get_loc(instr)
        loc_no = self._get_loc(instr)
        loc_no.loc_label = ''
        loc_tramp = copy.deepcopy(loc_no)
        loc_tramp.loc_label = tramp_label + ': '

        i0 = DoubleInstr((self._ops['jmp'], Label(tramp_label), loc, None))
        i1 = DoubleInstr((self._ops['jmp'], des, loc_tramp, None))

        self.replace_instrs(i0, loc, instr)
        self.append_instrs(i1, loc)
        self.update_process()
        try:
            print '[cfg_uncond_trampoline_diversify.py] split uncond edge at 0x%X in %s' % (
                loc.loc_addr, finfo.func_name
            )
        except Exception:
            print '[cfg_uncond_trampoline_diversify.py] split one unconditional edge'

    def visit(self, instrs, target_addr=None):
        print 'start unconditional-edge trampoline diversification'
        self.instrs = self._clone_instrs_for_edit(instrs)

        if isinstance(target_addr, str) and target_addr.strip() == '':
            target_addr = None

        cands = self._collect_candidates(target_addr)
        if len(cands) == 0:
            if target_addr is not None:
                print '[cfg_uncond_trampoline_diversify.py] Warning: target_addr %s has no splittable unconditional jump, skip' % target_addr
            else:
                print '[cfg_uncond_trampoline_diversify.py] Warning: no splittable unconditional jump, skip'
            return self.instrs

        chosen = random.choice(cands)
        self._rewrite_one(chosen[0], chosen[1], chosen[2])
        return self.instrs
