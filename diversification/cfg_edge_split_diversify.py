# -*- coding: utf-8 -*-
"""
【变异操作类型】关键边风格跳板分裂 (Critical-Edge-Style Trampoline Split)
【框架序号】Action ID: 15
【功能说明】优先选择目标入边较多（critical-like）的条件跳转边进行分裂；
           若不存在此类边，则回退到任意条件跳转边。
           原始形式:
             jcc target
           改写后:
             jcc EDGE_SPLIT_n
             ...
             EDGE_SPLIT_n:
               jmp target
           该变换在不改变语义的前提下，增加边上的中转节点并改变 CFG 拓扑。
"""
from analysis.visit import *
from disasm.Types import *
from utils.ail_utils import *
from utils.pp_print import *
from addr_utils import addr_equals


class cfg_edge_split_diversify(ailVisitor):

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

    def _incoming_label_count(self):
        incoming = {}
        for f in self.funcs:
            try:
                fil = self.func_instrs(f)
            except Exception:
                continue
            for ins in fil:
                if not isinstance(ins, DoubleInstr):
                    continue
                op = get_op(ins)
                if not (Opcode_utils.is_jmp(op) or Opcode_utils.is_cond_jmp(op)):
                    continue
                des = get_cf_des(ins)
                if isinstance(des, Label):
                    k = str(des)
                    incoming[k] = incoming.get(k, 0) + 1
        return incoming

    def _collect_conditional_candidates(self, incoming, target_addr=None):
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
                if not Opcode_utils.is_cond_jmp(op):
                    continue
                des = get_cf_des(ins)
                if not isinstance(des, Label):
                    continue
                in_deg = incoming.get(str(des), 0)
                is_critical_like = in_deg > 1
                cands.append((f, ins, des, is_critical_like, in_deg))
        return cands

    def _rewrite_one(self, finfo, instr, des):
        self._counter += 1
        prefix = self._sanitize_label_name(finfo.func_name)
        tramp_label = 'EDGE_SPLIT_%s_%d' % (prefix, self._counter)

        loc = get_loc(instr)
        op = get_op(instr)
        loc_no = self._get_loc(instr)
        loc_no.loc_label = ''
        loc_tramp = copy.deepcopy(loc_no)
        loc_tramp.loc_label = tramp_label + ': '

        i0 = DoubleInstr((op, Label(tramp_label), loc, None))
        i1 = DoubleInstr((self._ops['jmp'], des, loc_tramp, None))

        self.replace_instrs(i0, loc, instr)
        self.append_instrs(i1, loc)
        self.update_process()
        try:
            print '[cfg_edge_split_diversify.py] split edge at 0x%X in %s' % (
                loc.loc_addr, finfo.func_name
            )
        except Exception:
            print '[cfg_edge_split_diversify.py] split one conditional edge'

    def visit(self, instrs, target_addr=None):
        print 'start cfg edge split diversification'
        self.instrs = self._clone_instrs_for_edit(instrs)

        if isinstance(target_addr, str) and target_addr.strip() == '':
            target_addr = None

        incoming = self._incoming_label_count()
        cands = self._collect_conditional_candidates(incoming, target_addr)
        if len(cands) == 0:
            if target_addr is not None:
                print '[cfg_edge_split_diversify.py] Warning: target_addr %s has no splittable conditional jump, skip' % target_addr
            else:
                print '[cfg_edge_split_diversify.py] Warning: no splittable conditional jump, skip'
            return self.instrs

        critical_like = [c for c in cands if c[3]]
        pool = critical_like if len(critical_like) > 0 else cands
        chosen = random.choice(pool)
        self._rewrite_one(chosen[0], chosen[1], chosen[2])
        if chosen[3]:
            print '[cfg_edge_split_diversify.py] chosen edge indegree=%d (critical-like)' % chosen[4]
        else:
            print '[cfg_edge_split_diversify.py] fallback split (no critical-like edge found)'
        return self.instrs
