# -*- coding: utf-8 -*-
"""
【变异操作类型】函数内状态机扁平化 (Intra-function State-Machine Flattening)
【框架序号】Action ID: 12
【功能说明】将函数内的跳转边改写为“写入下一状态 + 跳转分发器”。
          分发器读取状态后再跳转到对应基本块标签，达到大幅改写函数内部 CFG 的目的。
"""
from analysis.visit import *
from disasm.Types import *
from utils.ail_utils import *
from utils.pp_print import *
from addr_utils import select_block_by_addr


class func_state_flatten_diversify(ailVisitor):

    def __init__(self, funcs, fb_tbl, cfg_tbl):
        ailVisitor.__init__(self)
        self.funcs = funcs
        self.fb_tbl = fb_tbl
        self.cfg_tbl = cfg_tbl
        self._fall_counter = 0

    def _sanitize_label_name(self, name):
        s = str(name)
        res = []
        for ch in s:
            if ch.isalnum() or ch == '_':
                res.append(ch)
            else:
                res.append('_')
        return ''.join(res)

    def _find_func_obj(self, fname):
        for f in self.funcs:
            if f.func_name == fname:
                return f
            sf = str(f)
            if sf.startswith(fname + "@"):
                return f
            if fname.startswith("S_0x"):
                try:
                    begin = sf.split("@")[1].split("-")[0]
                    if fname == ("S_" + begin):
                        return f
                except Exception:
                    pass
        return None

    def _choose_target_function(self, target_addr=None):
        # CLI 未传 --target_addr 时可能给空串，这里统一按 None 处理。
        if isinstance(target_addr, str):
            if target_addr.strip() == '':
                target_addr = None

        if target_addr is not None:
            for fname in self.fb_tbl.keys():
                bl = self.fb_tbl[fname]
                b, exact = select_block_by_addr(bl, target_addr)
                if b is not None:
                    if exact:
                        print '[func_state_flatten_diversify.py] Found target_addr: %s (matched with 0x%X) in %s' % (
                            target_addr, b.bblock_begin_loc.loc_addr, fname
                        )
                    else:
                        print '[func_state_flatten_diversify.py] Found target_addr: %s inside block (begin=0x%X) in %s' % (
                            target_addr, b.bblock_begin_loc.loc_addr, fname
                        )
                    return fname
            print '[func_state_flatten_diversify.py] Warning: target_addr %s not found, fallback to random' % target_addr

        # 未指定 target_addr：随机挑一个基本块，再锁定该块所在函数。
        block_candidates = []
        for fname in self.fb_tbl.keys():
            bl = self.fb_tbl[fname]
            if len(bl) < 3:
                continue
            for b in bl:
                try:
                    if len(self.bb_instrs(b)) > 0:
                        block_candidates.append((fname, b))
                except Exception:
                    continue
        if len(block_candidates) == 0:
            return None
        fname, b = random.choice(block_candidates)
        try:
            print '[func_state_flatten_diversify.py] Randomly selected block 0x%X in %s' % (
                b.bblock_begin_loc.loc_addr, fname
            )
        except Exception:
            print '[func_state_flatten_diversify.py] Randomly selected block in %s' % fname
        return fname

    def _cmov_for_jcc(self, op_name):
        table = {
            'je': 'cmove',
            'jz': 'cmovz',
            'jne': 'cmovne',
            'jnz': 'cmovnz',
            'jg': 'cmovg',
            'jge': 'cmovge',
            'jl': 'cmovl',
            'jle': 'cmovle',
            'ja': 'cmova',
            'jae': 'cmovae',
            'jb': 'cmovb',
            'jbe': 'cmovbe',
            'jo': 'cmovo',
            'jno': 'cmovno',
            'js': 'cmovs',
            'jns': 'cmovns',
            'jp': 'cmovp',
            'jnp': 'cmovnp',
        }
        return table.get(op_name)

    def _append_dispatcher(self, attach_loc, dispatcher_label, state_var, state_labels):
        """
        追加分发器逻辑：
        1) 从 state_var 读取下一个状态（标签地址）
        2) 通过一串 cmp/je 显式分发（便于把 CFG 改成星型）
        3) 兜底使用 jmp *state_var，确保功能不变
        """
        if len(state_labels) == 0:
            return

        loc_head = copy.deepcopy(attach_loc)
        loc_head.loc_label = dispatcher_label + ': '
        loc_no = copy.deepcopy(attach_loc)
        loc_no.loc_label = ''

        ridx = 0
        state_reg = self._regs[ridx]

        # mov tmp_value1, %reg
        self.append_instrs(TripleInstr((self._ops['mov'], state_reg, Label(state_var), loc_head, None)), attach_loc)
        for s in state_labels:
            self.append_instrs(
                TripleInstr(('cmp', state_reg, Label('$' + s), loc_no, None)),
                attach_loc
            )
            self.append_instrs(
                DoubleInstr(('je', Label(s), loc_no, None)),
                attach_loc
            )

        # fallback: jmp *tmp_value1
        self.append_instrs(
            DoubleInstr((self._ops['jmp'], Label('*' + state_var), loc_no, None)),
            attach_loc
        )

    def _rewrite_function_cfg(self, fname):
        f = self._find_func_obj(fname)
        if f is None:
            print '[func_state_flatten_diversify.py] Warning: cannot find function object for %s' % fname
            return False

        fil = self.func_instrs(f)
        if fil is None or len(fil) == 0:
            print '[func_state_flatten_diversify.py] Warning: empty function %s' % fname
            return False

        prefix = 'SMF_' + self._sanitize_label_name(fname)
        dispatcher_label = prefix + '_dispatch'
        state_var = 'tmp_value1'
        tmp_reg_a = self._regs[0]
        tmp_reg_b = self._regs[1]

        changed = 0
        state_labels = []

        # 注意：在原始函数指令快照上遍历，避免边改边遍历造成遗漏
        for i in list(fil):
            op = get_op(i)
            des = get_cf_des(i)
            if des is None or not isinstance(des, Label):
                continue

            op_name = p_op(op)
            loc = get_loc(i)
            if loc is None:
                continue

            # 无条件跳转：state=目标label；jmp dispatch
            if op_name == 'jmp' or op_name == self._ops['jmp']:
                loc_no = self._get_loc(i)
                loc_no.loc_label = ''
                repl = TripleInstr((self._ops['mov'], Label(state_var), Label('$' + str(des)), loc, None))
                jmp_disp = DoubleInstr((self._ops['jmp'], Label(dispatcher_label), loc_no, None))
                self.replace_instrs(repl, loc, i)
                self.append_instrs(jmp_disp, loc)
                state_labels.append(str(des))
                changed += 1
                continue

            # 条件跳转：使用 cmov 选择 true/false 状态，再 jmp dispatch
            cmov_op = self._cmov_for_jcc(op_name)
            if cmov_op is not None:
                self._fall_counter += 1
                fall_label = '%s_fall_%d' % (prefix, self._fall_counter)

                loc_no = self._get_loc(i)
                loc_no.loc_label = ''
                loc_fall = copy.deepcopy(loc_no)
                loc_fall.loc_label = fall_label + ': '

                seq = [
                    DoubleInstr((self._ops['push'], tmp_reg_a, loc, None)),
                    DoubleInstr((self._ops['push'], tmp_reg_b, loc_no, None)),
                    TripleInstr((self._ops['mov'], tmp_reg_a, Label('$' + fall_label), loc_no, None)),
                    TripleInstr((self._ops['mov'], tmp_reg_b, Label('$' + str(des)), loc_no, None)),
                    TripleInstr((cmov_op, tmp_reg_a, tmp_reg_b, loc_no, None)),
                    TripleInstr((self._ops['mov'], Label(state_var), tmp_reg_a, loc_no, None)),
                    DoubleInstr((self._ops['pop'], tmp_reg_b, loc_no, None)),
                    DoubleInstr((self._ops['pop'], tmp_reg_a, loc_no, None)),
                    DoubleInstr((self._ops['jmp'], Label(dispatcher_label), loc_no, None)),
                    SingleInstr((self._ops['nop'], loc_fall, None)),
                ]

                self.replace_instrs(seq[0], loc, i)
                for ni in seq[1:]:
                    self.append_instrs(ni, loc)
                state_labels.append(str(des))
                state_labels.append(fall_label)
                changed += 1

        if changed == 0:
            print '[func_state_flatten_diversify.py] Warning: no transformable jump in %s' % fname
            return False

        # 去重但保持顺序
        uniq_states = []
        seen = set()
        for s in state_labels:
            if s in seen:
                continue
            seen.add(s)
            uniq_states.append(s)

        attach_loc = get_loc(fil[-1])
        self._append_dispatcher(attach_loc, dispatcher_label, state_var, uniq_states)
        self.update_process()
        print '[func_state_flatten_diversify.py] flattened %s, rewritten_edges=%d' % (fname, changed)
        return True

    def visit(self, instrs, target_addr=None):
        print 'start function state-machine flatten diversification'
        self.instrs = copy.deepcopy(instrs)

        fname = self._choose_target_function(target_addr)
        if fname is None:
            print '[func_state_flatten_diversify.py] Warning: no candidate function, skip'
            return self.instrs

        self._rewrite_function_cfg(fname)
        return self.instrs
