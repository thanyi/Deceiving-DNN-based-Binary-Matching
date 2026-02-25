# -*- coding: utf-8 -*-
"""
【变异操作类型】基本块扁平化 (Basic Block Flattening)
【框架序号】Action ID: 5
【功能说明】将控制流图（CFG）扁平化，通过 switch-case 结构将所有基本块的跳转统一到一个分发器（dispatcher）。
           将所有跳转指令替换为将目标地址存入 global_des，然后跳转到 switch_bb 分发器，
           分发器根据 global_des 的值进行间接跳转。这样可以消除明显的控制流结构，
           使逆向分析者难以理解程序的执行流程。
           
           处理流程：
           1. 检查 CFG 是否可扁平化（is_flattenable_cfg）
           2. 将所有跳转指令替换为 mov global_des + jmp switch_bb
           3. 在代码中插入 switch_bb 分发器，执行 jmp *global_des
"""
from analysis.visit import *
from disasm.Types import *
from utils.ail_utils import *
from utils.pp_print import *
from addr_utils import addr_equals, select_block_by_addr
from junkcodes import get_junk_codes


obfs_proportion = 0.08


class bb_flatten_diversify(ailVisitor):

    def __init__(self, funcs, fb_tbl, cfg_tbl):
        ailVisitor.__init__(self)
        self.fb_tbl = fb_tbl
        self.funcs = funcs
        self.cfg_tbl = cfg_tbl

    @staticmethod
    def is_flattenable_cfg(cfg):
        acc = True
        f, v = cfg
        for t in v:
            if t is not None and not acc:
                return False
            elif len(t) == 2 and len(t[1]) == 2:
                if t[1][1] is None:
                    return False
                elif t[1][1] == 'T':
                    return False
        return True

    def flatten_cfg(self, cfg):
        fname, _ = cfg
        bbl = self.fb_tbl[fname]
        name_eloc_dict = {}
        for b in bbl:
            name_eloc_dict[b.bblock_name] = b.bblock_end_loc

        f = None
        for _f in self.funcs:
            '''
            Fname S_0x804985D
            self name set_suffix_length@0x804985D-0x8049A4A
            '''
            if _f.func_name == fname:
                f = _f
                break
            elif "@" in str(_f):
                tmp_f = str(_f)
                tmp_f = tmp_f.split("@")[-1]
                tmp_f = str(tmp_f.split("-")[0])
                tmp_f = "S_"+tmp_f
                if tmp_f  == fname:
                    f = _f
                    break 

        assert f is not None
        func_instrs = self.func_instrs(f)
        for i in func_instrs:
            op = get_op(i)
            if op in JumpOp:
                des = get_cf_des(i)
                if isinstance(des, Label):
                    i0_loc = get_loc(i)
                    i1_loc = self._get_loc(i)
                    i1_loc.loc_label = ''
                    i0 = TripleInstr((self._ops['mov'], Label('global_des'), Label('$' + des), i0_loc, None))
                    junk = get_junk_codes(i1_loc)
                    i1 = DoubleInstr((op, Label('switch_bb'), i1_loc, None))
                    self.replace_instrs(i0, get_loc(i), i)
                    for _i in junk:
                        self.append_instrs(_i, get_loc(i))
                    self.append_instrs(i1, get_loc(i))
        self.update_process()

    def bb_div_flatten(self, target_addr=None):
        """
        对控制流图执行扁平化变异
        
        如果指定 target_addr，找到包含该地址的函数并扁平化
        如果未指定 target_addr，对所有可扁平化的 CFG 进行扁平化
        """
        cfgs = []
        for cfg in self.cfg_tbl:
            if self.is_flattenable_cfg(cfg):
                cfgs.append(cfg)
        if len(cfgs) <= 0:
            #print 'no flattenable block, quit'
            raise Exception('no flattenable block, quit')
            #return
        
        # 如果指定了 target_addr，找到包含该地址的函数并扁平化
        if target_addr is not None:
            for cfg in cfgs:
                fname = cfg[0]
                if fname not in self.fb_tbl.keys():
                    continue
                bl = self.fb_tbl[fname]
                b, exact = select_block_by_addr(bl, target_addr)
                if b is not None:
                    if exact:
                        print '[bb_flatten_diversify.py:bb_div_flatten] Found target_addr: %s (matched with 0x%X) in function %s' % (target_addr, b.bblock_begin_loc.loc_addr, fname)
                    else:
                        print '[bb_flatten_diversify.py:bb_div_flatten] Found target_addr: %s inside block (begin=0x%X) in function %s' % (target_addr, b.bblock_begin_loc.loc_addr, fname)
                    self.flatten_cfg(cfg)
                    self.insert_switch_routine()
                    return
            print '[bb_flatten_diversify.py:bb_div_flatten] Warning: target_addr %s not found in flattenable functions' % target_addr
            return
        
        # 未指定 target_addr，对所有可扁平化的 CFG 进行扁平化
        for cfg in cfgs:
            if cfg[0] in self.fb_tbl.keys():
                self.flatten_cfg(cfg)
        self.insert_switch_routine()

    def get_switch_routine(self, loc):
        loc_without_label = copy.deepcopy(loc)
        loc_without_label.loc_label = ''
        junk = get_junk_codes(loc)
        i0 = DoubleInstr((self._ops['jmp'], Label('*global_des'), loc_without_label, None))
        junk.append(i0)
        # note junk can be length 0, the label modification must locate after the appending
        junk[0][-2].loc_label = ".globl switch_bb\nswitch_bb:"
        return junk

    def insert_switch_routine(self):
        bb_starts = []
        if not self.instrs:
            print '[bb_flatten_diversify.py:insert_switch_routine] Warning: empty instrs, skip switch routine'
            return
        for i in range(len(self.instrs)):
            if get_op(self.instrs[i]) in ControlOp and 'ret' in p_op(self.instrs[i]):
                # Note: do not use 'jmp', because it may result in collision with bb_branchfunc_diversify
                if i + 1 < len(self.instrs):
                    bb_starts.append(i + 1)
        if bb_starts:
            selected_i = self.instrs[random.choice(bb_starts)]
            # the location of switch routines should be carefully selected
            selected_loc = get_loc(selected_i)
            routine = self.get_switch_routine(selected_loc)
            for ins in routine:
                self.insert_instrs(ins, selected_loc)
            self.update_process()
            return

        # fallback: no ret found, append routine at end
        print '[bb_flatten_diversify.py:insert_switch_routine] Warning: no ret found, append routine at end'
        selected_loc = get_loc(self.instrs[-1])
        routine = self.get_switch_routine(selected_loc)
        for ins in routine:
            self.append_instrs(ins, selected_loc)
        self.update_process()

    def visit(self, instrs, target_addr = None):
        self.instrs = self._clone_instrs_for_edit(instrs)
        self.bb_div_flatten(target_addr)
        return self.instrs
