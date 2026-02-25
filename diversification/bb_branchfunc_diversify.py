# -*- coding: utf-8 -*-
"""
【变异操作类型】分支函数化 (Branch Function Diversification)
【框架序号】Action ID: 3
【功能说明】将直接跳转指令（jmp/条件跳转）转换为通过分支函数（branch function）的间接跳转。
           通过将跳转目标地址存储到变量中，然后调用统一的分支函数来执行跳转，
           从而隐藏控制流结构，增加逆向分析难度。同时插入垃圾代码增强混淆效果。
           
           处理两种跳转类型：
           1. 无条件跳转（jmp）：直接替换为分支函数调用
           2. 条件跳转（je/jne/jl等）：使用条件移动指令（cmov）选择跳转目标，再调用分支函数
"""
from analysis.visit import *
from disasm.Types import *
from utils.ail_utils import *
from utils.pp_print import *
from addr_utils import addr_equals
from junkcodes import get_junk_codes

obfs_proportion = 0.02


class bb_branchfunc_diversify(ailVisitor):

    def __init__(self, funcs, fb_tbl, cfg_tbl):
        ailVisitor.__init__(self)
        self.funcs = funcs
        self._new_des_id = 0

    def _branch_a_func(self, f, target_addr=None):
        """
        对单个函数执行分支函数化
        
        如果指定 target_addr，只处理该地址的跳转指令
        如果未指定 target_addr，处理函数中所有的跳转指令
        """
        fil = self.func_instrs(f)
        find_a_valid_func = False
        for instr in fil:
            # 如果指定了 target_addr，只处理匹配地址的指令
            if target_addr:
                loc = self._get_loc(instr)
                if not loc or not addr_equals(loc.loc_addr, target_addr):
                    continue
            
            op = get_op(instr)
            des = get_cf_des(instr)
            if des is not None and isinstance(des, Label):
                if op in JumpOp:
                    #if random.random() > obfs_proportion:
                    #if True: 
                    #    continue
                    # here we modify the process of 2 situations, jmp and conditional jmp
                    if p_op(op) == 'jmp' or p_op(op) == self._ops['jmp']:
                        # this is a simple jump, we simply cache the des and call the routine
                        find_a_valid_func = True
                        loc = self._get_loc(instr)
                        i0 = TripleInstr((self._ops['mov'], Label('branch_des'), Label('$' + str(des)), loc, None))
                        loc1 = copy.deepcopy(loc)
                        loc1.loc_label = ''
                        i1 = DoubleInstr((self._ops['call'], Label('branch_routine'), loc1, None))
                        junk1 = get_junk_codes(loc1)
                        junk2 = get_junk_codes(loc1)

                        self.insert_instrs(i0, loc)
                        for _i in junk1:
                            self.insert_instrs(_i, loc)
                        self.replace_instrs(i1, loc, instr)
                        for _i in junk2:
                            self.append_instrs(_i, loc)
                    elif p_op(op) in {'je', 'jne', 'jl', 'jle', 'jg', 'jge'}:
                        # we only handle with these conditional jmp
                        find_a_valid_func = True
                        loc = self._get_loc(instr)
                        postfix = p_op(op)[1:]
                        # we ues conditional move the modify a conditional jmp
                        self._new_des_id += 1
                        fall_through_label = 'fall_through_label_%d' % self._new_des_id
                        loc_no_label = copy.deepcopy(loc)
                        loc_no_label.loc_label = ''
                        loc_fall_through = copy.deepcopy(loc)
                        loc_fall_through.loc_label = fall_through_label + ':'
                        tmp = [
                            DoubleInstr((self._ops['push'], self._regs[0], loc, None)),  # 0  replace
                            DoubleInstr((self._ops['push'], self._regs[1], loc_no_label, None)),
                            TripleInstr((self._ops['mov'], self._regs[0], Label('$' + fall_through_label), loc_no_label, None)),
                            TripleInstr((self._ops['mov'], self._regs[1], Label('$' + str(des)), loc_no_label, None)),
                            TripleInstr(('cmov' + postfix, self._regs[0], self._regs[1], loc_no_label, None)),
                            TripleInstr((self._ops['mov'], Label('branch_des'), self._regs[0], loc_no_label, None)),
                            DoubleInstr((self._ops['pop'], self._regs[1], loc_no_label, None)),
                            DoubleInstr((self._ops['pop'], self._regs[0], loc_no_label, None)),
                            DoubleInstr((self._ops['call'], Label('branch_routine'), loc_no_label, None)),
                            SingleInstr((self._ops['nop'], loc_fall_through, None))
                        ]
                        self.replace_instrs(tmp[0], loc, instr)
                        for _i in tmp[1:]:
                            self.append_instrs(_i, loc)
                    
                    # 如果指定了 target_addr，找到目标后立即返回
                    if target_addr:
                        print '[bb_branchfunc_diversify.py:_branch_a_func] Found and processed target_addr: %s' % target_addr
                        return True
        return find_a_valid_func

    def branch_func(self, target_addr=None):
        """
        对函数执行分支函数化
        
        如果指定 target_addr，只处理包含该地址的函数
        如果未指定 target_addr，处理所有函数
        """
        # print 'bb branch on %d candidate function' % len(self.funcs)
        # select the 1st obfs_proportion functions
        # for f in self.funcs[:int(obfs_proportion * len(self.funcs))]:
        do_branch = False
        for f in self.funcs:
        #for f in random.sample(self.funcs, int(obfs_proportion * len(self.funcs)) + 1):
            if self._branch_a_func(f, target_addr):
                do_branch = True
                self.update_process()
                # 如果指定了 target_addr，找到后立即返回
                if target_addr:
                    return True
        if not do_branch:
            if target_addr:
                print '[bb_branchfunc_diversify.py:branch_func] Warning: target_addr %s not found' % target_addr
            else:
                # 连续变异时这属于正常情况：当前轮没有可改写的直接跳转。
                print '[bb_branchfunc_diversify.py:branch_func] Warning: no valid function is selected, skip this round'
            return False
        return True

    def bb_div_branch(self, target_addr=None):
        return self.branch_func(target_addr)

    def get_branch_routine(self, iloc):
        """
        return the list of routine instructions for branch functions
        :param iloc: the location of instruction that routine being inserted
        :return: the list of routine instructions
        """
        loc_with_branch_label = copy.deepcopy(iloc)
        loc_with_branch_label.loc_label = 'branch_routine: '
        loc = copy.deepcopy(iloc)
        loc.loc_label = ''
        i0 = DoubleInstr((self._ops['pop'], Label('global_des'), loc_with_branch_label, None))
        junk = get_junk_codes(loc)
        i1 = DoubleInstr((self._ops['jmp'], Label('*branch_des'), loc, None))
        res = [i0]
        res.extend(junk)
        res.append(i1)
        return res

    def attach_branch_routine(self):
        if not self.instrs:
            print '[bb_branchfunc_diversify.py:attach_branch_routine] Warning: empty instrs, skip'
            return
        loc = get_loc(self.instrs[-1])
        routine_instrs = self.get_branch_routine(loc)
        self.instrs.extend(routine_instrs)

    def bb_div_process(self, target_addr=None):
        changed = self.bb_div_branch(target_addr)
        if changed:
            self.attach_branch_routine()

    def visit(self, instrs, target_addr = None):
        print 'start bb branch function'
        self.instrs = self._clone_instrs_for_edit(instrs)
        self.bb_div_process(target_addr)
        return self.instrs
