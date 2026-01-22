# -*- coding: utf-8 -*-
"""
【变异操作类型】基本块不透明谓词混淆 (Basic Block Opaque Predicate Obfuscation)
【框架序号】Action ID: 1
【功能说明】在基本块中插入不透明谓词（opaque predicates），即总是为真或假的谓词条件，
           通过添加虚假的控制流分支来混淆代码逻辑，增加逆向分析的难度。
"""
from analysis.visit import *
from disasm.Types import *
from utils.ail_utils import *
from utils.pp_print import *
from junkcodes import get_junk_codes
from addr_utils import addr_equals, select_block_by_addr

obfs_proportion = 0.2


class bb_opaque_diversify(ailVisitor):
    """
    基本块不透明谓词混淆类
    
    通过插入总是为真或假的谓词条件，添加虚假的控制流分支来混淆代码。
    支持两种不透明谓词模式：
    - header1: 基于数学恒等式 (y < 10 || x*(x-1) % 2 == 0)，总是为真
    - header2: 基于函数调用返回值，通过 opaque_func 设置寄存器值，然后比较
    """

    def __init__(self, funcs, fb_tbl, cfg_tbl):
        """
        初始化不透明谓词混淆器
        
        参数:
            funcs: 函数列表
            fb_tbl: 函数到基本块列表的映射表
            cfg_tbl: 控制流图表
        """
        ailVisitor.__init__(self)
        self.fb_tbl = fb_tbl
        self.routine_constant = random.randint(-1000, 1000)

    def _change_instrs_with_changelist(self, changelist):
        """
        根据修改列表批量应用指令修改操作
        
        参数:
            changelist: 修改操作列表，每个元素为 (操作类型, 新指令, 位置, 旧指令)
        """
        for c in changelist:
            op = c[0]
            if op == instr_update.INSERT:
                self.insert_instrs(c[1], c[2])
            elif op == instr_update.REPLACE:
                self.replace_instrs(c[1], c[2], c[3])
            else:
                assert False, 'unknown operation to do.'

    def get_opaque_header2(self, b, spt_pos=0):
        """
        生成第二种不透明谓词头部（基于函数调用）
        
        原理：调用 opaque_func 函数，该函数会将 %eax 设置为 routine_constant，
             然后比较 %eax 和 routine_constant，结果总是相等，条件总是为真。
             如果条件为假（理论上不会发生），则调用 halt_func 终止程序。
        
        参数:
            b: 要插入不透明谓词的基本块
            spt_pos: 在基本块中的插入位置（第几条指令之前）
            
        返回:
            list: 修改操作列表，包含插入和替换指令的操作
        """
        opaque_symbol = b.bblock_name + '_opaque_next'
        bil = self.bb_instrs(b)
        i = bil[spt_pos]

        # print 'basic block opaque transformation: ' + b.bblock_name + ' ' + dec_hex(b.bblock_begin_loc.loc_addr) \
        #       + '->' + dec_hex(b.bblock_end_loc.loc_addr)
        # print 'instruction be added opaque block is: %s' % pp_print_instr(i)

        iloc = self._get_loc(i)
        tmp_iloc = copy.deepcopy(iloc)
        tmp_iloc.loc_label = ''
        tmp_iloc2 = copy.deepcopy(iloc)
        tmp_iloc2.loc_label = opaque_symbol + ': '

        # store the %eax to stack
        i1 = DoubleInstr(('push', self._regs[0], iloc, None))
        save_flag = SingleInstr((self._ops['pushf'], tmp_iloc, None))
        junk1 = get_junk_codes(tmp_iloc)
        i2 = DoubleInstr(('call', Types.Label('opaque_func'), tmp_iloc, None))
        # in attach_opaque_routines, we set the value of %eax=0x0
        # cmp is supposed to be true, and the code will always jump to opaque_symbol
        i3 = TripleInstr(('cmp', self._regs[0], Types.Normal(self.routine_constant), tmp_iloc, None))
        i4 = DoubleInstr(('je', Types.Label(opaque_symbol), tmp_iloc, None))
        # this junk should never be executed
        junk2 = get_junk_codes(tmp_iloc)
        i5 = DoubleInstr(('call', Types.Label('halt_func'), tmp_iloc, None))
        # recover the value of %eax, and here is the start of opaque_symbol
        recover_flag = SingleInstr((self._ops['popf'], tmp_iloc2, None))
        i6 = DoubleInstr(('pop', self._regs[0], tmp_iloc, None))
        i0 = set_loc(i, tmp_iloc)

        res = list()
        res.append((instr_update.INSERT, i1, iloc))
        res.append((instr_update.INSERT, save_flag, iloc))
        res.extend([(instr_update.INSERT, j, iloc) for j in junk1])
        res.append((instr_update.INSERT, i2, iloc))
        res.append((instr_update.INSERT, i3, iloc))
        res.append((instr_update.INSERT, i4, iloc))
        res.extend([(instr_update.INSERT, j, iloc) for j in junk2])
        res.append((instr_update.INSERT, i5, iloc))
        res.append((instr_update.INSERT, recover_flag, iloc))
        res.append((instr_update.INSERT, i6, iloc))
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
        # print '[bb_opaque_diversify.py:get_opaque_header1] b = ', b
        bil = self.bb_instrs(b)
        # print '[bb_opaque_diversify.py:get_opaque_header1] bil = ', bil
        i = bil[spt_pos]
        # print '[bb_opaque_diversify.py:get_opaque_header1] i = ', i
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
        # y < 10
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
        res.append((instr_update.INSERT, DoubleInstr(('call', Types.Label('abort'), iloc_without_label, None)),
                    iloc_with_block_label))
        # true branch
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

    def bb_div_opaque(self, target_addr = None):
        """
        对基本块执行不透明谓词混淆
        
        流程：
        1. 随机选择一种不透明谓词模式（header1 或 header2）
        2. 如果指定 target_addr，仅对该地址的基本块进行变异
        3. 如果未指定 target_addr，对每个函数随机选择一个基本块进行变异
        4. 批量应用所有修改
        """
        print 'do basic block transformation on: %d functions' % len(self.fb_tbl)
        # print 'fb_tbl = ', self.fb_tbl
        # add new opaque header method here
        # modes = [self.get_opaque_header1, self.get_opaque_header2]
        # header1 may cause errors in disassembling when compiler is gcc-4.8 (x64? I am not sure)
        modes = [self.get_opaque_header1]
        
        # 如果指定了 target_addr，只处理该地址的基本块
        if target_addr is not None:
            for f in self.fb_tbl.keys():
                bl = self.fb_tbl[f]
                b, exact = select_block_by_addr(bl, target_addr)
                if b is not None:
                    if exact:
                        print '[bb_opaque_diversify.py:bb_div_opaque] Found target_addr: %s (matched with 0x%X)' % (target_addr, b.bblock_begin_loc.loc_addr)
                    else:
                        print '[bb_opaque_diversify.py:bb_div_opaque] Found target_addr: %s inside block (begin=0x%X)' % (target_addr, b.bblock_begin_loc.loc_addr)
                    n_mode = random.randint(0, len(modes) - 1)
                    changelist = modes[n_mode](b, 0)
                    self._change_instrs_with_changelist(changelist)
                    self.update_process()
                    return  # 找到目标后立即返回
            print '[bb_opaque_diversify.py:bb_div_opaque] Warning: target_addr %s not found' % target_addr
            print '[bb_opaque_diversify.py:bb_div_opaque] start to random mutate' % target_addr
            # return
        
        # 未指定 target_addr或者没有找到tartget_addr，随机选择基本块进行变异
        for f in self.fb_tbl.keys():
            bl = self.fb_tbl[f]
            if len(bl) > 0:
                n = random.randint(0, len(bl) - 1)
                n_mode = random.randint(0, len(modes) - 1)
                changelist = modes[n_mode](bl[n], 0)
                self._change_instrs_with_changelist(changelist)
        self.update_process()

    def get_opaque_routines(self, iloc):
        """
        生成不透明谓词的辅助函数（opaque_func 和 halt_func）
        
        opaque_func: 设置 %eax = routine_constant，然后返回
        halt_func: 执行 hlt 指令终止程序（用于处理理论上不会发生的假分支）
        
        参数:
            iloc: 插入位置的位置信息
            
        返回:
            list: 辅助函数的指令列表
        """
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
        selected_i = self.instrs[random.choice(bb_starts)]
        # to avoid possible error in disassembling
        # selected_i = self.instrs[-1]
        # the location of opaque routines should be carefully selected
        selected_loc = get_loc(selected_i)
        opaque_routines = self.get_opaque_routines(selected_loc)
        # should be changed to call insert_instr, then call update_process
        for ins in opaque_routines:
            self.append_instrs(ins, selected_loc)
        self.update_process()

    def bb_opaque_process(self, target_addr = None):
        """
        执行完整的不透明谓词混淆流程
        
        步骤：
        1. 在基本块中插入不透明谓词（bb_div_opaque）
        2. 在代码末尾附加辅助函数（attach_opaque_routines）
        """
        self.bb_div_opaque(target_addr)
        # remove header1 mode when compiler is gcc-4.8
        self.attach_opaque_routines()

    def visit(self, instrs, target_addr = None):
        """
        访问者模式的入口方法
        
        参数:
            instrs: 输入的指令序列
            
        返回:
            list: 处理后的指令序列（已插入不透明谓词）
        """
        print 'start basic block opaque diversification'
        self.instrs = copy.deepcopy(instrs)
        self.bb_opaque_process(target_addr)
        return self.instrs
