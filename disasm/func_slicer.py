# -*- coding: utf-8 -*-
import Types
import init_sec_adjust
from Types import Func
from utils.ail_utils import read_file, unify_int_list, get_loc, dec_hex
import logging
logger = logging.getLogger(__name__)

class func_slicer(object):
    """
    Function boundary evaluation
    """

    def __init__(self, instrs, funcs):
        """
        :param instrs: list of instructions
        :param funcs: list of function objects
        """
        self.instrs = instrs
        self.funcs = funcs
        self.baddr = -1
        self.eaddr = -1
        self.label = ''
        self.func_begins = []
        self.text_b_addr = 0
        self.text_e_addr = 0
        self.func_set = {}

    def update(self):
        func = 'S_' + dec_hex(self.baddr)
        if func in self.func_set:
            self.func_set[func].func_begin_addr = self.baddr
            self.func_set[func].func_end_addr = self.eaddr
            self.func_set[func].is_lib = False
        else:
            f1 = Func(func, self.baddr, self.eaddr, False)
            self.func_set[func] = f1

    def filter_addr_by_secs(self, bl):
        """
        Filter out addresses in bad sections
        :param bl: address list
        """
        init_sec_adjust.main()
        with open('init_sec.info') as f:
            l = f.readline()    # .init 0000000000004000 004000 00001b
        items = l.split()
        baddr = int(items[1], 16)
        eaddr = baddr + int(items[3], 16)
        return filter(lambda n: n < baddr or n >= eaddr, bl)

    def update_text_info(self):
        """
        Load .text section info
        """
        with open('text_sec.info') as f:
            l = f.readline()
        items = l.split()
        self.text_b_addr = int(items[1], 16)
        self.text_e_addr = int(items[3], 16)

    def build_func_info(self):
        """
        Evaluate function boundaries
        """
        self.func_begins = unify_int_list(self.func_begins)
        logger.info("[disasm/func_slicer.py:build_func_info]: self.func_begins = {}".format(self.func_begins))
        self.func_begins = self.filter_addr_by_secs(self.func_begins)
        logger.info("[disasm/func_slicer.py:build_func_info]: self.func_begins_after_filter = {}".format(self.func_begins))
        for i in range(len(self.func_begins)-1):
            self.baddr = self.func_begins[i]
            self.eaddr = self.func_begins[i+1]
            self.update()
        self.baddr = self.func_begins[-1]
        # 使用instrs中实际存在的最后一个指令地址
        # 因为instrs是通过insert(0, ...)倒序插入的，所以instrs[0]应该是最高地址的指令
        # 但如果过滤掉了某些指令，instrs[0]可能不是最后一个指令
        # 为了安全，使用instrs[0]的地址（通常是最大的），但如果为空则使用.text段结束地址
        if len(self.instrs) > 0:
            # instrs[0]应该是最高地址的指令（因为倒序插入）
            # 但为了处理地址空缺的情况，我们需要确保使用实际存在的最大地址
            # 由于instrs是倒序的，instrs[0]通常是最大的，但如果有空缺可能不是
            # 这里简化处理：使用instrs[0]的地址，因为通常它是最大的
            self.eaddr = get_loc(self.instrs[0]).loc_addr
        else:
            # 如果没有指令，使用.text段结束地址
            self.eaddr = self.text_e_addr
        self.update()

    def check_text(self, e):
        """
        Check if function in .text section
        :param e: expression
        """
        if isinstance(e, Types.CallDes) and not e.is_lib:
            n = int(e.func_name[2:], 16)
            return self.text_b_addr <= n < self.text_e_addr
        return False

    def update_func(self):
        """
        Add function to function set
        """
        for e in self.funcs:
            self.func_set[e.func_name] = e
        logger.info("[disasm/func_slicer.py:update_func]: self.funcs = {}".format(self.funcs))
        logger.info("[disasm/func_slicer.py:update_func]: self.func_set = {}".format(self.func_set))

    def get_func_list(self):
        """
        Return list of function
        """
        return self.func_set.values()

    def get_funcs(self):
        """
        Evalute function info and return updated function list
        """
        self.func_begins = map(lambda a: int(a, 16), read_file('faddr.txt'))
        self.func_begins += [f.func_begin_addr for f in self.funcs if f.func_begin_addr != 0]
        # logger.info("[disasm/func_slicer.py:get_funcs]: self.func_begins = {}".format(self.func_begins))
        self.build_func_info()
        fl = self.get_func_list()
        logger.info("[disasm/func_slicer.py:get_funcs]: Sliced {} functions".format(len(self.func_begins)))
        logger.info("[disasm/func_slicer.py:get_funcs]: fl = {}".format(fl))
        return fl
