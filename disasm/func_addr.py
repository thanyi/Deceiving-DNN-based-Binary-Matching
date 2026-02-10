# -*- coding: utf-8 -*-
from math import log
import os
import re
import shutil
import config
import logging
logger = logging.getLogger(__name__)

def func_addr(filename, count, fexclude=''):
    """
    Dump function symbols and addresses
    :param filename: path to target executable
    :param count: unused
    :param fexclude: path to file of function symbols to exclude from dump
    """
    objdump_cmd = config.objdump + ' -Dr -j .text ' + filename + ' > dump.s'
    os.system(objdump_cmd)
    # os.system('pwd > dump.s.txt')
    logger.debug("[func_addr.py:func_addr]: objdump cmd = {}".format(objdump_cmd))
    grep_cmd = 'grep ">:" dump.s > fl'
    os.system(grep_cmd)
    logger.debug("[func_addr.py:func_addr]: grep cmd = {}".format(grep_cmd))

    # useless_func_discover(filename)  # 生成useless_func.info
    
    if len(fexclude) > 0 and os.path.isfile(fexclude):
        os.system('grep -v -f ' + fexclude + ' fl > fl.filtered')
        shutil.move('fl.filtered', 'fl')

    with open('fl') as f: fnl = f.readlines()

    fnl_old = []
    if os.path.isfile('faddr_old.txt'):
        with open('faddr_old.txt') as f:
            fnl_old = f.readlines()

    fnl_old = map(lambda l : int(l.split()[0], 16), fnl_old)    
    #fnl_old = map(lambda l : l.split()[0], fnl_old)
    #print fnl_old

    blacklist = ['__libc_csu_init', '__libc_csu_fini', '__i686.get_pc_thunk.bx', '__do_global_ctors_aux', '_start', '__do_global_dtors_aux', 'frame_dummy']
    addrs = []
    addrs_2 = []
    regex = re.compile(r'S_(0x[0-9A-F]+)', re.I)
    regex1 = re.compile(r'<(.*)>:', re.I)

    for fn in fnl:
        # ad-hoc solution, we don't consider basic block labels as functions
        if not "BB_" in fn:
            if "S_" in fn:
                m = regex.search(fn)    # "0000000000401160 <S_0x401160>:\n" --> S_0x401160
                if m:
                    d = m.groups()[0]   # '0x401160'
                    d1 = int(d,16)
                    if d1 in fnl_old:
                        addr = fn.split('<')[0].strip()
                        addrs.append("0x" + addr + '\n')
                        addrs_2.append(fn)
            elif count > 0:
                m = regex1.search(fn)
                if m:
                    d = m.groups()[0]
                    if not d in blacklist:
                        addr = fn.split('<')[0].strip()
                        addrs.append("0x" + addr + '\n')
                        addrs_2.append(fn)
                    else:
                        logger.debug("[func_addr.py:func_addr]: fn = {}, d = {}, count = {}".format(fn, d, count))
            else:
                addr = fn.split('<')[0].strip()
                addrs.append("0x" + addr + '\n')
                addrs_2.append(fn)

    with open('faddr.txt', 'w') as f:
        f.writelines(addrs)

    with open('faddr_old.txt', 'w') as f:
        f.writelines(addrs_2)

    shutil.copy('faddr.txt', 'faddr.txt.' + str(count))
    shutil.copy('faddr_old.txt', 'faddr_old.txt.' + str(count))



def useless_func_discover(filename):

    black_list = ('_start', '__do_global_dtors_aux', 'frame_dummy', '__do_global_ctors_aux', '__i686.get_pc_thunk.bx', '__libc_csu_fini', '__libc_csu_init')

    os.system(config.objdump + ' -Dr -j .text ' + filename + ' > ' + filename + '.temp')

    with open(filename + '.temp') as f:
        lines = f.readlines()

    lines.append('')
    start_addr = 0
    end_addr = 0
    in_func = 'NULL'
    last_addr = 0

    def check (l):
        for b in black_list:
            if '<'+b+'>:' in l: return b
        return 'NULL'

    res = {}
    for l in lines:
        if l.strip() == "":
            if in_func != "NULL":
                end_addr = last_addr
                if end_addr[-1] == ':':
                    end_addr = end_addr[:-1]
                res[in_func] = (start_addr, end_addr)
                in_func = "NULL"
        else:
            if check (l) != "NULL":
                in_func = check(l)
                start_addr = l.split()[0]
                last_addr = start_addr
            else:
                last_addr = l.split()[0]

    res_list = []
    key_list = []
    for key, value in res.items():
        res_list.append(key + " " + value[0] + " " + value[1] +'\n')
        key_list.append(key + '\n')

    with open("useless_func.info", 'w') as f:
        f.writelines(res_list)

    if len(key_list) > 0:
        with open("useless_func_key.info", 'w') as f:   # 用于剔除黑名单中的函数，防止在反汇编过程中出现
            f.writelines(key_list)
