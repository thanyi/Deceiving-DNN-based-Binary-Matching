# -*- coding: utf-8 -*-
"""
Workfiles initialization
"""

import os
import sys

from termcolor import colored

import ail
import config
from disasm import pic_process, extern_symbol_process, arm_process
from utils.ail_utils import ELF_utils
import logging
logger = logging.getLogger(__name__)
class Init(object):
    """
    Processing initializer
    """

    def __init__(self, filepath):
        """
        :param filepath: path to executable
        """
        self.file = filepath

    def disassemble(self):
        """
        Dump .text, .rodata, .data, .data.rel.ro, .eh_frame, .got to file
        """
        print colored('1: DISASSEMBLE', 'green')
        logger.info("[init.py:disassemble]: 1: DISASSEMBLE")
        print config.objdump + ' -Dr -j .text ' + self.file + ' > ' + self.file + '.temp'
        ret = os.system(config.objdump + ' -Dr -j .text ' + self.file + ' > ' + self.file + '.temp')
        self.checkret(ret, self.file + '.temp')

        if not ELF_utils.elf_arm():
            if ELF_utils.elf_32():
                pic_process.picprocess32(self.file)
            else:
                extern_symbol_process.globalvar(self.file)
                pic_process.picprocess64(self.file)
        ret = os.system(config.objdump + " -s -j .rodata " + self.file + " | grep \"^ \" | cut -d \" \" -f3,4,5,6 > rodata.info")
        self.checkret(ret, 'rodata.info')
        ret = os.system(config.objdump + " -s -j .data " + self.file + " | grep \"^ \" | cut -d \" \" -f3,4,5,6 > data.info")
        self.checkret(ret, 'data.info')
        # 提取 .data.rel.ro 段 (包含重定位的只读数据,如 longopts 等结构体)
        os.system(config.objdump + " -s -j .data.rel.ro " + self.file + " | grep \"^ \" | cut -d \" \" -f3,4,5,6 > data.rel.ro.info")
        # 提取重定位信息用于符号引用转换
        os.system("readelf -r " + self.file + " | grep 'R_X86_64_RELATIVE\\|R_386_RELATIVE' | awk '{print $1,$NF}' > reloc.info")
        os.system(config.objdump + " -s -j .eh_frame " + self.file + " | grep \"^ \" | cut -d \" \" -f3,4,5,6 > eh_frame.info")
        if not ELF_utils.elf_arm(): os.system(config.objdump + " -s -j .eh_frame_hdr " + self.file + " | grep \"^ \" | cut -d \" \" -f3,4,5,6 > eh_frame_hdr.info")
        os.system(config.objdump + " -s -j .got " + self.file + " | grep \"^ \" | cut -d \" \" -f3,4,5,6 > got.info")

    def process(self):
        """
        Process sections
        """
        self.pltProcess()
        self.textProcess()
        self.sectionProcess()
        self.bssHandler()
        self.export_tbl_dump()
        self.userFuncProcess()

    def bssHandler(self):
        """
        Generate .bss dump and extract global bss symbols
        """
        with open("sections.info") as f:
            bssinfo = next((l for l in f if '.bss' in l), None)
            size = int(bssinfo.split()[3], 16) if bssinfo is not None else 0
        with open("bss.info", 'w') as f:
            f.write(".byte 0x00\n" * size)
        os.system('readelf -sW ' + self.file + ' | grep OBJECT | awk \'/GLOBAL/ {print $2,$8}\' > globalbss.info')
        os.system('readelf -rW ' + self.file + ' | grep _GLOB_DAT | grep -v __gmon_start__ | awk \'{print $1,$5}\' > gotglobals.info')

    def textProcess(self):
        """
        Code disassembly dump
        """
        # useless_func_del.main(self.file)
        if ELF_utils.elf_arm(): arm_process.arm_process(self.file)
        else:
            extern_symbol_process.pltgot(self.file)
            os.system("cat " + self.file + ".temp | grep \"^ \" | cut -f1,3 > instrs.info")
        
        # 过滤掉系统函数的指令
        self._filter_system_func_instrs()
        
        os.system("cut -f 1 instrs.info > text_mem.info")
    
    def _filter_system_func_instrs(self):
        """
        Filter out instructions from system functions in instrs.info
        """
        # 系统函数名称列表（与func_addr.py中的blacklist一致）
        system_func_names = ['__libc_csu_init', '__libc_csu_fini', '__i686.get_pc_thunk.bx', 
                            '__do_global_ctors_aux', '_start', '__do_global_dtors_aux', 'frame_dummy',
                            'deregister_tm_clones', 'register_tm_clones']
        
        # 从fl文件中读取系统函数地址范围
        system_func_ranges = []  # [(start_addr, end_addr), ...]
        func_list = []  # [(addr, name), ...]
        
        if os.path.isfile('fl'):
            with open('fl') as f:
                for line in f:
                    # 格式: "00000000000181f0 <__libc_csu_init>:"
                    if ' <' in line and '>:' in line:
                        parts = line.split(' <')
                        if len(parts) == 2:
                            addr_str = parts[0].strip()
                            name_part = parts[1].split('>')[0]
                            try:
                                addr = int(addr_str, 16)
                                func_list.append((addr, name_part))
                            except ValueError:
                                pass
        
        # 按地址排序，确定每个系统函数的结束地址（下一个函数的开始地址）
        func_list.sort(key=lambda x: x[0])
        
        # 读取.text段结束地址
        text_end = 0
        if os.path.isfile('text_sec.info'):
            with open('text_sec.info') as f:
                l = f.readline()
                items = l.split()
                if len(items) >= 4:
                    text_start = int(items[1], 16)
                    text_size = int(items[3], 16)
                    text_end = text_start + text_size
        
        # 确定系统函数的地址范围
        for i, (addr, name) in enumerate(func_list):
            if name in system_func_names:
                # 确定结束地址：下一个函数的开始地址，或.text段结束
                if i + 1 < len(func_list):
                    end_addr = func_list[i + 1][0]
                else:
                    end_addr = text_end if text_end > 0 else addr + 0x1000  # 默认1KB范围
                system_func_ranges.append((addr, end_addr))
        
        if not system_func_ranges:
            return
        
        # 读取instrs.info，过滤掉系统函数地址范围内的指令
        filtered_lines = []
        with open('instrs.info', 'r') as f:
            for line in f:
                if ':' in line:
                    addr_str = line.split(':')[0].strip()
                    try:
                        addr = int(addr_str, 16)
                        # 检查是否在系统函数地址范围内
                        in_range = False
                        for start_addr, end_addr in system_func_ranges:
                            if start_addr <= addr < end_addr:
                                in_range = True
                                break
                        if not in_range:
                            filtered_lines.append(line)
                    except ValueError:
                        filtered_lines.append(line)
                else:
                    filtered_lines.append(line)
        
        # 写回过滤后的内容
        with open('instrs.info', 'w') as f:
            f.writelines(filtered_lines)

    def userFuncProcess(self):
        """
        Dump function symbols
        """
        os.system("cat " + self.file + ".temp | grep \"<\" | grep \">:\" > userfuncs.info")
        os.system("cat fl | grep -v \"<S_0x\" >> userfuncs.info")

    def sectionProcess(self):
        """
        Dump section boundaries
        """
        badsec = '.got.plt' 
        os.system("readelf -SW " + self.file + " | awk \'/data|bss|got/ {print $2,$4,$5,$6} \' | awk \ '$1 != \"" + badsec + "\" {print $1,$2,$3,$4}\' > sections.info")
        os.system("readelf -SW " + self.file + " | awk \'/text/ {print $2,$4,$5,$6} \' > text_sec.info")
        os.system("readelf -SW " + self.file + " | awk \'/init/ {print $2,$4,$5,$6} \' | awk \'$1 != \".init_array\" {print $1,$2,$3,$4}\' > init_sec.info")
        if os.path.isfile('init_array.info'): os.remove('init_array.info')
        os.system(config.objdump + " -s -j .init_array " + self.file + " >> init_array.info 2>&1")
        # Try .plt.sec first (modern ELF), fallback to .plt if .plt.sec doesn't exist
        ret = os.system("readelf -SW " + self.file + " | awk '$2==\".plt.sec\" {print $2,$4,$5,$6}' > plt_sec.info")
        # If .plt.sec doesn't exist or is empty (only contains newline), use .plt instead
        if ret != 0 or os.path.getsize("plt_sec.info") <= 1:
            os.system("readelf -SW " + self.file + " | awk '$2==\".plt\" {print $2,$4,$5,$6}' > plt_sec.info")

    def export_tbl_dump(self):
        """
        Dump global symbols
        """
        os.system("readelf -s " + self.file + " | grep GLOBAL > export_tbl.info")

    def pltProcess(self):
        """
        Dump plt section
        """
        # Try .plt.sec first (modern ELF), fallback to .plt if needed
        cmd = config.objdump + " -j .plt.sec -Dr " + self.file + " | grep \">:\" > plts.info"
        ret = os.system(cmd)
        # If .plt.sec doesn't exist or is empty, try .plt
        if ret != 0 or os.path.getsize("plts.info") == 0:
            logger.warning("[init.py:pltProcess]: .plt.sec section not found, trying .plt section")
            cmd = config.objdump + " -j .plt -Dr " + self.file + " | grep \">:\" > plts.info"
            os.system(cmd)
        logger.debug("[init.py:pltProcess]: pltcmd = {}".format(cmd))

    def ailProcess(self, instrument=False,specific_function=None):
        """
        Invoke processing skeleton
        :param instrument: True to apply instrumentations
        """
        logger.info("[init.py:ailProcess]: start ailProcess ...")
        logger.debug("[init.py:ailProcess]: self.file = {}".format(self.file))
        logger.debug("[init.py:ailProcess]: instrument = {}, specific_function = {}".format(instrument, specific_function))
        processor = ail.Ail(self.file)
        processor.sections()    # load section info
        processor.userfuncs()   # load function symbols
        processor.global_bss()  # load global bss symbols
        # try to construct control flow graph and call graph
        # which can help to obfuscating process
        processor.instrProcess(instrument, docfg=True,specific_function=specific_function)
        logger.info("[init.py:ailProcess]: processor.instrProcess function done ...")

    def checkret(self, ret, path):
        """
        Check return of dump operation
        :param ret: shell return code
        :param path: dump file path
        """
        if ret != 0 and os.path.isfile(path):
            os.remove(path)


def main(filepath, instrument=False,specific_function=None):
    """
    Init processing
    :param filepath: path to executable
    :param instrument: True to apply instrumentation
    """
    logger.info("[init.py:main]: start the init.main...")
    if ELF_utils.elf_strip() and ELF_utils.elf_exe():
        init = Init(filepath)
        init.disassemble()
        init.process()
        init.ailProcess(instrument,specific_function=specific_function)
    else:
        logger.error("[init.py:main]: binary is not stripped or is a shared library")
        sys.stderr.write('Error: binary is not stripped or is a shared library\n')
    
    logger.info("[init.py:main]: end the init.main...")
