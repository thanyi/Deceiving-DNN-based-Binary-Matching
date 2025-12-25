# -*- coding: utf-8 -*-
"""
Discover main function address
"""

import os
import config
from utils.ail_utils import ELF_utils
import logging
logger = logging.getLogger(__name__)

def main_discover(filename):
    """
    Find main function address and store it to file
    :param filename: path to target executable
    """
    os.system('file ' + filename + ' > elf.info')
    if ELF_utils.elf_exe():
        objdmp_cmd = config.objdump + ' -Dr -j .text '+ filename + ' > ' + filename + '.temp'
        logger.debug("[main_discover.py:main_discover]: objdmp cmd = {}".format(objdmp_cmd))
        os.system(objdmp_cmd)

        with open(filename + '.temp') as f:
            lines = f.readlines()

        ll = len(lines)
        main_symbol = ""

        if config.arch == config.ARCH_X86:
            for i in xrange(ll):
                l = lines[i]
                # when not using O2 to compile the original binary, we will remove all the _start code,
                # including the routine attached on the original program. In that case, we can not discover the
                # main function
                if "<__libc_start_main@plt>" in l:
                    main_symbol = lines[i-1].split()[-1] if ELF_utils.elf_32() \
                        else lines[i-1].split()[-1].split(',')[0]
                    if main_symbol == '%eax':
                        # to fit gcc-4.8 -m32, the address is mov to %eax, then push to stack
                        main_symbol = lines[i - 2].split()[-1].split(',')[0].split('0x')[1]
                    else:
                        main_symbol = main_symbol.split('0x')[1]
                    break
                #lines[i-1] = lines[i-1].replace(main_symbol, "main")
                #main_symbol = main_symbol[1:].strip()
                #print main_symbol

            ## Some of the PIC code/module rely on typical pattern to locate
            ## such as:

            ##	804c460: push   %ebx
            ##	804c461: call   804c452 <__i686.get_pc_thunk.bx>
            ##	804c466: add    $0x2b8e,%ebx
            ##	804c46c: sub    $0x18,%esp

            ## What we can do this pattern match `<__i686.get_pc_thunk.bx>` and calculate
            ## the address by plusing 0x2b8e and  0x804c466, which equals to the begin address of GOT.PLT table

            ## symbols can be leveraged in re-assemble are
            ##	_GLOBAL_OFFSET_TABLE_   ==    ** .got.plt **
            ##	....
        elif config.arch == config.ARCH_ARMT:
            ## 1035c:       4803            ldr     r0, [pc, #12]   ; (1036c <_start+0x28>)
            ## 1035e:       4b04            ldr     r3, [pc, #16]   ; (10370 <_start+0x2c>)
            ## 10360:       f7ff efde       blx     10320 <__libc_start_main@plt>
            ## 10364:       f7ff efe8       blx     10338 <abort@plt>
            ## ...
            ## 1036c:       0001052d
            for i in xrange(ll):
                l = lines[i]
                if '<__libc_start_main@plt>' in l:
                    j = i - 1
                    while j > 0:
                        if 'ldr' in lines[j] and 'r0' in lines[j]:
                            pcraddr = lines[j].split(';')[1].strip().split()[0][1:]
                            break
                        j -= 1
                    j = i + 1
                    while j < ll:
                        if lines[j].strip().startswith(pcraddr):
                            main_symbol = lines[j].split()[1]
                            if len(main_symbol) < 8: main_symbol = lines[j+1].split()[1] + main_symbol
                            main_symbol = int(main_symbol.lstrip('0'), 16) & (-2)
                            main_symbol = '%X' % main_symbol
                            break
                        j += 1
                    break
        if not main_symbol: 
            # find <main>: symbol
            for i in xrange(ll):
                l = lines[i]
                if "<main>:" in l: 
                    # i.e : "00000000000026f0"
                    addr_str = l.split('<')[0].strip()
                    # remove '0x' profix and uppercase
                    main_symbol = addr_str.lstrip('0').upper() 
                    if not main_symbol:     # if 0x00000000, lstrip('0') return 0
                        main_symbol = '0'
                    logger.debug("[main_discover.py:main_discover]: main_symbol = {}".format(main_symbol))
                    break
        
        # 如果仍然找不到，尝试从args.folder下的unstriped.out中查找
        if not main_symbol:
            folder = os.environ.get('UROBOROS_FOLDER', '')
            logger.info("[main_discover.py:main_discover]: folder = {}".format(folder))
            if folder:
                unstriped_path = os.path.join(folder, 'unstriped.out')
                logger.info("[main_discover.py:main_discover]: unstriped_path = {}".format(unstriped_path))
                if os.path.isfile(unstriped_path):
                    logger.info("[main_discover.py:main_discover]: unstriped_path is file")
                    os.system(config.objdump + ' -Dr -j .text ' + unstriped_path + ' > ' + unstriped_path + '.temp')
                    try:
                        with open(unstriped_path + '.temp') as f:
                            for l in f:
                                if "<main>:" in l:
                                    addr_str = l.split('<')[0].strip()
                                    main_symbol = addr_str.lstrip('0').upper()
                                    if not main_symbol:
                                        main_symbol = '0'
                                    break
                    except:
                        pass
                    if os.path.isfile(unstriped_path + '.temp'):
                        os.remove(unstriped_path + '.temp')
        
        if config.arch != config.ARCH_X86 and config.arch != config.ARCH_ARMT:
            raise Exception('Unknown arch')

        with open("main.info", 'w') as f:
            f.write('S_0x' + main_symbol.upper() + '\n')
