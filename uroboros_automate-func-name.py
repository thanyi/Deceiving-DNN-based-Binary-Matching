# -*- coding: utf-8 -*-
"""
Main module
"""

import glob
import os
import shutil
import sys
from argparse import ArgumentParser, RawTextHelpFormatter
import logging
from termcolor import colored
import fileinput
import config
import pickle 
import gen_address_mapping
import pickle_gen_mapping

_script_dir = os.path.dirname(os.path.abspath(__file__))
_default_log_path = os.path.join(_script_dir, 'log', 'uroboro.log')
log_file_path = os.environ.get('UROBORO_LOG_PATH', _default_log_path)
_log_dir = os.path.dirname(log_file_path)
if _log_dir and not os.path.exists(_log_dir):
    os.makedirs(_log_dir)
# 【日志优化】将级别从 DEBUG 改为 INFO，减少冗余日志输出
logging.basicConfig(filename=log_file_path,
                    level=logging.INFO,  # 从 DEBUG 改为 INFO
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='a') # 'a' for append, 'w' for overwrite
logger = logging.getLogger(__name__)

def open_path(path):
    if not os.path.exists(path):
        os.makedirs(path)



def fix_final_s(filepath):
    _main_globl_added = False
    for line in fileinput.input(filepath, inplace=True):
        # line = line.replace(' w ', ' nopw ') 
        # line = line.replace(' l ', ' nopl ') 
        # line = line.replace('notrack %ra', 'notrack jmpq *%rax')
        # line = line.replace('notrack %rb', 'notrack jmpq *%rbx')
        # line = line.replace('notrack %rc', 'notrack jmpq *%rcx')
        # line = line.replace('notrack %rd', 'notrack jmpq *%rdx')

        if "main :" in line and not _main_globl_added:
            sys.stdout.write(".globl main\n")
            _main_globl_added = True 
        sys.stdout.write(line)

def process(filepath, instrument=False, fexclude='',specific_function=None,target_addr=None):
    """
    Start file processing
    :param filepath: path to executable
    :param instrument: True to apply instrumentation
    :param fexclude: path to file of symbol exclusions
    :param specific_function: diversify specific function
    :param target_addr: target address of one of the top critical blocks
    :return: True if everything ok
    """
    import init
    import traceback
    from postprocess import compile_process
    from disasm import main_discover, func_addr
    logger.info("[uroboros_automate-func-name.py:process]: process function start ... ")
    print "Starting to process binary '" + filepath + "'"
    try:
        
        func_addr.func_addr(filepath, 1, fexclude = "useless_func_key.info")  # count = 1可以开启黑名单

        logger.info("[uroboros_automate-func-name.py:process]: finished dump.s, faddr.txt, faddr_old.txt")
        logger.debug("[uroboros_automate-func-name.py:process]: filepath = {}".format(filepath))
        
        main_discover.main_discover(filepath)
        logger.info("[uroboros_automate-func-name.py:process]: main_discover function done ...")
        os.system(config.strip + ' ' + filepath)
        os.system('file ' + filepath + ' > elf.info')

        init.main(filepath, instrument,specific_function = specific_function,target_addr=target_addr)
        logger.info("[uroboros_automate-func-name.py:process]: init.main function done ...")
        if not os.path.isfile("final.s"): 
            logger.info("[uroboros_automate-func-name.py:process]: write final_data.s failed!")
            return False

        with open('final_data.s', 'a') as f:
            f.write('\n.section .eh_frame\n')
            with open('eh_frame_split.info') as eh: f.write(eh.read())
            f.write('\n.section .eh_frame_hdr\n')
            with open('eh_frame_hdr_split.info') as eh: f.write(eh.read())
        with open('final.s', 'a') as f:
            with open('final_data.s', 'r') as fd: f.write(fd.read())
            if instrument: f.write('\n\n'.join(map(lambda e: e['plain'].instrdata, config.instrumentors)))
        logger.info("[uroboros_automate-func-name.py:process]: write final_data.s done!")
        logger.info("[uroboros_automate-func-name.py:process]: compile_process.main start!")

        compile_process.main(filepath)
        logger.info("[uroboros_automate-func-name.py:process]: compile_process.main done!")
        if instrument:
            for worker in config.instrumentors:
                worker['main'].aftercompile()
        if compile_process.reassemble() != 0: 
            logger.error("[uroboros_automate-func-name.py:process]: compile_process.reassemble() != 0, return False...")
            return False

    except Exception as e:
        print e
        traceback.print_exc()
        logger.error("[uroboros_automate-func-name.py:process]: %s.", e)
        return False
    logger.info("[uroboros_automate-func-name.py:process]: process function end ... ")
    return True


def check(filepath, assumptions, gccopt='', excludedata='', instrument=False):
    """
    Perform basic check on analyzed executable and set configuration values
    :param filepath: path to executable
    :param assumptions: list of assumption codes
    :param gccopt: additional options for the compiler
    :param excludedata: path to file of address exclusions
    :param instrument: True if instrumentation enabled
    :return: True if everything ok
    """
    if not assumptions: assumptions = []
    logger.debug('[uroboros_automate-func-name.py:check]:filepath = {}, assumptions = {}, gccopt = {}'.format(filepath, assumptions, gccopt))   # the mutated bin path
    if not os.path.isfile(filepath):
        sys.stderr.write("Cannot find input binary\n")
        return False

    if len(excludedata) != 0 and not os.path.isfile(excludedata):
        sys.stderr.write("File with exclusions not found\n")
        return False

    for f in glob.glob('*'): os.remove(f)

    if os.path.dirname(filepath) != os.getcwd():
        shutil.copy(filepath, '.')

    os.system('file ' + filepath + ' > elf.info')
    config.setup(filepath, gccopt, excludedata, instrument)

    if config.is_lib:
        sys.stderr.write("Uroboros doesn't support shared libraries\n")
        logger.error('binary is stripped, function boundaries evaluation may not be precise')
        return False

    # if assumption three is utilized, then input binary should be unstripped.
    if ('3' in assumptions or instrument) and not config.is_unstrip:
        # print colored('Warning:', 'yellow'), 'binary is stripped, function boundaries evaluation may not be precise'
        logger.warning('binary is stripped, function boundaries evaluation may not be precise')

    return True


def set_assumption (assumptions):
    """
    Save assumptions to file
    2 -> assumption two: fix data section starting address
    3 -> assumption three: function starting address + jump table
    :param assumptions: list of assumptions codes
    :return: True if everything ok
    """
    if not assumptions:
        with open('assumption_set.info', 'w') as f:
            f.write('1\n')
    else:
        chk = (i in ['2', '3'] for i in assumptions)
        if any(chk) == False:
            print "assumption undefined!"
            print "accepted assumptions: 2 for assumption two and 3 for assumption three"
            return False
        with open('assumption_set.info', 'w') as f:
            f.write(' '.join(assumptions) + '\n')
    return True


def set_diversification(d, iter):
    if d not in config.all_diver_modes:
        assert False, 'unrecognizible diversification mode %d for iteration %d' % (d, iter)
    config.diversification_mode = d
    print 'set diversification mode %d for iteration %d' % (d, iter)
    os.system('echo %d > mode' % d)


def reset_diversification():
    config.diversification_mode = 0


def main():
    """
    Main function
    """
    p = ArgumentParser(formatter_class=RawTextHelpFormatter)
    p.add_argument("binary", help="path to the input binary")
    p.add_argument("-o", "--output", help="destination output file")
    p.add_argument("-g", "--instrument", action='store_true', help="apply instrumentations to output")
    p.add_argument("-a", "--assumption", action="append",
                   help='''this option configures three addtional assumption,
note that two basic assumptions and addtional assumption one
(n-byte alignment) are set by default,
while assumption two and three need to be configured. For example, setting
assumption two and three: -a 2 -a 3''')
    p.add_argument("-gcc", "--gccopt", action="store", default="", help="A string of additional arguments for GCC")
    p.add_argument("-ex", "--exclude", default="", help="""File where each line is either a single value which must not be
a label or an address range of data section to exclude from symbol search""")
    p.add_argument("-fex", "--functionexclude", default="", help="File with a list of symbols not representing functions")
    p.add_argument("--version", action="version", version="Uroboros 0.2b")

    p.add_argument('-i', '--iteration', type=int, help='the number of disassemble-diversify-reassemble iterations')
    p.add_argument('-d', '--diversification', type=int, action='append',
                   help='the sequence of diversification for every iteration \n' + config.diversification_description)
    
    p.add_argument('-m', '--mode',default="mutated" ,
                   help='original or mutated ')

    p.add_argument('-f', '--folder', required=True ,
                   help='Folder to save everything,must be absolute path')

    p.add_argument('-div', '--function', type=str ,default="",nargs='+',
                   help='Function to modify')
    p.add_argument('-t', '--target_addr', type=str, default="", help='target address of one of the top critical blocks')

    args = p.parse_args()
    args.folder = os.path.realpath(args.folder)
    filepath = os.path.realpath(args.binary)
    outpath = os.path.realpath(args.output) if args.output is not None else None
    exclude = os.path.realpath(args.exclude) if len(args.exclude) > 0 else ''
    fexclude = os.path.realpath(args.functionexclude) if len(args.functionexclude) > 0 else ''

    # 【日志优化】注释掉详细的参数日志
    # logger.debug("[uroboros_automate-func-name.py:main]: outpath = {},".format(outpath))
    num_iteration = args.iteration
    if num_iteration is None or num_iteration <= 0:
        num_iteration = 1

    diversifications = args.diversification
    if diversifications is None:
        diversifications = [0]
    while len(diversifications) < num_iteration:
        diversifications.append(diversifications[-1])
    logger.info("[uroboros_automate-func-name.py:main]: diversifications = {}".format(diversifications))
    abs_path = os.path.dirname(os.path.abspath(__file__))
    open_path(args.folder)
    # 【日志优化】注释掉详细的参数日志
    # logger.debug("[uroboros_automate-func-name.py:main]: args.binary = {}," \
    #                         " args.mode = {}, args.function = {}".format(args.binary, args.mode, args.function))
    # Prefer per-run workdir under args.folder to avoid collisions across experiments.
    work_root = os.environ.get('UROBOROS_WORKDIR_ROOT', args.folder)
    work_root = os.path.realpath(work_root)
    if not os.path.isabs(work_root):
        work_root = os.path.abspath(work_root)
    open_path(work_root)
    for i in range(1, num_iteration + 1):
        workdir = os.path.join(work_root, 'workdir_' + str(i))
        if not os.path.isdir(workdir):
            os.mkdir(workdir)
        # 【日志优化】注释掉工作目录日志
        # logger.debug("[uroboros_automate-func-name.py:main]: workdir= {}".format(workdir))
        # print colored(('iteration %d dir:' + workdir) % i, 'green')
        # logger.info(('iteration %d dir:' + workdir) % i, 'green')
        os.chdir(workdir)

        # run after setting the work directory
        set_diversification(diversifications[i - 1], i)
        sym_to_addr = {}
        sym_to_cur_sym = {}
        failure_seed = []
        failure = []
        if args.mode == "mutated" :
            # resolve the folder
            gg = args.binary
            gg = "/".join(gg.split("/")[:-1])
            # 设置环境变量，让main_discover能够找到unstriped.out
            os.environ['UROBOROS_FOLDER'] = gg
            # 【日志优化】注释掉 pickle 路径日志
            # logger.debug("[uroboros_automate-func-name.py:main]: gg+'/sym_to_addr.pickle' = {},".format(gg+'/sym_to_addr.pickle'))

            sym_to_addr = pickle.load(open(gg+'/sym_to_addr.pickle','rb'))
            sym_to_cur_sym = pickle.load(open(gg+'/sym_to_cur_sym.pickle','rb'))
            failure_seed = pickle.load(open(gg+'/failure.pickle','rb'))
        
        '''
        sym_to_addr : 
        {'set_custom_quoting': '0x8056CBB', 'cp_option_init': '0x804B539
        
        ali.py : 
        S_0x8051BAC@0x8051BAC-0x8051BE3 OR dir_len@0x8053B72-0x8053BE3, mdir_name@0x8053BE3-0x8053C65
        '''
        specific_function = None 
        if len(args.function) > 0 and args.mode=='mutated' :
            specific_function = []
            func_list = args.function
            # 【日志优化】sym_to_addr 可能很大，注释掉
            # logger.info("[uroboros_automate-func-name.py:main]: sym_to_addr = {}".format(sym_to_addr))
            for function_name in func_list:
                addr_tmp = ""
                if function_name in sym_to_addr.keys():
                    addr_tmp = "S_"+sym_to_addr[function_name]
                    # name, address structure 
                    specific_function.append([function_name,addr_tmp])
            if len(specific_function) == 0:
                logger.error("[uroboros_automate-func-name.py:main]: No functions found in sym_to_addr and cannot find from binary. sym_to_addr keys: {}".format(list(sym_to_addr.keys())))
                raise Exception('Fail: Cannot find function(s) {} in sym_to_addr or binary'.format(func_list))

        elif len(args.function) > 0 and args.mode=='original':
            specific_function = []
            func_list = args.function
            for function_name in func_list:
                addr_tmp = "xxxxsidshidhafafaoja" # random junk 
                specific_function.append([function_name,addr_tmp])
            if len(specific_function) == 0:
                raise Exception('Fail')
        # 【日志优化】specific_function 已在上方打印，这里注释掉
        # logger.debug("[uroboros_automate-func-name.py:main]: specific_function = {},".format(specific_function))

        if check(filepath, args.assumption, args.gccopt, exclude, args.instrument) and set_assumption(args.assumption):
            # process(filepath, instrument=False, fexclude='',specific_function=None)
            if process(os.path.basename(filepath), args.instrument, fexclude = fexclude,specific_function=specific_function,target_addr=args.target_addr):
                # this will parse and return the pickled object 
                logger.info("[uroboros_automate-func-name.py:main]: process func returns True !")
                '''
                mapping_dict = [new_symbol_to_new_addr,
                                new_addr_to_new_symbol,
                                new_symbol_to_seed_symbol,
                                seed_symbol_to_new_symbol,
                                new_symbol_to_old_addr,
                                old_addr_to_new_symbol]
                '''
                mapping_dict = gen_address_mapping.gen_mapping(args.mode)
                # 【日志优化】注释掉 folder 路径日志
                # logger.debug("[uroboros_automate-func-name.py:main]]: args.folder is {}".format(args.folder))
                with open(args.folder+'/mapping_dict.pickle', 'wb') as handle:
                    pickle.dump(mapping_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                # 【日志优化】mapping_dict 可能很大，只记录长度
                logger.info("[uroboros_automate-func-name.py:main]: write mapping_dict ({} entries) in folder {}".format(len(mapping_dict) if isinstance(mapping_dict, (list, dict)) else 'N/A', args.folder))
                if args.mode == "original" :
                    sym_to_addr,sym_to_cur_sym = pickle_gen_mapping.gen_seed(mapping_dict)
                else:
                    sym_to_addr,sym_to_cur_sym,failure = pickle_gen_mapping.gen_mutated(sym_to_addr,sym_to_cur_sym,failure_seed,mapping_dict)
                
                # 【日志优化】sym_to_addr 可能有数千个条目，只记录数量
                logger.info("[uroboros_automate-func-name.py:main]: sym_to_addr contains {} symbols".format(len(sym_to_addr)))
                with open(args.folder+'/sym_to_addr.pickle', 'wb') as handle:
                    pickle.dump(sym_to_addr, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(args.folder+'/sym_to_cur_sym.pickle', 'wb') as handle:
                    pickle.dump(sym_to_cur_sym, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(args.folder+'/failure.pickle', 'wb') as handle:
                    pickle.dump(failure, handle, protocol=pickle.HIGHEST_PROTOCOL)

                os.system('cp a.out '+args.folder+"/unstriped.out")
                os.system('cp final.s '+args.folder+"/final.s")
                os.system('strip a.out')
                print colored("Processing %d succeeded" % i, "blue")
                if outpath is not None:
                    # 确保目标目录存在
                    outdir = os.path.dirname(outpath)
                    if outdir and not os.path.exists(outdir):
                        os.makedirs(outdir)
                    shutil.copy('a.out', outpath + '_' + str(i) + '_mode_' + str(config.diversification_mode))
                    shutil.copy('a.out', outpath)
                # change source file path
                filepath = workdir + '/a.out'
            else:
                print colored("Processing %d failed" % i, "red")
                exit(1)
        reset_diversification()


if __name__ == "__main__":
    logger.info('[uroboros_automate-func-name.main]: ============================================================ uroboros start ============================================================')
    main()
    logger.info('[uroboros_automate-func-name.main]: ============================================================ uroboros end ============================================================')
