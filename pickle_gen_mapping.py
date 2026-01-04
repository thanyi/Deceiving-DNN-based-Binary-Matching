# -*- coding: utf-8 -*-
import pandas as pd 
import pickle 
import copy 
import ast 
import glob 
import os 
import logging
logger = logging.getLogger(__name__)

def normalize_addr(addr):
    """规范化地址格式：去除前导零 (0x0000000004049D6 -> 0x4049D6)"""
    if isinstance(addr, str) and addr.startswith('0x'):
        return '0x' + hex(int(addr, 16))[2:].upper()
    return addr

def gen_seed(s):
    '''
    s = mapping_dict = [new_symbol_to_new_addr,
                        new_addr_to_new_symbol,
                        new_symbol_to_seed_symbol,
                        seed_symbol_to_new_symbol,
                        new_symbol_to_old_addr,
                        old_addr_to_new_symbol]
    '''
    sym_to_addr = {}
    sym_to_cur_sym = {}
    logger.debug("[pickle_gen_mapping.py:gen_seed]: s[3] = {}".format(s[3]))
    for symbol in s[3].keys():
        try:
            new_symbol = s[3][symbol]
            if new_symbol not in s[0]:
                logger.warning("[pickle_gen_mapping.py:gen_seed]: Symbol '{}' maps to '{}' but not found in s[0]. Skipping.".format(symbol, new_symbol))
                continue
            addr = s[0][new_symbol].upper().replace('X','x')
            addr = normalize_addr(addr)  # ✅ 规范化地址格式
            sym_to_addr[symbol] = addr
            sym_to_cur_sym[symbol] = new_symbol
        except KeyError as e:
            logger.warning("[pickle_gen_mapping.py:gen_seed]: KeyError for symbol '{}': {}. Skipping.".format(symbol, str(e)))
            continue
        except Exception as e:
            logger.warning("[pickle_gen_mapping.py:gen_seed]: Exception for symbol '{}': {}. Skipping.".format(symbol, str(e)))
            continue
    return sym_to_addr,sym_to_cur_sym

def gen_mutated(sym_to_addr,sym_to_cur_sym,failure_seed,s):
    """
    正确的映射逻辑:
    1. 从 sym_to_cur_sym 获取上一代的 cur_sym (如 'BB_179')
    2. 在 s[2]  中查找对应的 seed_symbol (地址)
    3. 在 s[3]  中查找新的 new_symbol
    4. 在 s[0]  中查找新地址
    """
    failure = []
    logger.debug("[pickle_gen_mapping.py:gen_mutated]: sym_to_addr = {}".format(sym_to_addr))
    logger.debug("[pickle_gen_mapping.py:gen_mutated]: sym_to_cur_sym = {}".format(sym_to_cur_sym))
    logger.debug("[pickle_gen_mapping.py:gen_mutated]: failure_seed = {}".format(failure_seed))
    logger.debug("[pickle_gen_mapping.py:gen_mutated]: s[0] sample = {}".format(list(s[0])))
    logger.debug("[pickle_gen_mapping.py:gen_mutated]: s[3] sample = {}".format(list(s[3])))
    logger.debug("[pickle_gen_mapping.py:gen_mutated]: s[3] total keys = {}".format(len(s[3])))
    
    for symbol in sym_to_addr.keys():
        old_addr = sym_to_addr[symbol]
        # ✅ 规范化地址格式：去除前导零 (0x0000000004049D6 -> 0x4049D6)
        old_addr = normalize_addr(old_addr)
        logger.debug("[pickle_gen_mapping.py:gen_mutated]: old_addr = {} (normalized)".format(old_addr))
        try:
            new_symbol = s[3][old_addr]
            addr = s[0][new_symbol].upper().replace('X','x')
            addr = normalize_addr(addr)  # ✅ 规范化新地址格式
            sym_to_addr[symbol] = addr
            sym_to_cur_sym[symbol] = new_symbol
            logger.debug("[pickle_gen_mapping.py:gen_mutated]: SUCCESS: {} => {} => {}".format(symbol, old_addr, new_symbol))
        except Exception as e:
            failure.append(symbol)
            logger.warning("[pickle_gen_mapping.py:gen_mutated]: FAILURE: {} (old_addr={}) - {}".format(symbol, old_addr, str(e)))
    for symbol in failure:
        tmp = sym_to_addr.pop(symbol, None)
        tmp = sym_to_cur_sym.pop(symbol, None)
    failure += failure_seed
    return sym_to_addr,sym_to_cur_sym,failure


def run_pickle(folder,step_size = None):
    if step_size == None:
        df = pd.read_csv(folder+'/record.csv')
    else:
        df = pd.read_csv(folder+'/record_step_'+str(step_size)+'.csv')

    chk = sorted(list(set(df['generation'])))

    df['mapping_addr'] = 1
    df['mapping_symm'] = 1
    df['failure'] = 1

    for gen in chk:
        disappeared = copy.copy(df[df['generation']==gen])
        damaged_index = list(disappeared.index)
        for i in range(len(disappeared)):
            path = disappeared.iloc[i]['output']
            exe = "/".join(path.split("/")[:-1])
            #exe = exe.replace("/container/","/container_cat/")
            s = pickle.load(open(exe+'/mapping_dict.pickle','rb'))
            if gen==1:
                sym_to_addr = {}
                sym_to_cur_sym = {}
                for symbol in s[3].keys():
                    try:
                        new_symbol = s[3][symbol]
                        if new_symbol not in s[0]:
                            logger.warning("[pickle_gen_mapping.py:run_pickle]: Symbol '{}' maps to '{}' but not found in s[0]. Skipping.".format(symbol, new_symbol))
                            continue
                        addr = s[0][new_symbol].upper().replace('X','x').replace('0x0','0x')
                        sym_to_addr[symbol] = addr
                        sym_to_cur_sym[symbol] = new_symbol
                    except KeyError as e:
                        logger.warning("[pickle_gen_mapping.py:run_pickle]: KeyError for symbol '{}': {}. Skipping.".format(symbol, str(e)))
                        continue
                    except Exception as e:
                        logger.warning("[pickle_gen_mapping.py:run_pickle]: Exception for symbol '{}': {}. Skipping.".format(symbol, str(e)))
                        continue
                df['mapping_addr'].iloc[damaged_index[i]] = str(sym_to_addr)
                df['mapping_symm'].iloc[damaged_index[i]] = str(sym_to_cur_sym)
                df['failure'].iloc[damaged_index[i]] = str([])
            else:
                get_seed = disappeared.iloc[i]['seed']
                sym_to_addr = ast.literal_eval(df[df['output']==get_seed]['mapping_addr'].values[0])
                sym_to_cur_sym = ast.literal_eval(df[df['output']==get_seed]['mapping_symm'].values[0])
                failure_seed = ast.literal_eval(df[df['output']==get_seed]['failure'].values[0])
                failure = []
                for symbol in sym_to_addr.keys():
                    old_addr = sym_to_addr[symbol]
                    try:
                        new_symbol = s[3][old_addr]
                        addr = s[0][new_symbol].upper().replace('X','x').replace('0x0','0x')
                        sym_to_addr[symbol] = addr
                        sym_to_cur_sym[symbol] = new_symbol
                    except:
                        failure.append(symbol)
                for symbol in failure:
                    tmp = sym_to_addr.pop(symbol, None)
                    tmp = sym_to_cur_sym.pop(symbol, None)
                failure += failure_seed
                df['mapping_addr'].iloc[damaged_index[i]] = str(sym_to_addr)
                df['mapping_symm'].iloc[damaged_index[i]] = str(sym_to_cur_sym)
                df['failure'].iloc[damaged_index[i]] = str(failure)
    if step_size == None:
        df.to_csv(folder+'/record_pickle.csv',index=False)
    else:
        df.to_csv(folder+'/record_pickle_'+str(step_size)+'.csv',index=False)


if __name__ == "__main__":
    ssssss = glob.glob("*")
    fail_list = []
    ok = []
    for gg in ssssss:
        if os.path.exists(gg+"/record.csv") == True and os.path.exists(gg+"/record_pickle.csv") == False:
            try:
                run_pickle(gg)
                ok.append(gg)
            except:
                fail_list.append(gg)

    print("[+] Failed : ")
    print(fail_list)

    print("[+] OK : ")
    for xxxx in ok:
        print(xxxx)
