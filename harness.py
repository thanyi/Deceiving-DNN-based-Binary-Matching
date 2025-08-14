"""
用于欺骗DNN二进制匹配系统的主程序
包含了变异生成、评估和优化的核心功能
"""
import subprocess
from subprocess import check_output
import random 
import os
import string
import datetime
import random 
import pandas as pd 
import hashlib
import glob 
import r2_cfg
import run_objdump
import pickle 
from pickle_gen_mapping import * 
import ast 
from argparse import ArgumentParser, RawTextHelpFormatter
import time
from loguru import logger
# from run_utils import run_one, train_pickle

SURVIVIE = 1 
TARGET = 0.40 
mutant = {}
tmp_bin_name = "/tmp/output_bin_"+str(random.randint(0,300000))

# st_time = None 

def open_path(path):
    """
    创建目录函数
    如果目录不存在则创建该目录
    
    参数:
        path: 需要创建的目录路径
    """
    if not os.path.exists(path):
        os.makedirs(path)

def generate(seed,save_bin_folder,mode,F_mode,function_name,tmp_bin = tmp_bin_name,iter=1):
    """
    生成变异二进制文件的函数
    
    参数:
        seed: 种子二进制文件
        save_bin_folder: 保存生成文件的目录
        mode: 变异模式
        F_mode: 功能模式(original/mutated)
        function_name: 目标函数名
        tmp_bin: 临时二进制文件名
        iter: 迭代次数
    
    返回:
        生成的变异二进制文件的路径
    """
    logger.debug(f'generate start: seed = {seed}, mode = {mode}, F_mode = {F_mode}, tmp_bin = {tmp_bin}')
    p = ['python2','./uroboros_automate-func-name.py',seed,'-i',str(iter),'-o',tmp_bin,'-d',str(mode),'-m',F_mode,'-f',save_bin_folder+"/tmp/","--function",function_name]
    logger.debug("uroboros cmd is :"+" ".join(p))
    s = check_output(p) # 执行命令，生成变异二进制文件
    logger.info('check_output function runs successfully!')
    h = hashlib.md5(open(tmp_bin, 'rb').read()).hexdigest() # 计算变异二进制文件的hash值
    ctr = 1 
    if not os.path.exists(save_bin_folder+"/"+str(h)+"_container/"):
        os.system("mv "+save_bin_folder+"/tmp/"+" "+save_bin_folder+"/"+str(h)+"_container/")
        os.system("mv "+tmp_bin+" "+save_bin_folder+"/"+str(h)+"_container/"+str(h))
        return save_bin_folder+"/"+str(h)+"_container/"+str(h)
    else:
        saved_flag = False 
        while saved_flag == False:
            h_tmp = str(h)+"_ctr"+str(ctr)
            if not os.path.exists(save_bin_folder+"/"+str(h_tmp)+"_container/"):
                os.system("mv "+save_bin_folder+"/tmp/"+" "+save_bin_folder+"/"+str(h_tmp)+"_container/")
                os.system("mv "+tmp_bin+" "+save_bin_folder+"/"+str(h_tmp)+"_container/"+str(h_tmp))
                saved_flag == True 
                return save_bin_folder+"/"+str(h_tmp)+"_container/"+str(h_tmp)
            else:
                ctr += 1 


def ret_top5(gen_1,score_list,grad_list,SAVE_PATH,STEP_SIZE,metric = 'grad_list'):
    """
    选择并返回得分最好的变异样本
    
    参数:
        gen_1: 当前代的变异样本列表
        score_list: 得分列表
        grad_list: 梯度列表
        SAVE_PATH: 日志保存路径
        STEP_SIZE: 步长大小
        metric: 评估指标('grad_list'或'score_list')
    
    返回:
        筛选后的变异样本、得分和梯度
    """
    df_tmp = pd.DataFrame()
    df_tmp['gen_1'] = gen_1
    df_tmp['score_list'] = score_list
    df_tmp['grad_list'] = grad_list
    if metric == 'score_list':
        df_tmp = df_tmp.sort_values(metric,ascending=True)
    else:
        df_tmp = df_tmp.sort_values(metric,ascending=False)
    if os.path.exists(SAVE_PATH+'/big_log_'+str(STEP_SIZE)+'.csv'):
        logging_df = pd.read_csv(SAVE_PATH+'/big_log_'+str(STEP_SIZE)+'.csv')
        logging_df = logging_df.append(df_tmp)
        logging_df.to_csv(SAVE_PATH+'/big_log_'+str(STEP_SIZE)+'.csv', index=False )
    else:
        df_tmp.to_csv(SAVE_PATH+'/big_log_'+str(STEP_SIZE)+'.csv', index=False )
    gen_1 = df_tmp['gen_1'].values[:SURVIVIE]
    score_list = df_tmp['score_list'].values[:SURVIVIE]
    grad_list = df_tmp['grad_list'].values[:SURVIVIE]
    print(df_tmp)
    df_tmp.to_csv(SAVE_PATH+'/log_'+str(STEP_SIZE)+'.csv')
    return gen_1,score_list,grad_list


def wrapper(MAIN_SEED,SAVE_PATH,function_name):
    """
    主要的执行流程控制函数
    
    实现了整个变异和评估的流程:
    1. 加载原始二进制的模型
    2. 生成初始变异样本集
    3. 迭代优化变异样本
    4. 评估变异效果
    
    参数:
        MAIN_SEED: 原始二进制文件路径
        SAVE_PATH: 结果保存路径
        function_name: 目标函数名
    """
    # 初始化存储列表
    seed_corpus = []      # 存储所有生成的变异样本
    seed_of_output = []   # 存储每个变异的源种子
    output = []          # 存储生成的变异输出
    generation = []      # 存储每个变异的代数
    operand = []         # 存储每个变异使用的操作

    open_path(SAVE_PATH)

    # 定义可用的变异模式
    # no use 0 , 4, 10 
    diversification = [1,2,3,4,5,7,8,9,10,11]

    # 检查是否已经找到成功的对抗样本
    if os.path.isfile(SAVE_PATH+"/bypassed.log") == True:
        print("Done :)")
        exit() 

    if os.path.isfile(MAIN_SEED+".s") == False:
        logger.info(f'routine start... binary ={MAIN_SEED} ')
        seed_s = run_objdump.routine(MAIN_SEED)

    # train seed's model 
    if True: #os.path.isfile(MAIN_SEED+".pickle"):
        # we gen the matching model first 
        #model_original = pickle.load(open(MAIN_SEED+".pickle",'rb'))
        model_original = pickle.load(open("/home/ycy/ours/Deceiving-DNN-based-Binary-Matching/gnu.pickle",'rb'))
    else:
        model_original = train_pickle(MAIN_SEED+".s")
        with open(MAIN_SEED+".pickle", 'wb') as handle:
            pickle.dump(model_original, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info('model_original loaded')
    # every time gen 10 
    # 

    def main(function_name,NUM_GEN):
        """
        主循环函数
        
        工作流程:
        1. 生成初始变异样本集
        2. 评估每个样本的效果(计算score和grad)
        3. 如果找到成功的对抗样本(score < TARGET)，则返回结果
        4. 否则，选择最好的样本继续变异
        5. 重复这个过程直到找到成功的对抗样本或达到迭代次数限制
        
        参数:
            function_name: 目标函数名
            NUM_GEN: 生成的变异样本数量
        
        返回:
            变异过程的记录和成功的对抗样本(如果有)


        关键指标:
        - score: 衡量变异样本与原始样本的相似度，越低越好
        - grad: 梯度值，用于指导变异方向
        - TARGET = 0.40: 目标相似度阈值，低于此值表示成功欺骗DNN系统
        """
        generation_dict = {}
        operand_dict = {}
        gen_1 = []
        for ctr in range(0,1):
            got_sample = False 
            retry_ctr = 0 
            while got_sample == False and retry_ctr <1:
                i = random.choice(diversification)  # 随机选出一种变异策略i
                try:
                    logger.info(f"now the action idx is {i}")
                    # 生成一个新的二进制文件，返回文件hash值
                    hash_ = generate(MAIN_SEED,SAVE_PATH,i,'original',function_name=function_name,iter=1)
                    logger.info('generate a new binary! generate func done.')
                    if len(generation_dict.keys())==0:
                        generation_dict[MAIN_SEED] = 0
                        generation_dict[hash_] = 1
                        operand_dict[MAIN_SEED] = []
                        operand_dict[hash_] = [i]
                        cur_gen = 1
                        cur_op = str(operand_dict[hash_])
                    elif MAIN_SEED in generation_dict.keys():
                        old_gen = generation_dict[MAIN_SEED] +1
                        generation_dict[hash_] = old_gen
                        cur_gen = old_gen
                        old_operand = operand_dict[MAIN_SEED].copy()
                        old_operand.append(i)
                        operand_dict[hash_] = old_operand
                        cur_op = str(operand_dict[hash_])
                    seed_of_output.append(MAIN_SEED)
                    output.append(hash_)
                    generation.append(cur_gen)
                    operand.append(cur_op)
                    seed_corpus.append(hash_)
                    gen_1.append(hash_)
                    got_sample= True 
                except:
                    # print("[+] Failed")
                    logger.error("error occurred in generate ...")
                    retry_ctr+=1 
                    pass 
            if retry_ctr == 5 :
                fs = open(SAVE_PATH+"/fail.log",'w')
                fs.write(str(i))
                fs.close()
                exit()

        df = pd.DataFrame()
        df['seed'] = seed_of_output
        df['output'] = output
        df['generation'] = generation
        df['operand'] = operand
        df.to_csv(SAVE_PATH+'/record_step_'+str(NUM_GEN)+'.csv')
        run_pickle(SAVE_PATH,step_size =NUM_GEN)
        
        logger.info("[+] Finished initial corpus generation ")

        bypassed = False 
        bypassed_sample = ""
        score_list = []
        grad_list = []
        xx = pd.read_csv(SAVE_PATH+'/record_pickle_'+str(NUM_GEN)+'.csv')
        for items in gen_1:
            #get id 
            id_ = xx[xx['output']==items].index.values[0]
            checkdict = ast.literal_eval(xx['mapping_symm'].iloc[id_])
            logger.debug(f'args in run_one function: MAIN_SEED = {MAIN_SEED}, items = {items}, model_original = {model_original}, \
                         checkdict = {checkdict}, function_name = {function_name}')
            score,grad = run_one(MAIN_SEED,items,model_original,checkdict,function_name)
            if score is None or grad is None:
                logger.warning(f"run_one returned None values, skipping sample: {items}")
                continue
            grad = abs(grad)
            score = abs(score)
            # run checker 
            # if works, quit 
            # if score <TARGET:
            #     bypassed = True 
            #     bypassed_sample = items
            #     #break 
            #     return seed_of_output,output,generation,operand,bypassed_sample 
            # elif score!= None :
            score_list.append(score)
            grad_list.append(grad)
        if bypassed == False :
            gen_1,score_list,grad_list = ret_top5(gen_1,score_list,grad_list,SAVE_PATH,NUM_GEN,metric='score_list')
            gen_1= gen_1.tolist()
            score_list= score_list.tolist()
            grad_list = grad_list.tolist()
            gen_1 = [gen_1[0]]
            score_list = [score_list[0]]
            grad_list = [grad_list[0]]
        tmp_gen = []
        if bypassed == False :
            for i in range(0,NUM_GEN):
                cur_gen = 0
                cur_op = ""
                pass_flag = False 
                fail_ctr = 0
                # max items 
                gen_max =  None 
                score_max = 999999999 
                grad_max = -999999999 
                #first we draw one seed from seed files
                while pass_flag == False:
                    if fail_ctr > 0 :
                        print("[+] Trying to escape")
                    seed = gen_1[0]#random.choice(gen_1)
                    mode = random.choice(diversification)
                    iter_ = 1 #random.randint(1,10)
                    #mode = random.randint(0,10)
                    try:
                        if seed == MAIN_SEED:
                            fmode = 'original'
                        else:
                            fmode = 'mutated'
                        hash_ = generate(seed,SAVE_PATH,mode,fmode,function_name=function_name,iter=iter_)
                        # consider the initailization case 
                        if len(generation_dict.keys())==0:
                            generation_dict[seed] = 0
                            generation_dict[hash_] = 1
                            operand_dict[seed] = []
                            operand_dict[hash_] = [mode]*iter_
                            cur_gen = 1
                            cur_op = str(operand_dict[hash_])
                        elif seed in generation_dict.keys():
                            old_gen = generation_dict[seed] +1
                            generation_dict[hash_] = old_gen
                            cur_gen = old_gen
                            old_operand = operand_dict[seed].copy()
                            gg = [mode]*iter_
                            old_operand +=gg
                            operand_dict[hash_] = old_operand
                            cur_op = str(operand_dict[hash_])
                        seed_of_output.append(seed)
                        output.append(hash_)
                        generation.append(cur_gen)
                        operand.append(cur_op)
                        seed_corpus.append(hash_)
                        #tmp_gen.append(hash_)
                        # One by one 
                        # do evaluation 
                        df = pd.DataFrame()
                        df['seed'] = seed_of_output
                        df['output'] = output
                        df['generation'] = generation
                        df['operand'] = operand
                        df.to_csv(SAVE_PATH+'/record_step_'+str(NUM_GEN)+'.csv')
                        # update the old 
                        #                     
                        run_pickle(SAVE_PATH,step_size =NUM_GEN)
                        xx = pd.read_csv(SAVE_PATH+'/record_pickle_'+str(NUM_GEN)+'.csv')
                        items = hash_
                        #get id 
                        print(items)
                        id_ = xx[xx['output']==items].index.values[0]
                        checkdict = ast.literal_eval(xx['mapping_symm'].iloc[id_])
                        score,grad = run_one(MAIN_SEED,items,model_original,checkdict,function_name)
                        if score is None or grad is None:
                            logger.warning(f"run_one returned None values, skipping sample: {items}")
                            continue
                        grad = abs(grad)
                        score = abs(score)
                        # run checker 
                        # if works, quit 
                        if score <TARGET:
                            bypassed = True 
                            bypassed_sample = items
                            #break 
                            return seed_of_output,output,generation,operand,bypassed_sample 
                        if score!= None :
                            if grad > 0 and pass_flag == False and grad >= grad_list[0]:
                                pass_flag = True 
                                gen_1 = [items]
                                score_list = [score]
                                grad_list = [grad]
                            elif fail_ctr > 10 and grad > 0 :
                                # let it go 
                                if score<score_max:
                                    gen_max = items 
                                    score_max = score
                                    grad_max = grad
                                pass_flag = True 
                                gen_1 = [gen_max]
                                score_list = [score_max]
                                grad_list = [grad_max]
                            else:
                                if grad>0:
                                    if score<score_max:
                                        gen_max = items 
                                        score_max = score
                                        grad_max = grad
                                fail_ctr +=1 
                                score = None
                    except:
                        pass 
                tmp_gen = []
                gen_1,score_list,grad_list = ret_top5(gen_1,score_list,grad_list,SAVE_PATH,NUM_GEN)
                gen_1= gen_1.tolist()
                score_list= score_list.tolist()
                grad_list = grad_list.tolist()
                gen_1 = [gen_1[0]]
                score_list = [score_list[0]]
                grad_list = [grad_list[0]]


        return seed_of_output,output,generation,operand,bypassed_sample 

    # 设置不同的步长进行迭代
    STEP = [10,20,40,50,100,200]
    STEP = [20]
    out_pd = pd.DataFrame()
    
    # 对每个步长执行变异和评估
    for step_size in STEP:
        # 执行主要的变异和评估过程
        seed_of_output,output,generation,operand,bypassed_sample = main(function_name,step_size)
        
        # 记录变异过程
        df = pd.DataFrame()
        df['seed'] = seed_of_output
        df['output'] = output
        df['generation'] = generation
        df['operand'] = operand 
        out_pd = out_pd.append(df)
        
        # 如果找到成功的对抗样本
        if len(bypassed_sample)>0:
            # 保存成功的对抗样本信息
            fs = open(SAVE_PATH+'/bypassed.log','w')
            fs.write(bypassed_sample)
            fs.close()
            
            # 保存完整的变异记录
            out_pd.to_csv(SAVE_PATH+'/record.csv')
            run_pickle(SAVE_PATH)
            
            # 记录执行时间
            global st_time
            end_time = time.time()
            duration = end_time - st_time 
            time_log = open(SAVE_PATH+'/duration.log','w')
            time_log.write(str(duration))
            time_log.close()
            exit()
    



if __name__ == "__main__":

    p = ArgumentParser(formatter_class=RawTextHelpFormatter)
    p.add_argument("-binary_seed", help="name of the binary")   # 原始二进制文件名
    p.add_argument("-function_name", help="name of the targetted function")     # 目标函数名
    args = p.parse_args()

    binary_seed = args.binary_seed
    function_name = args.function_name
    MAIN_SEED = "/home/ycy/ours/Deceiving-DNN-based-Binary-Matching/bin_bk/"+binary_seed
    SAVE_PATH ="/home/ycy/ours/Deceiving-DNN-based-Binary-Matching/function_container_"+function_name+"_"+MAIN_SEED.split("/")[-1]
    global st_time
    st_time = time.time()
    wrapper(MAIN_SEED,SAVE_PATH,function_name)

