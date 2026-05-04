from argparse import Namespace
from logging import Logger
import numpy as np
import os
from train import fold_train
#from train_3cl import fold_train
from args import set_train_argument
from tool import set_log,  get_task_name, mkdir

import warnings
warnings.filterwarnings("ignore")


def training(args,log):   #交叉验证训练
    info = log.info    #记录信息
    
    seed_first = args.seed       #获取初始随机种子
    data_path = args.data_path
    save_path = args.save_path     #model保存路径
    args.task_names = get_task_name(data_path)     #获取任务名列表
    
    score = []        #交叉验证的分数
    
    for num_fold in range(args.num_folds):      #交叉验证的折数
        info(f'Seed {args.seed}')
        args.seed = seed_first + num_fold       #更新随机种子为初始种子加上当前的折数
        args.save_path = os.path.join(save_path,f'Seed_{args.seed}')    #更新保存路径
        mkdir(args.save_path)             #创建保存路径的目录
        
        fold_score = fold_train(args,log)  # 进行训练，并获取当前折数的分数

        score.append(fold_score)        #将当前折数的分数添加到 score 列表中
    score = np.array(score)   #n-折下的所有分数，将分数列表转换为numpy
    
    info(f'Running {args.num_folds} folds in total.')    #记录总的折数
    if args.num_folds > 1:
        for num_fold, fold_score in enumerate(score):
            info(f'Seed {seed_first + num_fold} : test {args.metric} = {np.nanmean(fold_score):.6f}')
            if args.task_num > 1:
                for one_name, one_score in zip(args.task_names, fold_score):
                    info(f' Task {one_name} {args.metric} = {one_score:.6f}')

    ave_task_score = np.nanmean(score, axis=1)  #每个任务在所有折数下的平均分数
    score_ave = np.nanmean(ave_task_score)  #所有任务的平均分数
    score_std = np.nanstd(ave_task_score)   #所有任务分数的标准差
    info(f'final_all-task average test {args.metric} = {score_ave:.6f} +/- {score_std:.6f}') # 最终的分数，n-折-ntask下的一个平均值
    
    if args.task_num > 1:
        for i,one_name in enumerate(args.task_names):
            info(f'final every task Average test {one_name} {args.metric} = {np.nanmean(score[:, i]):.6f} +/- {np.nanstd(score[:, i]):.6f}')
            #每个task的平均值
    
    return score_ave, score_std

if __name__ == '__main__':
    args = set_train_argument()
    log = set_log('train',args.log_path)
    training(args,log)
