from argparse import Namespace
from logging import Logger
import os
from sklearn import metrics
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR

from tool import mkdir, get_task_name, load_data, split_data, get_label_scaler, get_loss, get_metric, save_model, NoamLR, load_model
from model import FPGNN
from data import MoleDataSet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def epoch_train(model,data,loss_f,optimizer,scheduler,args):
    model.train()
    data.random_data(args.seed)
    loss_sum = 0
    data_used = 0
    iter_step = args.batch_size
    
    for i in range(0, len(data), iter_step):
        if data_used + iter_step > len(data):
            #iter_step = len(data) - data_used
            break
        
        data_now = MoleDataSet(data[i:i+iter_step])

        smile = data_now.smile()
        label = data_now.label()
        #print(f"Labels: {label}")  # 打印标签以检查其有效性

        mask = torch.Tensor([[x is not None for x in tb] for tb in label])
        target = torch.Tensor([[0 if x is None else x for x in tb] for tb in label])
        # # 确保目标维度为 [batch_size, 1]
        # if target.size(1) > 1:
        #     target = torch.max(target, dim=1, keepdim=True)[0]  # 或使用其他合并方法
        # if mask.size(1) > 1:
        #     mask = torch.max(mask, dim=1, keepdim=True)[0]  # 或使用其他合并方法
        # 计算 mask 和 target
        #mask, target = compute_mask_target(label)

        if next(model.parameters()).is_cuda:
            mask, target = mask.cuda(), target.cuda()
        
        weight = torch.ones(target.shape)
        if args.cuda:
            weight = weight.cuda()
        
        model.zero_grad()
        pred = model(smile)   #(args,smile)


        # 确保批量大小一致
        if pred.size(0) != target.size(0):
            raise ValueError(f"Mismatch in batch size: pred size {pred.size(0)}, target size {target.size(0)}")
        # #处理二分类任务
        # if pred.size(1) == 1:  # 二分类任务
        #     target = target.view(-1, 1)  # 转换为 [batch_size, 1] 形状
        #     loss_f = nn.BCEWithLogitsLoss()
        # elif pred.size(1) > 1:  # 多分类任务
        #     target = target.long()  # 确保 target 是整数型
        #     if target.size(1) != pred.size(1):  # 转换为类别索引
        #         target = torch.argmax(target, dim=1)
        #     loss_f = nn.CrossEntropyLoss()
        # else:
        #     raise ValueError(
        #         "Unsupported task type. Ensure the model output dimensions match the loss function requirements.")

        if torch.isnan(pred).any():
            raise ValueError("NaN detected in gcn_out1")
        #print("gcn_out1 shape:", pred.shape)

        # 计算损失
        #if loss_f.__class__ == nn.CrossEntropyLoss:
            # CrossEntropyLoss 不需要 mask 和 weight
            #loss = loss_f(pred, target)
        #else:
            #loss = loss_f(pred, target) * weight * mask
            #loss = loss.sum() / mask.sum()


        loss = loss_f(pred,target) * weight * mask
        loss = loss.sum() / mask.sum()
        loss_sum += loss.item()
        data_used += len(smile)
        loss.backward()
        optimizer.step()
        if isinstance(scheduler, NoamLR):
            scheduler.step()
    if isinstance(scheduler, ExponentialLR):
        scheduler.step()
    return loss_sum
def predict(model, data,batch_size,scaler,args):
    model.eval()   #模型设置为评估模式
    pred = []
    data_total = len(data)    #据集中样本的总数
    
    for i in range(0, data_total, batch_size):
        data_now = MoleDataSet(data[i:i+batch_size])
        smile = data_now.smile()    #当前批次数据的 SMILES 字符串

        # # 检查输入数据是否包含空值或 NaN 值
        # if any(s is None for s in smile):
        #     print(f"None values found in input data for batch starting at index {i}")
        #     continue  # 跳过包含 None 值的批次


        with torch.no_grad():      #进入一个上下文管理器,为了在预测时减少内存消耗
            #pred_now,fpn_out, gcn_out1,egnn_out,image_out = model(smile)
            pred_now = model(smile)   #(args,smile)
        
        pred_now = pred_now.data.cpu().numpy()
        # if np.any(pred_now == None):
        #     print("Warning: pred_now contains None values.")
        #     pred_now = np.where(pred_now == None, 0.0, pred_now)  # 替换 None 值为 0.0

        if np.isnan(pred_now).any():
            # print(f"NaN values found in model output for batch starting at index {i}")
            # print(pred_now)
            continue  # Skip this batch if NaN values are found

        if scaler is not None:
            ave = scaler[0]
            std = scaler[1]
            pred_now = np.array(pred_now).astype(float)
            change_1 = pred_now * std + ave
            pred_now = np.where(np.isnan(change_1), None, change_1)    #np.where 函数替换NAN值
        
        pred_now = pred_now.tolist()      #预测结果转换为列表
        pred.extend(pred_now)
        # 确保预测值中没有 NaN 或 None
        # pred = []
        # for pred_list in pred:
        #     cleaned_list = [0.0 if x is None or np.isnan(x) else x for x in pred_list]
        #     pred.append(cleaned_list)

    return pred

def compute_score(pred,label,metric_f,args,log):
    info = log.info

    if isinstance(pred, list):
        pred = torch.tensor(pred)
    if isinstance(label, list):
        label = torch.tensor(label)
    print(f"pred shape: {pred.shape}")
    print(f"label shape: {label.shape}")

    batch_size = args.batch_size
    task_num = args.task_num
    data_type = args.dataset_type
    batch_size,num_classes = pred.shape
    
    if len(pred) == 0:
        return [float('nan')] * task_num  #如果预测结果为空，则返回一个包含 nan 的列表，长度为任务数量
    
    pred_val = []   #存储筛选后的预测值和标签值
    label_val = []
    for i in range(task_num):   #筛选出非none的预测结果和标签
        pred_val_i = []    #存储该任务的预测值和标签值
        label_val_i = []
        for j in range(len(pred)):
            if label[j][i] is not None and pred[j][i] is not None:
                pred_val_i.append(pred[j][i])
                label_val_i.append(label[j][i])

    for j in range(batch_size):
        for i in range(num_classes):
            if i < label.shape[1] and j < pred.shape[0]:  # 检查索引是否超出范围
                if label[j][i] is not None and pred[j][i] is not None:
                    # 你的计算逻辑
                    pass
            else:
                print(f"Index out of range: label.shape={label.shape}, pred.shape={pred.shape}, i={i}, j={j}")
        # if len(pred_val_i) > 0:  # 确保至少有一个有效值
        #     pred_val.append(pred_val_i)
        #     label_val.append(label_val_i)
        # else:
        # pred_val.append([float('nan')])
        # label_val.append([float('nan')])
        # if any(np.isnan(pred_val_i)):
        #     info(f'Warning: NaN found in predictions for task {i}.')
        # if any(np.isnan(label_val_i)):
        #     info(f'Warning: NaN found in labels for task {i}.')
        pred_val.append(pred_val_i)
        label_val.append(label_val_i)
    #print(f"Index j={j},Index i={i},pred length=len{pred},pred[j] length={len(pred[j])}")
    result = []
    for i in range(task_num):
        if data_type == 'classification':
            if all(one == 0 for one in label_val[i]) or all(one == 1 for one in label_val[i]):
                info('Warning: All labels are 1 or 0.')
                result.append(float('nan'))
                continue
            if all(one == 0 for one in pred_val[i]) or all(one == 1 for one in pred_val[i]):
                info('Warning: All predictions are 1 or 0.')
                result.append(float('nan'))
                continue
        # valid_label_val_i = [x for x in label_val[i] if not np.isnan(x)]
        # valid_pred_val_i = [x for x in pred_val[i] if not np.isnan(x)]

        # if not valid_label_val_i or not valid_pred_val_i:  # 确保列表不为空
        #     info(f'Warning: No valid predictions or labels for task {i}.')
        #     result.append(float('nan'))
        #     continue
        # if len(label_val[i]) == 0 or len(pred_val[i]) == 0:
        #     info(f'Warning: No valid predictions or labels for task {i}.')
        #     result.append(float('nan'))
        # else:
        #     re = metric_f(label_val[i], pred_val[i])
        #     result.append(re)

        # clean_pred_val_i = [0.0 if x is None or np.isnan(x) else x for x in pred_val[i]]
        # clean_label_val_i = [0.0 if x is None or np.isnan(x) else x for x in label_val[i]]

        re = metric_f(label_val[i], pred_val[i])
        result.append(re)

    return result


def fold_train(args, log):   #用于单个交叉验证折训练
    info = log.info
    debug = log.debug
    
    debug('Start loading data')      #加载数据的调试信息
    
    args.task_names = get_task_name(args.data_path)     #取任务名称并赋值到 args.task_names
    data = load_data(args.data_path,args)           #加载数据并赋值到 data
    args.task_num = data.task_num()              #从数据集中获取任务数量并赋值到 args.task_num
    data_type = args.dataset_type
    if args.task_num > 1:                         #如果任务数大于1，则设置多任务标志
        args.is_multitask = 1
    
    debug(f'Splitting dataset with Seed = {args.seed}.')
    if args.val_path:
        val_data = load_data(args.val_path,args)
    if args.test_path:
        test_data = load_data(args.test_path,args)
    if args.val_path and args.test_path:
        train_data = data
    elif args.val_path:
        split_ratio = (args.split_ratio[0],0,args.split_ratio[2])
        train_data, _, test_data = split_data(data,args.split_type,split_ratio,args.seed,log)
    elif args.test_path:
        split_ratio = (args.split_ratio[0],args.split_ratio[1],0)
        train_data, val_data, _ = split_data(data,args.split_type,split_ratio,args.seed,log)
    else:
        train_data, val_data, test_data = split_data(data,args.split_type,args.split_ratio,args.seed,log)
    debug(f'Dataset size: {len(data)}    Train size: {len(train_data)}    Val size: {len(val_data)}    Test size: {len(test_data)}')


    if data_type == 'regression':
        label_scaler = get_label_scaler(train_data)    #根据数据类型，获取标签缩放器
    else:
        label_scaler = None
    args.train_data_size = len(train_data)     #设置训练数据大小
    
    loss_f = get_loss(data_type)      #获取损失函数 loss_f
    metric_f = get_metric(args.metric)      #评估指标函数 metric_f
    
    debug('Training Model')
    model = FPGNN(args)               #创建模型实例
    #debug(model)
    if args.cuda:
        model = model.to(device)
    save_model(os.path.join(args.save_path, 'model.pt'),model,label_scaler,args)     #保存初始模型

    optimizer = Adam(params=model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)    #初始化优化器 Adam 和学习率调度器 NoamLR
    scheduler = NoamLR( optimizer=optimizer,  warmup_epochs=[args.warmup_epochs], total_epochs=None or [args.epochs] * args.num_lrs, \
                        steps_per_epoch=args.train_data_size // args.batch_size, init_lr=[args.init_lr], max_lr=[args.max_lr], \
                        final_lr=[args.final_lr] )

    if data_type == 'classification':    #最高分数为正值
        best_score = -float('inf')
    else:
        best_score = float('inf')
    best_epoch = 0    #初始化最佳周期计数器为 0
    n_iter = 0   #初始化迭代次数

    # train_losses = []
    # val_losses = []
    # train_scores = []
    # val_scores = []
    for epoch in range(args.epochs):
        info(f'Epoch {epoch}')     #记录当前周期
        
        train_loss = epoch_train(model,train_data,loss_f,optimizer,scheduler,args)
        train_pred = predict(model,train_data,args.batch_size,label_scaler,args)
        train_label = train_data.label()      #训练数据的真实标签
        train_score = compute_score(train_pred,train_label,metric_f,args,log)

        val_loss = epoch_train(model, val_data, loss_f, optimizer, scheduler, args)
        val_pred = predict(model,val_data,args.batch_size,label_scaler,args)
        val_label = val_data.label()
        val_score = compute_score(val_pred,val_label,metric_f,args,log)

        # train_losses.append(train_loss)
        # val_losses.append(val_loss)
        # train_scores.append(np.nanmean(train_score))
        # val_scores.append(np.nanmean(val_score))

        ave_train_score = np.nanmean(train_score)
        info(f'Train {args.metric} = {ave_train_score:.6f}')
        info(f'Train_loss = {train_loss:.6f}')
        
        ave_val_score = np.nanmean(val_score)
        info(f'Validation {args.metric} = {ave_val_score:.6f}')
        info(f'val_loss = {val_loss:.6f}')

        if args.task_num > 1:
            for one_name,one_score in zip(args.task_names,val_score):
                info(f'Validation {one_name} {args.metric} = {one_score:.6f}')
        
        if data_type == 'classification' and ave_val_score > best_score:
            best_score = ave_val_score
            best_epoch = epoch
            save_model(os.path.join(args.save_path, 'model.pt'),model,label_scaler,args)
        elif data_type == 'regression' and ave_val_score < best_score:
            best_score = ave_val_score
            best_epoch = epoch
            save_model(os.path.join(args.save_path, 'model.pt'),model,label_scaler,args)


    info(f'Best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')

    # 绘制损失和得分曲线
    # import matplotlib.pyplot as plt
    #
    # epochs = range(args.epochs)
    # plt.figure()
    # plt.plot(epochs, train_losses, 'b', label='Training loss')
    # plt.plot(epochs, val_losses, 'r', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.legend()
    # plt.savefig(os.path.join(args.save_path, 'loss_curve.png'))
    #
    # plt.figure()
    # plt.plot(epochs, train_scores, 'b', label='Training score')
    # plt.plot(epochs, val_scores, 'r', label='Validation score')
    # plt.title('Training and validation score')
    # plt.legend()
    # plt.savefig(os.path.join(args.save_path, 'score_curve.png'))

    model = load_model(os.path.join(args.save_path, 'model.pt'),args.cuda,log)
    test_label = test_data.label()    #测试数据的真实标签
    
    test_pred = predict(model,test_data,args.batch_size,label_scaler,args)
    test_score = compute_score(test_pred,test_label,metric_f,args,log)
    ave_test_score = np.nanmean(test_score)    #测试分数的平均值，忽略 NaN 值
    info(f'Seed {args.seed} : test {args.metric} = {ave_test_score:.6f}')
    
    if args.task_num > 1:
        for one_name,one_score in zip(args.task_names,test_score):
            info(f'Task {one_name} {args.metric} = {one_score:.6f}')


    # 记录每个任务的预测结果
    assert len(test_data) == len(test_pred)
    test_pred = np.array(test_pred)
    test_pred = test_pred.tolist()
    test_label = np.array(test_label)
    test_label = test_label.tolist()

    print('Write result.')
    write_smile = test_data.smile()

    #with open(exp_path + args.dataset + "_seed_" + str(args.seed) + "_pred_label.csv", "w", newline='') as file:
    with open(args.dataset + "_seed_" + str(args.seed) + "_pred_label.csv", "w", newline='') as file:   #生成的测试数据集
        writer = csv.writer(file)
        line = ['Smiles']
        line.extend(args.task_names)
        line.extend(list(range(1,args.task_num)))
        writer.writerow(line)
        # for i in range(fir_data_len):
        for i in range(len(test_data)):
            line = []
            line.append(write_smile[i])
            line.extend(test_pred[i])
            line.extend(test_label[i])
            writer.writerow(line)

    return test_score
