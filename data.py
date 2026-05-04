from argparse import Namespace
from rdkit import Chem
import torch
from torch.utils.data.dataset import Dataset
from collections import defaultdict
import random
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedKFold

class MoleData:
    def __init__(self,line,args):    #它接受三个参数：self（类的实例），line（表示分子数据的行，通常是字符串列表），和 args（包含额外参数的对象）
        self.args = args
        self.smile = line[0]      #从 line 列表中取出第一个元素（预期是 SMILES 字符串），并将其赋值给实例变量 smile
        self.mol = Chem.MolFromSmiles(self.smile)    #将smile转换为分子对象，并赋值给实例变量mol
        self.label = [float(x) if x != '' else None for x in line[1:]]   #尝试将每个元素转换为浮点数。如果元素为空字符串，则将其赋值为 None
        
    def task_num(self):      #定义了一个名为 task_num 的实例方法，用于返回与分子数据相关的任务数量

        return len(self.label)
    
    def change_label(self,label):   # 定义了一个名为 change_label 的实例方法，它接受一个参数 label
        self.label = label


class MoleDataSet(Dataset):   #用于封装分子数据集，并提供一些操作这些数据的方法。
    def __init__(self,data):
        self.data = data
        if len(self.data) > 0:
            self.args = self.data[0].args
        else:
            self.args = None
        self.scaler = None
    
    def smile(self):     #smile 方法返回数据集中所有分子的 SMILES 字符串列表
        smile_list = []
        for one in self.data:
            smile_list.append(one.smile)
        return smile_list
    
    def mol(self):      #mol 方法返回数据集中所有分子的 RDKit Mol 对象列表
        mol_list = []
        for one in self.data:
            mol_list.append(one.mol)
        return mol_list
    
    def label(self):     #label 方法返回数据集中所有分子的标签列表
        label_list = []
        for one in self.data:
            label_list.append(one.label)
        return label_list
    
    def task_num(self):    #task_num 方法返回数据集中任务的数量，如果数据集为空，则返回 None
        if len(self.data) > 0:
            return self.data[0].task_num()
        else:
            return None
    
    def __len__(self):    #__len__ 方法返回数据集的长度
        return len(self.data)
    
    def __getitem__(self,key):   #__getitem__ 方法允许通过索引访问数据集中的单个项
        return self.data[key]
    
    def random_data(self,seed):
        random.seed(seed)
        random.shuffle(self.data)
    
    def change_label(self,label):   #change_label 方法允许更改整个数据集中分子的标签
        assert len(self.data) == len(label)
        for i in range(len(label)):
            self.data[i].change_label(label[i])


def generate_scaffold(mol, include_chirality=False):
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

    return scaffold
#generate_scaffold 函数接收一个分子（可以是 SMILES 字符串或 RDKit Mol 对象）和一个布尔值 include_chirality。
# 它使用 RDKit 的 MurckoScaffold 来生成给定分子的基本骨架 SMILES，可以选择性地包含手性信息。

def scaffold_to_smiles(mol, use_indices=False):
    scaffolds = defaultdict(set)
    for i, one in enumerate(mol):
        scaffold = generate_scaffold(one)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(one)

    return scaffolds

#scaffold_to_smiles 函数接收一个分子数据集和 use_indices 布尔值
#它生成一个从骨架 SMILES 到分子数据集索引的映射，如果 use_indices 为 True，则映射到索引集合；否则，映射到分子集合

def scaffold_split(data, size, seed, log):
    assert sum(size) == 1

    # Split
    train_size, val_size, test_size = size[0] * len(data), size[1] * len(data), size[2] * len(data)
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index in the data
    scaffold_to_indices = scaffold_to_smiles(data.mol(), use_indices=True)

    index_sets = list(scaffold_to_indices.values())
    big_index_sets = []
    small_index_sets = []
    for index_set in index_sets:
        if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
            big_index_sets.append(index_set)
        else:
            small_index_sets.append(index_set)
    random.seed(seed)
    random.shuffle(big_index_sets)
    random.shuffle(small_index_sets)
    index_sets = big_index_sets + small_index_sets

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1

    log.debug(f'Total scaffolds = {len(scaffold_to_indices):,} | '
              f'train scaffolds = {train_scaffold_count:,} | '
              f'val scaffolds = {val_scaffold_count:,} | '
              f'test scaffolds = {test_scaffold_count:,}')

    # Map from indices to data
    train = [data[i] for i in train]
    val = [data[i] for i in val]
    test = [data[i] for i in test]

    return MoleDataSet(train), MoleDataSet(val), MoleDataSet(test)

def scaffold_split_balanced(data,size,seed,log):
    """
    Split a dataset by scaffold so that no molecules sharing a scaffold are in the same split.

    :param data: A MoleculeDataset.
    :param sizes: A length-3 tuple with the proportions of data in the
    train, validation, and test sets.
    :param balanced: Try to balance sizes of scaffolds in each set, rather than just putting smallest in test set.
    :param seed: Seed for shuffling when doing balanced splitting.
    :param logger: A logger.
    :return: A tuple containing the train, validation, and test splits of the data.
    """

    assert sum(size) == 1

    # Split
    train_size, val_size, test_size = size[0] * len(data), size[1] * len(data), size[2] * len(data)
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index in the smiles
    scaffold_to_indices = scaffold_to_smiles(data.mol(), use_indices=True)

    #if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
    index_sets = list(scaffold_to_indices.values())
    big_index_sets = []
    small_index_sets = []
    for index_set in index_sets:
        if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
            big_index_sets.append(index_set)
        else:
            small_index_sets.append(index_set)
    random.seed(seed)
    random.shuffle(big_index_sets)
    random.shuffle(small_index_sets)
    index_sets = big_index_sets + small_index_sets
    #else:  # Sort from largest to smallest scaffold sets
        #index_sets = sorted(list(scaffold_to_indices.values()), key=lambda index_set: len(index_set), reverse=True)

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1
    log.debug(f'Total scaffolds = {len(scaffold_to_indices):,} | '
              f'train scaffolds = {train_scaffold_count:,} | '
              f'val scaffolds = {val_scaffold_count:,} | '
              f'test scaffolds = {test_scaffold_count:,}')

    # Map from indices to data
    train = [data[i] for i in train]
    val = [data[i] for i in val]
    test = [data[i] for i in test]

    return MoleDataSet(train), MoleDataSet(val), MoleDataSet(test)

