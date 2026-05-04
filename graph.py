from argparse import Namespace
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
import numpy as np
from torch.nn import Linear

from torch_geometric.data import Data
import random
from torch.utils.data.dataset import Dataset
from collections import defaultdict
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch

allowable_features = {
    'possible_atomic_num_list': list(range(0, 119)),
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

smile_changed = {}
def get_atom_poses(mol, conf):
    """tbd"""
    atom_poses = []
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 0:
            return [[0.0, 0.0, 0.0]] * len(mol.GetAtoms())
        pos = conf.GetAtomPosition(i)
        atom_poses.append([pos.x, pos.y, pos.z])
    return atom_poses

def get_MMFF_atom_poses(mol, numConfs=None, return_energy=False):      #Merck分子力场
    """the atoms of mol will be changed in some cases."""
    try:
        new_mol = Chem.AddHs(mol)
        res = AllChem.EmbedMultipleConfs(new_mol, numConfs=numConfs)
        ### MMFF generates multiple conformations
        res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
        new_mol = Chem.RemoveHs(new_mol)
        index = np.argmin([x[1] for x in res])
        #energy = res[index][1]
        conf = new_mol.GetConformer(id=int(index))
    except:
        new_mol = mol
        AllChem.Compute2DCoords(new_mol)
        #energy = 0
        conf = new_mol.GetConformer()

    atom_poses = get_atom_poses(new_mol, conf)

    return atom_poses


def get_two_graph(mol,args):   #它用于将一个 RDKit 分子对象转换成一个图表示
    atom_feature_list = []              #存储原子特征atom_feature_list:{list:4}[[35,0],[6,0],[35,0],[35,0]]
    atom_size = mol.GetNumAtoms()       #获取原子数量atom_size:4
    for atom in mol.GetAtoms():       #遍历分子中所有的原子
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
                                         'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_feature_list.append(atom_feature)
        #获取原子的原子序数，并使用 allowable_features['possible_atomic_num_list'] 中的索引作为原子特征的第一个元素。
        #获取原子的手性标签，并使用 allowable_features['possible_chirality_list'] 中的索引作为原子特征的第二个元素。
        #将这两个特征组合成一个列表，并将其添加到 atom_feature_list 中
    x = torch.tensor(np.array(atom_feature_list), dtype=torch.long)    #列表转换为张量，数据类型为长整型Tensor(4,2)

    row, col = [], []       #用于存储图的边的行索引和列索引 col:[1,0,2,1,3,1]  row:[0,1,1,2,1,3]
    for bond in mol.GetBonds():
        i,j = bond.GetBeginAtomIdx(),  bond.GetEndAtomIdx()      #键的起始原子索引:1，结束原子索引:3
        row += [i, j]
        col += [j, i]
    # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
    edge_index = torch.tensor(([row, col]), dtype=torch.long)  #edge_index:tensor([[0,1,1,2,1,3],\n   [1,0,2,1,3,1]])

    return x, atom_size, edge_index   #原子特征，原子数，边索引


def one_hot_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def encoding_unk(x, allowable_set):
    list = [False for i in range(len(allowable_set))]
    i = 0
    for atom in x:
        if atom in allowable_set:
            list[allowable_set.index(atom)] = True
            i += 1
    if i != len(x):
        list[-1] = True
    return list

def cluster_graph(mol):
    n_atoms = mol.GetNumAtoms()  #获得分子图中原子数
    cliques = []  # 所有的的边（化学键和一对原子组成）和环
    for bond in mol.GetBonds():  #获得分子图中所有化学键
        a1 = bond.GetBeginAtom().GetIdx() #获得开始原子索引
        a2 = bond.GetEndAtom().GetIdx() #获得结束原子索引
        if not bond.IsInRing():   #判断化学键是否在环中
            cliques.append([a1, a2])  #将边加入到集合中

    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]  # 获得分子图中的所有环
    cliques.extend(ssr)  #添加环信息
    # cliques 所有的的边（化学键和一对原子组成）和环
    # nei_list为原子属于哪个基团\子结构
    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)

    edges = []
    for i in range(len(cliques)-1):
        for j in range(i+1,len(cliques)):
            if len(set(cliques[i]) & set(cliques[j]))!= 0:
                edges.append([i,j])
                edges.append([j,i])
    return cliques, edges  #子结构集合和子结构之间的边

def get_atom_features(atom):
    atom_type = atom.GetAtomicNum()
    chirality_tag = int(atom.GetChiralTag())
    return atom_type, chirality_tag

    # 填充 features 到指定长度
    # if len(features) < feature_length:
    #     features.extend([0] * (feature_length - len(features)))

    # 如果没有指定 feature_length，则自动使用 features 的长度
    if feature_length is None:
        feature_length = max(len(features), 2)

        # 计算 features 的均值
    mean_value = sum(features) / len(features) if features else 0

    # 填充 features 到指定长度
    if len(features) < feature_length:
        features.extend([mean_value] * (feature_length - len(features)))
    elif len(features) > feature_length:
        features = features[:feature_length]



    return np.array(features)



    #x = torch.tensor(x, dtype=torch.long)
    # 调试信息
    # print(f"Generated features for {smile}:")
    # print(f"x shape: {x.shape}")
    # print(x)

    atom_size = len(clique)  # 子结构图节点数:3
    # print("Returning from clique_to_graph: edge_index={}, atom_size={}, x.shape={}, edge={}".format(
    #     edge_index.shape, atom_size, x.shape, edge))
    return atom_size, x,edge_index  #子结构的数量、子结构的特征列表和边的列表
    #return clique_size, c_features, edge


def get_three_graph(mol,args):
    atom_feature_list = []
    edges_list = []
    atom_size = mol.GetNumAtoms()
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
                                         'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_feature_list.append(atom_feature)
    x = torch.tensor(np.array(atom_feature_list), dtype=torch.long)

    if len(mol.GetBonds()) > 0:
        for i in range(atom_size):
            for j in range(atom_size):
                if i != j:
                    edges_list.append((i, j))

                # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list), dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.zeros((0, 2), dtype=torch.long)
    atom_3dcoords = get_MMFF_atom_poses(mol, numConfs=None, return_energy=False)
    pos = torch.tensor(np.array(atom_3dcoords), dtype=torch.float)

    return x, atom_size, edge_index, pos
