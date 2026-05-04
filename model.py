from argparse import Namespace
import torch
import torch.nn as nn
from rdkit.Chem import AllChem
from graph import get_two_graph,get_three_graph,clique_to_graph
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
from rdkit import Chem
import numpy as np
from rdkit.Chem import Draw
import torchvision.transforms as transforms
from rdkit import RDLogger
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
RDLogger.DisableLog('rdApp.*')
num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3#3


class FPN(nn.Module):  #FPN_GRU_ATT
    def __init__(self,args):
        super(FPN, self).__init__()
        self.fp_dim = args.fp_dim         #从 args 对象中提取网络配置参数，并赋值给相应的实例变量：输入特征的维度
        self.fp_2_dim=args.fp_2_dim       #第一层全连接层输出的维度
        self.dropout_fpn = args.dropout_fpn   #0.2
        self.hidden_dim = args.hidden_size     #128

        self.num_lstm_layers = args.num_lstm_layers
        self.lstm = nn.LSTM(self.fp_dim, self.hidden_dim, num_layers=self.num_lstm_layers,
                          bidirectional=True, batch_first=True,dropout=self.dropout_fpn)
        # self.dropout = nn.Dropout()  # 防止过拟合
        # self.fc = nn.Linear(self.fp_2_dim, self.hidden_dim)  # 全连接层，用于将网络层输出映射到最终的输出维度

        self.fc1=nn.Linear(self.fp_dim, self.fp_2_dim)       #第一层全连接层，将输入特征从 fp_dim 维度转换到 fp_2_dim 维度  fc1=in_feature=1024,out_features=256
        self.act_func = nn.ReLU()                            #激活函数
        self.fc2 = nn.Linear(self.fp_2_dim, self.hidden_dim)  #第二层全连接层，将特征从 fp_2_dim 维度转换到 hidden_dim 维度  fc2=in_feature=256,out_features=128

        self.dropout = nn.Dropout(p=self.dropout_fpn)        #dropout 层，用于正则化以防止过拟合  dropout:p=0.2

        self.cuda = args.cuda
        self.args = args

        # Multi-head attention layers
        self.num_attention_heads = args.num_attention_heads
        self.ep_attention_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.fp_2_dim)
        ) for _ in range(self.num_attention_heads)])
        self.ep_fc_layers = nn.ModuleList([nn.Linear(self.hidden_dim * 2, self.fp_2_dim) for _ in range(self.num_attention_heads)])
        # 每个注意力头对应的全连接层，将 GRU 的输出维度从 self.hidden_dim_lstm * 2 转换到 self.output_dim


    def forward(self, smile):    #接受一个参数 smile，预期是一个包含 SMILES 字符串的列表  smile=list[30]
        fp_list = []               #用于存储提取的特征 fp_list[30,1024]
        for i, one in enumerate(smile):   #遍历 smile 列表中的每个 SMILES 字符串 i=29
            fp = []
            RDLogger.DisableLog('rdApp.*')
            mol = Chem.MolFromSmiles(one)    #smiles字符串转换为分子对象
            fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)   #生成 Morgan 指纹（一种圆形指纹），半径为 2，1024 位
            fp.extend(fp_morgan)
            fp_list.append(fp)
        fp_list = torch.Tensor(fp_list)  #将 fp_list 转换为 PyTorch 张量

        if self.cuda:
            fp_list = fp_list.cuda()

        # Add GRU layer
        lstm_out, _ = self.lstm(fp_list)  # (30,1,256) (4,30,128)Ensure input is of shape (batch_size, seq_len, features)

        # Apply attention mechanism
        attention_out = 0
        #attention_out = self.attention(gru_out)
        for i in range(self.num_attention_heads):
            ep_attention_weights = torch.softmax(self.ep_attention_layers[i](lstm_out), dim=1)
            ep_linear = self.ep_fc_layers[i](lstm_out)
            attention_out += ep_attention_weights * ep_linear
        attention_out /= self.num_attention_heads  #将注意力头的加权输出取平均，得到最终的注意力输出。
        attention_out = attention_out.squeeze(1)
        #print(attention_out.shape)


        # Final output layer
        fpn_out = self.fc2(attention_out)  # Remove the sequence length dimension
        # fpn_out = fpn_out.view(1, -1)
        fpn_out = self.dropout(fpn_out)
        #print(fpn_out.shape)

        return fpn_out   #[30,128]



class GAT(torch.nn.Module):
    def __init__(self, args):
        super(GAT, self).__init__()
        #torch.manual_seed(12345)
        self.emb_dim= args.emb_dim_gnn       #嵌入层的维度
        self.hidden_gnn = args.hidden_gnn    #gnn隐藏层维度
        self.dropout_gnn = args.dropout_gnn   #gnn中的dropout比率
        self.args = args
        self.cuda = args.cuda

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, self.emb_dim)   #原子类型的嵌入层    维度(120,32)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, self.emb_dim)    #手性标签的嵌入层   (3,32)
        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)    #初始化两个嵌入层的权重
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.conv1 = GATConv(self.emb_dim, self.hidden_gnn, heads=8)              #三个图卷积层用于图结构上执行卷积操作(32,128)
        self.conv2 = GATConv(self.hidden_gnn*8, self.hidden_gnn,heads=4)           #(128,128)
        self.conv3 = GATConv(self.hidden_gnn*4, self.hidden_gnn,heads=1)           #(128,128)

        #self.conv4 = GCNConv(self.hidden_gnn, self.hidden_gnn)

        #self.conv5 = GINConv(nn.Sequential(nn.Linear(self.emb_dim, self.hidden_gnn),nn.ReLU(),nn.Linear(self.hidden_gnn, self.hidden_gnn)))

        self.l1 = nn.Linear(self.emb_dim,self.emb_dim)       #线性层(l1): Linear(in_features=32, out_features=32, bias=True)
        self.dropout = nn.Dropout(p=self.dropout_gnn)        #Dropout(p=0.05, inplace=False)
        self.AN1 = torch.nn.LayerNorm(self.emb_dim)      #归一化层 BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.l2 = nn.Linear(self.emb_dim, self.hidden_gnn*4)   #Linear(in_features=32, out_features=128, bias=True)
        self.dropout = nn.Dropout(p=self.dropout_gnn)
        self.l3 = nn.Linear(self.hidden_gnn*4, self.hidden_gnn)
        #self.l4 = nn.Linear(self.hidden_gnn, self.hidden_gnn)
        self.AN2 = torch.nn.BatchNorm1d(self.hidden_gnn)     #BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.attn_linear = nn.Linear(self.hidden_gnn, self.hidden_gnn)

    def forward(self,smiles):
        gcn_outs = []
        for i, one in enumerate(smiles):
            #print('gnn_i', i, 'smile', one)
            RDLogger.DisableLog('rdApp.*')
            mol = Chem.MolFromSmiles(one)
            x, atom_size, edge_index = get_two_graph(mol,self.args)     #调用 get_two_graph 函数来获取分子的图表示，包括原子特征 x、原子数量 atom_size 和边索引 edge_index，此处的图为无向图
            #print("x shape:", x.shape)
            if self.cuda:
                x, edge_index= x.cuda(), edge_index.cuda()


            x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])     #原子特征通过两个嵌入层并将结果相加

            x = self.dropout(F.relu(self.l1(x)))     #应用 dropout 和 ReLU 激活函数

            x = self.AN1(x)

            # 1. Obtain node embeddings
            h_1 =self.conv1(x, edge_index)
            h_1 = F.relu(h_1)       #执行第一层图卷积

            x = self.l2(x)
            #print('x_1gcn', x)
            h_1 = self.dropout(h_1)

            h_2 =self.conv2(h_1, edge_index)
            h_2 = F.relu(h_2)    #执行第二层图卷积
            #x = self.l3(x)
            #print('x_2gcn', x)
            h_2 = self.dropout(h_2+x)


            h_3 = F.relu(self.conv3(h_2, edge_index))
            x = self.l3(x)
            h_3 =self.dropout(h_3+x)

            #h_4 =self.conv4(h_3, edge_index)
            #h_4 = F.relu(h_4)

            #h_5 = self.conv5(x, edge_index)
            #h_5 = F.relu(h_5)

            #h_combined = h_3 + h_4 + h_5




            gcn_out = torch.sum(h_3,dim=0) / atom_size    #对所有原子的隐藏表示进行求和，并除以原子数量，得到分子的图表示

            gcn_outs.append(gcn_out)

        gcn_outs = torch.stack(gcn_outs, dim=0)    #将gcn_outs列表中的所有图表示堆叠成一个张量
        #print('x_gcn', x.size())
        #graph_gnn = global_mean_pool(x, self.batch)  # [batch_size, hidden_channels]
        #print('gnn_g',  gcn_outs .size())
        # g = self.lin(g)

        attn_weights = F.softmax(self.attn_linear(gcn_outs), dim=1)
        gcn_outs = attn_weights * gcn_outs

        return gcn_outs     #(30,128)(44,128)


#3D 分子图特征学习
def index_sum(agg_size, source, idx,cuda):  #(结果张量的大小；一个包含数值的源张量，其形状为 N x hid_dim，其中 N 是索引的数量，hid_dim 是隐藏维度的大小；一个包含整数索引的张量，用于 source 中行的聚合，相当于邻居节点索引；)
    """
        source is N x hid_dim [float]
        idx    is N           [int]

        Sums the rows source[.] with the same idx[.];
    """
    tmp = torch.zeros((agg_size, source.shape[1]))   #创建一个形状为 (agg_size, source.shape[1]) 的零张量 tmp，用于存储聚合结果的临时张量
    tmp = tmp.cuda() if cuda else tmp
    res = torch.index_add(tmp, 0, idx, source)
    return res   #聚合求和后的结果张量res

class ConvEGNN(nn.Module):
    def __init__(self,hid_dim,cuda=True):
        super().__init__()
        self.hid_dim = hid_dim
        self.cuda = cuda
        # computes messages based on hidden representations -> [0, 1]
        self.f_e = nn.Sequential(
            nn.Linear(self.hid_dim * 2 + 1, self.hid_dim), nn.SiLU(),
            nn.Linear(self.hid_dim, self.hid_dim), nn.SiLU())

        # preducts "soft" edges based on messages
        self.f_inf = nn.Sequential(
            nn.Linear(self.hid_dim, 1),
            nn.Sigmoid())

        # updates hidden representations -> [0, 1]
        self.f_h = nn.Sequential(
            nn.Linear(self.hid_dim + self.hid_dim, self.hid_dim), nn.SiLU(),
            nn.Linear(self.hid_dim, self.hid_dim))

        self.f_pos = nn.Sequential(
            nn.Linear(self.hid_dim * 2 + 1, 3), nn.SiLU(),
            nn.Linear(3, 3), nn.SiLU())
        # preducts "soft" edges based on messages
        self.f_h_pos = nn.Sequential(
            nn.Linear(6, 3), nn.SiLU(),
            nn.Linear(3, 3))

    def forward(self, x, edge_index, pos):
        e_st, e_end = edge_index[:, 0], edge_index[:, 1]     #获取边起点和结束节点索引
        dists = torch.norm(pos[e_st] - pos[e_end], dim=1).reshape(-1, 1)   #计算起始节点和结束节点之间的距离

        # compute messages
        tmp = torch.hstack([x[e_st], x[e_end], dists])
        m_ij = self.f_e(tmp)
        m_ij_pos = self.f_pos(tmp)

        # predict edges
        e_ij = self.f_inf(m_ij)

        # average e_ij-weighted messages
        # m_i is num_nodes x hid_dim
        # m_i_pos is num_nodes x 3  (zuobiao)坐标
        m_i = index_sum(x.shape[0], e_ij * m_ij,edge_index[:, 0],self.cuda)     #index_sum聚合求和函数，并将结果按照顺序放入一个新的张量中
        m_i_pos = 1 / pos.size(0) * index_sum(pos.shape[0], dists * m_ij_pos, edge_index[:, 0],self.cuda)

        # update hidden representations
        x += self.f_h(torch.hstack([x, m_i]))
        pos += self.f_h_pos(torch.hstack([pos, m_i_pos]))     #更新位置信息

        return  x,  edge_index, pos


class NetEGNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.emb_dim = args.emb_egnn      #从 args 中获取 EGNN 的嵌入维度
        self.batch = args.batch_size      #获取批量大小
        self.pool = global_mean_pool
        self.dropout_egnn = args.dropout_egnn
        self.cuda = args.cuda

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, self.emb_dim )        #定义两个嵌入层，用于将原子类型和手性标签嵌入到高维空间
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, self.emb_dim )

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)          #使用 Xavier 均匀分布初始化两个嵌入层的权重
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.conv1 = ConvEGNN(self.emb_dim,self.cuda)
        self.conv2 = ConvEGNN(self.emb_dim,self.cuda)
        self.conv3 = ConvEGNN(self.emb_dim,self.cuda)
        self.l1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.AN1 = torch.nn.BatchNorm1d(self.emb_dim)
        self.dropout = nn.Dropout(p=self.dropout_egnn)

    def forward(self, smiles):
        egnn_outs = []
        for i, one in enumerate(smiles):
            #print('i', i,'smile',one)
            RDLogger.DisableLog('rdApp.*')
            mol = Chem.MolFromSmiles(one)

            x, atom_size, edge_index, pos = get_three_graph(mol, self.args)      #pos位置
            #print('i_edge_index', i,edge_index)
            if self.cuda:
                x, edge_index, pos = x.cuda(), edge_index.cuda(), pos.cuda()

            x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

            x = self.dropout(F.relu(self.l1(x)))
            #h = self.emb(h)
            h_1, edge_index, pos = self.conv1(x, edge_index,pos)
            #print('x_1',x)

            h_1 = self.dropout(h_1+x)

            h_2, edge_index, pos = self.conv2(h_1, edge_index,pos)
            #print('x_2', x)
            #x, edge_index, pos = self.conv3(x, edge_index,pos)
            h_2 = self.dropout(h_2+x)

            egnn_out = h_2.sum(dim=0) / atom_size
            #print('one_egnn_out', egnn_out.size())
            egnn_outs.append(egnn_out)

        egnn_outs = torch.stack(egnn_outs, dim=0)

        #print('egnn_g', egnn_outs.size())

        return egnn_outs

# 图像特征学习模型
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#准备测试所用的模型
class VggNet(nn.Module):
    def __init__(self, args):
        super(VggNet, self).__init__()
        self.image_dim = args.linear_dim
        self.cuda = args.cuda
        self.Conv = torch.nn.Sequential(
            # 3*224*224  conv1
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # 64*112*112   conv2
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # 128*56*56    conv3
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # 256*28*28    conv4
            torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        # 512*14*14   conv5
            torch.nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2))
        # 512*7*7
        self.Classes = torch.nn.Sequential(
            torch.nn.Linear(512*7*7, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096, 1060),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            nn.Linear(1060, self.image_dim))

    def forward(self, smile):
        img_list = []
        for i, one in enumerate(smile):
            img_f = []
            RDLogger.DisableLog('rdApp.*')
            mol = Chem.MolFromSmiles(one)
            img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(224, 224))
            img = np.array(img)
            img = transform(img)
            # img = img.unsqueeze(0)
            # print(img.size())
            # print('img',img)
            img_list.append(img)
        # print('img_list', img_list.shape)
        img_list = torch.stack(img_list)
        # print('img_list', img_list.size())
        if self.cuda:
            img_list = img_list.cuda()

        x = self.Conv(img_list)
        #print('x_conv', x.size())  # [batchsize,2048]
        fc_input = x.view(x.size(0), -1)
        #print('fc_input', fc_input.size())  # [batchsize,2048]
        #x = x.view(-1, 14 * 14 * 512)
        x = self.Classes(fc_input)
        return x

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, n_head,output_dim):
        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.n_heads = n_head
        self.output_dim = output_dim
        self.d_k = self.d_v = self.input_dim // self.n_heads

        self.W_Q = torch.nn.Linear(self.input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(self.input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(self.input_dim, self.d_v * self.n_heads, bias=False)

        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.output_dim, bias=False)
        #self.AN1 = torch.nn.LayerNorm(self.input_dim )
        self.l1 = torch.nn.Linear(self.output_dim , self.output_dim // 2)

    def forward(self, X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        #batch_size = X.shape[0]
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)
        # batch_size = X.size(0)
        # Q = self.W_Q(X).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # K = self.W_K(X).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # V = self.W_V(X).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        #context: [len_q, n_heads * d_v]

        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        #context = context.transpose(1, 2).contiguous().view(batch_size,-1, self.n_heads * self.d_v)

        output = self.fc(context)
        #output = self.AN1(output)
        output =self.l1(output)
        return output

class CrossAttention(nn.Module):
    def __init__(self, input_dim):
        super(CrossAttention, self).__init__()
        self.query_fc = nn.Linear(input_dim, input_dim)
        self.key_fc = nn.Linear(input_dim, input_dim)
        self.value_fc = nn.Linear(input_dim, input_dim)
        self.attn = nn.MultiheadAttention(input_dim, num_heads=1)

    def forward(self, query, key, value):
        # 将Q、K、V输入到前馈网络
        query = self.query_fc(query).unsqueeze(0)  # (1, batch_size, input_dim)
        key = self.key_fc(key).unsqueeze(0)        # (1, batch_size, input_dim)
        value = self.value_fc(value).unsqueeze(0)  # (1, batch_size, input_dim)
        attn_output, _ = self.attn(query, key, value)
        return attn_output.squeeze(0)  # 输出shape: (batch_size, input_dim)


# 定义最终模型
feature_out = []


class FpgnnModel(nn.Module):
    def __init__(self, is_classif,cuda, dropout_fpn, input_dim):
        super(FpgnnModel, self).__init__()
        self.is_classif = is_classif
        self.dropout_fpn = dropout_fpn
        self.cuda = cuda
        self.input_dim = input_dim
        if self.is_classif:
            self.sigmoid = nn.Sigmoid()

        self.cross_attention1 = CrossAttention(input_dim)
        # 新增门控层
        self.gate_layer = nn.Linear(64, 1)  # 生成融合权重
    def create_fpn(self, args):
        self.encoder1 = FPN(args)

    def create_gnn(self, args):
        self.encoder2 = GAT(args)

    # def create_gnn1(self, args):
    #     self.encoder5 = clique_GAT(args)

    def create_egnn(self, args):
        self.encoder3 = NetEGNN(args)

    def create_imagecnn(self, args):
        self.encoder4 = VggNet(args)

    def create_fc(self, args):
        fpn_dim = args.hidden_size
        gnn_dim = args.hidden_gnn
        gnn_dim1 = args.hidden_gnn
        egnn_dim = args.emb_egnn
        linear_dim = int(args.linear_dim)

        encoder1_FC =nn.Sequential()
        encoder1_FC.add_module('fc',  nn.Linear(fpn_dim, linear_dim)) #2048=128*4*4,此处维数维初始维数*最后一层输出维数
        #encoder1_FC.add_module('LayerNorm_', nn.BatchNorm1d(linear_dim))
        self.encoder1_FC = encoder1_FC

        encoder2_FC = nn.Sequential()
        encoder2_FC.add_module('fc', nn.Linear(gnn_dim, linear_dim))
        #encoder2_FC.add_module('LayerNorm', nn.BatchNorm1d(linear_dim))
        self.encoder2_FC = encoder2_FC

        encoder5_FC = nn.Sequential()
        encoder5_FC.add_module('fc', nn.Linear(gnn_dim1, linear_dim))
        # encoder2_FC.add_module('LayerNorm', nn.BatchNorm1d(linear_dim))
        self.encoder5_FC = encoder5_FC

        encoder3_FC = nn.Sequential()
        encoder3_FC.add_module('fc', nn.Linear(egnn_dim, linear_dim))
        #encoder3_FC.add_module('LayerNorm_', nn.BatchNorm1d(linear_dim))
        self.encoder3_FC = encoder3_FC

        encoder4_FC = nn.Sequential()
        encoder4_FC.add_module('fc', nn.Linear(linear_dim, linear_dim))
        #encoder4_FC.add_module('LayerNorm_', nn.BatchNorm1d(linear_dim))
        self.encoder4_FC = encoder4_FC

        self.act_func = nn.ReLU()
        self.fpn_attn = MultiHeadAttention(fpn_dim, 1, linear_dim)
        self.gnn_attn = MultiHeadAttention(gnn_dim, 1, linear_dim)
        self.gnn_attn1 = MultiHeadAttention(gnn_dim1, 1, linear_dim)
        self.egnn_attn = MultiHeadAttention(egnn_dim, 1, linear_dim)
        self.vggnet_attn = MultiHeadAttention(linear_dim, 1, linear_dim)

    def create_fc1(self, args):
        gcn_dim = 64  # Ensure input dimensions are correct
        linear_dim = 64  # output dimension after FC layer

        # Define encoder1_FC
        encoder1_FC1 = nn.Sequential()
        encoder1_FC1.add_module('fc', nn.Linear(gcn_dim, linear_dim))
        self.encoder1_FC1 = encoder1_FC1


        # Similarly for other layers (if needed):
        self.encoder2_FC1 = nn.Linear(gcn_dim, linear_dim)  # 64 -> 128 (for GCN or other models)
        self.encoder5_FC1 = nn.Linear(gcn_dim, linear_dim)  # 64 -> 128 (for GCN or other models)

        self.act_func1 = nn.ReLU()

    def create_ffn(self, args):
        linear_dim = int(args.linear_dim)
        self.ffn = nn.Sequential(
        nn.Linear(in_features=linear_dim*4, out_features=linear_dim, bias=True),
        nn.ReLU(),
        #nn.BatchNorm1d(linear_dim),
        nn.Dropout(self.dropout_fpn),
        nn.Linear(in_features=linear_dim, out_features=args.task_num, bias=True)#out_features=args.task_num

        )


    def forward(self, input):
        fpn_out = self.encoder1(input)  #[8,128]
        #print("fpn_out shape:", fpn_out.shape)
        # print("fpn_out:", fpn_out)
        # if torch.isnan(fpn_out).any():
        #     raise ValueError("NaN detected in fpn_out")
        #print("fpn_out shape:", fpn_out.shape)
        gcn_out = self.encoder2(input) #[8,128]
        #print("gcn_out shape:", gcn_out.shape)
        # print("gcn_out:", gcn_out)
        # if torch.isnan(gcn_out).any():
        #     raise ValueError("NaN detected in gcn_out")
        #print("gcn_out shape:", gcn_out.shape)
        gcn_out1, valid_indices = self.encoder5(input) #[8,128]

        #print("gcn_out1 shape:", gcn_out1.shape)
        #print("gcn_out1:", gcn_out1)
        # if torch.isnan(gcn_out1).any():
        #     raise ValueError("NaN detected in gcn_out1")
        # print("gcn_out1 shape:", gcn_out1.shape)
        egnn_out = self.encoder3(input)  #[8,64]
        #print("egnn_out shape:", egnn_out.shape)
        # if torch.isnan(egnn_out).any():
        #     raise ValueError("NaN detected in egnn_out")
        #print("egnn_out shape:", egnn_out.shape)
        image_out = self.encoder4(input)  #[8,64]
        #print("image_out shape:", image_out.shape)
        # if torch.isnan(image_out).any():
        #     raise ValueError("NaN detected in image_out")
        #print("image_out shape:", image_out.shape)

        # fpn_out = fpn_out[valid_indices]
        # egnn_out = egnn_out[valid_indices]
        # image_out = image_out[valid_indices]
        # gcn_out = gcn_out[valid_indices]

        # Check for NaN values
        if torch.isnan(fpn_out).any():
            raise ValueError("NaN detected in fpn_out")
        if torch.isnan(gcn_out).any():
            raise ValueError("NaN detected in gcn_out")
        if torch.isnan(gcn_out1).any():
            raise ValueError("NaN detected in gcn_out1")
        if torch.isnan(egnn_out).any():
            raise ValueError("NaN detected in egnn_out")
        if torch.isnan(image_out).any():
            raise ValueError("NaN detected in image_out")

        fpn_out = self.encoder1_FC(fpn_out)#[8,64]
        #print("fpn_out after FC shape:", fpn_out.shape)
        #fpn_out = self.act_func(fpn_out)

        gcn_out = self.encoder2_FC(gcn_out)#[8,64]
        #print("gcn_out after FC shape:", gcn_out.shape)
        gcn_out = self.act_func(gcn_out)

        gcn_out1 = self.encoder5_FC(gcn_out1) #[8,64]
        #print("gcn_out1 after FC shape:", gcn_out1.shape)
        gcn_out1 = self.act_func(gcn_out1)

        egnn_out = self.encoder3_FC(egnn_out) #[8,64]
        #print("egnn_out after FC shape:", egnn_out.shape)
        egnn_out = self.act_func(egnn_out)

        image_out = self.encoder4_FC(image_out) #[8,64]
        #print("image_out after FC shape:", image_out.shape)
        image_out = self.act_func(image_out)

        # More checks
        if torch.isnan(fpn_out).any():
            raise ValueError("NaN detected in fpn_out after FC")
        if torch.isnan(gcn_out).any():
            raise ValueError("NaN detected in gcn_out after FC")
        if torch.isnan(gcn_out1).any():
            raise ValueError("NaN detected in gcn_out1 after FC")
        if torch.isnan(egnn_out).any():
            raise ValueError("NaN detected in egnn_out after FC")
        if torch.isnan(image_out).any():
            raise ValueError("NaN detected in image_out after FC")

        #print('img_out:', image_out.size())    #[30,64]
        #print('egnn_out:', egnn_out.size())    #[30,64]
        #print('gcn_out:', gcn_out.size())      #[30,64]
        #print('fpn_out:', fpn_out.size())      #[30,64]
        #fpn_out = self.fpn_attn(fpn_out)
        # print("fpn_out after attn shape:", fpn_out.shape)
        #print('fpn_out:', fpn_out.size())
        #gcn_out = self.gnn_attn(gcn_out)
        #print('gcn_out:', gcn_out.size())
        # gcn_out1 = self.gnn_attn1(gcn_out1)
        # egnn_out = self.egnn_attn(egnn_out)
        # image_out = self.vggnet_attn(image_out)
        # Find the maximum size along the dimension that needs to match
        max_size = max(fpn_out.size(0), gcn_out.size(0), gcn_out1.size(0), egnn_out.size(0), image_out.size(0))

        # Pad tensors to match the maximum size
        def pad_tensor(tensor, max_size):
            if tensor.size(0) < max_size:
                pad_size = max_size - tensor.size(0)
                padding = torch.zeros(pad_size, tensor.size(1)).to(tensor.device)
                tensor = torch.cat((tensor, padding), dim=0)
            return tensor

        fpn_out = pad_tensor(fpn_out, max_size)
        gcn_out = pad_tensor(gcn_out, max_size)
        gcn_out1 = pad_tensor(gcn_out1, max_size)
        egnn_out = pad_tensor(egnn_out, max_size)
        image_out = pad_tensor(image_out, max_size)

        # Ensure no NaN values are present
        def replace_nan(tensor):
            return torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)

        fpn_out = replace_nan(fpn_out)
        gcn_out = replace_nan(gcn_out)
        gcn_out1 = replace_nan(gcn_out1)
        egnn_out = replace_nan(egnn_out)
        image_out = replace_nan(image_out)

        #output = torch.cat([fpn_out, gcn_out1, egnn_out, image_out], axis=1)

        #output = self.attn(output)
        #print('output:', output.size())
        #output =torch.cat([gcn_out,image_out])
        # output = fpn_out+gcn_out+egnn_out+image_out+gcn_out1
        gcn_out = torch.where(torch.isnan(gcn_out), torch.zeros_like(gcn_out), gcn_out)
        gcn_out1 = torch.where(torch.isnan(gcn_out1), torch.zeros_like(gcn_out1), gcn_out1)
        # 第一轮交叉注意力  代码SG-ATT
        gcn_out = gcn_out + self.cross_attention1(gcn_out, gcn_out1, gcn_out1)  # fpn_out作为Q，gcn_out作为K和V
        gcn_out1 = gcn_out1 + self.cross_attention1(gcn_out1, gcn_out, gcn_out)  # fpn_out作为Q，gcn_out作为K和V

        # FC层
        gcn_out = self.encoder2_FC1(gcn_out)
        gcn_out = self.act_func1(gcn_out)
        gcn_out1 = self.encoder5_FC1(gcn_out1)
        gcn_out1 = self.act_func1(gcn_out1)
        # 门控融合
        weights = torch.sigmoid(self.gate_layer(gcn_out + gcn_out1))  # (30, 1)
        features = weights * gcn_out + (1 - weights) * gcn_out1  # (30, 64)

        #features = torch.cat([gcn_out, gcn_out1], dim=1)  # (30,192)(30,96)
        #output = fpn_out + egnn_out + image_out + gcn_out + gcn_out1
        #output = fpn_out + egnn_out + image_out + features
        output = torch.cat([fpn_out, egnn_out, image_out, features], dim=1)
        #output = fpn_out + egnn_out + image_out + gcn_out
        output = self.ffn(output)

        #保存特征
        #graph_out = output
        #graph = graph_out.cpu().numpy()
        #graph = graph.tolist()
        #feature_out.append(graph)

        if torch.isnan(output).any():
            raise ValueError("NaN detected in output")

        if self.is_classif and not self.training:
            output = self.sigmoid(output)

        return output


def get_features_out():
    return feature_out


def FPGNN(args):  #初始化和配置一个图神经网络模型FPGNN
    if args.dataset_type == 'classification':
        is_classif = 1
    else:
        is_classif = 0     #回归任务设为0
    model = FpgnnModel(is_classif,args.cuda,args.dropout_fpn, args.input_dim)   #创建 FpgnnModel 类的实例 model，并将分类标志、是否使用 CUDA 以及 FPN（特征金字塔网络）的 dropout 比率作为参数传递
    model.create_fpn(args)    #调用 model 的 create_fpn 方法来创建特征金字塔网络部分，使用 args 中的参数
    model.create_gnn(args)
    model.create_gnn1(args)
    model.create_egnn(args)
    model.create_imagecnn(args)
    model.create_fc(args)
    model.create_ffn(args)
    model.create_fc1(args)
    #model.create_att(args)

    for param in model.parameters():     #循环遍历模型 model 中的所有参数
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

    return model