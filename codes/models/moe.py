import torch
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from bootstrap import *


class MLP(nn.Module):
    def __init__(self,dim,hidden_dim):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(self.dim,self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim,self.dim)

    def forward(self,x):
        x = self.linear1(x)
        nn.SiLU()
        x = self.linear2(x)
        nn.SiLU()
        return x

class moe(nn.Module):
    def __init__(self,num_expert,unified_dim,depth):
        super().__init__()
        self.num_expert = num_expert
        self.unified_dim = unified_dim
        self.depth = depth
        self.token_attention = TokenAttention(self.unified_dim)
        self.expert_list = []
        for i in range(self.num_expert):
            experts = []
            for j in range(self.depth):
                # experts.append(Block(dim=self.unified_dim, num_heads=8))  # note: need to output model[:,0]
                experts.append(MLP(dim=self.unified_dim, hidden_dim=self.unified_dim*2))
            experts = nn.ModuleList(experts)
            self.expert_list.append(experts)
        self.experts = nn.ModuleList(self.expert_list)
        self.gate1 = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                            nn.SiLU(),
                                            nn.Linear(self.unified_dim, self.num_expert),
                                            )
        self.gate2 = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                       nn.SiLU(),
                                       nn.Linear(self.unified_dim, self.num_expert),
                                       )

    def forward(self, x):
        x_attention,_ = self.token_attention(x)
        gate_weights1 = self.gate1(x_attention)
        gate_weights2 = self.gate2(x_attention)
        feature, feature1 = 0, 0
        for i in range(self.num_expert):
            expert = self.experts[i]
            tmp_feature = x
            for j in range(self.depth):
                tmp_feature = expert[j](tmp_feature)
            feature += (tmp_feature * gate_weights1[:, i].unsqueeze(1).unsqueeze(1))
            feature1 += (tmp_feature * gate_weights2[:, i].unsqueeze(1).unsqueeze(1))
        feature = feature[:, 0]
        # feature1 = feature1[:, 0]
        return feature,feature1



# input=torch.randn(16,50,128)
# model = moe(3,128,1)
# out,out2=model(input)
class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_hid))
        self.beta = nn.Parameter(torch.zeros(d_hid))
        self.eps = eps

    def forward(self, z):
        mean = z.mean(dim=-1, keepdim=True, )
        std = z.std(dim=-1, keepdim=True, )
        ln_out = (z - mean) / (std + self.eps)
        ln_out = self.gamma * ln_out + self.beta

        return ln_out
class Transformer(nn.Module):
    def __init__(self, model_dimension, number_of_heads, dropout_probability,
                 log_attention_weights=False):
        super().__init__()
        encoder_layer = EncoderLayer(model_dimension, dropout_probability,number_of_heads)
        self.encoder = Encoder(encoder_layer)
        self.init_params()

    def init_params(self, default_initialization=False):
        if not default_initialization:
            # model.named_parameters 每一次迭代元素的名字和param。
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    # 初始化均匀分布的网络参数
                    nn.init.xavier_uniform_(p)

    def forward(self, input1, input2):
        # Q:input1 K,V:input2
        src_representations_batch1 = self.encoder(input1, input2)
        return src_representations_batch1
class Encoder(nn.Module):
    def __init__(self, encoder_layer):
        super().__init__()
        assert isinstance(encoder_layer, EncoderLayer), f'Expected EncoderLayer got {type(encoder_layer)}.'
        self.encoder_layer = encoder_layer
        self.ffn_layer = PoswiseFeedForwardNet(encoder_layer.model_dimension,encoder_layer.model_dimension*2)
        # self.ffn_layer = FeedForward(encoder_layer.model_dimension,encoder_layer.model_dimension*2)
    def forward(self, src1, src2):
        # Forward pass through the encoder stack
        src_representations_batch = self.encoder_layer(src1, src2)
        representations = self.ffn_layer(src_representations_batch)
        return representations

class EncoderLayer(nn.Module):
    def __init__(self, model_dimension, dropout_probability,number_of_heads):
        super().__init__()
        self.sublayer1 = SublayerLogic(model_dimension, dropout_probability)
        # self.sublayer2 = SublayerLogic(model_dimension, dropout_probability)
        self.mha1 = MultiHeadedAttention(model_dimension=model_dimension,number_of_heads=number_of_heads,dropout_probability=dropout_probability,log_attention_weights=False)
        # self.mha2 = MultiHeadedAttention(model_dimension=model_dimension,number_of_heads=number_of_heads,dropout_probability=dropout_probability,log_attention_weights=False)
        self.model_dimension = model_dimension
        self.norm = nn.LayerNorm(model_dimension)

    def forward(self, srb1, srb2):
        # 两层的多头注意
        encoder_self_attention1 = lambda srb1, srb2: self.mha1(query=srb1, key=srb2, value=srb2)
        # encoder_self_attention2 = lambda srb1, srb2: self.mha2(query=srb1, key=srb2, value=srb2)
        src_representations_batch = self.norm(self.sublayer1(srb1, srb2, encoder_self_attention1))
        # src_representations_batch = self.norm(self.sublayer2(src_representations_batch, srb2, encoder_self_attention2))
        return src_representations_batch
class SublayerLogic(nn.Module):
    def __init__(self, model_dimension, dropout_probability):
        super().__init__()
        self.norm = nn.LayerNorm(model_dimension)
        self.dropout = nn.Dropout(p=dropout_probability)
        # self.pos_emb_v = PosEncoding(input1_len * 10, model_dimension)
        # self.pos_emb_s = PosEncoding(input2_len * 10, model_dimension)
    def forward(self, srb1, srb2, mha):
        return srb1 + self.dropout(mha(self.norm(srb1), self.norm(srb2)))
class MultiHeadedAttention(nn.Module):
    def __init__(self, model_dimension, number_of_heads, dropout_probability, log_attention_weights):
        super().__init__()
        assert model_dimension % number_of_heads == 0, f'Model dimension must be divisible by the number of heads.'

        self.head_dimension = int(model_dimension / number_of_heads)
        self.number_of_heads = number_of_heads

        self.qkv_nets = get_clones(nn.Linear(model_dimension, model_dimension), 3)  # identity activation hence "nets"
        self.out_projection_net = nn.Linear(model_dimension, model_dimension)

        self.attention_dropout = nn.Dropout(p=dropout_probability)  # no pun intended, not explicitly mentioned in paper
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the softmax along the last dimension

        self.log_attention_weights = log_attention_weights  # should we log attention weights
        self.attention_weights = None  # for visualization purposes, I cache the weights here (translation_script.py)

    def attention(self, query, key, value):
        # query b*8*300*128, key b*8*49*128, value b*8*49*128
        # 送入softmax前对点积结果进行缩放
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dimension)
        # scores b*8*512*50
        attention_weights = self.softmax(scores)
        attention_weights = self.attention_dropout(attention_weights)
        # 调整attention维度
        # matrix = torch.ones(attention_weights.shape[3],attention_weights.shape[2]).cuda()
        # attention_weights = torch.matmul(matrix,attention_weights)
        # 根据mask_att从中取值
        intermediate_token_representations = torch.matmul(attention_weights, value)
        # intermediate_token_representations b*8*300*49

        return intermediate_token_representations, attention_weights  # attention weights for visualization purposes

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        # query b*300*1024, key b*49*1024, value b*49*1024
        query, key, value = [net(x).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
                             for net, x in zip(self.qkv_nets, (query, key, value))]
        # query b*8*300*128, key b*8*49*128, value b*8*49*128
        intermediate_token_representations, attention_weights = self.attention(query, key, value)

        if self.log_attention_weights:
            self.attention_weights = attention_weights
        reshaped = intermediate_token_representations.transpose(1, 2).reshape(batch_size, -1,
                                                                              self.number_of_heads * self.head_dimension)
        # forward
        # 合并多头注意力的结果，使用一个用于拼接的线性层
        token_representations = self.out_projection_net(reshaped)
        return token_representations
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, inputs):
        # inputs: [b_size x len_q x d_model]
        residual = inputs
        output = self.relu(self.conv1(inputs.transpose(1, 2)))
        # outputs: [b_size x len_q x d_model]
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)

        return self.layer_norm(residual + output)
def get_clones(module, num_of_deep_copies):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_deep_copies)])