import copy
import json
import os
import time
import resampy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import tqdm
import soundfile as sf
from sklearn.metrics import *
from tqdm import tqdm
from transformers import AutoConfig, BertModel
from transformers.models.bert.modeling_bert import BertLayer
from zmq import device
from layers import *
from coattention import *
# from bootstrap import *
from tools import *
from moe import *
class SVFENDModel(torch.nn.Module):
    def __init__(self,bert_model,fea_dim,dropout):
        super(SVFENDModel, self).__init__()
        #加载预训练bert模型、停止梯度计算
        self.bert = pretrain_bert_models().requires_grad_(False)
        self.text_dim = 1024
        # self.comment_dim = 768
        self.img_dim = 1024
        self.video_dim = 4096
        self.num_frames = 83
        self.num_audioframes = 50
        self.num_comments = 23
        self.dim = fea_dim
        self.num_heads = 12
        out_dim = 1

        self.dropout = dropout
        self.audio_dim = 1024
        self.attention = Attention(dim=self.dim,heads=4,dropout=dropout)
        self.unified_dim=512
        self.trans_dim = 512
        #对音频的处理
        # torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        # self.vggish_layer = torch.hub.load('harritaylor/torchvggish', 'vggish', source = 'github', force_reload ='True')
        # net_structure = list(self.vggish_layer.children())
        # self.vggish_modified = nn.Sequential(*net_structure[-2:-1])

        self.num_expert = 3
        #共同注意力机制
        #文本和音频的
        # self.co_attention_ta = co_attention(d_k=fea_dim, d_v=fea_dim, n_heads=self.num_heads, dropout=self.dropout, d_model=fea_dim,
        #                                 visual_len=self.num_audioframes, sen_len=512, fea_v=self.dim, fea_s=self.dim, pos=False)
        # self.co_attention_tv = co_attention(d_k=fea_dim, d_v=fea_dim, n_heads=self.num_heads, dropout=self.dropout, d_model=fea_dim,
        #                                 visual_len=self.num_frames, sen_len=512, fea_v=self.dim, fea_s=self.dim, pos=False)

        #transformer,d_model:输入维度，nhead:头

        # self.co_attention_ca = co_attention(d_k=fea_dim, d_v=fea_dim, n_heads=self.num_heads, dropout=self.dropout, d_model=fea_dim,
        #                                  visual_len=self.num_audioframes, sen_len=512, fea_v=self.dim, fea_s=self.dim, pos=False)



        self.trm = nn.TransformerEncoderLayer(d_model = self.trans_dim, nhead = 2, batch_first = True)
        self.linear_text = nn.Sequential(torch.nn.Linear(self.text_dim, self.trans_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        # self.linear_comment = nn.Sequential(torch.nn.Linear(self.comment_dim, fea_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_img = nn.Sequential(torch.nn.Linear(self.img_dim, self.trans_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        # self.linear_video = nn.Sequential(torch.nn.Linear(self.video_dim, fea_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        # self.linear_intro = nn.Sequential(torch.nn.Linear(self.text_dim, fea_dim),torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_audio = nn.Sequential(torch.nn.Linear(self.audio_dim, self.trans_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))

        self.classifier = nn.Linear(fea_dim,2)
        self.final_linear = nn.Sequential(torch.nn.Linear(self.trans_dim, fea_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.depth = 1

        self.txt_moe = moe(self.num_expert,self.unified_dim,self.depth)
        self.img_moe = moe(self.num_expert, self.unified_dim, self.depth)
        self.audio_moe = moe(self.num_expert, self.unified_dim, self.depth)
        # self.trans_dim = 128
        # 共注意力模块
        self.tv = Transformer(model_dimension=self.trans_dim, number_of_heads=self.num_heads,
                              dropout_probability=self.dropout)
        # self.vt = Transformer(model_dimension=self.trans_dim, number_of_heads=self.num_heads,
                              # dropout_probability=self.dropout)
        
        
        
        self.at = Transformer(model_dimension=self.trans_dim, number_of_heads=self.num_heads,
                                  dropout_probability=self.dropout)
        # self.ta = Transformer(model_dimension=self.trans_dim, number_of_heads=self.num_heads,
                              # dropout_probability=self.dropout)
        

        #TokenAttention
        # self.image_attention = TokenAttention(self.unified_dim)
        # self.text_attention = TokenAttention(self.unified_dim)
        # self.audio_attention = TokenAttention(self.unified_dim)
        #image experts,text experts and mm experts
        # image_expert_list, text_expert_list = [], []
        # image_expert_list,text_expert_list,audio_expert_list = [], [], []
        # for i in range(self.num_expert):
        #     image_expert = []
        #     for j in range(self.depth):
        #         image_expert.append(Block(dim=self.unified_dim, num_heads=8))  # note: need to output model[:,0]
        #
        #     image_expert = nn.ModuleList(image_expert)
        #     image_expert_list.append(image_expert)

        # for i in range(self.num_expert):
        #     text_expert = []
        #     for j in range(self.depth):
        #         text_expert.append(Block(dim=self.unified_dim, num_heads=8))
        #
        #     text_expert = nn.ModuleList(text_expert)
        #     text_expert_list.append(text_expert)

        # for i in range(self.num_expert):
        #     audio_expert = []
        #     for j in range(self.depth):
        #         audio_expert.append(Block(dim=self.unified_dim, num_heads=8))
        #
        #     audio_expert = nn.ModuleList(audio_expert)
        #     audio_expert_list.append(audio_expert)



        # self.image_experts = nn.ModuleList(image_expert_list)
        # self.text_experts = nn.ModuleList(text_expert_list)
        # self.audio_experts = nn.ModuleList(audio_expert_list)

        #gates
        # self.image_gate_mae = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
        #                                     nn.SiLU(),
        #                                     nn.Linear(self.unified_dim, self.num_expert),
        #                                     )
        # self.text_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
        #                                nn.SiLU(),
        #                                nn.Linear(self.unified_dim, self.num_expert),
        #                                )
        # self.audio_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
        #                                nn.SiLU(),
        #                                nn.Linear(self.unified_dim, self.num_expert),
        #                                )
        #
        # self.image_gate_mae_1 = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
        #                                       nn.SiLU(),
        #                                       # SimpleGate(),
        #                                       # nn.BatchNorm1d(int(self.unified_dim/2)),
        #                                       nn.Linear(self.unified_dim, self.num_expert),
        #                                       )
        #
        # self.text_gate_1 = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
        #                                  nn.SiLU(),
        #                                  # SimpleGate(),
        #                                  # nn.BatchNorm1d(int(self.unified_dim/2)),
        #                                  nn.Linear(self.unified_dim, self.num_expert),
        #                                  )
        # self.audio_gate_1 = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
        #                                  nn.SiLU(),
        #                                  nn.Linear(self.unified_dim, self.num_expert),
        #                                  )

        #分类头
        #文本分类头
        # self.text_trim = nn.Sequential(
        #     nn.Linear(self.unified_dim, 64),
        #     nn.SiLU(),
        #     # SimpleGate(),
        #     # nn.BatchNorm1d(64),
        #     # nn.Dropout(0.2),
        # )
        self.text_alone_classifier = nn.Sequential(
            nn.Linear(self.unified_dim, 2),
        )
        # 图片分类头
        # self.image_trim = nn.Sequential(
        #     nn.Linear(self.unified_dim, 64),
        #     nn.SiLU(),
        #     # SimpleGate(),
        #     # nn.BatchNorm1d(64),
        #     # nn.Dropout(0.2),
        # )
        self.image_alone_classifier = nn.Sequential(
            nn.Linear(self.unified_dim,2),
        )
        #语音分类头
        # self.audio_trim = nn.Sequential(
        #     nn.Linear(self.unified_dim, 64),
        #     nn.SiLU(),
        # )
        self.audio_alone_classifier = nn.Sequential(
            nn.Linear(self.unified_dim, 2),
        )
        #混合分类头
#         self.mix_trim = nn.Sequential(
#             nn.Linear(self.unified_dim, 64),
#             nn.SiLU(),

#         )
        self.mix_classifier = nn.Sequential(
            nn.Linear(self.unified_dim, 2),
        )


       

    def forward(self,  **kwargs):

        ### User Intro ###
        # intro_inputid = kwargs['intro_inputid']
        # intro_mask = kwargs['intro_mask']
        # fea_intro = self.bert(intro_inputid,attention_mask=intro_mask)[1]
        # fea_intro = self.linear_intro(fea_intro)

        ### Title ###
        title_inputid = kwargs['title_inputid']  # (batch,512)
        title_mask = kwargs['title_mask']  # (batch,512)

        fea_text = self.bert(title_inputid, attention_mask=title_mask)['last_hidden_state']  # (batch,sequence,768)
        #text_atn_feature, _ = self.text_attention(fea_text)
        fea_text = self.linear_text(fea_text)
        # text_atn_feature, _ = self.text_attention(fea_text)


        ### Audio Frames ###
        fea_audio = kwargs['audio_feas']
        # audioframes=kwargs['audioframes']#(batch,36,12288)
        # audioframes_masks = kwargs['audioframes_masks']
        # fea_audio = self.vggish_modified(audioframes) #(batch, frames, 128)
        # fea_audio = F.avg_pool1d(fea_audio.transpose(1,2),128)
        fea_audio = self.linear_audio(fea_audio)
        # audio_atn_feature, _ = self.audio_attention(fea_audio)




        ### Image Frames ###
        frames=kwargs['frames']#(batch,30,4096)
        frames_masks = kwargs['frames_masks']
        fea_img = self.linear_img(frames)
        # fea_img, fea_text = self.co_attention_tv(v=fea_img, s=fea_text, v_len=fea_img.shape[1], s_len=fea_text.shape[1])
        # fea_img = torch.mean(fea_img, -2)
        # fea_text = torch.mean(fea_text, -2)
        # image_atn_feature, _ = self.image_attention(fea_img)


        # shared_image_feature_1 = fea_img
        # shared_image_feature = torch.mean(fea_img, dim=-2)
        # shared_text_feature_1 = fea_text
        # shared_text_feature = torch.mean(fea_img, dim=-2)
        # shared_audio_feature_1 = fea_audio
        # shared_audio_feature = torch.mean(fea_audio, dim=-2)
        
        
        shared_image_feature, shared_image_feature_1 = self.img_moe(fea_img)
        shared_text_feature, shared_text_feature_1 = self.txt_moe(fea_text)
        shared_audio_feature,shared_audio_feature_1 = self.audio_moe(fea_audio)#2*50*128
        
        
        
        # 共注意层
        fea_ta = self.at(shared_text_feature_1, shared_audio_feature_1)
        # fea_at = self.ta(shared_audio_feature_1, shared_text_feature_1)
        # fea_co1 = torch.cat((fea_ta, fea_at), 1)
        fea_tav = self.tv(fea_ta, shared_image_feature_1)
        # fea_vt = self.vt(shared_image_feature_1, fea_co1)
        # fea_co = torch.cat((fea_vt, fea_tv), 1)
        
        
        
        cc_fea = torch.mean(fea_tav,dim=-2)
        # cc_fea = self.cc_liner(torch.cat((shared_text_feature_1,shared_image_feature_1,shared_audio_feature_1),dim=1))
      
        cc_only_output = self.mix_classifier(cc_fea)
        image_only_output = self.image_alone_classifier(shared_image_feature)
        text_only_output = self.text_alone_classifier(shared_text_feature)
        audio_only_output = self.audio_alone_classifier(shared_audio_feature)
       

        shared_image_feature = shared_image_feature.unsqueeze(1)
        shared_text_feature = shared_text_feature.unsqueeze(1)
        shared_audio_feature = shared_audio_feature.unsqueeze(1)
        
        cc_fea = cc_fea.unsqueeze(1)
        #全连接
        fea = torch.cat((shared_text_feature,shared_audio_feature,shared_image_feature,cc_fea),1) # (bs, 6, 128)
        #transformer
        fea = self.trm(fea)
        #求平均
        fea = torch.mean(fea, -2)

        
        output = self.classifier(self.final_linear(fea))

        return image_only_output, text_only_output,audio_only_output,cc_only_output,output, fea


