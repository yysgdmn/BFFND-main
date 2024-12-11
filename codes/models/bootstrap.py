import torch
import torch.nn as nn
# import SVFEND
from timm.models.vision_transformer import Block


class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()


    #空间维度平均输出
    def mu(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the average across
        it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
        return torch.sum(x,(1))/(x.shape[1])

    #标准差表示
    def sigma(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
        across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
        the permutations are required for broadcasting"""
        return torch.sqrt((torch.sum((x.permute([1,0])-self.mu(x)).permute([1,0])**2,(1))+0.000000023)/(x.shape[1]))

    def forward(self, x, mu, sigma):
        """ Takes a content embeding x and a style embeding y and changes
        transforms the mean and standard deviation of the content embedding to
        that of the style. [See eq. 8 of paper] Note the permutations are
        required for broadcasting"""
        # print(mu.shape) # 12
        x_mean = self.mu(x)
        x_std = self.sigma(x)
        x_reduce_mean = x.permute([1, 0]) - x_mean
        x_norm = x_reduce_mean/x_std
        # print(x_mean.shape) # 768, 12
        return (sigma.squeeze(1)*(x_norm + mu.squeeze(1))).permute([1,0])

class TokenAttention(torch.nn.Module):
    def __init__(self, input_shape):
        super(TokenAttention, self).__init__()
        self.attention_layer = nn.Sequential(
                            torch.nn.Linear(input_shape, input_shape),
                            nn.SiLU(),
                            torch.nn.Linear(input_shape, 1),
        )

    def forward(self, inputs):
        scores = self.attention_layer(inputs)
        scores = scores.view(inputs.size(0),scores.shape[2], inputs.size(1))
        # scores = torch.softmax(scores, dim=-1).unsqueeze(1)
        # scores = scores.unsqueeze(1)
        #tensor连乘操作
        outputs = torch.matmul(scores, inputs).squeeze(1)
        # scores = self.attention_layer(inputs)
        # outputs = scores*inputs
        return outputs, scores
class MULTIVIEW(nn.Module):
    def __init__(self,
                 batch_size=64, text_token_len=197, imgframes_token_len=197, is_use_bce=True,
                 thresh=0.5
                 ):
        self.thresh = thresh
        self.batch_size = 64
        self.text_token_len, self.imgframes_token_len = text_token_len, imgframes_token_len
        self.unified_dim, self.text_dim = 768, 768
        self.is_use_bce = is_use_bce
        out_dim = 1 if self.is_use_bce else 2
        self.num_expert = 2
        self.depth = 1
        super(MULTIVIEW, self).__init__()




        self.image_attention = TokenAttention(self.unified_dim)
        self.mm_attention = TokenAttention(self.unified_dim)
        #experts
        image_expert_list, text_expert_list, mm_expert_list = [], [], []
        for i in range(self.num_expert):
            image_expert = []
            for j in range(self.depth):
                image_expert.append(Block(dim=self.unified_dim, num_heads=8))  # note: need to output model[:,0]

            image_expert = nn.ModuleList(image_expert)
            image_expert_list.append(image_expert)

        for i in range(self.num_expert):
            text_expert = []
            mm_expert = []
            for j in range(self.depth):
                text_expert.append(Block(dim=self.unified_dim, num_heads=8))
                mm_expert.append(Block(dim=self.unified_dim, num_heads=8))

            text_expert = nn.ModuleList(text_expert)
            text_expert_list.append(text_expert)
            mm_expert = nn.ModuleList(mm_expert)
            mm_expert_list.append(mm_expert)

        self.image_experts = nn.ModuleList(image_expert_list)
        self.text_experts = nn.ModuleList(text_expert_list)
        self.mm_experts = nn.ModuleList(mm_expert_list)
        #门
        self.image_gate_mae = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                            nn.SiLU(),
                                            nn.Linear(self.unified_dim, self.num_expert),
                                            )
        self.text_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                       nn.SiLU(),
                                       nn.Linear(self.unified_dim, self.num_expert),
                                       )
        self.mm_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                     nn.SiLU(),
                                     # SimpleGate(),
                                     # nn.BatchNorm1d(int(self.unified_dim/2)),
                                     nn.Linear(self.unified_dim, self.num_expert),
                                     )
        self.mm_gate_1 = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                       nn.SiLU(),
                                       # SimpleGate(),
                                       # nn.BatchNorm1d(int(self.unified_dim/2)),
                                       nn.Linear(self.unified_dim, self.num_expert),
                                       )

        self.image_gate_mae_1 = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                              nn.SiLU(),
                                              # SimpleGate(),
                                              # nn.BatchNorm1d(int(self.unified_dim/2)),
                                              nn.Linear(self.unified_dim, self.num_expert),
                                              )

        self.text_gate_1 = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                         nn.SiLU(),
                                         # SimpleGate(),
                                         # nn.BatchNorm1d(int(self.unified_dim/2)),
                                         nn.Linear(self.unified_dim, self.num_expert),
                                         )
        #不一致性
        self.irrelevant_tensor = nn.Parameter(torch.ones((1, self.unified_dim)), requires_grad=True)
        #分类头
        # 文本分类头
        self.text_trim = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
            # SimpleGate(),
            # nn.BatchNorm1d(64),
            # nn.Dropout(0.2),
        )
        self.text_alone_classifier = nn.Sequential(
            nn.Linear(64, out_dim),
        )
        # 图片分类头
        self.image_trim = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
            # SimpleGate(),
            # nn.BatchNorm1d(64),
            # nn.Dropout(0.2),
        )
        self.image_alone_classifier = nn.Sequential(
            nn.Linear(64, out_dim),
        )
        self.mix_trim = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),

        )
        self.mix_classifier = nn.Sequential(
            nn.Linear(64, out_dim),
        )


        #单视图预测器MLPs
        #keyframes
        self.mapping_Img_MLP_mu = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_Img_MLP_sigma = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        #Title
        self.mapping_T_MLP_mu = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_T_MLP_sigma = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_CC_MLP_mu = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_CC_MLP_sigma = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.adaIN = AdaIN()




########foward
    def forward(self, input_ids, attention_mask, token_type_ids, image, no_ambiguity,
                 category=None,
                calc_ambiguity=False, image_feature=None, text_feature=None,
                return_features=False):

        # print(input_ids.shape) # (24,197)
        # print(attention_mask.shape) # (24,197)
        # print(token_type_ids.shape) # (24,197)
        batch_size = image.shape[0]
        #if image_aug is None:
        #    image_aug = image


        # BASE FEATURE AND ATTENTION(base feature and attention)
        # if category is None:
        #     category = torch.zeros((batch_size))

        # IMAGE MAE:  OUTPUT IS (BATCH, 197, 768)
        ## FILTER OUT INVALID MODAL INFORMATION
        ##  NEWS TYPE: 0 MULTIMODAL 1 IMAGE 2 TEXT
        if image_feature is None:
          image_feature = self.image_model.forward_ying(image)

    #        image_feature = self.linear_img(frames)


        # TEXT:  INPUT IS (BATCH, WORDLEN, 768)
        ##  NEWS TYPE: 0 MULTIMODAL 1 IMAGE 2 TEXT
        # if self.dataset in self.LOW_BATCH_SIZE_AND_LR:
        #     text_feature = self.text_model(input_ids)
        # else:
        if text_feature is None:
            text_feature = self.text_model(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           token_type_ids=token_type_ids)[0]

        # print("text_feature size {}".format(text_feature.shape)) # 64,170,768
        # print("image_feature size {}".format(image_feature.shape)) # 64,197,1024
        text_atn_feature, _ = self.text_attention(text_feature)

        # IMAGE ATTENTION: OUTPUT IS (BATCH, 768)
        image_atn_feature, _ = self.image_attention(image_feature)

        mm_atn_feature, _ = self.mm_attention(torch.cat((image_feature, text_feature), dim=1))

        # print("text_atn_feature size {}".format(text_atn_feature.shape)) # 64, 768
        # print("image_atn_feature size {}".format(image_atn_feature.shape))
        # GATE
        gate_image_feature = self.image_gate_mae(image_atn_feature)
        gate_text_feature = self.text_gate(text_atn_feature)  # 64 320

        gate_mm_feature = self.mm_gate(mm_atn_feature)
        gate_mm_feature_1 = self.mm_gate_1(mm_atn_feature)

        # gate_image_feature_1 = self.image_gate_mae_1(image_atn_feature)
        # gate_text_feature_1 = self.text_gate_1(text_atn_feature)  # 64 320

        # IMAGE EXPERTS
        # NOTE: IMAGE/TEXT/MM EXPERTS WILL BE MLPS IF WE USE WWW LOADER
        shared_image_feature, shared_image_feature_1 = 0, 0
        for i in range(self.num_expert):
            image_expert = self.image_experts[i]
            tmp_image_feature = image_feature
            for j in range(self.depth):
                tmp_image_feature = image_expert[j](tmp_image_feature + self.positional_image)
            shared_image_feature += (tmp_image_feature * gate_image_feature[:, i].unsqueeze(1).unsqueeze(1))
        shared_image_feature = shared_image_feature[:, 0]
        # shared_image_feature_1 = shared_image_feature_1[:, 0]

        ## TEXT AND MM EXPERTS
        shared_text_feature, shared_text_feature_1 = 0, 0
        for i in range(self.num_expert):
            text_expert = self.text_experts[i]
            tmp_text_feature = text_feature
            for j in range(self.depth):
                tmp_text_feature = text_expert[j](tmp_text_feature + self.positional_text)  # text_feature: 64, 170, 768
            shared_text_feature += (tmp_text_feature * gate_text_feature[:, i].unsqueeze(1).unsqueeze(1))
            # shared_text_feature_1 += (tmp_text_feature * gate_text_feature_1[:, i].unsqueeze(1).unsqueeze(1))
        shared_text_feature = shared_text_feature[:, 0]
        # shared_text_feature_1 = shared_text_feature_1[:, 0]

        mm_feature = torch.cat((image_feature, text_feature), dim=1)
        # mm_feature = torch.cat((shared_image_feature_1, shared_text_feature_1), dim=1)
        shared_mm_feature, shared_mm_feature_CC = 0, 0
        for i in range(self.num_expert):
            mm_expert = self.mm_experts[i]
            tmp_mm_feature = mm_feature
            for j in range(self.depth):
                tmp_mm_feature = mm_expert[j](tmp_mm_feature + self.positional_mm)
            shared_mm_feature += (tmp_mm_feature * gate_mm_feature[:, i].unsqueeze(1).unsqueeze(1))
            shared_mm_feature_CC += (tmp_mm_feature * gate_mm_feature_1[:, i].unsqueeze(1).unsqueeze(1))
        shared_mm_feature = shared_mm_feature[:, 0]
        shared_mm_feature_CC = shared_mm_feature_CC[:, 0]
        shared_mm_feature_lite = self.aux_trim(shared_mm_feature_CC)
        aux_output = self.aux_classifier(shared_mm_feature_lite)  # final_feature_aux_task


        if calc_ambiguity:
            return aux_output, aux_output, aux_output

        ## UNIMODAL BRANCHES, NOT USED ANY MORE(unimodal branches,not used any more)
        # aux_output = aux_output.clone().detach()
        shared_image_feature_lite = self.image_trim(shared_image_feature)
        shared_text_feature_lite = self.text_trim(shared_text_feature)

        image_only_output = self.image_alone_classifier(shared_image_feature_lite)
        text_only_output = self.text_alone_classifier(shared_text_feature_lite)

        ## WEIGHTED MULTIMODAL FEATURES, REMEMBER TO DETACH AUX_OUTPUT
        # soft_scores = torch.softmax(torch.cat((aux_output,image_only_output,text_only_output,vgg_only_output),dim=1),dim=1)
        ## IF IMAGE-TEXT MATCHES, aux_output WOULD BE 0, OTHERWISE 1.
        #weighted multimodal features,remember to detach aux_output.if image-text matches
        aux_atn_score = 1 - torch.sigmoid(aux_output).clone().detach()  # torch.abs((torch.sigmoid(aux_output).clone().detach()-0.5)*2)
        is_mu = self.mapping_IS_MLP_mu(torch.sigmoid(image_only_output).clone().detach())
        t_mu = self.mapping_T_MLP_mu(torch.sigmoid(text_only_output).clone().detach())
        cc_mu = self.mapping_CC_MLP_mu(aux_atn_score.clone().detach())  # 1-aux_atn_score
        is_sigma = self.mapping_IS_MLP_sigma(torch.sigmoid(image_only_output).clone().detach())
        t_sigma = self.mapping_T_MLP_sigma(torch.sigmoid(text_only_output).clone().detach())
        cc_sigma = self.mapping_CC_MLP_sigma(aux_atn_score.clone().detach())  # 1-aux_atn_score

        shared_image_feature = self.adaIN(shared_image_feature,is_mu,is_sigma) #shared_image_feature * (image_atn_score)
        shared_text_feature = self.adaIN(shared_text_feature,t_mu,t_sigma) #shared_text_feature * (text_atn_score)
        shared_mm_feature = shared_mm_feature #shared_mm_feature #* (aux_atn_score)
        irr_score = torch.ones_like(shared_mm_feature)*self.irrelevant_tensor #torch.ones_like(shared_mm_feature).cuda()
        irrelevant_token = self.adaIN(irr_score,cc_mu,cc_sigma)
        concat_feature_main_biased = torch.stack((shared_image_feature,
                                                  shared_text_feature,
                                                  shared_mm_feature,
                                                  irrelevant_token
                                                  ), dim=1)


