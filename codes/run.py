
import collections

from distutils.version import LooseVersion
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from gensim.models import KeyedVectors
from models.SVFEND import SVFENDModel


import sys
sys.path.append('models/utils')
from models.utils.dataloader import *
from models.Trainer import Trainer
from models.Trainer_3set import Trainer3


def pad_sequence(seq_len,lst, emb):
    result=[]
    for video in lst:
        if isinstance(video, list):
            video = torch.stack(video)
        ori_len=video.shape[0]
        if ori_len == 0:
            video = torch.zeros([seq_len,emb],dtype=torch.long)
        elif ori_len>=seq_len:
            if emb == 200:
                video=torch.FloatTensor(video[:seq_len])
            else:
                video=torch.LongTensor(video[:seq_len])
        else:
            video=torch.cat([video,torch.zeros([seq_len-ori_len,video.shape[1]],dtype=torch.long)],dim=0)
            if emb == 200:
                video=torch.FloatTensor(video)
            else:
                video=torch.LongTensor(video)
        result.append(video)
    return torch.stack(result)

# def pad_sequence_bbox(seq_len,lst):
#     result=[]
#     for video in lst:
#         if isinstance(video, list):
#             video = torch.stack(video)
#         ori_len=video.shape[0]
#         if ori_len == 0:
#             video = torch.zeros([seq_len,45,4096],dtype=torch.float)
#         elif ori_len>=seq_len:
#             video=torch.FloatTensor(video[:seq_len])
#         else:
#             video=torch.cat([video,torch.zeros([seq_len-ori_len,45,4096],dtype=torch.float)],dim=0)
#         result.append(video)
#     return torch.stack(result)

def pad_frame_sequence(seq_len,lst):
    attention_masks = []
    result=[]
    for video in lst:
        video=torch.FloatTensor(video)
        ori_len=video.shape[0]
        if ori_len>=seq_len:
            gap=ori_len//seq_len
            video=video[::gap][:seq_len]
            mask = np.ones((seq_len))
        else:
            video=torch.cat((video,torch.zeros([seq_len-ori_len,video.shape[1]],dtype=torch.float)),dim=0)
            mask = np.append(np.ones(ori_len), np.zeros(seq_len-ori_len))
        result.append(video)
        mask = torch.IntTensor(mask)
        attention_masks.append(mask)
    return torch.stack(result), torch.stack(attention_masks)

def pad_audio_sequence(seq_len,lst):
    attention_masks = []
    result=[]
    for video in lst:
        video = torch.squeeze(video)
        # video=torch.FloatTensor(video)
        ori_len=video.shape[0]
        if ori_len>=seq_len:
            gap=ori_len//seq_len
            video=video[::gap][:seq_len]
            mask = np.ones((seq_len))
        else:
            video=torch.cat((video,torch.zeros([seq_len-ori_len,video.shape[1]],dtype=torch.float)),dim=0)
            mask = np.append(np.ones(ori_len), np.zeros(seq_len-ori_len))
        result.append(video)
        mask = torch.IntTensor(mask)
        attention_masks.append(mask)
    return torch.stack(result), torch.stack(attention_masks)

def _init_fn(worker_id):
    np.random.seed(2022)

def SVFEND_collate_fn(batch): 
    # num_comments = 23
    num_frames = 83
    num_audioframes = 50 

    # intro_inputid = [item['intro_inputid'] for item in batch]
    # intro_mask = [item['intro_mask'] for item in batch]

    title_inputid = [item['title_inputid'] for item in batch]
    title_mask = [item['title_mask'] for item in batch]

    # comments_like = [item['comments_like'] for item in batch]
    # comments_inputid = [item['comments_inputid'] for item in batch]
    # comments_mask = [item['comments_mask'] for item in batch]
    #
    # comments_inputid_resorted = []
    # comments_mask_resorted = []
    # comments_like_resorted = []

    # for idx in range(len(comments_like)):
    #     comments_like_one = comments_like[idx]
    #     comments_inputid_one = comments_inputid[idx]
    #     comments_mask_one = comments_mask[idx]
    #     if comments_like_one.shape != torch.Size([0]):
    #         comments_inputid_one, comments_mask_one, comments_like_one = (list(t) for t in zip(*sorted(zip(comments_inputid_one, comments_mask_one, comments_like_one), key=lambda s: s[2], reverse=True)))
    #     comments_inputid_resorted.append(comments_inputid_one)
    #     comments_mask_resorted.append(comments_mask_one)
    #     comments_like_resorted.append(comments_like_one)
    #
    # comments_inputid = pad_sequence(num_comments,comments_inputid_resorted,250)
    # comments_mask = pad_sequence(num_comments,comments_mask_resorted,250)
    # comments_like=[]
    # for idx in range(len(comments_like_resorted)):
    #     comments_like_resorted_one = comments_like_resorted[idx]
    #     if len(comments_like_resorted_one)>=num_comments:
    #         comments_like.append(torch.tensor(comments_like_resorted_one[:num_comments]))
    #     else:
    #         if isinstance(comments_like_resorted_one, list):
    #             comments_like.append(torch.tensor(comments_like_resorted_one+[0]*(num_comments-len(comments_like_resorted_one))))
    #         else:
    #             comments_like.append(torch.tensor(comments_like_resorted_one.tolist()+[0]*(num_comments-len(comments_like_resorted_one))))

    frames = [item['frames'] for item in batch]
    frames, frames_masks = pad_frame_sequence(num_frames, frames)

    # audioframes  = [item['audioframes'] for item in batch]
    # audioframes, audioframes_masks = pad_frame_sequence(num_audioframes, audioframes)

    audio_feas = [item['audio_fea'] for item in batch]
    audio_feas, audiofeas_masks = pad_audio_sequence(num_audioframes, audio_feas)

    c3d  = [item['c3d'] for item in batch]
    c3d, c3d_masks = pad_frame_sequence(num_frames, c3d)

    label = [item['label'] for item in batch]

    return {
        'label': torch.stack(label),
        # 'intro_inputid': torch.stack(intro_inputid),
        # 'intro_mask': torch.stack(intro_mask),
        'title_inputid': torch.stack(title_inputid),
        'title_mask': torch.stack(title_mask),
        # 'comments_inputid': comments_inputid,
        # 'comments_mask': comments_mask,
        # 'comments_like': torch.stack(comments_like),
        'audio_feas': audio_feas,
        'audiofeas_masks': audiofeas_masks,
        # 'audioframes': audioframes,
        # 'audioframes_masks': audioframes_masks,
        'frames':frames,
        'frames_masks': frames_masks,
        'c3d': c3d,
        'c3d_masks': c3d_masks,
    }

class Run():
    def __init__(self,
                 config
                 ):

        self.model_name = config['model_name']
        self.mode_eval = config['mode_eval']
        self.fold = config['fold']
        self.data_type = 'SVFEND'

        self.epoches = config['epoches']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.epoch_stop = config['epoch_stop']
        self.seed = config['seed']
        self.device = config['device']
        self.lr = config['lr']
        self.lambd=config['lambd']
        self.save_param_dir = config['path_param']
        self.path_tensorboard = config['path_tensorboard']
        self.dropout = config['dropout']
        self.weight_decay = config['weight_decay']
        self.event_num = 616 
        self.mode ='normal'
    

    def get_dataloader(self,data_type,data_fold):
        collate_fn=None

        if data_type=='SVFEND':
            dataset_train = SVFENDDataset(f'vid_fold_no_{data_fold}.txt')
            dataset_test = SVFENDDataset(f'vid_fold_{data_fold}.txt')
            collate_fn=SVFEND_collate_fn

        train_dataloader = DataLoader(dataset_train, batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            worker_init_fn=_init_fn,
            collate_fn=collate_fn)

        test_dataloader=DataLoader(dataset_test, batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            worker_init_fn=_init_fn,
            collate_fn=collate_fn)

        dataloaders =  dict(zip(['train', 'test'],[train_dataloader, test_dataloader]))

        return dataloaders


    def get_dataloader_temporal(self, data_type):
        collate_fn=None
        bert_token = pretrain_bert_token()
        if data_type=='SVFEND':
            dataset_train = SVFENDDataset(tokenizer=bert_token,path_vid='vid_time3_train.txt')
            dataset_val = SVFENDDataset(tokenizer=bert_token,path_vid='vid_time3_val.txt')
            dataset_test = SVFENDDataset(tokenizer=bert_token,path_vid='vid_time3_test.txt')
            collate_fn=SVFEND_collate_fn
        else:
            # can be added
            print ("Not available")

        train_dataloader = DataLoader(dataset_train, batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            worker_init_fn=_init_fn,
            collate_fn=collate_fn)
        val_dataloader = DataLoader(dataset_val, batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            worker_init_fn=_init_fn,
            collate_fn=collate_fn)
        test_dataloader=DataLoader(dataset_test, batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            worker_init_fn=_init_fn,
            collate_fn=collate_fn)
 
        dataloaders =  dict(zip(['train', 'val', 'test'],[train_dataloader, val_dataloader, test_dataloader]))

        return dataloaders

    def get_model(self):
        if self.model_name == 'SVFEND':
            self.model = SVFENDModel(bert_model='bert-base-chinese', fea_dim=128,dropout=self.dropout)
        return self.model


    def main(self):

        if self.mode_eval == "nocv":
        #with torch.no_grad():
            self.model = self.get_model()
            dataloaders = self.get_dataloader(data_type=self.data_type, data_fold=self.fold)
            trainer = Trainer(model=self.model, device = self.device, lr = self.lr, dataloaders = dataloaders, epoches = self.epoches, dropout = self.dropout, weight_decay = self.weight_decay, model_name = self.model_name, event_num = self.event_num,
                    epoch_stop = self.epoch_stop, save_param_path = self.save_param_dir+self.data_type+"/"+self.model_name+"/", writer = SummaryWriter(self.path_tensorboard))
            result=trainer.train()

        elif self.mode_eval == "temporal":
            self.model = self.get_model()
            dataloaders = self.get_dataloader_temporal(data_type=self.data_type)
            trainer = Trainer3(model=self.model, device = self.device, lr = self.lr, dataloaders = dataloaders, epoches = self.epoches, dropout = self.dropout, weight_decay = self.weight_decay, model_name = self.model_name, event_num = self.event_num,
                    epoch_stop = self.epoch_stop, save_param_path = self.save_param_dir+self.data_type+"/"+self.model_name+"/", writer = SummaryWriter(self.path_tensorboard))
            result=trainer.train()
            return result
        
        elif self.mode_eval == "cv":
            collate_fn=None
            if self.model_name == 'TextCNN':
                wv_from_text = KeyedVectors.load_word2vec_format("./stores/tencent-ailab-embedding-zh-d100-v0.2.0-s/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt", binary=False)

            history = collections.defaultdict(list)


            for fold in range(1, 6): 
                print('-' * 50)
                print ('fold %d:' % fold)
                print('-' * 50)
                self.model = self.get_model()
                dataloaders = self.get_dataloader(data_type=self.data_type, data_fold=fold)
                trainer = Trainer(model = self.model, device = self.device, lr = self.lr, dataloaders = dataloaders, epoches = self.epoches, dropout = self.dropout, weight_decay = self.weight_decay, mode = self.mode, model_name = self.model_name, event_num = self.event_num, 
                    epoch_stop = self.epoch_stop, save_param_path = self.save_param_dir+self.data_type+"/"+self.model_name+"/", writer = SummaryWriter(self.path_tensorboard+"fold_"+str(fold)+"/"))
       
                result = trainer.train()

                history['auc'].append(result['auc'])
                history['f1'].append(result['f1'])
                history['recall'].append(result['recall'])
                history['precision'].append(result['precision'])
                history['acc'].append(result['acc'])
                
            print ('results on 5-fold cross-validation: ')
            for metric in ['acc', 'f1', 'precision', 'recall', 'auc']:
                print ('%s : %.4f +/- %.4f' % (metric, np.mean(history[metric]), np.std(history[metric])))
        
        else:
            print ("Not Available")
