import copy
import os
import time

import tqdm
from sklearn.metrics import *
from tqdm import tqdm
#from utils.metrics import *
from zmq import device

from .layers import *
from utils.metrics import metrics, get_confusionmatrix_fnd


class Trainer3():
    def __init__(self,
                model, 
                 device,
                 lr,
                 dropout,
                 dataloaders,
                 weight_decay,
                 save_param_path,
                 writer, 
                 epoch_stop,
                 epoches,
                 model_name, 
                 event_num,
                 save_threshold = 0.0, 
                 start_epoch = 0,
                 ):
        
        self.model = model
        self.device = device
        self.model_name = model_name
        self.event_num = event_num
        self.dataloaders = dataloaders
        self.start_epoch = start_epoch
        self.num_epochs = epoches
        self.epoch_stop = epoch_stop
        self.save_threshold = save_threshold
        self.writer = writer

        if os.path.exists(save_param_path):
            self.save_param_path = save_param_path
        else:
            self.save_param_path = os.makedirs(save_param_path)
            self.save_param_path= save_param_path

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout

        self.criterion = nn.CrossEntropyLoss()
        

    def train(self):

        since = time.time()

        self.model.cuda()

        best_model_wts_val = copy.deepcopy(self.model.state_dict())
        best_acc_val = 0.0
        best_epoch_val = 0

        is_earlystop = False

        # if self.mode == "eann":
        #     best_acc_val_event = 0.0
        #     best_epoch_val_event = 0

        for epoch in range(self.start_epoch, self.start_epoch+self.num_epochs):
            if is_earlystop:
                break
            print('-' * 50)
            print('Epoch {}/{}'.format(epoch+1, self.start_epoch+self.num_epochs))
            print('-' * 50)

            p = float(epoch) / 100
            lr = self.lr / (1. + 10 * p) ** 0.75
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
            
            for phase in ['train', 'val', 'test']:
                if phase == 'train':
                    self.model.train()  
                else:
                    self.model.eval()   
                print('-' * 10)
                print (phase.upper())
                print('-' * 10)

                running_loss_fnd = 0.0
                running_loss = 0.0 
                tpred = []
                tlabel = []


                for batch in tqdm(self.dataloaders[phase]):
                    batch_data=batch
                    for k,v in batch_data.items():
                        batch_data[k]=v.cuda()
                    label = batch_data['label']


                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        image_only_output,text_only_output,audio_only_output,cc_only_output,outputs,fea = self.model(**batch_data)
                        _, preds = torch.max(outputs, 1)
                        
                        # loss = self.criterion(outputs, label)
                        
                        loss_ce= self.criterion(outputs, label)
                        loss_t = self.criterion(text_only_output, label)
                        loss_i = self.criterion(image_only_output, label)
                        loss_au = self.criterion(audio_only_output,label)
                        loss_cc = self.criterion(cc_only_output,label)
                        coarse_loss = (loss_t + loss_i + loss_au )/3
                        loss = loss_ce + 0.5 * coarse_loss + 0.5 * loss_cc
                        # print(loss_ce)
                        # print(loss_t)
                        # print(loss_i)
                        # print(loss_au)
                        # print(loss_cc)
                        # if hasattr(torch.cuda, 'empty_cache'):
                        #     torch.cuda.empty_cache()
                        
                        
                        if phase == 'train':
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            #if hasattr(torch.cuda, 'empty_cache'):
                            #    torch.cuda.empty_cache()
                            self.optimizer.step()
                            self.optimizer.zero_grad()

                    tlabel.extend(label.detach().cpu().numpy().tolist())
                    tpred.extend(preds.detach().cpu().numpy().tolist())
                    running_loss += loss.item() * label.size(0)

                    
                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)            
                print('Loss: {:.4f} '.format(epoch_loss))
                results = metrics(tlabel, tpred)
                print (results)
                self.writer.add_scalar('Loss/'+phase, epoch_loss, epoch+1)
                self.writer.add_scalar('Acc/'+phase, results['acc'], epoch+1)
                self.writer.add_scalar('F1/'+phase, results['f1'], epoch+1)


                if phase == 'test' and results['acc'] > best_acc_val:
                    best_acc_val = results['acc']
                    best_model_wts_val = copy.deepcopy(self.model.state_dict())
                    best_epoch_val = epoch+1
                    if best_acc_val > self.save_threshold:
                        torch.save(self.model.state_dict(), self.save_param_path + "_val_epoch" + str(best_epoch_val) + "_{0:.4f}".format(best_acc_val))
                        print ("saved " + self.save_param_path + "_val_epoch" + str(best_epoch_val) + "_{0:.4f}".format(best_acc_val) )
                    else:
                        if epoch-best_epoch_val >= self.epoch_stop-1:
                            is_earlystop = True
                            print ("early stopping...")
                
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print("Best model on val: epoch" + str(best_epoch_val) + "_" + str(best_acc_val))

        self.model.load_state_dict(best_model_wts_val)

        print ("test result when using best model on val")
        return self.test()   



    def test(self):
        since = time.time()

        self.model.cuda()
        self.model.eval()   

        pred = []
        label = []

        # if self.mode == "eann":
        #     pred_event = []
        #     label_event = []

        for batch in tqdm(self.dataloaders['test']):
            with torch.no_grad(): 
                batch_data=batch
                for k,v in batch_data.items():
                    batch_data[k]=v.cuda()
                batch_label = batch_data['label']
                batch_outputs = self.model(**batch_data)

                _, batch_preds = torch.max(batch_outputs, 1)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_preds.detach().cpu().numpy().tolist())


        print (get_confusionmatrix_fnd(np.array(pred), np.array(label)))
        print (metrics(label, pred))

        return metrics(label, pred)
    
