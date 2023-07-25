"""
EEG Conformer 

Convolutional Transformer for EEG decoding

Couple CNN and Transformer in a concise manner with amazing results

Paper: https://ieeexplore.ieee.org/document/9991178

The dataset used: BCI Competition 2008 – Graz data set A, BCI Competition IV, 4 class motor imagery
Description: https://www.bbci.de/competition/iv/desc_2a.pdf
The dataset is 18 GDF files for 9 subjects. 2 files per subject (train and test) 

This code has been modified by Anton ANDREEV in order to access the dataset.

The code is hard coded for BCICIV_2a_gdf dataset.
Values that are hardcoded:
    - fs=250
    - epoch length to 1000
    - time epoch length (4 - (1/250))
    - the markers that generate the epochs

"""
# remember to change paths

import argparse
import os
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import numpy as np
import math
import glob
import random
import itertools
import datetime
import time
import datetime
import sys
import scipy.io

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchsummary import summary
import torch.autograd as autograd
from torchvision.models import vgg19

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
# from common_spatial_pattern import csp

import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True
#from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split

# writer = SummaryWriter('./TensorBoardX/')


# Convolution module
# use conv to capture local features, instead of postion embedding.
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (22, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )


    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        
        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(2440, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out


class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, n_classes=4, **kwargs):
        super().__init__(

            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
class ExP():
    def __init__(self, nsub):
        super(ExP, self).__init__()
        self.batch_size = 72
        self.n_epochs = 3 #default 2000
        self.c_dim = 4
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.dimension = (190, 50)
        self.nSub = nsub

        self.start_epoch = 0
        #self.root = '/Data/strict_TE/'

        self.log_write = open("C:\\Temp\\MI\\BCICIV_2a_gdf\\results\\log_subject%d.txt" % self.nSub, "w")


        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.model = Conformer().cuda()
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()
        # summary(self.model, (1, 22, 1000))


    # Segmentation and Reconstruction (S&R) data augmentation
    def interaug(self, timg, label):  
        aug_data = []
        aug_label = []
        for cls4aug in range(4):
            cls_idx = np.where(label == cls4aug + 1)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]

            tmp_aug_data = np.zeros((int(self.batch_size / 4), 1, 22, 1000)) #1000 length of epoch that we have selected 
            for ri in range(int(self.batch_size / 4)):
                for rj in range(8):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    
                    tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :,rj * 125:(rj + 1) * 125]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:int(self.batch_size / 4)])
            
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).cuda()
        aug_data = aug_data.float()
        aug_label = torch.from_numpy(aug_label-1).cuda()
        aug_label = aug_label.long()
        return aug_data, aug_label

    def get_source_data(self):

        self.root = 'C:\\Temp\\MI\\BCICIV_2a_gdf\\'
        
        import mne
        
        #TRAIN##################################################################################################
        
        #this is designed only for the BCICIV_2a_gdf
        filename = self.root + 'A0%dT.gdf' % self.nSub
        print("Train subject data",filename)

        raw = mne.io.read_raw_gdf(filename)


        #print(raw.info)
        #print(raw.ch_names)
        
        # Find the events time positions
        events, _ = mne.events_from_annotations(raw)

        # Pre-load the data
        raw.load_data()

        # Filter the raw signal with a band pass filter in 7-35 Hz
        raw.filter(7., 35., fir_design='firwin')

        # Remove the EOG channels and pick only desired EEG channels

        raw.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']

        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False,
                       exclude='bads')

        # Extracts epochs of 3s time period from the datset into 288 events for all 4 classes

        #tmin, tmax = 1.0, 4.0
        tmin, tmax = 0, 4 - (1/250) #because fs = 250
        # left_hand = 769,right_hand = 770,foot = 771,tongue = 772
        event_id = dict({'769': 7,'770': 8,'771': 9,'772': 10})

        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                baseline=None, preload=True)

        train_data = epochs.get_data()
        train_labels = epochs.events[:,-1] - 7 + 1         
        
        # train data
        self.allData =  train_data
        self.allLabel = train_labels
        
        print("Done loading train data for subject: ", self.nSub)
        #TEST######################################################################################################
        
        filename = self.root + 'A0%dE.gdf' % self.nSub
        print("Test subject data",filename)

        raw = mne.io.read_raw_gdf(filename)

        #print(raw.info)
        #print(raw.ch_names)
        
        # Find the events time positions
        events, _ = mne.events_from_annotations(raw)

        # Pre-load the data
        raw.load_data()

        # Filter the raw signal with a band pass filter in 7-35 Hz
        raw.filter(7., 35., fir_design='firwin')

        # Remove the EOG channels and pick only desired EEG channels

        raw.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']

        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False,
                        exclude='bads')

        # Extracts epochs of 3s time period from the datset into 288 events for all 4 classes

        tmin, tmax = 0, 4 - (1/250) #because fs = 250
        # left_hand = 769,right_hand = 770,foot = 771,tongue = 772
        event_id = dict({'769': 7,'770': 8,'771': 9,'772': 10})

        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                baseline=None, preload=True,on_missing="warn")

        test_data = epochs.get_data()
        test_labels = epochs.events[:,-1] - 7 + 1         
        
        self.testData  = test_data 
        self.testLabel = test_labels
        
        print("Done loading test data for subject: " ,self.nSub)
        
        #Expand axis as in the original code to add the conv channel
        #our data is (trial number, electrode channel, time series data)
        #we need to convert it from 3 to 4 dim
        #we need to convert it to: (trial, conv channel, electrode channel, time samples)
        self.allData =  np.expand_dims(self.allData,  axis=1)
        self.testData = np.expand_dims(self.testData, axis=1)

        # shuffle train data
        shuffle_num   = np.random.permutation(len(self.allData))
        self.allData  = self.allData[shuffle_num, :, :, :]
        self.allLabel = self.allLabel[shuffle_num]
        
        #produce the validation dataset from the train data
        self.allData, self.valData, self.allLabel, self.valLabel = train_test_split(self.allData, self.allLabel, test_size=0.23, random_state=42, shuffle = True)

        #shuffle test data
        shuffle_num    = np.random.permutation(len(self.testData))
        self.testData  = self.testData[shuffle_num, :, :, :]
        self.testLabel = self.testLabel[shuffle_num]

        # standardize
        target_mean = np.mean(self.allData)
        target_std =  np.std(self.allData)
        
        self.allData =  (self.allData  - target_mean) / target_std
        self.valData =  (self.valData  - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std

        # data shape: (trial, conv channel, electrode channel, time samples)
        return self.allData, self.allLabel, self.valData, self.valLabel, self.testData, self.testLabel

    def train(self):

        train_data, train_label, val_data, val_label, test_data, test_label = self.get_source_data()

        #train
        train_data = torch.from_numpy(train_data)
        train_label = torch.from_numpy(train_label - 1)

        train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
        self.train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

        #validation
        val_data = torch.from_numpy(val_data)
        val_label = torch.from_numpy(val_label - 1)
        val_dataset = torch.utils.data.TensorDataset(val_data, val_label)
        self.val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=True)

        #test
        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label - 1)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))

        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0

        # Train the cnn model
        total_step = len(self.train_dataloader)
        curr_lr = self.lr
        early_stopper = EarlyStopper(patience=3, min_delta=2) #neds to be adjusted !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        for e in range(self.n_epochs):
            # in_epoch = time.time()
            
            self.model.train() #sets the model in training mode
            
            for i, (img, label) in enumerate(self.train_dataloader):

                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))

                # data augmentation
                aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                img = torch.cat((img, aug_data))
                label = torch.cat((label, aug_label))


                tok, train_outputs = self.model(img)

                train_loss = self.criterion_cls(train_outputs, label) 

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                
                #print(i)

            # Evaluate the model on the validation set
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(self.val_dataloader):
                    
                    inputs = Variable(inputs.cuda().type(self.Tensor))
                    labels = Variable(labels.cuda().type(self.LongTensor))
                    
                    tok, val_outputs = self.model(inputs)
                    
                    validation_loss = self.criterion_cls(val_outputs,labels)#loss_fn(val_outputs, labels) #loss_fn = nn.CrossEntropyLoss()
                    #validation_loss += loss.item()
        
            #early stopping 
            #validation_loss = validate_one_epoch(model, validation_loader)
            if early_stopper.early_stop(validation_loss):
                print("Early stop @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                break
    
            # out_epoch = time.time()
            print("Done epoch")


        # test process - SHOULD NOT BE AFTER EACH EPOCH, BUT AFTER THE TRAINING(with validation)
        self.model.eval() #sets the model in evaluation mode
        Tok, Cls = self.model(test_data)
        loss_test = self.criterion_cls(Cls, test_label)
        y_pred = torch.max(Cls, 1)[1]
        acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
        
        #train_pred = torch.max(train_outputs, 1)[1]
        #train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))

        print('Subject:', self.nSub,'  Test acc: %.6f' % acc)
        
        # print('Epoch:', e,
        #       '  Train loss: %.6f' % train_loss.detach().cpu().numpy(),
        #       '  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
        #       '  Train accuracy %.6f' % train_acc,
        #       '  Test accuracy is %.6f' % acc)

        self.log_write.write(str(e) + "    " + str(acc) + "\n")
       
        Y_true = test_label

        torch.save(self.model.module.state_dict(), 'model.pth')
        #averAcc = averAcc / num
        #print('The average accuracy is:', averAcc)
        #print('The best accuracy is:', bestAcc)
        #self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        #self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")

        return acc, Y_true, Y_pred
        # writer.close()


def main():
    #best = 0
    #aver = 0
    result_write = open("C:\\Temp\\MI\\BCICIV_2a_gdf\\results\\sub_result.txt", "w")

    accuracies = []
    
    for i in range(9):
        starttime = datetime.datetime.now()


        seed_n = np.random.randint(2021)
        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)


        print('Subject %d' % (i+1))
        exp = ExP(i + 1)
        
        acc, Y_true, Y_pred = exp.train()
        
        accuracies.append(acc)
        
        print('THE ACCURACY IS ' + str(acc))
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'Seed is: ' + str(seed_n) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The accuracy is: ' + str(acc) + "\n")

        endtime = datetime.datetime.now()
        print('subject %d duration: '%(i+1) + str(endtime - starttime))
        
        if i == 0:
            yt = Y_true
            yp = Y_pred
        else:
            yt = torch.cat((yt, Y_true))
            yp = torch.cat((yp, Y_pred))


    #best = best / 9
    #aver = aver / 9

    #result_write.write('**The average Best accuracy is: ' + str(best) + "\n")
    #result_write.write('The average Aver accuracy is: ' + str(aver) + "\n")
    result_write.close()


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))