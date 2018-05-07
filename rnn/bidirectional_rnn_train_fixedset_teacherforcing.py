import numpy as np
np.random.seed(42)
import pickle 
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


import os
import argparse
import torch
import pickle
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from torch.nn import LSTM as nnLSTM
import pickle as pkl
from datetime import datetime
from sklearn.model_selection import train_test_split


print("loading datasets...")
Tr = pkl.load(open('checkpoints/bidirectional_exp8/Tr_2.pkl', 'rb'))
Te = pkl.load(open('checkpoints/bidirectional_exp8/Te_2.pkl', 'rb'))
print("finished loading datasets")

num_samples, num_symbols = Tr.shape

# params for now
nepochs = 20000 
n_hidden = 128 
learning_rate = 0.001
n_embed = 128
batch_size= 512
print_every = 3
save_every = 25
use_cuda = torch.cuda.is_available()
save_path = './checkpoints/bidirectional_exp18_tf/'





max_ind = np.max(np.unique(Tr.flatten()))


#Tr, Te = train_test_split(dataset_tokens, train_size=0.9, random_state=42)



# Tr, Te = train_test_split(dataset, train_size=0.9, random_state=42)

about_info = "Adam Opt,Fixed train test split, using 90% teacher forcing no LR reduction, dropout 50%"
print(about_info)
pkl.dump(about_info, open(save_path+'about.pkl', 'wb'))
print(nepochs, n_hidden, learning_rate, n_embed, batch_size, print_every, save_every, use_cuda, save_path)
pkl.dump(Tr, open(save_path+'Tr_2.pkl', 'wb'))
pkl.dump(Te, open(save_path+'Te_2.pkl', 'wb'))


# load data
dataloader = DataLoader(Tr, batch_size=batch_size)

# use cuda
# if use_cuda:
#     dataloader.data.cuda()


all_letters = np.unique(Tr.flatten())
n_letters = len(all_letters)+1 # total number of possible symbols

print("Total number of letters: ", n_letters)




class RNN(nn.Module):
    def __init__(self, nsymbols, hidden_size, embed_size,batch_size=32):
        #  nsymbols: number of possible input/output symbols
        super(RNN, self).__init__()
        self.embed = nn.Embedding(nsymbols, embed_size)
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(nsymbols + hidden_size, hidden_size)
        self.h2o = nn.Linear(2*hidden_size, nsymbols)
        self.softmax = nn.LogSoftmax()
        self.batch_size=batch_size
        
        self.lstm = nnLSTM(3*hidden_size, 
                           hidden_size, 
                           #num_layers=2, 
                           bidirectional=True,
                           dropout=0.5,
                           batch_first=True)
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.i2h.weight.data.uniform_(-0.1, 0.1)
        self.i2h.bias.data.fill_(0)
        self.h2o.weight.data.uniform_(-0.1, 0.1)
        self.h2o.bias.data.fill_(0)

    def forward(self, input_x, hidden):
        x = self.embed(input_x)
#         print("a", x.size())

        combined = torch.cat((x, hidden), 1) 
#         print(x.size(), hidden.size())
#         print("b", combined.size())
        
        output, hidden = self.lstm(combined.view(self.batch_size, 1, -1))
#         print("c", output.size(), hidden[0].size())
        output = output.view(self.batch_size,-1)
#         print("d", output.size())
        output = self.h2o(output) 
#         print("e", output.size())
        output = self.softmax(output)        
#         print("f", output.size())
        
        return output, hidden[0].view(self.batch_size,-1)
                                
    def initHidden(self):
        return Variable(torch.zeros((self.batch_size, 2*self.hidden_size)).cuda())
    
    
# Turn a one-hot Variable of <1 x n_letters> into a Variable of size <1> 
def variableToIndex(tensor):
    idx = np.nonzero(tensor.data.numpy().flatten())[0]
    assert len(idx) == 1
    return Variable(torch.cuda.LongTensor(idx))

def train_batch(seq_tensor, rnn):
    hidden = rnn.initHidden()
    rnn.zero_grad()
    loss = 0
#     loss.cuda()
    seq_length = seq_tensor.size()[0]
    last_output=None
    for i in range(seq_length-1):
        tf_this_time = np.random.uniform(0,1) < 0.1
        if (i ==0 or tf_this_time):
            next_item = seq_tensor[i]
        else:
            next_item = last_output
        output, hidden = rnn(next_item, hidden)        
        this_loss = criterion(output, seq_tensor[i+1]) 
        last_output = output.max(dim=1)[1]
        loss += this_loss
#     stop
    loss.backward()
    optimizer.step()
#     return loss[0] / float(seq_length-1)
    return loss.cpu().data.numpy()[0] / float(seq_length-1)


# training
rnn = RNN(n_letters,n_hidden,n_embed,batch_size=batch_size) 


rnn.cuda()

rnn.train()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate, momentum=0.75)
# optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
losses = []
avg_losses = []
for epoch in range(1,nepochs+1):
    print(epoch)
    this_losses = []
    epoch_start = datetime.now()
    for batch, data in enumerate(dataloader):
        if data.size()[0] == batch_size:
            pat = Variable(torch.transpose(data,0,1).cuda())
            loss = train_batch(pat, rnn)
            losses.append(loss)
            this_losses.append(loss)
    if epoch % print_every == 0:
        print("avg loss overall at epoch {}: {} during epoch: {} duration: {}".format(epoch, 
                                                                                     np.mean(losses), 
                                                                                     np.mean(this_losses),
                                                                                      str(datetime.now() - epoch_start)
                                                                                    ))
        avg_losses.append(np.mean(this_losses))
    if epoch > 1000 and (avg_losses[-1] - avg_losses[-2]) > 0.001:
        print("shrinking LR")
        optimizer.param_groups[0]['lr'] *= 0.9
        print("new learning rate: ", optimizer.param_groups[0]['lr'])
        
    if (epoch + 1) % save_every == 0: 
        print("Saving models at epoch: {}".format(epoch))


        param_save = {'nepochs': nepochs,
                    'n_hidden': n_hidden, 
                    'learning_rate' : optimizer.param_groups[0]['lr'],
                    'n_embed' : n_embed,
                    'batch_size':batch_size,
                     'use_cuda': use_cuda
        }

        pkl.dump(rnn, open(save_path+'rnn_checkpoint_{}.pkl'.format(epoch), 'wb'))
        pkl.dump(optimizer, open(save_path+'opt_checkpoint_{}.pkl'.format(epoch), 'wb'))
        pkl.dump(param_save, open(save_path+'param_checkpoint.pkl', 'wb'))


pkl.dump(rnn, open(save_path+'rnn_checkpoint_{}.pkl'.format(epoch), 'wb'))
pkl.dump(optimizer, open(save_path+'opt_checkpoint_{}.pkl'.format(epoch), 'wb'))
pkl.dump(param_save, open(save_path+'param_checkpoint.pkl', 'wb'))

