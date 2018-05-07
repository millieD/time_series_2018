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
    
    
