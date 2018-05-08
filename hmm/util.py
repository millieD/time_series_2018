import numpy as np
import numpy.random as npr 
import matplotlib.pyplot as plt
import pickle 

from hmmlearn import hmm
np.random.seed(42)

from sklearn.metrics import hamming_loss  
from sklearn.externals import joblib
from sklearn.decomposition import PCA


'''
important functions for model training/retrieval, data loading, reading txt files into numpy arrays
'''

def ReadTxt(file_path):
        reals, predicted = [], []
        pred = True
        with open(file_path) as f:
                for line in f:
                        if 'PRED' in line:
                                pred =True
                        if 'REAL' in line:
                                pred = False
                        
                        if ',' in line:
                                entry = [ int(f) for f in  line[:-1].split(',') ]
                                if pred:
                                        predicted.append(entry)
                                else:
                                        reals.append(entry[:-1])
                                
                predicted, reals = np.array(predicted), np.array(reals)
                print('predicted and real sizes: ', predicted.shape, reals.shape)
        return predicted, reals 

def ComputeError(predicted,  real):
    loss = hamming_loss( real.reshape(-1), predicted.reshape(-1) )
    return loss

def Data():
        data_pickle_path = 'len100_decrement5_123.p'
        dataset = pickle.load(open(data_pickle_path, "rb"))
        dataset, structure_indices = dataset
        dataset = dataset - 1
        return dataset, structure_indices

def GetModel(load = True):
        
        model = hmm.MultinomialHMM(n_components=10)
        model_path = 'syn_model.pkl'
        if load:
                model = joblib.load(model_path)#'len100_decrement5_123.pkl')
        
        else:
                dataset, structure_indices = Data()
                num_samples, num_symbols = dataset.shape

                dataset_vec = np.concatenate(dataset).reshape(-1, 1) # 
                dataset = dataset_vec.reshape(100000, 100) 
                lengths = [ num_symbols for x in range(num_samples) ]

                model.fit(dataset_vec, lengths)
                joblib.dump(model, model_path)# "len100_decrement5_123.pkl")

        return model
#def  PlotDynamic( datas  ):
