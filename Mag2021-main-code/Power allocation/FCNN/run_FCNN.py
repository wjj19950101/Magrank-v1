#  #################################################################
#  Python code FCNN for power allocation
#  Codes have been tested successfully on Python 3.6.0 with TensorFlow 1.14.0.
#  #################################################################
import numpy as np
import scipy.io as sio
import runner
import math


# function for sorting samples
def sort_samp(X,Y):
    num = X.shape[1]
    for i in range(num):
        Inde_s = np.argsort(-X[:,i])
        X[:,i] = X[Inde_s,i]
        Y[:,i] = Y[Inde_s,i]     
    return X,Y 


K = 10                         # number of channels
num_H = 300                      # number of training samples
num_val = math.ceil(0.1*num_H) # number of validation samples
training_epochs = 3000         # number of training epochs
N_mont = 10                    # number of Montercalo simulations
LR = 0.1                       # initial learning rate
layer = [K,100,K]                # number of nodes for each layer
batch_size = min(num_H, 32)    # batch size
Is_rank = 0                    # ranking indicator: 1: rank; 0: no-rank

# load data
Xtrain = sio.loadmat('../Data/Trainingdata_Nc'+str(K)+'.mat')['H_tr']
Ytrain = sio.loadmat('../Data/Trainingdata_Nc'+str(K)+'.mat')['P_tr']
X = sio.loadmat('../Data/Trainingdata_Nc'+str(K)+'.mat')['H_te']
Y = sio.loadmat('../Data/Trainingdata_Nc'+str(K)+'.mat')['P_te']

# sort sample
if Is_rank:
    Xtrain, Ytrain = sort_samp(Xtrain,Ytrain)
    X, Y = sort_samp(X,Y)
    
Ratio, Time = runner.run(Xtrain,Ytrain,X,Y,num_H,num_val,N_mont,K,batch_size,training_epochs,LR,Is_rank,layer)
Sort_Ratio = np.sort(Ratio)
print('The second worst ratio is:  %f ' % Sort_Ratio[1] )
print('Average time for each training is:  %f  s' % (np.mean(Time)) )
