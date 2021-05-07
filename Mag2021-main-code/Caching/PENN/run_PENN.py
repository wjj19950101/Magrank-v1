#  #################################################################
#  Python codes PENN for caching 
#  Codes have been tested successfully on Python 3.6.0 with TensorFlow 1.14.0.
#  #################################################################
import scipy.io as sio                    
import numpy as np                         
import runner  
import math
import sys

K = 10                         # number of files
num_H = 15                     # number of training samples
num_val = math.ceil(0.1*num_H) # number of validation samples
training_epochs = 10000         # number of training epochs, 10000 for K=10 and 20, 15000 for K=30
N_mont = 10                    # number of Montercalo simulations
layernum = [K, 100,K]          # layer size
LR = 0.01                      # initial learning rate
batch_size = min(num_H, 1000)  # batch size

# load data
Xtrain = sio.loadmat('../Data/Sup_WFpol_Nf'+str(K)+'.mat')['X_train']
Ytrain = sio.loadmat('../Data/Sup_WFpol_Nf'+str(K)+'.mat')['pol_tr']
X = sio.loadmat('../Data/Sup_WFpol_Nf'+str(K)+'.mat')['X_test']
Y = sio.loadmat('../Data/Sup_WFpol_Nf'+str(K)+'.mat')['pol_te']

# training
Ratio,Time = runner.run(Xtrain, Ytrain,X,Y,num_H,num_val,N_mont, training_epochs=training_epochs, LR=LR,
                                    batch_size=batch_size, K=K, layernum=layernum)

# performance
Sort_Ratio = np.sort(Ratio)
print('The second worst ratio is:  %f ' % Sort_Ratio[1] )
print('Average time for each training is:  %f  s' % (np.mean(Time)) )



