#  #################################################################
#  Python code PENN for power allocation
#  Codes have been tested successfully on Python 3.6.0 with TensorFlow 1.14.0.
#  #################################################################
import scipy.io as sio                      
import numpy as np           
import runner                         
import math

K = 10                         # number of channels
num_H = 20                     # number of training samples
num_val = math.ceil(0.1*num_H) # number of validation samples
training_epochs = 10000         # number of training epochs, 10000 for K=10 and 20, 15000 for K=30
N_mont = 10                    # number of Montercalo simulations
layernum = [K,100,K]           # layer size
LR = 0.001                     # initial learning rate
batch_size = min(num_H, 1000)  # batch size

# load data
Xtrain = sio.loadmat('../Data/Trainingdata_Nc'+str(K)+'.mat')['H_tr']
Ytrain = sio.loadmat('../Data/Trainingdata_Nc'+str(K)+'.mat')['P_tr']
X = sio.loadmat('../Data/Trainingdata_Nc'+str(K)+'.mat')['H_te']
Y = sio.loadmat('../Data/Trainingdata_Nc'+str(K)+'.mat')['P_te']
H = np.reshape(X, (K, X.shape[1]))

# training
Ratio,Time = runner.run(Xtrain, Ytrain,X,Y,H,num_H, num_val,N_mont, training_epochs=training_epochs, LR=LR, batch_size=batch_size,
                                   K=K, layernum=layernum)
# performance
Sort_Ratio = np.sort(Ratio)
print('The second worst ratio is:  %f ' % Sort_Ratio[1] )
print('Average time for each training is:  %f  s' % (np.mean(Time)) )



    