#  #################################################################
#  Codes have been tested successfully on Python 3.6.0 with TensorFlow 1.14.0.
#  Based on the reference: 
#   [1] Haoran Sun, Xiangyi Chen, Qingjiang Shi, Mingyi Hong, Xiao Fu, and Nikos D. Sidiropoulos. 
#       "LEARNING TO OPTIMIZE: TRAINING DEEP NEURAL NETWORKS FOR WIRELESS RESOURCE MANAGEMENT."
#  #################################################################

import scipy.io as sio                      
import numpy as np                         
import runner 
import sys
import math
import Gen_data as GD

K = 10                           # number of users
num_H = 120                  # number of training samples
num_val = math.ceil(0.1*num_H)   # number of validation samples 
num_ini = num_H+num_val+1        # number of total samples 
num_test = 1000                  # number of testing  samples
trainseed = 0                    # set random seed for training set
testseed = 7                     # set random seed for test set
training_epochs = 3000            # number of training epochs: 500 for K=10, 3000 for K=20, 8000 for K=30
N_mont = 3                       # number of Monton carlo simulations
layernum = [K, 10*K, 10*K,K]     # layer size
LR = 0.01                        # initial learning rate
batch_size = min(num_H, 1000)    # batch size


# Generate Data
X_ini, Y_ini = GD.generate_Gaussian(K, num_ini, seed=trainseed)
X, Y = GD.generate_Gaussian(K, num_test, seed=testseed)
H = np.reshape(X, (K, K, X.shape[1]))

# training and testing
ratio, Time = runner.run(N_mont,num_val,num_H,X_ini, Y_ini, X,Y,H, training_epochs=training_epochs, LR=LR,
                                   traintestsplit = 0, batch_size=batch_size,
                                   K=K, isparamshare=True, layernum=layernum)
# system performance
Sort_Ratio = np.sort(ratio)
print('The best performance is:  %f ' % Sort_Ratio[-1] )
print('Average time for each training is:  %f  s' % (np.mean(Time)) )
 