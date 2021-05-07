# ###############################################
# Codes have been tested successfully on Python 3.6.0 with TensorFlow 1.14.0.
# Codes based on the following Reference: [1] Haoran Sun, Xiangyi Chen, Qingjiang Shi, Mingyi Hong, Xiao Fu, and Nikos D. Sidiropoulos. 
# "Learning to optimize: Training deep neural networks for wireless resource management." 
# Signal Processing Advances in Wireless Communications (SPAWC), 2017 IEEE 18th International Workshop on. IEEE, 2017.
# ###############################################
import scipy.io as sio                      
import numpy as np                         
import runner 
import random
import math
import time
import Gen_data as GD


K = 10                           # number of users
num_H = 10                  # number of training samples
num_val = math.ceil(0.1*num_H)   # number of validation samples 
num_ini = num_H+num_val+1        # number of total samples 
num_test = 1000                  # number of testing  samples
training_epochs = 300            # number of training epochs: 300 for no-rank, 500 for rank
N_mont = 3                       # number of Monte Carlo simulations
batch_size = min(num_H, 1000)    # batch size
Is_rank = 0                      # indicator of ranking, 0:no-ranking, 1:ranking


# Generate Data
trainseed = 0                    # set random seed for training set
testseed = 7                     # set random seed for test set
X_ini, Y_ini = GD.generate_Gaussian(K, num_ini, seed=trainseed)
X, Y = GD.generate_Gaussian(K, num_test, seed=testseed)


Ratio = np.zeros((N_mont))
MSE_train = np.zeros((training_epochs,N_mont))
MSE_val = np.zeros((training_epochs,N_mont))
Time = np.zeros((N_mont))

# Sort testing samples
if Is_rank:
    X, Y = GD.sort_samp(K, X, Y)
    
for i in range(N_mont):    
    
    start_time = time.time()
    # randomly select training and validation samples
    f = random.randint(0,num_ini-num_H-num_val-1)
    Xtrain = X_ini[:, f: f+num_H] 
    Ytrain = Y_ini[:, f: f+num_H]
    X_val = X_ini[:, f+num_H: f+num_H+num_val]
    Y_val = Y_ini[:, f+num_H: f+num_H+num_val]
    H = np.reshape(X, (K, K, X.shape[1]), order="F")    
    # Sort training and validation sample
    if Is_rank:
        Xtrain, Ytrain = GD.sort_samp(K,Xtrain,Ytrain)
        X_val, Y_val = GD.sort_samp(K, X_val, Y_val)
    # Training and testing deep neural networks
    Ratio[i], MSE_train[:,i], MSE_val[:,i] = runner.run(Xtrain, Ytrain,X_val,Y_val, X,Y,H, training_epochs=training_epochs, traintestsplit = 0, batch_size=batch_size)
    Time[i] = time.time() - start_time
    print("training time: %0.2f s" % (Time[i]))  
    
    print("The system performance is: %f " % (Ratio[i]))
    

Sort_Ratio = np.sort(Ratio)
print('The best performance is:  %f ' % Sort_Ratio[-1] )
print('Average time for each training is:  %f  s' % (np.mean(Time)) )

if Is_rank:
    sio.savemat('../Experiments/FCNN/rank/NN_IC_Nc'+str(K)+'.mat', {'MSE_train':MSE_train, 'MSE_val': MSE_val, 'Ratio':Ratio} )
else:
    sio.savemat('../Experiments/FCNN/no-rank/NN_IC_Nc'+str(K)+'.mat', {'MSE_train':MSE_train, 'MSE_val': MSE_val, 'Ratio':Ratio} )
        

