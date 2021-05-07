#  #################################################################
#  Codes have been tested successfully on Python 3.6.0 with TensorFlow 1.14.0.
#  #################################################################
import tensorflow as tf
import scipy.io as sio
import numpy as np
import math
from scipy.integrate import quad
import time
from FCNN import  MLP



def run(X_ini,Y_ini,X_test,Y_test,num_H,num_val,N_mont,K,batch_size,training_epochs,LR,Is_rank,layer):
    
    MSE_train = np.zeros((training_epochs,N_mont))
    MSE_val = np.zeros((training_epochs,N_mont))
    Time = np.zeros((N_mont))
    Ratio = np.zeros((N_mont))    
    N_ini = X_ini.shape[1]                       
    total_batch = int(num_H / batch_size)    
    X_test = np.transpose(X_test) 
    Y_test = np.transpose(Y_test)
            
    for i_mont in range(N_mont):
        
        start_time = time.time()       
        # randomly selecting training and validation samples
        flag=np.random.randint(0,N_ini-num_H-num_val)
        X = X_ini[:,flag:flag+num_H]
        Y = Y_ini[:,flag:flag+num_H]
        X_val = X_ini[:,flag+num_H:flag+num_H+num_val]
        Y_val = Y_ini[:,flag+num_H:flag+num_H+num_val]        
        X_train = np.transpose(X[:, 0:num_H])      
        Y_train = np.transpose(Y[:, 0:num_H])      
        X_val_ = np.transpose(X_val[:, 0:num_val])     
        Y_val_ = np.transpose(Y_val[:, 0:num_val])
        
        # network initialization        
        mlp = MLP(layer, 0, transfer_function = tf.nn.softplus, optimizer = tf.train.AdagradOptimizer(LR, 0.9))
   
        
        for epoch in range(training_epochs):
            # training
            for i in range(total_batch):
                idx = np.random.randint(num_H, size=batch_size)                      
                mlp.optimizer.run({mlp.x: X_train[idx, :], mlp.y_: Y_train[idx, :], mlp.keep_prob: 1})  
                
            MSE_train[epoch,i_mont]  = mlp.getcost(X =  X_train, Y = Y_train,  keep_prob = 1)/len(X_train)   
            MSE_val[epoch,i_mont] = mlp.getcost(X = X_val_, Y = Y_val_, keep_prob = 1)/len(X_val_)    
        
            if epoch % 500 == 0:
                print('i_mont:%d, ' % i_mont, 'epoch:%d, ' % epoch, 'MSE_train:%f, ' %(MSE_train[epoch,i_mont]),
                      'MSE_val:%f.' %(MSE_val[epoch,i_mont] ))   
                
        Time[i_mont] = time.time() - start_time
        print("training time: %0.2f s" % (Time[i_mont]))       
        
        # testing        
        y_pred = mlp.getoutputs(X=X_test,keep_prob=1)
        pyrate, nnrate = perf_eval(np.transpose(X_test), np.transpose(Y_test), y_pred, K)
        Ratio[i_mont] = np.mean(nnrate)/ np.mean(pyrate)
        print('Ratio:  %f ' % (Ratio[i_mont]))
        
        

    if Is_rank:
        sio.savemat('../Experiments/FCNN/rank/FCNN_WF_Nc'+str(K)+'.mat', {'MSE_train':MSE_train, 'MSE_val': MSE_val, 'Ratio':Ratio,'y_pred': y_pred, 'Y_test':Y_test} )
    else:
        sio.savemat('../Experiments/FCNN/no-rank/FCNN_WF_Nc'+str(K)+'.mat', {'MSE_train':MSE_train, 'MSE_val': MSE_val, 'Ratio':Ratio,'y_pred': y_pred, 'Y_test':Y_test} )
    
    return Ratio,Time

# Functions for performance evaluation
def perf_eval(H, Py_p, NN_p, K, var_noise=1):
    num_sample = H.shape[1]
    pyrate = np.zeros(num_sample)
    nnrate = np.zeros(num_sample)
    for i in range(num_sample):
        pyrate[i] = obj_IA_sum_rate(H[:, i], Py_p[:, i], var_noise, K)
        nnrate[i] = obj_IA_sum_rate(H[:, i], NN_p[i, :], var_noise, K)
    return pyrate, nnrate

# Functions for objective (Data-rate) calculation
def obj_IA_sum_rate(H, p, var_noise, K):
    y = 0.0
    for i in range(K):      
        y = y+math.log2(1+H[i]*p[i]/var_noise)
    return y

