# ###############################################
# Codes have been tested successfully on Python 3.6.0 with TensorFlow 1.14.0.
###############################################

from __future__ import print_function
import matplotlib.pyplot as plt 
import tensorflow as tf
import numpy as np
import scipy.io as sio
import time
import random
import math
from PENN import MLP, standard_scale, get_random_block_from_data


# Functions for objective (sum-rate) calculation
def obj_IA_sum_rate(H, p, var_noise, K):
    y = 0.0
    for i in range(K):
        s = var_noise
        for j in range(K):
            if j!=i:
                s = s+H[i,j]**2*p[j]
        y = y+math.log2(1+H[i,i]**2*p[i]/s)
    return y

# Functions for performance evaluation
def perf_eval(H, Py_p, NN_p, K, var_noise=1, isprint=1):
    num_sample = H.shape[2]
    pyrate = np.zeros(num_sample)
    nnrate = np.zeros(num_sample)
    for i in range(num_sample):
        pyrate[i] = obj_IA_sum_rate(H[:, :, i], Py_p[:, i], var_noise, K)
        nnrate[i] = obj_IA_sum_rate(H[:, :, i], NN_p[i, :], var_noise, K)
    return sum(pyrate)/num_sample, sum(nnrate)/num_sample

# Functions for deep neural network training and testing 
def run(N_mont,num_val,num_H,X, Y,X_test,Y_test, H, training_epochs=300, batch_size=5000, LR=0.001, traintestsplit=0.01, LRdecay=0, K=1, isparamshare=False, isdimgeneral=False, layernum=[2]):
    global mlp
    num_ini = X.shape[1]
    n_input = X.shape[0]                          # input size
    n_output = Y.shape[0]                         # output size        
    X_test = np.transpose(X_test)     # training data
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_output])
    is_train = tf.placeholder("bool")
    learning_rate = tf.placeholder(tf.float32, shape=[])
    total_batch = int(num_H / batch_size)
    input_keep_prob = tf.placeholder(tf.float32)
    hidden_keep_prob = tf.placeholder(tf.float32)
    block_onehot_train = np.ones([num_H, K])
    block_onehot_val = np.ones([num_val, K])
    ratio = np.zeros([N_mont])
    MSE_train =  np.zeros([N_mont,training_epochs])
    MSE_val = np.zeros([N_mont,training_epochs])
    Time = np.zeros((N_mont))
    
    
    for i_mont in range(N_mont):
        
        start_time = time.time()
        # randomly selecting training and validation samples
        f = random.randint(0,num_ini-num_H-num_val-1)
        X_train = np.transpose(X[:, f:f+num_H])      
        Y_train = np.transpose(Y[:, f:f+num_H])      
        X_val = np.transpose(X[:, f+num_H:f+num_H+num_val])  
        Y_val = np.transpose(Y[:, f+num_H:f+num_H+num_val])      
        mlp = MLP(layernum, [1, 10, 1],
                  0.1, K, K, transfer_function=tf.nn.relu,
                  optimizer=tf.train.AdamOptimizer(LR, 0.9),
                  isparamshare=isparamshare, isdimgeneral=False)             
        for epoch in range(training_epochs):
            for i in range(total_batch):
                idx = np.random.randint(num_H, size=batch_size)
                start = time.clock()
                mlp.optimizer_MSE.run({mlp.x: X_train[idx, :], mlp.y_: Y_train[idx, :],
                                       mlp.block_onehot: block_onehot_train[idx, :],
                                       mlp.keep_prob: 1, mlp.is_train: 1})
                c = mlp.getcost(X=X_train[idx, :], Y=Y_train[idx, :], block_onehot=block_onehot_train[idx, :], keep_prob=1, is_train=0)
    
            MSE_train[i_mont,epoch] = c
            MSE_val[i_mont,epoch] = mlp.getcost(X=X_val, Y=Y_val, block_onehot=block_onehot_val, keep_prob=1, is_train=0)
                  
            if epoch % 50 == 0:
                print('epoch:%d, ' % epoch, 'MSE_train:%2f, ' % (c), 'MSE_val:%f.' % MSE_val[i_mont,epoch])
        
        Time[i_mont] = time.time() - start_time
        print("training time: %0.2f s" % (Time[i_mont]))         
        # testing        
        y_pred = mlp.getoutputs(X=X_test,block_onehot=np.ones([X_test.shape[0],K]),keep_prob=1,is_train=0)
        y_pred = np.round(y_pred)
        pyrate, nnrate = perf_eval(H, Y_test, y_pred, K)
        ratio[i_mont] = nnrate/pyrate
        print("The system performance is: %f" % (ratio[i_mont]))
    # save data    
    sio.savemat('../Experiments/PENN/PENN_Nc'+str(K)+'.mat', {'MSE_train': MSE_train, 'MSE_val': MSE_val, 'ratio': ratio})
    return ratio,Time




