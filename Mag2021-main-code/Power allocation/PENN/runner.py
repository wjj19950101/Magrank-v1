#  Training and test
#  Codes have been tested successfully on Python 3.6.0 with TensorFlow 1.14.0.
import tensorflow as tf  
import numpy as np
import scipy.io as sio
import time
import math
from PENN import MLP, standard_scale, get_random_block_from_data


def run(X_ini, Y_ini, X_test,Y_test,H,num_H,num_val,N_mont, training_epochs=300, batch_size=5000, LR=0.001, traintestsplit=0.01,  K=1, isdimgeneral=False, layernum=[2]):
    global mlp
    MSE_train = np.zeros((training_epochs,N_mont))
    MSE_val = np.zeros((training_epochs,N_mont))
    Time = np.zeros((N_mont))
    Ratio = np.zeros((N_mont))    
    N_ini = X_ini.shape[1]                       
    total_batch = int(num_H / batch_size)
    block_onehot_train = np.ones([num_H, K])
    block_onehot_val = np.ones([num_val, K])
    X_test = np.transpose(X_test) 
    Y_test = np.transpose(Y_test)
    
    x = tf.placeholder("float", [None, K])
    y = tf.placeholder("float", [None, K])
    is_train = tf.placeholder("bool")        
    input_keep_prob = tf.placeholder(tf.float32)
    hidden_keep_prob = tf.placeholder(tf.float32)
    
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
              
        # Initializing network
        mlp = MLP(layernum, [1, 10, 1],
                  0.1, K, K, transfer_function=tf.nn.softplus,
                  optimizer=tf.train.AdamOptimizer(LR, 0.9), isdimgeneral=False)
        
        for epoch in range(training_epochs):
            for i in range(total_batch):
                idx = np.random.randint(num_H, size=batch_size)
                # Trainining 
                mlp.optimizer_MSE.run({mlp.x: X_train[idx, :], mlp.y_: Y_train[idx, :],
                                       mlp.block_onehot: block_onehot_train[idx, :],
                                       mlp.keep_prob: 1, mlp.is_train: 1})
                
                c = mlp.getcost(X=X_train[idx, :], Y=Y_train[idx, :], block_onehot=block_onehot_train[idx, :], keep_prob=1, is_train=0)
            
            MSE_train[epoch,i_mont] = mlp.getcost(X=X_train, Y=Y_train, block_onehot=block_onehot_train, keep_prob=1, is_train=0) / len(X_train)
            MSE_val[epoch,i_mont] = mlp.getcost(X=X_val_, Y=Y_val_, block_onehot=block_onehot_val, keep_prob=1, is_train=0) / len(X_val)
            
            if epoch % 500 == 0:
                print('i_mont:%d, ' % i_mont, 'epoch:%d, ' % epoch, 'MSE_train:%f, ' %(MSE_train[epoch,i_mont]),
                      'MSE_val:%f.' %(MSE_val[epoch,i_mont] ))  
                
        Time[i_mont] = time.time() - start_time
        print("training time: %0.2f s" % (Time[i_mont]))
        # Testing        
        y_pred = mlp.getoutputs(X=X_test,block_onehot=np.ones([X_test.shape[0],K]),keep_prob=1,is_train=0)
        pyrate, nnrate = perf_eval(H, np.transpose(Y_test), y_pred, K)
        Ratio[i_mont] = np.mean(nnrate)/ np.mean(pyrate)
        print('Ratio:  %f ' % (Ratio[i_mont]))
        
            
    sio.savemat('../Experiments/PENN/PENN_WF_Nc'+str(K)+'.mat', {'MSE_train':MSE_train, 'MSE_val': MSE_val, 'Ratio':Ratio,'y_pred': y_pred, 'Y_test':Y_test} )
    
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




