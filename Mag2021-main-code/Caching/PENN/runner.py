#  Training and test
#  Codes have been tested successfully on Python 3.6.0 with TensorFlow 1.14.0.
from __future__ import print_function
import tensorflow as tf 
import numpy as np
import scipy.io as sio
from scipy import integrate
import time
import math
from PENN import MLP, standard_scale, get_random_block_from_data

# constant for caculating SOP
def f(x):
    return 1/(1+pow(x,3.7/2))
Lamd = 5/pow(250,2)/math.pi
Lamd_u = 5/pow(250,2)/math.pi
R0 = 2 * pow(10,6)
W= 20 * pow(10,6)
gama0 = pow(2,R0*(1+1.28*Lamd_u/Lamd)/W)-1
p0 = pow(1+Lamd_u/3.5/Lamd,-3.5)
quad_z1,err = integrate.quad(f,pow(gama0,-2/3.7),1000)
Z1 = pow(gama0,2/3.7)*quad_z1
Z2 = math.gamma(1-2/3.7)*math.gamma(1+2/3.7)/(math.gamma(1))*pow(gama0,2/3.7)

# Functions for deep neural network training
def run(X_ini, Y_ini, X_test,Y_test,num_H,num_val,N_mont, training_epochs=300, batch_size=5000, LR=0.001, LRdecay=0, K=1, layernum=[2]):
    global mlp
    num_te = X_test.shape[1]
    x = tf.placeholder("float", [None, K])
    y = tf.placeholder("float", [None, K])
    is_train = tf.placeholder("bool")                
    block_onehot_train = np.ones([num_H, K])
    block_onehot_val = np.ones([num_val, K])
    
    total_batch = int(num_H / batch_size)
    N_ini = X_ini.shape[1]
    MSE_train = np.zeros((training_epochs,N_mont))
    MSE_val = np.zeros((training_epochs,N_mont))
    Time = np.zeros((N_mont))
    Ratio = np.zeros((N_mont))    
    Ypred = np.zeros((num_te,K,N_mont))  
    X_test = np.transpose(X_test) 
    Y_test = np.transpose(Y_test) 
    Ora_meansop = SOP(X_test,Y_test,0.1*K)
    for i_mont in range(N_mont):
        
        start_time = time.time() 
        # randomly selecting training and validation samples
        flag=np.random.randint(0,N_ini-num_H-num_val)
        X = X_ini[:,flag:flag+num_H]
        Y = Y_ini[:,flag:flag+num_H] 
        X_val_ = X_ini[:,flag+num_H:flag+num_H+num_val]
        Y_val_ = Y_ini[:,flag+num_H:flag+num_H+num_val]           
        X_train = np.transpose(X)     
        Y_train = np.transpose(Y)         
        X_val = np.transpose(X_val_)     
        Y_val = np.transpose(Y_val_)           
        
        # initilazing neural network
        mlp = MLP(layernum, [1, 10, 1],
                  0.1, K, K, transfer_function=tf.nn.softplus,
                  optimizer=tf.train.AdamOptimizer(LR, 0.9), isdimgeneral=False)
        # training
        for epoch in range(training_epochs):
            for i in range(total_batch):
                idx = np.random.randint(num_H, size=batch_size)
                mlp.optimizer_MSE.run({mlp.x: X_train[idx, :], mlp.y_: Y_train[idx, :],
                                       mlp.block_onehot: block_onehot_train[idx, :],
                                       mlp.keep_prob: 1, mlp.is_train: 1})
                c = mlp.getcost(X=X_train[idx, :], Y=Y_train[idx, :], block_onehot=block_onehot_train[idx, :], keep_prob=1, is_train=0)

            MSE_train[epoch,i_mont] = mlp.getcost(X=X_train, Y=Y_train, block_onehot=block_onehot_train, keep_prob=1, is_train=0) / num_H
            MSE_val[epoch,i_mont] = mlp.getcost(X=X_val, Y=Y_val, block_onehot=block_onehot_val, keep_prob=1, is_train=0) / num_val

            
            if epoch % (500) == 0:
                print('i_mont:%d, ' % i_mont,'epoch:%d, ' % epoch, 'MSE_train:%f, ' %(MSE_train[epoch,i_mont]),
                      'MSE_val:%f.' %(MSE_val[epoch,i_mont] ))    
        
        Time[i_mont] = time.time() - start_time
        print("training time: %0.2f s" % (Time[i_mont]))
        
        # testing 
        y_pred = mlp.getoutputs(X=X_test,block_onehot=np.ones([X_test.shape[0],K]),keep_prob=1,is_train=0)
        Ypred[:,:,i_mont] = y_pred
        
        NN_meansop = SOP(X_test,y_pred,0.1*K)        
        Ratio[i_mont] = NN_meansop/Ora_meansop
        print('i_mont:%d, ' % i_mont,'Sop_ratio:%f.' %(Ratio[i_mont])) 
        sio.savemat('../Experiments/PENN/NN_Nf'+str(K)+'.mat', {'Ratio': Ratio, 'Pol': Ypred,'MSE_train':MSE_train,'MSE_val':MSE_val,'y_pred': y_pred, 'Y_test':Y_test})
    return Ratio, Time

def SOP(pf,Y,Nc):
    N = Y.shape[0]
    ps = np.zeros((1,N))
    for i in range(N):
        pol = Y[i,:]
        if sum(pol)>Nc:
            pol = pol/sum(pol)*Nc
        pf_one = pf[i,:]
        ps[0,i] = sum( pf_one*pol / ((1-p0)*pol*Z1+(1-p0)*(1-pol)*Z2+pol))        
    Mean_SOP = np.mean(ps)            
    return Mean_SOP



