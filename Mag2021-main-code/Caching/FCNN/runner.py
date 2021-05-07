#  #################################################################
#  Codes have been tested successfully on Python 3.6.0 with TensorFlow 1.14.0.
#  #################################################################
import tensorflow as tf
import scipy.io as sio
import numpy as np
import math
from scipy.special import gamma
from scipy.integrate import quad
import time
from FCNN import MLP


# constant for caculating SOP
def f(x):
    return 1/(1+pow(x,3.7/2))
Lamd = 5/pow(250,2)/math.pi
Lamd_u = 5/pow(250,2)/math.pi
R0 = 2 * pow(10,6)
W= 20 * pow(10,6)
gama0 = pow(2,R0*(1+1.28*Lamd_u/Lamd)/W)-1
p0 = pow(1+Lamd_u/3.5/Lamd,-3.5)
quad_z1,err = quad(f,pow(gama0,-2/3.7),1000)
Z1 = pow(gama0,2/3.7)*quad_z1
Z2 = math.gamma(1-2/3.7)*math.gamma(1+2/3.7)/(math.gamma(1))*pow(gama0,2/3.7)


def run(X_ini,Y_ini,X_test,Y_test,num_H,num_val,N_mont,K,batch_size,training_epochs,LR,Is_rank,layer):
    
    MSE_train = np.zeros((training_epochs,N_mont))
    MSE_val = np.zeros((training_epochs,N_mont))
    Time = np.zeros((N_mont))
    Ratio = np.zeros((N_mont))    
    N_ini = X_ini.shape[1]                       
    total_batch = int(num_H / batch_size)    
    X_test = np.transpose(X_test) 
    Y_test = np.transpose(Y_test)
    Ora_meansop = SOP(X_test,Y_test,0.1*K)


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
        mlp = MLP(layer, 0, transfer_function = tf.nn.sigmoid, optimizer = tf.train.AdamOptimizer(LR, 0.9))
           
        for epoch in range(training_epochs):
            
            # training
            
            for i in range(total_batch):
                idx = np.random.randint(num_H, size=batch_size)                      
                mlp.optimizer.run({mlp.x: X_train[idx, :], mlp.y_: Y_train[idx, :], mlp.keep_prob: 1})  
                
            MSE_train[epoch,i_mont]  = mlp.getcost(X =  X_train, Y = Y_train,  keep_prob = 1)/len(X_train)   
            MSE_val[epoch,i_mont] = mlp.getcost(X = X_val_, Y = Y_val_, keep_prob = 1)/len(X_val_)    
        
            if epoch % 100 == 0:
                print('i_mont:%d, ' % i_mont, 'epoch:%d, ' % epoch, 'MSE_train:%f, ' %(MSE_train[epoch,i_mont]),
                      'MSE_val:%f.' %(MSE_val[epoch,i_mont] ))    
        
        Time[i_mont] = time.time() - start_time
        print("training time: %0.2f s" % (Time[i_mont]))
        
        # testing        
        y_pred = mlp.getoutputs(X=X_test,keep_prob=1)
        NN_meansop = SOP(X_test,y_pred,0.1*K)        
        Ratio[i_mont] = NN_meansop/Ora_meansop
        print('i_mont:%d, ' % i_mont,'Sop_ratio:%f.' %(Ratio[i_mont])) 
        
        

    if Is_rank:
        sio.savemat('../Experiments/FCNN/rank/FCNN_Cach_Nc'+str(K)+'.mat', {'MSE_train':MSE_train, 'MSE_val': MSE_val, 'Ratio':Ratio,'y_pred': y_pred, 'Y_test':Y_test} )
    else:
        sio.savemat('../Experiments/FCNN/no-rank/FCNN_Cach_Nc'+str(K)+'.mat', {'MSE_train':MSE_train, 'MSE_val': MSE_val, 'Ratio':Ratio,'y_pred': y_pred, 'Y_test':Y_test} )
    
    return Ratio,Time

# function for calculating SOP
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
