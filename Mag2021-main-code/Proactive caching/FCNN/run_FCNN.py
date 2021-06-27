#  #################################################################
#  Python code FCNN for power allocation
#  Codes have been tested successfully on Python 3.6.0 with TensorFlow 1.14.0.
#  #################################################################
import numpy as np
import scipy.io as sio
import runner
import math

# function for sorting samples
def sort_samp_tr(X,Y,d_past):
    num = X.shape[2]
    for i in range(num):
        Inde_s = np.argsort(-X[:,d_past-1,i])
#        Inde_s = np.argsort(-Y[:,i])
        X[:,:,i] = X[Inde_s,:,i]
#        X[:,i] = X[Inde_s,i]
        Y[:,i] = Y[Inde_s,i]     
    return X,Y 

def sort_samp_te(X,Y,pf,d_past):
    num = X.shape[2]
    for i in range(num):
        Inde_s = np.argsort(-X[:,d_past-1,i])
#        Inde_s = np.argsort(-Y[:,i])
        X[:,:,i] = X[Inde_s,:,i]
#        X[:,i] = X[Inde_s,i]
        Y[:,i] = Y[Inde_s,i] 
        pf[:,i] = pf[Inde_s,i]
    return X,Y,pf 


K = 10                         # number of files
num_H = 100000                      # number of training samples
num_val = math.ceil(0.1*num_H) # number of validation samples
training_epochs = 1000         # number of training epochs, rank: 10000; no-rank: 3000
N_mont = 10                    # number of Montercalo simulations
LR = 0.01                         # initial learning rate

batch_size = min(num_H, 1000)    # batch size
Is_rank = 0                   # ranking indicator: 1: rank; 0: no-rank

# load data
Xtrain = sio.loadmat('../Data/Sup_WFpol_Nf'+str(K)+'.mat')['X_train']
Ytrain = sio.loadmat('../Data/Sup_WFpol_Nf'+str(K)+'.mat')['pol_tr']
X = sio.loadmat('../Data/Sup_WFpol_Nf'+str(K)+'.mat')['X_test']
Y = sio.loadmat('../Data/Sup_WFpol_Nf'+str(K)+'.mat')['pol_te']
pf_test = sio.loadmat('../Data/Sup_WFpol_Nf'+str(K)+'.mat')['pf_test']
num_tr = Xtrain.shape[2] 
num_te = X.shape[2] 
d_past= Xtrain.shape[1] 


#Xtrain = sio.loadmat('../Data/Sup_WFpol_Nf'+str(K)+'.mat')['pf_train']
#Ytrain = sio.loadmat('../Data/Sup_WFpol_Nf'+str(K)+'.mat')['pol_tr']
#X = sio.loadmat('../Data/Sup_WFpol_Nf'+str(K)+'.mat')['pf_test']
#Y = sio.loadmat('../Data/Sup_WFpol_Nf'+str(K)+'.mat')['pol_te']
#pf_test = sio.loadmat('../Data/Sup_WFpol_Nf'+str(K)+'.mat')['pf_test']
#num_tr = Xtrain.shape[1] 
#num_te = X.shape[1] 
#d_past= 1 


layer = [d_past*K,10*K,K]                # number of nodes for each layer



# sort sample
if Is_rank:
    Xtrain, Ytrain = sort_samp_tr(Xtrain,Ytrain,d_past)
    X, Y, pf_test = sort_samp_te(X,Y,pf_test,d_past)

Xtrain = np.reshape(Xtrain,(d_past*K,num_tr))
X = np.reshape(X,(d_past*K,num_te))   

Ratio, Time = runner.run(Xtrain,Ytrain,X,Y,pf_test,num_H,num_val,N_mont,K,batch_size,training_epochs,LR,Is_rank,layer)

Sort_Ratio = np.sort(Ratio)
print('The second worst ratio is:  %f ' % Sort_Ratio[1] )
print('Average time for each training is:  %f  s' % (np.mean(Time)) )
