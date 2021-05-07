#  #################################################################
#  Codes have been tested successfully on Python 3.6.0 with TensorFlow 1.14.0.
#  #################################################################
import scipy.io as sio                     
import numpy as np                          
import math

#Sort sample
def sort_samp(K,X,Y):
    num=X.shape[1]
    for loop in range(num):
        H_x=np.reshape(X[:,loop], (K,K), order="F")
        H_x_s=np.zeros((K,K))
        h_kk=np.zeros((1,K))
        for i_k in range(K):
            h_kk[0,i_k]=H_x[i_k,i_k]
        Inde_s=np.argsort(-h_kk[0,:])
        for i_k in range(K):
            for j_k in range(K):
                H_x_s[i_k,j_k]=H_x[Inde_s[i_k],Inde_s[j_k]]
        X[:,loop]=np.reshape(H_x_s ,(K**2,), order="F")
        Y[:,loop]=Y[Inde_s,loop]
    return X, Y

# Functions for data generation, Gaussian IC case
def generate_Gaussian(K, num_H, Pmax=1, Pmin=0, seed=2017):
    print('Generate Data ... (seed = %d)' % seed)
    np.random.seed(seed)
    Pini = Pmax*np.ones(K)
    var_noise = 1
    X=np.zeros((K**2,num_H))
    Y=np.zeros((K,num_H))
    for loop in range(num_H):
        CH = 1/np.sqrt(2)*(np.random.randn(K,K)+1j*np.random.randn(K,K))
        H=abs(CH)
        X[:,loop] = np.reshape(H, (K**2,), order="F")
        H=np.reshape(X[:,loop], (K,K), order="F")
        Y[:,loop] = WMMSE_sum_rate(Pini, H, Pmax, var_noise)
    return X, Y

# WMMSE
def WMMSE_sum_rate(p_int, H, Pmax, var_noise):
    K = np.size(p_int)
    vnew = 0
    b = np.sqrt(p_int)
    f = np.zeros(K)
    w = np.zeros(K)
    for i in range(K):
        f[i] = H[i, i] * b[i] / (np.square(H[i, :]) @ np.square(b) + var_noise)
        w[i] = 1 / (1 - f[i] * b[i] * H[i, i])
        vnew = vnew + math.log2(w[i])

    VV = np.zeros(100)
    for iter in range(100):
        vold = vnew
        for i in range(K):
            btmp = w[i] * f[i] * H[i, i] / sum(w * np.square(f) * np.square(H[:, i]))
            b[i] = min(btmp, np.sqrt(Pmax)) + max(btmp, 0) - btmp

        vnew = 0
        for i in range(K):
            f[i] = H[i, i] * b[i] / ((np.square(H[i, :])) @ (np.square(b)) + var_noise)
            w[i] = 1 / (1 - f[i] * b[i] * H[i, i])
            vnew = vnew + math.log2(w[i])

        VV[iter] = vnew
        if vnew - vold <= 1e-3:
            break

    p_opt = np.square(b)
    return p_opt