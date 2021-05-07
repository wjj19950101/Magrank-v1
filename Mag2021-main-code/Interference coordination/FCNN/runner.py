# ###############################################
# Codes have been tested successfully on Python 3.6.0 with TensorFlow 1.14.0.
###############################################

from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.io as sio
import time
import math


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
def perf_eval(H, Py_p, NN_p, K, var_noise=1):
    num_sample = H.shape[2]
    pyrate = np.zeros(num_sample)
    nnrate = np.zeros(num_sample)
    mprate = np.zeros(num_sample)
    rdrate = np.zeros(num_sample)
    for i in range(num_sample):
        pyrate[i] = obj_IA_sum_rate(H[:, :, i], Py_p[:, i], var_noise, K)
        nnrate[i] = obj_IA_sum_rate(H[:, :, i], NN_p[i, :], var_noise, K)
        mprate[i] = obj_IA_sum_rate(H[:, :, i], np.ones(K), var_noise, K)
        rdrate[i] = obj_IA_sum_rate(H[:, :, i], np.random.rand(K,1), var_noise, K)
    return sum(pyrate)/num_sample, sum(nnrate)/num_sample


# Functions for deep neural network weights initialization
def ini_weights(n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_output):
    weights = {
        'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1]) / np.sqrt(n_input)),
        'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2]) / np.sqrt(n_hidden_1)),
        'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3]) / np.sqrt(n_hidden_2)),
        'out': tf.Variable(tf.truncated_normal([n_hidden_3, n_output])) / n_hidden_3,
    }
    biases = {
        'b1': tf.Variable(tf.ones([n_hidden_1]) * 0.1),
        'b2': tf.Variable(tf.ones([n_hidden_2]) * 0.1),
        'b3': tf.Variable(tf.ones([n_hidden_3]) * 0.1),
        'out': tf.Variable(tf.ones([n_output]) * 0.1),
    }
    return weights, biases

# Functions for deep neural network structure construction
def multilayer_perceptron(x, weights, biases,input_keep_prob,hidden_keep_prob):
    x = tf.nn.dropout(x, input_keep_prob)                         # dropout layer
    
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])   # x = wx+b
    layer_1 = tf.nn.relu(layer_1) 
    layer_1 = tf.nn.dropout(layer_1, hidden_keep_prob)            # dropout layer

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2, hidden_keep_prob)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)  
    layer_3 = tf.nn.dropout(layer_3, hidden_keep_prob)

    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    out_layer = tf.nn.relu6(out_layer) / 6
    return out_layer


# Functions for deep neural network training
def run(X, Y,X_val,Y_val,X_test,Y_test,H,training_epochs=100, batch_size=1000, LR= 0.001, n_hidden_1 = 200,n_hidden_2 = 80,n_hidden_3 = 80, traintestsplit = 0.01, LRdecay=0):
    num_train = X.shape[1]                         
    num_val = X_val.shape[1]
    n_input = X.shape[0]                          
    n_output = Y.shape[0]                         
    X_train = np.transpose(X)       
    Y_train = np.transpose(Y)     
    X_test = np.transpose(X_test)     
    Y_test = np.transpose(Y_test)     
    X_val = np.transpose(X_val)      
    Y_val = np.transpose(Y_val)  
    K = n_output

    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_output])
    is_train = tf.placeholder("bool")
    learning_rate = tf.placeholder(tf.float32, shape=[])
    total_batch = int(num_train / batch_size)
    print('train: %d ' % num_train, 'validation: %d ' % num_val)

    input_keep_prob = tf.placeholder(tf.float32)
    hidden_keep_prob = tf.placeholder(tf.float32)
    weights, biases = ini_weights(n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_output)
    pred = multilayer_perceptron(x, weights, biases, input_keep_prob, hidden_keep_prob)
    cost = tf.reduce_mean(tf.square(pred - y))    # cost function: MSE    
    optimizer = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(cost) # training algorithms: RMSprop
    init = tf.global_variables_initializer()
    MSE_train = np.zeros((training_epochs))
    MSE_val = np.zeros((training_epochs))
    with tf.Session() as sess:
        sess.run(init)
        # training
        for epoch in range(training_epochs):
            for i in range(total_batch):
                idx = np.random.randint(num_train,size=batch_size)
                if LRdecay==1:
                    _, c = sess.run([optimizer, cost], feed_dict={x: X_train[idx, :], y: Y_train[idx, :],
                                                                  input_keep_prob: 1, hidden_keep_prob: 1,
                                                                  learning_rate: LR/(epoch+1), is_train: True})
                elif LRdecay==0:
                    _, c = sess.run([optimizer, cost], feed_dict={x: X_train[idx, :], y: Y_train[idx, :],
                                                                      input_keep_prob: 1, hidden_keep_prob: 1,
                                                                      learning_rate: LR, is_train: True})
            MSE_train[epoch]= sess.run(cost, feed_dict={x: X_train, y: Y_train, input_keep_prob: 1, hidden_keep_prob: 1, is_train: False})
            MSE_val[epoch]= sess.run(cost, feed_dict={x: X_val, y: Y_val, input_keep_prob: 1, hidden_keep_prob: 1, is_train: False})            
            if epoch % 50 == 0:
                print('epoch:%d, '%epoch, 'MSE_train:%f, '%(MSE_train[epoch]),'MSE_val:%f.'%(MSE_val[epoch]))        
        # testing
        y_pred = multilayer_perceptron(x, weights, biases, input_keep_prob, hidden_keep_prob)
        Y_pred_test = sess.run(y_pred, feed_dict={x: X_test, input_keep_prob: 1, hidden_keep_prob: 1, is_train: False})
        Y_pred_test = np.round(Y_pred_test)
        pyrate, nnrate = perf_eval(H, np.transpose(Y_test), Y_pred_test, K)
        Ratio =  nnrate / pyrate 
    return Ratio,MSE_train,MSE_val

    
    
    
    
    
    
