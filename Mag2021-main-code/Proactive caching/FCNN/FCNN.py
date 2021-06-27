#  #################################################################
#  Codes have been tested successfully on Python 3.6.0 with TensorFlow 1.14.0.
#  #################################################################
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.contrib import layers
import random


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6 / (fan_in + fan_out))
    high = constant * np.sqrt(6 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


class MLP(object):
    def __init__(self, layernum, lamda, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer()):
        a0 = str(random.random())
        print(a0)
        
        self.layernum = layernum        
        self.transfer = transfer_function
        self.x = tf.placeholder(tf.float32, [None, self.layernum[0]])
        self.keep_prob = tf.placeholder(tf.float32)
       
        self.weights = dict()
        self.hidden = dict()

        for i in range(len(layernum)-1):
            if i==0:
                self.hidden[str(i)] = self._add_layer_with_loss(self.x, self.layernum[i], self.layernum[i+1],
                            0.1, self.keep_prob, lamda,'layer' + str(i+1),a0)
            elif i!=len(layernum)-2:
                self.hidden[str(i)] = self._add_layer_with_loss(self.hidden[str(i-1)], self.layernum[i], self.layernum[i+1],
                           0.1, self.keep_prob, lamda, 'layer' + str(i + 1),a0)
            else:
 
                 self.y = self._add_layer_with_loss(self.hidden[str(i-1)], self.layernum[i], self.layernum[i+1],
                            0.1, 1, lamda, 'layer' + str(i + 1), a0,transfer_function=self.transfer)

        self.y_ = tf.placeholder(tf.float32, [None, self.layernum[-1]])
       
        self.cost0 = tf.nn.l2_loss(tf.subtract(self.y_, self.y))     
        tf.add_to_collection(a0, self.cost0)
        self.cost = tf.add_n(tf.get_collection(a0), name='total_loss')
        tf.summary.scalar('loss', self.cost0)
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.merged = tf.summary.merge_all()
#        self.train_writer = tf.summary.FileWriter('2018-05-07/logs/train', self.sess.graph)
#        self.test_writer = tf.summary.FileWriter('2018-05-07/logs/test')
        
        self.sess.run(init)
        
    def _add_layer_with_loss(self, input, inshape, outshape, stddev, keep_prob, wl, name, a0, transfer_function=tf.nn.relu):
        with tf.variable_scope(name, reuse=True):
            with tf.variable_scope('weights', reuse=True):
                try:
                    self.weights['w_'+name] = tf.get_variable(name = 'weight')
                except:
                    self.weights['w_'+name] = tf.Variable(tf.truncated_normal([inshape, outshape], stddev=stddev), name = 'weight')
                   
                # self.weights['w_'+name] = tf.contrib.absolute_import
                tf.summary.histogram('histogram', self.weights['w_'+name])
            with tf.variable_scope('biases', reuse=True):
                try:
                    self.weights['b_'+name] = tf.get_variable(name = 'bias')
                except:
                    self.weights['b_'+name] = tf.Variable(tf.zeros([outshape]), name = 'bias')
            tf.summary.histogram('histogram', self.weights['b_'+name])
 
            Wx_plus_b = tf.matmul(input, self.weights['w_'+name]) + self.weights['b_'+name]
            output = transfer_function(Wx_plus_b)
            tf.summary.histogram('activations', output)
            output = tf.nn.dropout(output, keep_prob)
            if wl is not None:
                weight_loss = tf.multiply(tf.nn.l2_loss(self.weights['w_'+name]), wl, name='weight_loss')
                tf.add_to_collection(a0, weight_loss)
            return output

    def getoutputs(self, X, keep_prob):
        return self.sess.run(self.y, feed_dict={self.x: X, self.keep_prob: keep_prob})

    def getallparas(self):
        allparas = []
        for i in range(len(self.layernum)-1):
            allparas.append(self.weights['w_layer' + str(i + 1)])
            allparas.append(self.weights['b_layer' + str(i + 1)])
        return self.sess.run(tuple(allparas))

    def getcost(self, X, Y, keep_prob):
        return self.sess.run(self.cost0, feed_dict={self.x: X, self.y_: Y, self.keep_prob: keep_prob})

    def variable_summaries(self, var):
        with tf.name_scope('summaries'):
            mean = tf.reduce


def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test, preprocessor
