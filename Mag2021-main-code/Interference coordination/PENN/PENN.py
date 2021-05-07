#  #################################################################
#  Codes have been tested successfully on Python 3.6.0 with TensorFlow 1.14.0.
#  PENN
#  #################################################################
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
import random


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6 / (fan_in + fan_out))
    high = constant * np.sqrt(6 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


class MLP(object):
    def __init__(self, layernum, layernum_beta, lamda, blockdim, blockmaxnum, transfer_function=tf.nn.relu, optimizer=tf.train.AdamOptimizer(), isparamshare=False, isdimgeneral=False):
        a0 = str(random.random())
        print(a0)
        self.layernum = layernum
        self.layernum_beta = layernum_beta
        self.transfer = transfer_function
        self.is_train = tf.placeholder("bool")
        self.x = tf.placeholder(tf.float32, [None, self.layernum[0]*self.layernum[0]])
        self.keep_prob = tf.placeholder(tf.float32)
        self.blockdim = blockdim
        self.blockmaxnum = blockmaxnum
        self.block_onehot = tf.placeholder(tf.float32, [None, blockmaxnum])
        self.blocknum = tf.reshape(tf.reduce_sum(self.block_onehot, axis=1), [-1, 1])
        self.eye_mat = dict()
        self.weights = dict()
        self.hidden = dict()
        self.hidden_2 = dict()
        self.hidden_3 = dict()
        self.shade_mat = tf.reshape(tf.tile(tf.reshape(self.block_onehot, [-1, self.blockmaxnum, 1]), [1, 1, self.blockdim]),
                                    [-1, self.blockdim * self.blockmaxnum])
        self.x1 = tf.reshape(self.x, [-1, blockdim, blockdim])
        for i in range(len(layernum_beta) - 1):
            if i == 0:
                self.hidden_3[str(i)], _ = self._add_layer_with_loss(self.blocknum,
                                                                     self.layernum_beta[i], self.layernum_beta[i + 1],
                                                                     0.1, self.keep_prob, lamda, 'betaNet' + str(i + 1),
                                                                     a0, transfer_function=self.transfer
                                                                     )
            elif i != len(layernum_beta) - 2:
                self.hidden_3[str(i)], _ = self._add_layer_with_loss(self.hidden_3[str(i - 1)],
                                                                     self.layernum_beta[i], self.layernum_beta[i + 1],
                                                                     0.1, self.keep_prob, lamda, 'betaNet' + str(i + 1),
                                                                     a0, transfer_function=self.transfer
                                                                     )
            else:
                self.nMS_multiplier, _ = self._add_layer_with_loss(self.hidden_3[str(i - 1)], self.layernum_beta[i],
                                                                   self.layernum_beta[i + 1],
                                                                   0.1, 1, lamda, 'betaNet' + str(i + 1), a0,
                                                                   transfer_function=self.transfer, factor=1)
        if len(layernum) > 2:
            for i in range(len(layernum) - 1):
                if i == 0:
                    self.hidden[str(i)], _ = self._add_layer_with_loss(self.x1,
                                                                       self.layernum[i], self.layernum[i + 1],
                                                                       0.1, self.keep_prob, lamda,
                                                                       'mainNet' + str(i + 1), a0,
                                                                       transfer_function=self.transfer,
                                                                       isparamshare=isparamshare,
                                                                       isdimgeneral=isdimgeneral,
                                                                       is_BN=False,
                                                                       )
                elif i != len(layernum) - 2:
                    self.hidden[str(i)], _ = self._add_layer_with_loss(self.hidden[str(i - 1)],
                                                                       self.layernum[i], self.layernum[i + 1],
                                                                       0.1, self.keep_prob, lamda,
                                                                       'mainNet' + str(i + 1), a0,
                                                                       transfer_function=self.transfer,
                                                                       isparamshare=isparamshare,
                                                                       isdimgeneral=isdimgeneral,
                                                                       is_BN=False)
                else:
                    _, self.y0 = self._add_layer_with_loss(self.hidden[str(i - 1)],
                                                           self.layernum[i], self.layernum[i + 1],
                                                           0.1, 1, lamda,
                                                           'mainNet' + str(i + 1), a0,
                                                           transfer_function=tf.nn.sigmoid, factor=1,
                                                           isparamshare=isparamshare,
                                                           isdimgeneral=isdimgeneral,
                                                           is_BN=True)
        else:
            _, self.y0 = self._add_layer_with_loss(self.x, self.layernum[0], self.layernum[1],
                                                   0.1, 1, lamda, 'mainNet' + str(1), a0,
                                                   transfer_function=tf.nn.sigmoid, factor=1,
                                                   isparamshare=isparamshare, isdimgeneral=isdimgeneral,
                                                   is_BN=False,
                                                   islstlayer=True)
        self.y1 = tf.reshape(tf.reduce_sum(self.y0 * tf.reshape(tf.eye(self.layernum[-1]),[-1,self.layernum[-1],self.layernum[-1]]), reduction_indices=[1]),[-1,self.layernum[-1]])
        _, self.y = self._add_layer_with_loss_1(self.y1,
                                                self.layernum[-1], self.layernum[-1],
                                                0.1, self.keep_prob, lamda,
                                                'mainNetz1', a0,
                                                isparamshare=isparamshare,
                                                isdimgeneral=isdimgeneral,
                                                is_BN=True)

        self.y = tf.nn.sigmoid(self.y)
        self.y_ = tf.placeholder(tf.float32, [None, self.layernum[-1]])
        self.cost = tf.reduce_mean(tf.square(self.y_-self.y))
        self.gradients = optimizer.compute_gradients(self.cost)
        self.capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in self.gradients if grad is not None]

        self.optimizer_MSE = optimizer.apply_gradients(self.capped_gradients)


        init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.merged = tf.summary.merge_all()
        self.sess.run(init)

    def _add_layer_with_loss(self, input, inshape, outshape, stddev, keep_prob, wl, name, a0,
                             transfer_function=tf.nn.softplus, factor=1, isparamshare=False, isdimgeneral=False, is_BN=False, islstlayer=False):
        if isparamshare is False:
            with tf.variable_scope(name, reuse=True):
                with tf.variable_scope('weights', reuse=True):
                    try:
                        self.weights['w_' + name] = tf.get_variable(name='weight')
                    except:
                        self.weights['w_' + name] = tf.Variable(tf.truncated_normal([inshape, outshape], stddev=stddev),
                                                                name='weight')
                    tf.summary.histogram('histogram', self.weights['w_' + name])
                with tf.variable_scope('biases', reuse=True):
                    try:
                        self.weights['b_' + name] = tf.get_variable(name='bias')
                    except:
                        self.weights['b_' + name] = tf.Variable(tf.ones([outshape])*0.1, name='bias')
            tf.summary.histogram('histogram', self.weights['b_' + name])
            Wx_plus_b = tf.matmul(input, self.weights['w_' + name]) + self.weights['b_' + name]
            output = transfer_function(Wx_plus_b * factor)
            tf.summary.histogram('activations', output)
            output = tf.nn.dropout(output, keep_prob)
            if wl is not None:
                weight_loss = tf.multiply(tf.nn.l2_loss(self.weights['w_' + name]), wl, name='weight_loss')
                tf.add_to_collection(a0, weight_loss)
        else:
            with tf.variable_scope(name, reuse=True):
                with tf.variable_scope('weights', reuse=True):
                    try:
                        self.weights['w_' + name + '_l0'] = tf.get_variable(name='weight_l0')
                        self.weights['w_' + name + '_l1'] = tf.get_variable(name='weight_l1')
                        self.weights['w_' + name + '_r2'] = tf.get_variable(name='weight_r2')
                        self.weights['w_' + name + '_r3'] = tf.get_variable(name='weight_r3')
                    except:
                        self.weights['w_' + name + '_l0'] = tf.Variable(
                            tf.truncated_normal([int(inshape / self.blockmaxnum), int(outshape / self.blockmaxnum)],
                                                stddev=stddev),
                            name='weight_l0')
                        self.weights['w_' + name + '_l1'] = tf.Variable(
                            tf.truncated_normal([int(inshape / self.blockmaxnum), int(outshape / self.blockmaxnum)],
                                                stddev=stddev),
                            name='weight_l1')
                        self.weights['w_' + name + '_r2'] = tf.Variable(
                            tf.truncated_normal([int(inshape / self.blockmaxnum), int(outshape / self.blockmaxnum)],
                                                stddev=stddev),
                            name='weight_r2')
                        self.weights['w_' + name + '_r3'] = tf.Variable(
                            tf.truncated_normal([int(inshape / self.blockmaxnum), int(outshape / self.blockmaxnum)],
                                                stddev=stddev),
                            name='weight_r3')
                with tf.variable_scope('biases', reuse=True):
                    try:
                        self.weights['b_' + name + '_l0'] = tf.get_variable(name='bias_l0')
                        self.weights['b_' + name + '_r1'] = tf.get_variable(name='bias_r1')
                    except:
                        self.weights['b_' + name + '_l0'] = tf.Variable(tf.ones([int(outshape / self.blockmaxnum), int(outshape / self.blockmaxnum)]) * 0.1,
                                                                       name='bias_l0')
                        self.weights['b_' + name + '_r1'] = tf.Variable(tf.ones([1, int(inshape / self.blockmaxnum)]) * 0.1,
                                                                       name='bias_r1')

            self.weights['w_' + name + 'lbig1'] = tf.tile(self.weights['w_' + name + '_l0'],
                                                         [self.blockmaxnum, self.blockmaxnum])
            self.weights['w_' + name + 'lbig2'] = tf.tile(self.weights['w_' + name + '_l1'],
                                                         [self.blockmaxnum, self.blockmaxnum])
            self.weights['w_' + name + 'rbig3'] = tf.tile(self.weights['w_' + name + '_r2'],
                                                         [self.blockmaxnum, self.blockmaxnum])
            self.weights['w_' + name + 'rbig4'] = tf.tile(self.weights['w_' + name + '_r3'],
                                                         [self.blockmaxnum, self.blockmaxnum])
            self.eye_mat[name] = tf.reshape(tf.transpose(
                tf.tile(tf.reshape(tf.eye(self.blockmaxnum), [1, 1, self.blockmaxnum, self.blockmaxnum]),
                        [int(inshape / self.blockmaxnum), int(outshape / self.blockmaxnum), 1, 1]), perm=[3, 0, 2, 1]),
                                            [inshape, outshape])
            self.eye_mat[name] = tf.to_float(self.eye_mat[name])
            dim_submat_in = int(inshape / self.blockmaxnum)
            dim_submat_out = int(outshape / self.blockmaxnum)

            self.weights['w_' + name + '_l'] = self.eye_mat[name] * self.weights['w_' + name + 'lbig2'] + (
                        1 - self.eye_mat[name]) * self.weights['w_' + name + 'lbig1']
            self.weights['w_' + name + '_r'] = self.eye_mat[name] * self.weights['w_' + name + 'rbig4'] + (
                        1 - self.eye_mat[name]) * self.weights['w_' + name + 'rbig3']
            '''
            self.weights['w_' + name + '_l'] = tf.Variable(
                            tf.truncated_normal([int(inshape), int(outshape)],
                                                stddev=stddev),
                            name='weight_l')
            self.weights['w_' + name + '_r'] = tf.Variable(
                            tf.truncated_normal([int(inshape), int(outshape)],
                                                stddev=stddev),
                            name='weight_r')
            '''
            self.weights['b_' + name + '_l'] = tf.tile(self.weights['b_' + name + '_l0'], [self.blockmaxnum, self.blockmaxnum])
            Wx_plus_b = tf.matmul(tf.transpose(self.weights['w_' + name + '_l']),
                                  tf.matmul(input, self.weights['w_' + name + '_r'])) 
            if islstlayer is True:
                Wx_plus_b = tf.matmul(tf.transpose(self.weights['w_' + name + '_l']),
                                      tf.matmul(input, tf.transpose(self.weights['b_' + name + '_r'])))
            if is_BN is True:
                Wx_plus_b = tf.layers.batch_normalization(Wx_plus_b, training=self.is_train)
            output = transfer_function(Wx_plus_b)
            tf.summary.histogram('activations', output)
            output = tf.nn.dropout(output, keep_prob)
        return output, Wx_plus_b


    def _add_layer_with_loss_1(self, input, inshape, outshape, stddev, keep_prob, wl, name, a0,
                             transfer_function=tf.nn.softplus, factor=1, isparamshare=False, isdimgeneral=False, is_BN=False, islstlayer=False):
        if isparamshare is False:
            with tf.variable_scope(name, reuse=True):
                with tf.variable_scope('weights', reuse=True):
                    try:
                        self.weights['w_' + name] = tf.get_variable(name='weight')
                    except:
                        self.weights['w_' + name] = tf.Variable(tf.truncated_normal([inshape, outshape], stddev=stddev),
                                                                name='weight')
                    tf.summary.histogram('histogram', self.weights['w_' + name])
                with tf.variable_scope('biases', reuse=True):
                    try:
                        self.weights['b_' + name] = tf.get_variable(name='bias')
                    except:
                        self.weights['b_' + name] = tf.Variable(tf.ones([outshape])*0.1, name='bias')
            tf.summary.histogram('histogram', self.weights['b_' + name])
            Wx_plus_b = tf.matmul(input, self.weights['w_' + name]) + self.weights['b_' + name]
            output = transfer_function(Wx_plus_b * factor)
            tf.summary.histogram('activations', output)
            output = tf.nn.dropout(output, keep_prob)
            if wl is not None:
                weight_loss = tf.multiply(tf.nn.l2_loss(self.weights['w_' + name]), wl, name='weight_loss')
                tf.add_to_collection(a0, weight_loss)
        else:
            with tf.variable_scope(name, reuse=True):
                with tf.variable_scope('weights', reuse=True):
                    try:
                        self.weights['w_' + name + '_0'] = tf.get_variable(name='weight_0')
                        self.weights['w_' + name + '_1'] = tf.get_variable(name='weight_1')
                    except:
                        self.weights['w_' + name + '_0'] = tf.Variable(
                            tf.truncated_normal([int(inshape / self.blockmaxnum), int(outshape / self.blockmaxnum)],
                                                stddev=stddev),
                            name='weight_0')
                        self.weights['w_' + name + '_1'] = tf.Variable(
                            tf.truncated_normal([int(inshape / self.blockmaxnum), int(outshape / self.blockmaxnum)],
                                                stddev=stddev),
                            name='weight_1')
                with tf.variable_scope('biases', reuse=True):
                    try:
                        self.weights['b_' + name + '_0'] = tf.get_variable(name='bias_0')
                    except:
                        self.weights['b_' + name + '_0'] = tf.Variable(tf.ones([1, int(outshape / self.blockmaxnum)]) * 0.1,
                                                                       name='bias_0')

            self.weights['w_' + name + 'big1'] = tf.tile(self.weights['w_' + name + '_0'],
                                                         [self.blockmaxnum, self.blockmaxnum])
            self.weights['w_' + name + 'big2'] = tf.tile(self.weights['w_' + name + '_1'],
                                                         [self.blockmaxnum, self.blockmaxnum])
            self.eye_mat[name] = tf.reshape(tf.transpose(
                tf.tile(tf.reshape(tf.eye(self.blockmaxnum), [1, 1, self.blockmaxnum, self.blockmaxnum]),
                        [int(inshape / self.blockmaxnum), int(outshape / self.blockmaxnum), 1, 1]), perm=[3, 0, 2, 1]),
                                            [inshape, outshape])
            self.eye_mat[name] = tf.to_float(self.eye_mat[name])
            dim_submat_in = int(inshape / self.blockmaxnum)
            dim_submat_out = int(outshape / self.blockmaxnum)

            self.weights['w_' + name + '_l'] = self.eye_mat[name] * self.weights['w_' + name + 'big2'] + (
                        1 - self.eye_mat[name]) * self.weights['w_' + name + 'big1']
            self.weights['b_' + name + '_l'] = tf.tile(self.weights['b_' + name + '_0'], [1, self.blockmaxnum])

            Wx_plus_b = tf.matmul(input, self.weights['w_' + name + '_l']) + self.weights['b_' + name + '_l']
            if is_BN is True:
                Wx_plus_b = tf.layers.batch_normalization(Wx_plus_b, training=self.is_train)
            output = transfer_function(Wx_plus_b)
            tf.summary.histogram('activations', output)
            output = tf.nn.dropout(output, keep_prob)
        return output, Wx_plus_b

    def addRandomRelu(self, X):
        ispositive = tf.to_float(X > 0)
        randint = tf.to_float(tf.random_uniform(tf.shape(X), 0, 1)>0)
        door = tf.minimum((ispositive + randint), 1)
        return door * X

    def getoutputs(self, X, block_onehot, keep_prob, is_train):
        return self.sess.run(self.y, feed_dict={self.x: X, self.block_onehot: block_onehot,
                                                self.keep_prob: keep_prob, self.is_train: is_train})

    def getallparas(self):
        allparas = []
        for i in range(len(self.layernum) - 1):
            allparas.append(self.weights['w_O_layer' + str(i + 1)])
            allparas.append(self.weights['b_O_layer' + str(i + 1)])
        return self.sess.run(tuple(allparas))

    def getallparas_nMS(self):
        allparas = []
        for i in range(len(self.layernum_beta) - 1):
            allparas.append(self.weights['w_O_layerX' + str(i + 1)])
            allparas.append(self.weights['b_O_layerX' + str(i + 1)])
        return self.sess.run(tuple(allparas))

    def getcost(self, X, Y, block_onehot, keep_prob, is_train):
        return self.sess.run(self.cost, feed_dict={self.x: X, self.y_: Y, self.keep_prob: keep_prob,
                                                   self.block_onehot: block_onehot, self.is_train: is_train})

    def model_restore(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, 'Model/model.ckpt')


def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test, preprocessor


def get_random_block_from_data(dataX, dataY, batch_size):
    start_index = np.random.randint(0, len(dataX) - batch_size)
    return dataX[start_index: start_index + batch_size], dataY[start_index: start_index + batch_size]
