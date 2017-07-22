#coding=utf8
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import util
from util import Data
import getdata
#coding=utf8
'''
根据google multidigits论文内容构建的卷积网络，针对数据形式网络结构与原论文略有不同
'''

class MultiNet(object):
    """model"""
    def __init__(self, image_height, image_width, image_channel, keepPro, classNum, num_length):
        self.X = tf.placeholder(tf.float32, [None, image_height, image_width, image_channel])
        self.y = tf.placeholder(tf.int64, [None, num_length+1])#前7位是数字最后一位存储该真实数据长度
        self.keep_prob_train = keepPro
        self.keep_prob = tf.placeholder(tf.float32)
        self.CLASSNUM = classNum
        self.num_length = num_length
        self.buildCNN()
        self.score1 = self.num1
        self.score2 = self.num2
        self.score3 = self.num3
        self.score4 = self.num4
        self.score5 = self.num5
        self.score6 = self.num6
        self.score7 = self.num7
        self.length = self.len
        #确性度定义(即根据网络输出给出网络自身认为本次输出正确的概率值是多少)
        self.probabi = self.probability()

        # 损失函数定义
        #for i in xrange(self.num_length):
        #    lossi = tf.losses.softmax_cross_entropy(onehot_labels=util.makeonehot(self.y[:,i], self.CLASSNUM), logits=eval('self.score'+str(i+1)))
        #    tf.add_to_collection("losses",lossi)
        # 损失函数定义
        with tf.variable_scope('loss_scope'):
            for i in xrange(self.num_length):
                lossi = tf.losses.sparse_softmax_cross_entropy(labels=self.y[:,i], logits=eval('self.score'+str(i+1)))
                tf.add_to_collection("losses",lossi)
            # 增加了长度之后的损失函数
            length_loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y[:,num_length], logits=self.length)
            tf.add_to_collection("losses",length_loss)
            #总共的损失
            self.loss = tf.add_n(tf.get_collection('losses'),name="loss")

        # 优化器定义
        self.train_op = tf.train.MomentumOptimizer(0.001,0.9).minimize(self.loss)
        # 准确度定义
        with tf.variable_scope('accuracy_scope'):
            for i in range(self.num_length):
                accuracyi = tf.reduce_mean(tf.cast(tf.equal(tf.reshape(self.y[:, i], [1,-1]),
                              tf.reshape(tf.argmax(eval('self.score'+str(i+1)), axis=1), [1, -1])), tf.float32))
                #accuracyi = tf.metrics.accuracy(labels=self.y[:,i], predictions=tf.argmax(eval('self.score'+str(i+1)), axis=1),)[1]
                tf.add_to_collection('accuracys',accuracyi)
            self.accuracy = tf.div(tf.add_n(tf.get_collection('accuracys')),self.num_length,name="accuracy")
        # 初始化
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    def probability(self):
        #确性度定义(即根据网络输出给出网络自身认为本次输出正确的概率值是多少)
        with tf.variable_scope('probabi_scope'):
            length_acc = tf.reshape(tf.reduce_max(tf.nn.softmax(self.length),axis=1),[-1,1])
            score1_acc = tf.reshape(tf.reduce_max(tf.nn.softmax(self.score1),axis=1),[-1,1])
            score2_acc = tf.reshape(tf.reduce_max(tf.nn.softmax(self.score2),axis=1),[-1,1])
            score3_acc = tf.reshape(tf.reduce_max(tf.nn.softmax(self.score3),axis=1),[-1,1])
            score4_acc = tf.reshape(tf.reduce_max(tf.nn.softmax(self.score4),axis=1),[-1,1])
            score5_acc = tf.reshape(tf.reduce_max(tf.nn.softmax(self.score5),axis=1),[-1,1])
            score6_acc = tf.reshape(tf.reduce_max(tf.nn.softmax(self.score6),axis=1),[-1,1])
            score7_acc = tf.reshape(tf.reduce_max(tf.nn.softmax(self.score7),axis=1),[-1,1])
            probabi = length_acc*score1_acc*score2_acc*score3_acc*score4_acc*score5_acc*score6_acc*score7_acc*100
            probabi = tf.cast(probabi,tf.int64)
        return probabi
    def buildCNN(self):
        '''
        为了简洁使用tensorflow的layers包里的卷积层直接使用
        '''
        with tf.variable_scope('hidden1'):
            conv = tf.layers.conv2d(self.X, filters=48, kernel_size=[3, 3], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=self.keep_prob)
            hidden1 = dropout

        with tf.variable_scope('hidden2'):
            conv = tf.layers.conv2d(hidden1, filters=64, kernel_size=[3, 3], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=self.keep_prob)
            hidden2 = dropout

        with tf.variable_scope('hidden3'):
            conv = tf.layers.conv2d(hidden2, filters=128, kernel_size=[3, 3], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=self.keep_prob)
            hidden3 = dropout

        with tf.variable_scope('hidden4'):
            conv = tf.layers.conv2d(hidden3, filters=160, kernel_size=[3, 3], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=self.keep_prob)
            hidden4 = dropout

        with tf.variable_scope('hidden5'):
            conv = tf.layers.conv2d(hidden4, filters=192, kernel_size=[3, 3], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=self.keep_prob)
            hidden5 = dropout

        with tf.variable_scope('hidden6'):
            conv = tf.layers.conv2d(hidden5, filters=192, kernel_size=[3, 3], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=self.keep_prob)
            hidden6 = dropout

        with tf.variable_scope('hidden7'):
            conv = tf.layers.conv2d(hidden6, filters=192, kernel_size=[3, 3], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=self.keep_prob)
            hidden7 = dropout
        '''
        with tf.variable_scope('hidden8'):
            conv = tf.layers.conv2d(hidden7, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=self.keep_prob)
            hidden8 = dropout
        '''
        flatten = tf.reshape(hidden7, [-1, 4 * 12 * 192])

        with tf.variable_scope('hidden9'):
            dense = tf.layers.dense(flatten, units=2048, activation=tf.nn.relu)
            hidden9 = dense

        with tf.variable_scope('hidden10'):
            dense = tf.layers.dense(hidden9, units=2048, activation=tf.nn.relu)
            hidden10 = dense

        with tf.variable_scope('digit1'):
            dense = tf.layers.dense(hidden10, units=self.CLASSNUM)
            self.num1 = dense

        with tf.variable_scope('digit2'):
            dense = tf.layers.dense(hidden10, units=self.CLASSNUM)
            self.num2 = dense

        with tf.variable_scope('digit3'):
            dense = tf.layers.dense(hidden10, units=self.CLASSNUM)
            self.num3 = dense

        with tf.variable_scope('digit4'):
            dense = tf.layers.dense(hidden10, units=self.CLASSNUM)
            self.num4 = dense

        with tf.variable_scope('digit5'):
            dense = tf.layers.dense(hidden10, units=self.CLASSNUM)
            self.num5 = dense

        with tf.variable_scope('digit6'):
            dense = tf.layers.dense(hidden10, units=self.CLASSNUM)
            self.num6 = dense
        with tf.variable_scope('digit7'):
            dense = tf.layers.dense(hidden10, units=self.CLASSNUM)
            self.num7 = dense   
        with tf.variable_scope('length'):
            dense = tf.layers.dense(hidden10, units=self.num_length)
            self.len = dense

    def makeprint(self,scoreval1, scoreval2, scoreval3, scoreval4, scoreval5, scoreval6,scoreval7, lengthval, proval):
        predictions1 = tf.reshape(tf.argmax(scoreval1, axis=1), [-1, 1])
        predictions2 = tf.reshape(tf.argmax(scoreval2, axis=1), [-1, 1])
        predictions3 = tf.reshape(tf.argmax(scoreval3, axis=1), [-1, 1])
        predictions4 = tf.reshape(tf.argmax(scoreval4, axis=1), [-1, 1])
        predictions5 = tf.reshape(tf.argmax(scoreval5, axis=1), [-1, 1])
        predictions6 = tf.reshape(tf.argmax(scoreval6, axis=1), [-1, 1])
        predictions7 = tf.reshape(tf.argmax(scoreval7, axis=1), [-1, 1])
        predictions_len = tf.reshape(tf.argmax(lengthval, axis=1), [-1, 1])
        proval_1 = tf.reshape(proval, [-1, 1])
        predictions = tf.concat([predictions1, predictions2, predictions3,
                                             predictions4, predictions5, predictions6,predictions7, predictions_len, proval_1], axis=1)
        return predictions
    
    def train(self, sess, X_train, Y_train, X_test, Y_test, num_epoch=1000):
        sess.run(self.init_op)
     # 随机生成器需要fit总体样本数据
        datagen = util.get_generator()
        saver = tf.train.Saver()
        # 因为datagen的特殊需求(bacthsize需要能够整除训练集总个数，并且这里样本也少，直接全体当batchsize)
        split = 500
        batch_size = int(len(X_train)/split)#取split使得尽量为200左右
        if len(X_train)%split != 0:#这里仅仅是因为图片生成那里要求batchsize需要能够和输入数据个数 整除 所以如此做以确保这一点
            remove = len(X_train)%split
            X_train = X_train[:-1*remove]#如果后续数据多了大可以不必进行图片生成或者图片数据很多却依然做图片生成时则batch_size和这个可能需要再调整
            Y_train = Y_train[:-1*remove]
        print('batch_size:', batch_size)
        for e in range(num_epoch):
            yieldData = Data(X_train,Y_train)
            print('Epoch',e)
            batches = 0
            if e != 0 and e %5 == 0:
                saver.save(sess, 'savedmodel/'+str(e)+'model')
            for x_batch, y_batch in datagen.flow(X_train, Y_train, batch_size=batch_size, save_to_dir=None):
                #print(x_batch[0])
                #print(y_batch[0])
                if batches %100 == 0:
                    print('batch: ',batches)
                    '''训练集'''
                    accuval, scoreval1, scoreval2, scoreval3, scoreval4, scoreval5, scoreval6,scoreval7, lengthval, proval= sess.run(
                         [self.accuracy, self.score1, self.score2, self.score3, self.score4, self.score5, self.score6, self.score7,self.length, self.probabi], 
                         feed_dict={self.X: x_batch, self.keep_prob: 1, self.y:y_batch})
                    print("Train accuracy:",accuval)
                    if batches == split:
                        print('train')
                        predictions = self.makeprint(scoreval1, scoreval2, scoreval3, scoreval4, scoreval5, scoreval6,scoreval7, lengthval, proval)
                        a = sess.run(predictions)
                        print(a.shape)
                        print(a)
                        print(y_batch)
                    '''测试集'''
                    yieldTest = Data(X_test,Y_test)
                    tmp = 0
                    for b_test_x,b_test_y in yieldTest.get_next_batch(1000):
                        accuval, scoreval1, scoreval2, scoreval3, scoreval4, scoreval5, scoreval6, scoreval7,lengthval, proval= sess.run(
                           [self.accuracy, self.score1, self.score2, self.score3, self.score4, self.score5, self.score6,self.score7, self.length, self.probabi], 
                           feed_dict={self.X: b_test_x, self.keep_prob: 1, self.y:b_test_y})
                        tmp = tmp + accuval
                        if batches == split:
                            print('test')
                            predictions = self.makeprint(scoreval1, scoreval2, scoreval3, scoreval4, scoreval5, scoreval6,scoreval7, lengthval, proval)
                            a = sess.run(predictions)
                            #print(a.shape)
                            #print(a)
                            #print(b_test_y)
                    print("Test accuracy:", tmp/5)
                    if batches == split:
                        batches = 0
                        break
                if batches<=split:
                    _,lossval= sess.run([self.train_op,self.loss], 
                                         feed_dict={self.X: x_batch, self.keep_prob:self.keep_prob_train, self.y:y_batch})
                    if batches%25 == 0:
                        print("loss:",lossval)
                batches += 1
                        #使用原始数据进行迭代
            for i in xrange(1):
                for batch_X,batch_Y in yieldData.get_next_batch(batch_size):
                    lossval, scoreval = sess.run([self.train_op, self.loss],
                                                    feed_dict={self.X: batch_X, self.keep_prob:self.keep_prob_train, self.y:batch_Y})

    def predict(self,sess,X):
        scoreval1, scoreval2, scoreval3, scoreval4, scoreval5, scoreval6, scoreval7,lengthval, proval= sess.run(
                           [self.score1, self.score2, self.score3, self.score4, self.score5, self.score6, self.score7,self.length, self.probabi], 
                           feed_dict={self.X: X, self.keep_prob: 1.0})
        res = self.makeprint(scoreval1, scoreval2, scoreval3, scoreval4, scoreval5, scoreval6, scoreval7,lengthval, proval)
        res = sess.run(res)
        return res

    def predict_y(self,sess,X,Y):
        accuval= sess.run( self.accuracy, feed_dict={self.X: X, self.keep_prob: 1, self.y:Y})
        print("This Test accuracy:", accuval)     
