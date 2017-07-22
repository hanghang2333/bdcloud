#coding=utf8
from __future__ import print_function
import codecs
import numpy as np
import time
import MultiNumNet
import getdata
import os
import tensorflow as tf

this_time = time.time()
# 超参数
num_classes = 16
image_height = 60
image_width = 180
image_channel = 1
num_length = 7
# 初始化
#init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
sess = tf.Session()#全局load模型
model = MultiNumNet.MultiNet(image_height, image_width, image_channel, 0.5, num_classes,num_length)
saver = tf.train.Saver()
# 读取训练好的模型参数
saver.restore(sess, 'savedmodel/50model')
#print('初始化用时：%fs' % (time.time() - this_time))
#print('start predicting')

def predict_result(X):
    return model.predict(sess, X)

def preprocess(X_pre):
    '''
    输入图片也需要进行与训练时相同的预处理才能放到模型里传播
    这里假设X_pre是图片的完整路径列表,可以不只一张图片
    '''
    numbers = len(X_pre)
    X = np.zeros((numbers, image_height, image_width, image_channel), np.int8)
    inx = 0
    for i in xrange(numbers):
        data = getdata.get_image(X_pre[i], image_height, image_width)
        data = data.astype(np.int8)
        #mean = np.sum(data)/(image_height*image_width)
        #std = np.max(data) - np.min(data)
        #data = (data -mean)/std
        X[inx, :, :, :] = data
        inx = inx + 1
    return X

def predict(X_pre):
    
    a = predict_result(preprocess(X_pre))
    return a
wrong = 0
def makeresult(X_list):
    global wrong
    res = []
    for line in X_list:
        tmpres = ''
        length = line[7]+1
        line = line[0:length]
        for i in line:
            if i == 10:
                tmpres = tmpres+'+'
            elif i==11:
                tmpres = tmpres+'-'
            elif i==12:
                tmpres = tmpres+'*'
            elif i==13:
                tmpres = tmpres+'('
            elif i==14:
                tmpres = tmpres+')'
            else:
                tmpres = tmpres+ str(i)
        ans = 0
        try:
            ans = eval(tmpres)
            wrong = wrong + 1
        except Exception as e:
            wrong = wrong + 1
            print('wrong at',wrong-1)
            pass
        tmpres = tmpres+' '+str(ans)
        res.append(tmpres)
    return res

class Data():
    def __init__(self,X):
        self.X = X
        self.length = len(X)
        self.index = 0

    def get_next_batch(self, batch_size):
        '''
        以batch_size大小来取数据
        '''
        while self.index < self.length:
            returnX = self.X[self.index:self.index+batch_size]
            self.index = self.index + batch_size
            yield returnX

def test():
    path = '/home/lihang/2017/bdcloud/data/test/val/'
    namedict = {}
    for name in os.listdir(path):
        namedict[int(name.replace('.png',''))] = path+name
    namelist = []
    length = len(namedict)
    for i in range(length):
        namelist.append(namedict[i])
    namelistData = Data(namelist).get_next_batch(200)
    out = codecs.open('result.txt','w','utf8')
    for X in namelistData:
        res = predict(X)
        res = makeresult(res)
        #print(res)
        out.write('\n'.join(res)+'\n')
    out.close()
test()
