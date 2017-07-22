#coding=utf8
import numpy as np
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
from MultiNumNet import MultiNet
import getdata
import util
num_classes = 16 #0-9and +-*()and blank
image_height = 60#数字图片应该普遍长宽比例是这样
image_width = 180
image_channel = 1
num_length = 7
imgpath = '../data/image_contest_level_1/'
labelfile = '../data/labels.txt'
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
# 训练，测试，持久化
#with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
os.environ["CUDA_VISIBLE_DEVICES"]="0"
with tf.Session() as sess:
    model = MultiNet(image_height, image_width, image_channel, 0.3, num_classes, num_length)   
    # 完成数据的读取，使用的是tensorflow的读取图片
    X, Y = getdata.get_by_path(image_height, image_width, image_channel, num_length,labelfile, imgpath)
    # 将数据集shuffle
    X, Y = util.shuffledata(X,Y)
        # 将数据区分为测试集合和训练集合
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.95, random_state=33)
    model.train(sess, X_train, Y_train, X_test, Y_test, 101)
