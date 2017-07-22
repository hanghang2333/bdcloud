#coding=utf8
from __future__ import print_function
from PIL import Image
import os
import codecs
import numpy as np
import tensorflow as tf

def getlabel(name, length=8):
    '''
    将name字符串转换为相应的标签值.这里以10->+,11->-,12->*,13->(,14->)
    name:字符串，在这里定义成是相应的图片的名字(原始数据里每个图片的标签放在名字上)
    return:一维np array。初步定义长度是7.会进行补齐。用15来补齐
    eg:
    name:"(3-7)+5" return:[13,3,11,7,14,10,5,6]
    '''
    #开始转换
    resarray = np.zeros(length)
    resarray.fill(15)
    lengthhere = len(name)
    resarray[length-1] = lengthhere-1
    for i in xrange(lengthhere):
        if name[i] == '+':
            resarray[i] = 10
        elif name[i] == '-':
            resarray[i] = 11
        elif name[i] == '*':
            resarray[i] = 12
        elif name[i] == '(':
            resarray[i] = 13
        elif name[i] == ')':
            resarray[i] = 14
        else:
            try:
                resarray[i] = int(name[i])
            except Exception as e:
                print('wrong',name,i,' ',name[i])
    return resarray

def test_getlabel():
    print(getlabel("(3-7)+5"))
    print(getlabel("5-6+2"))
    print(getlabel("(6+7)*2"))
    print(getlabel("(4+2)+7"))
    print(getlabel("(6*4)*4"))
    print(getlabel('5+(2*5)'))
#test_getlabel()

def get_path_label(labelfile,image_path,numlength):
    '''
    给出相应目录下所以文件的文件名到标签值的一个映射字典{}
    labelfile:需要读取的文件
    image_path:需要读取的图片的目录
    return:字典{文件1:标签array,文件2:标签array,...}
    '''
    label = codecs.open(labelfile,'r','utf8').readlines()
    label = [i.split() for i in label]
    result = [i[1].replace('\n','') for i in label]#没啥用
    labels = [i[0] for i in label]
    res = {}
    for i,j in enumerate(labels):
        res[image_path+str(i)+'.png'] = getlabel(j,numlength+1)
    return res

#res = get_path_label('../data/labels.txt','',7)
#print(res)

def get_image(image_path,height,width):  
    """
    从给定路径中读取图片，返回的是numpy.ndarray
    image_path:string, height:图像像素高度 width:图像像素宽度
    return:numpy.ndarray的图片tensor 
    """ 
    im = Image.open(image_path).convert('L')
    b = reshape(im,height,width)
    return b

def reshape(im,height,width):
    '''
    resize
    im:PIL读取图片后的Image对象
    '''
    #b = np.reshape(im, [im.size[1], im.size[0], 1])
    b = im.resize((width,height),Image.BILINEAR)
    b = np.reshape(im, [im.size[1], im.size[0], 1])
    #b = tl.prepro.imresize(b, size=(height, width), interp='bilinear')
    return b

def normal(data,height,width):
    '''
    归一化
    '''
    data = data.astype(np.int8)
    #mean = np.sum(data)/(height*width)
    #std = np.max(data) - np.min(data)
    #data = (data -mean)/std
    return data

def get_by_path(image_height,image_width,image_channel,num_length,labelfile, image_path):
    pathlabel = get_path_label(labelfile,image_path,num_length)
    image_num = len(pathlabel)
    inx = 0
    X = np.zeros((image_num, image_height, image_width, image_channel), np.float32)
    Y = np.zeros((image_num, num_length+1), np.uint8)#最后还有一位存储数字串长度
    for path in pathlabel:#对每一个label
        data = get_image(path, image_height, image_width)
        data = normal(data,image_height,image_width)
        label = pathlabel[path]
        X[inx, :, :, :] = data
        Y[inx, :] = label
        inx = inx+1
    print(X.shape)
    print(Y.shape)
    return X, Y

#print(get_image('/home/lihang/ocr/data/num/0.10.jpeg',30,100))