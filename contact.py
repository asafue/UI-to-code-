import cv2
import numpy as np
import os
from random import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression  

def contact(img_data):
    #TEST_DIR = 'test2/'
    #img_name = '10.jpg'
    IMG_SIZE = 50
    LR = 1e-3
    #img_data = cv2.imread(os.path.join(TEST_DIR,img_name), cv2.IMREAD_GRAYSCALE)
    img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
    testing_data = np.array(img_data)
    with tf.Graph().as_default():
        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
        convnet = conv_2d(convnet, 32, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 64, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 128, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 64, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 32, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)
        convnet = fully_connected(convnet, 3, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
        newmodel = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)
        newmodel.load('algos/models/contact.tflearn')
        
        img_data = testing_data
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
        newmodel_out = newmodel.predict([data])[0]
        if np.argmax(newmodel_out) == 1:
            str_label = 'contact1'
        elif np.argmax(newmodel_out) == 0:
            str_label = 'contact2'
        else:
            str_label = 'contact3'
    return(str_label)