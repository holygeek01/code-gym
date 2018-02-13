import os
import skimage
import numpy as np
from skimage import data, io, transform
from skimage.transform import rescale
from skimage.color import rgb2gray
import matplotlib.pyplot as plt 
import tensorflow as tf

import cv2

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Layer, Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Deconvolution2D, Activation
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential, Model
from keras.optimizers import SGD, adadelta, adagrad, adam, adamax, nadam

import  PIL
from PIL import Image

import random

import h5py

LEARNING_RATE = 0.01
BATCH_SIZE = 16
NUM_EPOCHS = 1000
NUM_CHANNELS = 3

LOAD_PRE_TRAINED_MODEL = True
DO_TESTING = True

def charbonnier(y_true, y_pred):
    return K.sqrt(K.square(y_true - y_pred) + 0.01**2)

def get_unet():
    
    inputs = Input((80, 160, 6))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(3, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model

def get_input_frames(img1, img2):
    return(np.concatenate((img1, img2), axis=2))


def load_data(data_directory):
    files = os.listdir(data_directory)

    in_frames = np.zeros(shape=(0, 80, 160, 6))
    out_frames = np.zeros(shape=(0, 80, 160, 3))
    
    num_files = 0

    t1 = np.zeros(shape=(80, 160, 3))
    t2 = np.zeros(shape=(80, 160, 3))
    t = np.zeros(shape=(80, 160, 3))

    res_temp = np.zeros(shape=(80, 160, 3))
    
    for f in files:
        img = cv2.imread(data_directory + "\\" + f)
        
        if num_files%3==0:
            t1 = img
        elif num_files%3==1:               
            res_temp = img
        elif num_files%3==2:
            t2 = img
            
            linf = (t2 - t1)/2
            linf = res_temp - linf
            linf = linf.reshape((1, 80, 160, 3))
            
            t = get_input_frames(t1, t2)
            t = t.reshape((1, 80, 160, 6))
            
            out_frames = np.append(out_frames, linf, axis=0)

            #print("t2 dim " + str(t2.shape))
            #print("t dim " + str(t.shape))
            #print("in_frames dim " + str(in_frames.shape))
            
            in_frames = np.append(in_frames, t, axis=0)
            
        print("Reading Frame " + str(num_files))
            
        num_files+=1

        if num_files == 1500:
            break

    print("Found Total " + str(num_files) + " images")
    print("Input Shape " + str(in_frames.shape))
    print("Output Shape " + str(out_frames.shape))

    return in_frames, out_frames

model = get_unet()

optimizer = adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# loss = "hinge"
# loss = "mse"
# loss = "categorical_crossentropy"
# loss = "binary_crossentropy"

loss = charbonnier

model.compile(loss=loss, optimizer=optimizer)

#####################################################s########

data_dir = 'Cosmos\02'

in_frames, out_frames = load_data(data_dir)


model.fit(x=in_frames, y=out_frames, batch_size=8, epochs=50, verbose=1,
          callbacks=None, validation_split=0.0, validation_data=None,
          shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,
          steps_per_epoch=None, validation_steps=None)

model.save('vfi.h5')

