#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 13:40:41 2020

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling1D, Conv1D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--GPU",type=int, default=0)
parser.add_argument("--TIME",type=int, default=0)
args = parser.parse_args()
TIME = str(args.TIME)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
timenow = time.localtime(time.time())
TSTART = "V" + str(timenow[1]).zfill(2) + str(timenow[2]).zfill(2) + "_" + str(timenow[3]).zfill(2) + str(timenow[4]).zfill(2)
#%%save path 
save_path = "../"
fname = "Generator"
img_path = save_path +'image/' +fname+'_image_'+TIME
m_path = save_path  +'model/'+ fname+'_model/'
if not os.path.exists(img_path):
    os.makedirs(img_path)
if not os.path.exists(m_path):
    os.makedirs(m_path)
#%% with condition
def sample_images(epoch,x_train,feature):
    global img_path, beat_num
    sampled_labels = np.zeros((5,21))
    for i in range(5):
        index = i*beat_num
        sampled_labels[i] = feature[index]
    gen_imgs = Dnn.predict(sampled_labels)
    fig, axs = plt.subplots(1, 5)
    for j in range(5):
        index = beat_num*j
        axs[j].plot(x_train[index],linewidth=3.0, label='Original')
        
        axs[j].plot(gen_imgs[j],linewidth=1.0,c = 'r',label='Generated')
        axs[j].set_xticks([])
#        axs[0,j].set_title("Orginal ECG Person %d"%j)
        if j >0:
            axs[j].set_yticks([])
#        axs[1,j].set_title("Constructed ECG Person %d"%j)
    fig.set_size_inches(20, 5)
    fig.savefig(img_path+"/%d.png" % epoch)
    plt.close()
#%% data select 
def data_select(data,feature,subject):
    beats = int(len(data)/subject)
    new_data = []
    new_feature = [] 
    for sub in range(subject):
        avg = np.mean(feature[sub*beats:(sub+1)*beats], axis = 0)
        avg = avg.reshape(1,21)
        dis = np.linalg.norm(feature[sub*beats:(sub+1)*beats]-avg,axis = 1)
        Mean = np.mean(dis)
        std = np.std(dis,ddof = 1)
        index = np.where(dis<= Mean +2*std)[0]
        if len(index) < beats:
            lack_num = beats- len(index)
            select = np.random.choice(index,lack_num)
            Index = np.append(index,select)
        else:
            Index = index
        Index = Index + sub*beats
        print(sub*beats)
        print('-----')
        print(np.max(Index))
        new_data.extend(data[Index])  
        new_feature.extend(feature[Index])   
    new_data = np.array(new_data)
    new_feature = np.array(new_feature)
    return new_data,new_feature
#%% define generator 
def DNN_Structure(inp_shape=(28,28,1),dim = 100):
    global num_class,label_dim
    print("-----------------------Building model---------------------------------")
    
    # label input
    in_label = Input(shape=(label_dim,))
    in_signal = in_label

    struct = Dense(256)(in_signal)
    struct = BatchNormalization(momentum=0.8)(struct) 
    struct = LeakyReLU(alpha=0.2)(struct)  
    
    struct = Dense(512)(in_signal)
    struct = BatchNormalization(momentum=0.8)(struct) 
    struct = LeakyReLU(alpha=0.2)(struct)   
    
    struct = Dense(512)(in_signal)
    struct = BatchNormalization(momentum=0.8)(struct) 
    struct = LeakyReLU(alpha=0.2)(struct)   
    
    struct = Dense(1024)(struct)
    struct = BatchNormalization(momentum=0.8)(struct) 
    struct = LeakyReLU(alpha=0.2)(struct) 

    out_layer = Dense(np.prod(inp_shape), activation='tanh')(struct)
    # define model
    model = Model(in_label, out_layer)    
    opt = Adam(lr = 0.0002,beta_1 = 0.5)
    model.compile(loss='MSE', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model
#%%
path = '../data/SYN_noiseless/'
#%%hyperpara
beat_num = 500
label_dim = 21
batch_size = 32
sample_interval = 1000
data_shape = (200,1)
#%% Load the data
feature = np.load(path+"F_template.npy")
x_train = np.load(path+"new_train.npy")
## Rescale -1 to 1
x_train = x_train / 500 - 1 
# data select 
x_train, feature = data_select(x_train,feature,300)
## normalize 
for i in range(len(feature)):    
    feature[i,:15] = feature[i,:15]/np.max(feature[i,:15])*2-1
    feature[i,15:] = feature[i,15:]/np.max(feature[i,15:])*2-1
#%% construct model 
# Adversarial ground truths
Dnn = DNN_Structure(inp_shape = data_shape,dim = 100)
#%% training 
for epoch in range(20000):
    # ---------------------
    #  Train Dnn
    # ---------------------

    # Select a random batch of images
    idx = np.random.randint(0, x_train.shape[0], batch_size) 
    #imgs = x_train[idx]
    imgs, label1 = x_train[idx], feature[idx]
    targets = imgs.reshape((32,200))   
    Loss = Dnn.train_on_batch(label1, targets) 
    gen_imgs = Dnn.predict(label1)
    if epoch == 20000-1:
        Dnn.save(m_path+'Dnn_without_noise'+TSTART+'.h5')
    # Plot the progress
    if epoch % 100==0:
        print ("%d [G loss: %f]" % (epoch, Loss[0]))
    # If at save interval => save generated image samples
    if epoch % sample_interval == 0:
        sample_images(epoch,x_train,feature)

