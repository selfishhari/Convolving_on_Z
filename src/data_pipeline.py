"""
1. Get Data
2. Load Data
"""
from random import shuffle
import glob
import sys
import cv2
import numpy as np
#import skimage.io as io
import tensorflow as tf
from tfrecord_utils import *


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  # Load images from address
def load_image(addr,img_size):
    # read an image and resize
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    if img is None:
        return None
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
  
# Load MNIST,CIFAR10,CIFAR100 data and store it in x_train,x_test
def load_tfdata(Data):
  if Data == "CIFAR10":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
  elif Data == "CIFAR100":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
  return (x_train, y_train), (x_test, y_test)

def normalize_data(data, mean, std):
      normalize = lambda x: ((x - mean) / std).astype('float16')
      norm_data = normalize(data)
      return norm_data
    
pad4 = lambda x: np.pad(x, [(0, 0), (4, 4), (4, 4), (0, 0)], mode='reflect')

def do_onetime_processing(x_train, y_train, x_test, y_test, normalize=True, pad=False, 
                          pixel_normalize=False, use_imgnet_vals=False):
    
    len_train, len_test = len(x_train), len(x_test)
    img_size = x_train.shape[1]
    y_train = y_train.astype('int64').reshape(len_train)
    y_test = y_test.astype('int64').reshape(len_test)
    
    if pixel_normalize:
        
        x_train = x_train/255
        
        x_test = x_test/255
    
    if normalize:

        #normalize data
        train_mean = np.mean(x_train, axis=(0,1,2))
        train_std = np.std(x_train, axis=(0,1,2))

        
        if use_imgnet_vals:
            
            train_mean = [0.4914, 0.4822, 0.4465]
            
            train_std = [0.2023, 0.1994, 0.2010]

        x_train = normalize_data(x_train, train_mean, train_std)
        
        x_test = normalize_data(x_test, train_mean, train_std)
        
    if pad:
        
        x_train = pad4(x_train)
        
    return (x_train.astype(np.float16), y_train.astype(np.int64), x_test.astype(np.float16), y_test.astype(np.int64))
    

def get_data(dataset_name = "CIFAR10", tfrecords_flag=False, 
             tfrec_trn_pth='../data/train/train.tfrecords', tfrec_tst_pth='../data/test/test.tfrecords',
            np_trn_pth= ["../data/train/train_x.npy","../data/train/train_y.npy"] , 
             np_tst_pth=["../data/test/test_x.npy", "../data/test/test_y.npy"]):
    
    print("Downloading..")
    
    (x_train, y_train), (x_test, y_test) = load_tfdata(dataset_name)
    
    print("Preprocessing..")
    
    x_train, y_train, x_test, y_test = do_onetime_processing(x_train, y_train, x_test, y_test)
    
    print("Saving..")
    
    if tfrecords_flag:
        
        print("saving to tf records")
        
        createDataRecord(tfrec_trn_pth, x_train, y_train)
        
        createDataRecord(tfrec_tst_pth, x_test, y_test)
        
    else:
        
        print("saving to numpy pickle")
        
        np.save(np_trn_pth[0], x_train)
        
        np.save(np_trn_pth[1] ,y_train)
        
        np.save(np_tst_pth[0], x_test)
        
        np.save(np_tst_pth[1] ,y_test)
        
def load_saved_numpy_data(train_path=["../data/train/train_x.npy","../data/train/train_y.npy"],
                         test_path=["../data/test/test_x.npy", "../data/test/test_y.npy"]):
    
    x_train = np.load(train_path[0])
    
    y_train = np.load(train_path[1])
    
    x_test = np.load(test_path[0])
    
    y_test = np.load(test_path[1])
    
    return x_train, y_train, x_test, y_test
        
