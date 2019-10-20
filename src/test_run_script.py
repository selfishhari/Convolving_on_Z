#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 13:59:01 2019

@author: narahari b m
"""

import numpy as np
import os


import tensorflow as tf

import numpy as np
import time, math
from tqdm import tqdm_notebook as tqdm


import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

os.chdir("/home/fractaluser/Personal/Narahari/eva_research/v2/eva_research_team4/src")

from importlib import reload

from augmentation_utils import cutout


BATCH_SIZE = 10 #@param {type:"integer"}
MOMENTUM = 0.95 #@param {type:"number"}

MIN_MOMENTUM = 0.8 #@param {type:"number"}
LEARNING_RATE = 0.4 #@param {type:"number"}
WEIGHT_DECAY = 5e-4 #@param {type:"number"}
EPOCHS = 1 #@param {type:"integer"}


MIN_LEARNING_RATE = 0.000001 #@param {type:"number"}

END_LR_SMOOTHING_PERC = 0.15 #@param {type:"number"}

COMMENTS = "Densenext test" #@param {type:"string"}


params_tune = {
    
  "epochs":EPOCHS, 
 
  "batch_size" : BATCH_SIZE,

  "max_lr": LEARNING_RATE,

  "min_lr":MIN_LEARNING_RATE,

  "end_anneal_pc":END_LR_SMOOTHING_PERC,

  "max_mom":MOMENTUM,
 
  "min_mom":MIN_MOMENTUM,
 
  "wd":WEIGHT_DECAY,
  
  "skip_testing_epochs":0,
    
  "batches_per_epoch":3//BATCH_SIZE,
    
  "comments":COMMENTS
}

import data_pipeline

reload(data_pipeline)

#data_pipeline.get_data(dataset_name = "CIFAR10", tfrecords_flag=True)

loaded_tfrecs = data_pipeline.load_tfrecords(params_tune["batch_size"])

train_dataset = loaded_tfrecs["train"]

eval_dataset = loaded_tfrecs["eval"]


train_mean = np.array([125.30691805, 122.95039414, 113.86538318])

train_std= np.array([62.99321928, 62.08870764, 66.70489964])

normalize = lambda x: ((x - train_mean) / train_std)

import augmentation_utils
reload(augmentation_utils)

#from augmentation_utils import cutout

def data_aug(x, y):
    
    #x = tf.image.per_image_standardization(x)
    
    x = normalize(x)

    paddings = [(4, 4), (4, 4), (0, 0)]
    
    x = tf.pad(x, paddings, "REFLECT")
    
    x = tf.random_crop(x, [32, 32, 3])
    
    x = tf.image.random_flip_left_right(x)
    
    x = cutout(x)
    
    return (x, y)

def tst_data_supplier(epoch_num):
    
    batch_size = params_tune["batch_size"]
    
    global eval_dataset
  
    len_test = 30

    test_set = eval_dataset.take(len_test).map(data_aug).batch(batch_size).prefetch(1)
    
    return (test_set, len_test)

def trn_data_supplier(epoch_num):
    
    batch_size = params_tune["batch_size"]
    
    global train_dataset
    
    len_train = 1000

    train_set = train_dataset.take(len_train).map(data_aug).batch(batch_size).prefetch(1)
    
    return (train_set, len_train)


#data_aug = lambda x, y: (tf.image.random_flip_left_right(tf.random_crop(x, [32, 32, 3])), y)

#from matplotlib import pyplot as plt
#its, lens =trn_data_supplier(0)

#x, y = next(its.__iter__())

#plt.imshow(x[3,:,:,:])


import run_util

import zeedensenet
import model_blocks
reload(model_blocks)
reload(zeedensenet)
reload(run_util)

from run_util import Run

model = zeedensenet.ZeeDenseNet(f_filter=16, 
                                dimensions_dict= {"dimensions_to_sample":(8,8)}, 
                                layers_filters={0:32, 1:64, 2:128}, 
                                roots_flag = True, 
                                num_roots_dict={0:8,1:8,2:8},
                                multisoft_list=[0, 1,2]
                                )


obj = Run()

x = obj.run( params_tune, trn_data_supplier, tst_data_supplier, model=model)

run_util.early_inference_accuracy(model=obj.model, test_dataset = tst_data_supplier)

import visual_utils
reload(visual_utils)
    
#diff_df = visual_utils._get_batch_difference_df(obj.model, imgs, ys).reset_index(drop=True)

diff_df = visual_utils.grab_different_imgs(obj.model, trn_data_supplier)

all_df = visual_utils.grab_different_imgs(obj.model, trn_data_supplier, difference=False)

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

import warnings

warnings.filterwarnings("ignore")

visual_utils.plot_diff(diff_df, denormalize=True)

visual_utils.plot_diff(diff_df,
                       sm_col="sm1_correct", main_col="sm3_correct",
                       img_col="imgs",
              true_col="ys", 
              pred_col="sm1_class",
              pred_col2="sm3_class",
              prob_col1="sm1_probs",
              prob_col2="sm3_probs",
              ncols=10,
              denormalize=True)


visual_utils.classwise_display(df=diff_df, img_col="imgs", 
                               true_col="ys", pred_col="sm2_class", 
                               ncols=5, class_map=class_names)

visual_utils.plot_good_and_worst(df=all_df, sm_col="sm2_correct",
              img_col="imgs",
              true_col="ys", 
              pred_col="sm2_class",
              prob_col="sm2_probs",
              ncols=10,
              denormalize=True,
              CLASSWISE_SELECT_TOP_IMAGES = 10)


visual_utils.plot_cm(all_df.ys, all_df.sm1_class)

from matplotlib import pyplot as plt

plt.imshow(diff_df["imgs"][0].astype(np.uint8))


##################################################

model1 = zeedensenet.ZeeDenseNet(f_filter=2, dimensions_dict= {"dimensions_to_sample":(8,8)}, layers_filters={0:2})

model2 = zeedensenet.ZeeDenseNet(f_filter=3, dimensions_dict= {"dimensions_to_sample":(8,8)}, layers_filters={0:2, 1: 3})

model3 = zeedensenet.ZeeDenseNet(f_filter=4, dimensions_dict= {"dimensions_to_sample":(8,8)}, layers_filters={0:2, 1: 2, 2: 3})


params_tune_grid = {
        
  "model":[model1, model2, model3],
    
  "epochs": [1] , 
 
  "batch_size" : [5],

  "max_lr": [LEARNING_RATE],

  "min_lr":[MIN_LEARNING_RATE],

  "end_anneal_pc":[END_LR_SMOOTHING_PERC],

  "max_mom":[MOMENTUM],
 
  "min_mom":[MIN_MOMENTUM],
 
  "wd":[WEIGHT_DECAY],
  
  "skip_testing_epochs":[0],
    
  "batches_per_epoch":[100//5],
    
  "comments":[COMMENTS]
}

obj.grid_search(params_tune_grid, trn_data_supplier, tst_data_supplier)

import all_models

reload(all_models)
    


import densenext#DenseNext

reload(densenext)

model = densenext.DenseNext()
model(np.random.normal(size=(5,32,32,3)).astype(np.float16), 
              np.array([1, 2, 1, 1, 1]))

import model_blocks


l1 = np.random.normal(size=(3,16, 16, 32)).astype(np.float16)

l2 = np.random.normal(size=(3, 8, 8, 64)).astype(np.float16)

l3 = np.random.normal(size=(3, 4, 4,128)).astype(np.float16)

layers_dict = {0: l1, 1:l2, 2:l3}

layers_dict = {0: l1, 1:l2}

reload(model_blocks)

zee_dense_blk = model_blocks.ZeeConvBlk(gap_mode="channel_axis", roots_flag=True)

zee_block_output = zee_dense_blk(layers_dict)

tf.shape(zee_block_output)

import zeedensenet
reload(model_blocks)
reload(zeedensenet)
model = zeedensenet.ZeeDenseNet(
        dimensions_dict= {"dimensions_to_sample":(8,8)}, 
        layers_filters={0:16, 1:32, 2:64}, 
        gap_mode="channel_axis",
        multisoft_list = [0,2]
        )

m_o = model(np.random.normal(size=(15,64,64,5)).astype(np.float16), 
              np.array([1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1]))

print(tf.shape(m_o[2]))

m_o




from matplotlib import pyplot as plt


x, y = next(trn_data_supplier(0)[0].__iter__())

img = x[0, :, :, :]

plt.imshow(img.numpy().astype(np.uint8))

from matplotlib import pyplot as plt

plt.imshow(diff_df["imgs"][3].astype(np.uint8))
