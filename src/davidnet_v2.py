#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 23:32:33 2019

@author: narahari b m
"""
import tensorflow as tf
from importlib import reload
import model_blocks
#reload(model_blocks)
from model_blocks import *

import time

class DavidNetMultiSoft(tf.keras.Model):
  
  def __init__(self, num_classes= 10, f_filter=64, weight=0.125, 
               kernel_initializer='glorot_uniform', multisoft_list = [0, 1, 2],
               residual_strategy = [True, True, True]):
    
    super().__init__()
    
    self.multisoft_list = multisoft_list
    
    self.init_conv_bn = ConvBnRl(filters=f_filter, kernel_size=(3,3), strides=(1,1), padding="same" , dilation_rate=(1,1), 
                                  kernel_regularizer = None, kernel_initializer=kernel_initializer, conv_flag=True, bnflag=True,  relu=True)
    
    self.blk1 = ResBlk(cbr = ConvBnRl(filters=f_filter*2, kernel_size=(3,3), strides=(1,1), padding="same" , kernel_initializer=kernel_initializer, conv_flag=True, bnflag=True, relu=True),
               
               cbr_res1 = ConvBnRl(filters=f_filter*2, kernel_size=(3,3), strides=(1,1), padding="same" , kernel_initializer=kernel_initializer, conv_flag=True, bnflag=True, relu=True),
               
               cbr_res2 = ConvBnRl(filters=f_filter*2, kernel_size=(3,3), strides=(1,1), padding="same" , kernel_initializer=kernel_initializer, conv_flag=True, bnflag=True, relu=True),
               
               pool=tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='same'), 
               
               res=residual_strategy[0] )
    
    self.blk2 = ResBlk(cbr = ConvBnRl(filters=f_filter*4, kernel_size=(3,3), strides=(1,1), padding="same" , kernel_initializer=kernel_initializer, conv_flag=True, bnflag=True, relu=True),
               
               cbr_res1 = ConvBnRl(filters=f_filter*4, kernel_size=(3,3), strides=(1,1), padding="same" , kernel_initializer=kernel_initializer, conv_flag=True, bnflag=True, relu=True),
               
               cbr_res2 = ConvBnRl(filters=f_filter*4, kernel_size=(3,3), strides=(1,1), padding="same" , kernel_initializer=kernel_initializer, conv_flag=True, bnflag=True, relu=True),
               
               pool=tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='same'), 
               
               res=residual_strategy[1] )
    
    self.blk3 = ResBlk(cbr = ConvBnRl(filters=f_filter*8, kernel_size=(3,3), strides=(1,1), padding="same" , kernel_initializer=kernel_initializer, conv_flag=True, bnflag=True, relu=True),
               
               cbr_res1 = ConvBnRl(filters=f_filter*8, kernel_size=(3,3), strides=(1,1), padding="same" , kernel_initializer=kernel_initializer, conv_flag=True, bnflag=True, relu=True),
               
               cbr_res2 = ConvBnRl(filters=f_filter*8, kernel_size=(3,3), strides=(1,1), padding="same" , kernel_initializer=kernel_initializer, conv_flag=True, bnflag=True, relu=True),
               
               pool=tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='same'), 
               
               res=residual_strategy[2] )
    
    self.pool = tf.keras.layers.GlobalMaxPool2D()
    
    self.linear = {}
    
    for i in range(len(self.multisoft_list)):
        
        self.linear[i] = tf.keras.layers.Dense(num_classes, kernel_initializer='glorot_uniform', use_bias=False)
        
    self.linear[-1] = tf.keras.layers.Dense(num_classes, kernel_initializer='glorot_uniform', use_bias=False)
    
    self.weight = weight
    
  def get_softmax(self, y, layer, layer_num, last_layer_flag=False):
        
        gap_m = self.pool(layer)
        
        if last_layer_flag:
            
            gap = self.linear[-1](gap_m) * self.weight
            
            
        else:
            
            
            linear_idx = self.multisoft_list.index(layer_num)
                
            gap = self.linear[linear_idx](gap_m) * self.weight

            
            
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=gap, labels=y)
        
        loss = tf.reduce_sum(ce)
            
        return (gap, loss)

  def call(self, x, y, infer_multi=False):
      
    start_time = time.time()
      
    if infer_multi == True:
        
        multi_accuracies = {}
      
    layer1 = self.blk1(self.init_conv_bn(x))
    
    if 0 in self.multisoft_list:
        
        gap1, loss1 = self.get_softmax(y, layer1, 0)
        
        if infer_multi == True:
            
            multi_accuracies["sm1"] = {}
        
            multi_accuracies["sm1"]["acc"] = tf.reduce_sum(tf.cast(tf.math.equal(tf.argmax(gap1, axis = 1), y), tf.float16))
            
            curr_time = time.time()
            
            multi_accuracies["sm1"]["loss"] = loss1
            
            multi_accuracies["sm1"]["infer_time"] = curr_time - start_time
            
            multi_accuracies["sm1"]["prob"] = tf.nn.softmax(gap1, axis = 1)
        
    else:
        
        loss1 = tf.constant(0, dtype="float32") 
    
    layer2 = self.blk2(layer1)
    
    if 1 in self.multisoft_list:
        
        gap2, loss2 = self.get_softmax(y, layer2, 1)
        
        if infer_multi == True:
            
            multi_accuracies["sm2"] = {}
        
            multi_accuracies["sm2"]["acc"] = tf.reduce_sum(tf.cast(tf.math.equal(tf.argmax(gap2, axis = 1), y), tf.float16))
            
            curr_time = time.time()
            
            multi_accuracies["sm2"]["loss"] = loss2
            
            multi_accuracies["sm2"]["infer_time"] = curr_time - start_time
            
            multi_accuracies["sm2"]["prob"] = tf.nn.softmax(gap2, axis=1)
        
        
    else:
        
        loss2 = tf.constant(0, dtype="float32")
    
    layer3 = self.blk3(layer2)
    
    gap3, loss3 = self.get_softmax(y, layer3, 2, last_layer_flag=True)
    
    correct = tf.reduce_sum(tf.cast(tf.math.equal(tf.argmax(gap3, axis = 1), y), tf.float16))
        
    if infer_multi == True:
        
            multi_accuracies["sm3"] = {}
            
            multi_accuracies["sm3"]["acc"] = correct
            
            curr_time = time.time()
    
            multi_accuracies["sm3"]["loss"] = loss3
                    
            multi_accuracies["sm3"]["infer_time"] = curr_time - start_time
            
            multi_accuracies["sm3"]["prob"] = tf.nn.softmax(gap3, axis=1)
            
    if infer_multi == True:
        
            return loss3, correct, gap3, multi_accuracies
    
    
    return loss3, correct, gap3