#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 04:57:35 2019

@author: narahari b m
"""
import tensorflow as tf
from importlib import reload
import model_blocks
#reload(model_blocks)
from model_blocks import *
from model_blocks import ZeeConvBlk
import time

class ZeeDenseNet(tf.keras.Model):
    
  def __init__(self, num_classes= 10, f_filter=64, weight=0.125, gap_mode="x_axis",
               dimensions_dict = {"dimensions_to_sample":(8,8)}, 
               layers_filters = {0:16, 1:32, 2:64},
               multisoft_list = [0, 1, 2],
               roots_flag = False,
               num_roots_dict = {0:8, 1:8, 2:8},
               residuals_flag = False,
               reluz= True, bnz=True, convz=True,
               ls_coeff1 = 0.3,
               ls_coeff2 = 0.3,
               ls_coeff3 = 1.0,
               ):
    
    super().__init__()
    
    self.multisoft_list = multisoft_list
    
    self.roots_flag = roots_flag
    
    self.num_roots_dict = num_roots_dict
    
    self.layers_filters = layers_filters
    
    self.residuals_flag = residuals_flag
    
    self.ls_coeff1 = ls_coeff1
    
    self.ls_coeff2 = ls_coeff2
    
    self.ls_coeff3 = ls_coeff3
    
    self.init_conv_bn = ConvBnRl(filters=f_filter, kernel_size=(3,3), strides=(1,1), padding="same" , dilation_rate=(1,1), 
                                  kernel_regularizer = None, kernel_initializer='glorot_uniform', conv_flag=True, bnflag=True,  relu=True, kernel_name=str(random.random())+"conv")
    
    self.blk1 = ResBlk(cbr = ConvBnRl(filters=f_filter*2, kernel_size=(3,3), strides=(1,1), padding="same" , conv_flag=True, bnflag=True, relu=True, kernel_name=str(random.random())+"conv"),
               
               cbr_res1 = ConvBnRl(filters=f_filter*2, kernel_size=(3,3), strides=(1,1), padding="same" , conv_flag=True, bnflag=True, relu=True, kernel_name=str(random.random())+"conv"),
               
               cbr_res2 = ConvBnRl(filters=f_filter*2, kernel_size=(3,3), strides=(1,1), padding="same" , conv_flag=True, bnflag=True, relu=True, kernel_name=str(random.random())+"conv"),
               
               pool=tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='same'), 
               
               res=True )
    
    self.blk2 = ResBlk(cbr = ConvBnRl(filters=f_filter*4, kernel_size=(3,3), strides=(1,1), padding="same" , conv_flag=True, bnflag=True, relu=True, kernel_name=str(random.random())+"conv"),
               
               cbr_res1 = ConvBnRl(filters=f_filter*4, kernel_size=(3,3), strides=(1,1), padding="same" , conv_flag=True, bnflag=True, relu=True, kernel_name=str(random.random())+"conv"),
               
               cbr_res2 = ConvBnRl(filters=f_filter*4, kernel_size=(3,3), strides=(1,1), padding="same" , conv_flag=True, bnflag=True, relu=True, kernel_name=str(random.random())+"conv"),
               
               pool=tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='same'), 
               
               res=True )
    
    self.blk3 = ResBlk(cbr = ConvBnRl(filters=f_filter*8, kernel_size=(3,3), strides=(1,1), padding="same" , conv_flag=True, bnflag=True, relu=True, kernel_name=str(random.random())+"conv"),
               
               cbr_res1 = ConvBnRl(filters=f_filter*8, kernel_size=(3,3), strides=(1,1), padding="same" , conv_flag=True, bnflag=True, relu=True, kernel_name=str(random.random())+"conv"),
               
               cbr_res2 = ConvBnRl(filters=f_filter*8, kernel_size=(3,3), strides=(1,1), padding="same" , conv_flag=True, bnflag=True, relu=True, kernel_name=str(random.random())+"conv"),
               
               pool=tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='same'), 
               
               res=True )
    
    self.blk4 = ZeeConvBlk(dimensions_dict= dimensions_dict, layers_filters=layers_filters, 
                           gap_mode=gap_mode, roots_flag=self.roots_flag, 
                           num_roots_dict= self.num_roots_dict, reluz= True, bnz=True, 
                           convz=True, residuals_flag=self.residuals_flag)
    
    self.pool = tf.keras.layers.GlobalMaxPool2D()
    
    self.linear = {}
    
    for i in range(len(self.multisoft_list)):
        
        self.linear[i] = tf.keras.layers.Dense(num_classes, kernel_initializer='glorot_uniform', use_bias=False)
        
    self.linear[-1] = tf.keras.layers.Dense(num_classes, kernel_initializer='glorot_uniform', use_bias=False)
    
    self.weight = weight
    


  def get_softmax(self, y, layer_num, layer_dict, last_layer_flag=False):
        
        gap_m = self.pool(layer_dict[layer_num])
        
        gap_z = self.pool(self.blk4(layer_dict))
        
        gap_concat = tf.concat([gap_m, gap_z], axis= 1)
        
            
        if last_layer_flag:
            
            gap = self.linear[-1](gap_concat) * self.weight
            
            
        else:
            
            
            linear_idx = self.multisoft_list.index(layer_num)
                
            gap = self.linear[linear_idx](gap_concat) * self.weight

            
            
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=gap, labels=y)
        
        loss = tf.reduce_sum(ce)
            
        return (gap, loss)
        

  def call(self, x, y, infer_multi = False):
    
    #h = self.pool(self.blk3(self.blk2(self.blk1(self.init_conv_bn(x)))))
    
    start_time = time.time()
    
    if infer_multi == True:
        
        multi_accuracies = {}
    
    init_cbn = self.init_conv_bn(x)
    
    blk1 = self.blk1(init_cbn)
    
    if 0 in self.multisoft_list:
        
        gap1, loss1 = self.get_softmax(y, 0, layer_dict= {0:blk1})
        
        #print(loss1)
        
        if infer_multi == True:
            
            multi_accuracies["sm1"] = {}
        
            multi_accuracies["sm1"]["acc"] = tf.reduce_sum(tf.cast(tf.math.equal(tf.argmax(gap1, axis = 1), y), tf.float16))
            
            curr_time = time.time()
            
            multi_accuracies["sm1"]["loss"] = loss1
            
            multi_accuracies["sm1"]["infer_time"] = curr_time - start_time
            
            multi_accuracies["sm1"]["prob"] = tf.nn.softmax(gap1, axis = 1)
        
    else:
        
        loss1 = tf.constant(0, dtype="float32")    
        
    blk2 = self.blk2(blk1)
    
    if 1 in self.multisoft_list:
    
        gap2, loss2 = self.get_softmax(y, 1, layer_dict= {0:blk1, 1:blk2})
        
        
        
        if infer_multi == True:
            
            multi_accuracies["sm2"] = {}
        
            multi_accuracies["sm2"]["acc"] = tf.reduce_sum(tf.cast(tf.math.equal(tf.argmax(gap2, axis = 1), y), tf.float16))
            
            curr_time = time.time()
            
            multi_accuracies["sm2"]["loss"] = loss2
            
            multi_accuracies["sm2"]["infer_time"] = curr_time - start_time
            
            multi_accuracies["sm2"]["prob"] = tf.nn.softmax(gap2, axis=1)
        
        
    else:
        
        loss2 = tf.constant(0, dtype="float32")
        
    
    
    blk3 = self.blk3(blk2)
    
    gap3, loss3 = self.get_softmax(y, 2, layer_dict= {0:blk1, 1:blk2, 2:blk3}, last_layer_flag = True)
    
    loss = tf.math.add_n([self.ls_coeff1 * loss1 , self.ls_coeff2 * loss2 , self.ls_coeff3 * loss3])
    
    correct = tf.reduce_sum(tf.cast(tf.math.equal(tf.argmax(gap3, axis = 1), y), tf.float16))
    
    if infer_multi == True:
        
            multi_accuracies["sm3"] = {}
            
            multi_accuracies["sm3"]["acc"] = correct
            
            curr_time = time.time()
    
            multi_accuracies["sm3"]["loss"] = loss3
                    
            multi_accuracies["sm3"]["infer_time"] = curr_time - start_time
            
            multi_accuracies["sm3"]["prob"] = tf.nn.softmax(gap3, axis=1)
        
    if infer_multi == True:
        
            return loss, correct, gap3, multi_accuracies
    
    return loss, correct, gap3