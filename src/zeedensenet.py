#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 04:57:35 2019

@author: narahari
"""
import tensorflow as tf
from importlib import reload
import model_blocks
#reload(model_blocks)
from model_blocks import *
from model_blocks import ZeeConvBlk

class ZeeDenseNet(tf.keras.Model):
    
  def __init__(self, num_classes= 10, f_filter=64, weight=0.125, gap_mode="x_axis",
               dimensions_dict = {"dimensions_to_sample":(8,8)}, 
               layers_filters = {0:16, 1:32, 2:64},
               multisoft_list = [0, 1, 2]
               ):
    
    super().__init__()
    
    self.multisoft_list = multisoft_list
    
    self.layers_filters = layers_filters
    
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
    
    self.blk4 = ZeeConvBlk(dimensions_dict= dimensions_dict, layers_filters=layers_filters, gap_mode=gap_mode)
    
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
        

  def call(self, x, y):
    
    #h = self.pool(self.blk3(self.blk2(self.blk1(self.init_conv_bn(x)))))
    
    init_cbn = self.init_conv_bn(x)
    
    blk1 = self.blk1(init_cbn)
    
    if 0 in self.multisoft_list:
        
        gap1, loss1 = self.get_softmax(y, 0, layer_dict= {0:blk1})
        print(loss1)
        
    else:
        
        loss1 = tf.constant(0, dtype="float16")    
        
    blk2 = self.blk2(blk1)
    
    if 1 in self.multisoft_list:
    
        gap2, loss2 = self.get_softmax(y, 1, layer_dict= {0:blk1, 1:blk2})
        
    else:
        
        loss2 = tf.constant(0, dtype="float16")
        
    
    
    blk3 = self.blk3(blk2)
    
    gap3, loss3 = self.get_softmax(y, 2, layer_dict= {0:blk1, 1:blk2, 2:blk3}, last_layer_flag = True)
    
    loss = tf.math.add_n([loss1 , 0.3 * loss2 , 0.3 * loss3])
    
    correct = tf.reduce_sum(tf.cast(tf.math.equal(tf.argmax(gap3, axis = 1), y), tf.float16))
    
    return loss, correct, gap3