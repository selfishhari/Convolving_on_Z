#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 04:57:35 2019

@author: narahari
"""
import tensorflow as tf
from importlib import reload
import model_blocks
reload(model_blocks)
from model_blocks import *

class ZeeDenseNet(tf.keras.Model):
    
  def __init__(self, num_classes= 10, f_filter=64, weight=0.125, dimensions_dict = {"dimensions_to_sample":(8,8)}, layers_filters = {0:16, 1:32, 2:64}):
    
    super().__init__()
    
    self.init_conv_bn = ConvBnRl(filters=f_filter, kernel_size=(1,1), strides=(1,1), padding="same" , dilation_rate=(1,1), 
                                  kernel_regularizer = None, kernel_initializer='glorot_uniform', conv_flag=True, bnflag=True,  relu=True)
    
    self.blk1 = ResBlk(cbr = ConvBnRl(filters=f_filter*2, kernel_size=(3,3), strides=(1,1), padding="same" , conv_flag=True, bnflag=True, relu=True),
               
               cbr_res1 = ConvBnRl(filters=f_filter*2, kernel_size=(3,3), strides=(1,1), padding="same" , conv_flag=True, bnflag=True, relu=True),
               
               cbr_res2 = ConvBnRl(filters=f_filter*2, kernel_size=(3,3), strides=(1,1), padding="same" , conv_flag=True, bnflag=True, relu=True),
               
               pool=tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='same'), 
               
               res=True )
    
    self.blk2 = ResBlk(cbr = ConvBnRl(filters=f_filter*4, kernel_size=(3,3), strides=(1,1), padding="same" , conv_flag=True, bnflag=True, relu=True),
               
               cbr_res1 = ConvBnRl(filters=f_filter*4, kernel_size=(3,3), strides=(1,1), padding="same" , conv_flag=True, bnflag=True, relu=True),
               
               cbr_res2 = ConvBnRl(filters=f_filter*4, kernel_size=(3,3), strides=(1,1), padding="same" , conv_flag=True, bnflag=True, relu=True),
               
               pool=tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='same'), 
               
               res=True )
    
    self.blk3 = ResBlk(cbr = ConvBnRl(filters=f_filter*8, kernel_size=(3,3), strides=(1,1), padding="same" , conv_flag=True, bnflag=True, relu=True),
               
               cbr_res1 = ConvBnRl(filters=f_filter*8, kernel_size=(3,3), strides=(1,1), padding="same" , conv_flag=True, bnflag=True, relu=True),
               
               cbr_res2 = ConvBnRl(filters=f_filter*8, kernel_size=(3,3), strides=(1,1), padding="same" , conv_flag=True, bnflag=True, relu=True),
               
               pool=tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='same'), 
               
               res=True )
    
    self.blk4 = ConciseDenseBlk(dimensions_dict= dimensions_dict, layers_filters=layers_filters)
    
    self.pool = tf.keras.layers.GlobalMaxPool2D()
    
    self.linear = tf.keras.layers.Dense(num_classes, kernel_initializer='glorot_uniform', use_bias=False)
    
    self.weight = weight

  def call(self, x, y):
    
    #h = self.pool(self.blk3(self.blk2(self.blk1(self.init_conv_bn(x)))))
    
    init_cbn = self.init_conv_bn(x)
    
    blk1 = self.blk1(init_cbn)
    
    blk2 = self.blk2(blk1)
    
    blk3 = self.blk3(blk2)
    
    layers_dict = {0 :blk1, 1:blk2, 2:blk3}
    
    
    blk4 = self.blk4(layers_dict)    
    
    gap1 = self.pool(blk3)
    
    gap2 = self.pool(blk4) 
    
    gap = tf.concat([gap1, gap2], axis= 1)
    
    gap = self.linear(gap) * self.weight
    
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=gap, labels=y)
    
    loss = tf.reduce_sum(ce)
    
    correct = tf.reduce_sum(tf.cast(tf.math.equal(tf.argmax(gap, axis = 1), y), tf.float16))
    
    return loss, correct, gap    