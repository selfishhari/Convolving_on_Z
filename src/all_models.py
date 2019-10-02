from model_blocks import *
import tensorflow as tf

class ResNext_50(tf.keras.Model):
  
  def __init__(self, num_classes= 10, f_filter=32):
    
    super().__init__()

    
    self.first_layer = ConvBnRl(filters=f_filter, kernel_size=(3,3), strides=(1,1), padding="same" , dilation_rate=(1,1), 
                                  kernel_regularizer = None, kernel_initializer='glorot_uniform', conv_flag=True, bnflag=True,  relu=True)

    self.blk1 = ResNeXtBlk(filters=f_filter, layer_num='1')

    self.blk2 = ResNeXtBlk(filters=f_filter*2, layer_num='2')

    self.blk3 = ResNeXtBlk(filters=f_filter*4, layer_num='3')

    self.pool = tf.keras.layers.GlobalMaxPool2D()

    self.linear = tf.keras.layers.Dense(num_classes, kernel_initializer='glorot_uniform', use_bias=False)
  
  def call(self, x, y):
        
        x = self.first_layer(x)

        x = self.blk1(x)
        
        x = self.blk2(x)
        
        x = self.blk3(x)

        x = self.pool(x)
        
        x = self.linear(x)
        
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x, labels=y)
    
        loss = tf.reduce_sum(ce)
    
        correct = tf.reduce_sum(tf.cast(tf.math.equal(tf.argmax(x, axis = 1), y), tf.float32))
      
        return loss, correct, h

"""# DavidNet"""

class DavidNet(tf.keras.Model):
  
  def __init__(self, num_classes= 10, f_filter=64, weight=0.125):
    
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
    
    self.pool = tf.keras.layers.GlobalMaxPool2D()
    
    self.linear = tf.keras.layers.Dense(num_classes, kernel_initializer='glorot_uniform', use_bias=False)
    
    self.weight = weight

  def call(self, x, y):
    
    h = self.pool(self.blk3(self.blk2(self.blk1(self.init_conv_bn(x)))))
    
    h = self.linear(h) * self.weight
    
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=h, labels=y)
    
    loss = tf.reduce_sum(ce)
    
    correct = tf.reduce_sum(tf.cast(tf.math.equal(tf.argmax(h, axis = 1), y), tf.float16))
    
    return loss, correct, h


