# -*- coding: utf-8 -*-
"""model_apis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19GnSVrVlCUEckIh0-fQspehAiC5jAVFV

# Conv(Normal, Dilated, Depthwise separable, Spatially separable) BatchNorm Relu
"""

import tensorflow as tf
import numpy as np

class BatchNorm(tf.keras.Model):
  
  def __init__(self, momentum=0.9, epsilon=1e-5):
    
    super().__init__()
    
    self.momentum = momentum
    
    self.epsilon = epsilon
    
    self.bn = tf.keras.layers.BatchNormalization(momentum=self.momentum, epsilon=self.epsilon)
  
  def call(self, inputs):
    
    return self.bn(inputs)

class spatially_separable_conv(tf.keras.Model):
  
  def __init__(self, filters=32, kernel_size=(3,3), strides=(1,1), padding="same" , dilation_rate=(1,1), kernel_regularizer = None, 
               kernel_initializer="glorot_uniform"):
    
    super().__init__()
    
    k_size1 = (kernel_size[0], 1)
    
    k_size2 = (1, kernel_size[1])
    
    self.step1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=k_size1, strides=strides, padding=padding, dilation_rate=dilation_rate,
                                       kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer, use_bias=False)
    
    self.step2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=k_size1, strides=strides, padding=padding, dilation_rate=dilation_rate,
                                       kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer, use_bias=False)
    
  def call(self, inputs):
      
    return self.step2(self.step1(inputs))

class ConvBnRl(tf.keras.Model):
  
  def __init__(self, filters=32, kernel_size=(3,3), strides=(1,1), padding="same" , dilation_rate=(1,1), kernel_regularizer = None, 
               kernel_initializer="glorot_uniform", conv_flag=True, bnflag=True,  relu=True, depthwise_separable=False, spatial_separable=False):
    
    super().__init__()
    
    self.relu = relu
    
    self.bn_flag = bnflag
    
    self.conv_flag = conv_flag
    
    if depthwise_separable:
      
      self.conv = tf.keras.layers.SeparableConv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate,
                                       kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer, use_bias=False)
    elif spatial_separable:
      
      self.conv = spatially_separable_conv(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate, 
                                           kernel_regularizer = kernel_regularizer, kernel_initializer=kernel_initializer)
      
    else:
      
      self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate,
                                       kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer, use_bias=False)
    
    
    
    self.bn = BatchNorm(momentum=0.9, epsilon=1e-5)

  def call(self, inputs):
    
    if self.conv_flag:

      if self.relu:

        if self.bn_flag:
          return tf.nn.relu(self.bn(self.conv(inputs)))

        else:
          return tf.nn.relu(self.conv(inputs))
      else:

        if self.bn_flag:
          return self.bn(self.conv(inputs))

        else:
          return self.conv(inputs)
        
    else:
      
      if self.relu:

        if self.bn_flag:
          
          return tf.nn.relu(self.bn(inputs))

        else:
          
          return tf.nn.relu(inputs)
      else:

        if self.bn_flag:
          
          return self.bn(inputs)

        else:
          
          """if conv, bn, rl flags are False, then just return conv"""
          
          return self.conv(inputs)


"""# ResNet"""

class ResBlk(tf.keras.Model):
  
  def __init__(self, 
               
               cbr = ConvBnRl(filters=32, kernel_size=(3,3), strides=(1,1), padding="same" , conv_flag=True, bnflag=True, relu=True),
               
               cbr_res1 = ConvBnRl(filters=32, kernel_size=(3,3), strides=(1,1), padding="same" , conv_flag=True, bnflag=True, relu=True),
               
               cbr_res2 = ConvBnRl(filters=32, kernel_size=(3,3), strides=(1,1), padding="same" , conv_flag=True, bnflag=True, relu=True),
               
               pool=tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='same'), 
               
               res=True ):
    
    super().__init__()
    
    self.conv_bn = cbr
    
    self.pool = pool
    
    self.res = res
    
    if self.res:
      
      self.res1 = cbr_res1
      
      self.res2 = cbr_res2

  def call(self, inputs):
    
    h = self.pool(self.conv_bn(inputs))
    
    if self.res:
      
      h = h + self.res2(self.res1(h))
      
    return h


"""# ResNext"""

class ResNeXtBlk(tf.keras.Model):
  
    def __init__(self, 
                 
                 layer_num, 
                 
                 filters=32, 
                 
                 pool=tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='same'), 
                 
                 kernel_regularizer=None,
                 
                 kernel_initializer="glorot_uniform",
                 
                 res_block=1, cardinality=8):
      
      super().__init__()
      
      self.filters = filters
      
      self.layer_num = layer_num
      
      self.res_block=res_block
      
      self.kernel_regularizer = kernel_regularizer
      
      self.kernel_initializer = kernel_initializer
      
      self.conv_bn_1x1 = ConvBnRl(filters=self.filters, kernel_size=(1,1), strides=(1,1), padding="same" , dilation_rate=(1,1), 
                                  kernel_regularizer = self.kernel_regularizer, kernel_initializer=self.kernel_initializer, conv_flag=True, bnflag=True,  relu=True)
      
      self.conv_bn_3x3 = ConvBnRl(filters=self.filters, kernel_size=(3,3), strides=(1,1), padding="same" , dilation_rate=(1,1), 
                                  kernel_regularizer = self.kernel_regularizer, kernel_initializer=self.kernel_initializer, conv_flag=True, bnflag=True,  relu=True)
      
      
      self.cardinality = cardinality
      
      self.pool = pool
      
    def concatenation(self, layers) :
      
            return tf.keras.layers.concatenate(layers, axis=3)       

    def first_layer(self, x, scope):
        with tf.name_scope(scope) :
            
            x = self.conv_bn_3x3(x)

            return x

    def transform_layer(self, x, depth, pool_flag, scope):
        with tf.name_scope(scope) :
          
            if pool_flag:
              
              x = self.pool(x)
          
            cbr_1x1 = ConvBnRl(filters=depth, kernel_size=(1,1), strides=(1,1), padding="same" , dilation_rate=(1,1), 
                                  kernel_regularizer = self.kernel_regularizer, kernel_initializer=self.kernel_initializer, conv_flag=True, bnflag=True,  relu=True)
            
            x = cbr_1x1(x)
            
            cbr_3x3 = ConvBnRl(filters=depth, kernel_size=(3,3), strides=(1,1), padding="same" , dilation_rate=(1,1), 
                                  kernel_regularizer = self.kernel_regularizer, kernel_initializer=self.kernel_initializer, conv_flag=True, bnflag=True,  relu=True)
            
            x = cbr_3x3(x)

            return x

    def transition_layer(self, x, filters, scope):
        with tf.name_scope(scope):
          
            cb_1x1 = ConvBnRl(filters=filters, kernel_size=(1,1), strides=(1,1), padding="same" , dilation_rate=(1,1), 
                                  kernel_regularizer = self.kernel_regularizer, kernel_initializer=self.kernel_initializer, conv_flag=True, bnflag=True,  relu=True)
            
            x = cb_1x1(x)

            return x

    def split_layer(self, input_x, filters, pool_flag, layer_name):
      
        with tf.name_scope(layer_name) :
          
            layers_split = list()
            
            depth = filters//self.cardinality
            
            for i in range(self.cardinality) :
              
                splits = self.transform_layer(input_x, depth, pool_flag=pool_flag, scope=layer_name + '_splitN_' + str(i))
                
                layers_split.append(splits)

            return self.concatenation(layers_split)

    def residual_layer(self, input_x):
        # split + transform(bottleneck) + transition + merge

        for i in range(self.res_block):
            # input_dim = input_x.get_shape().as_list()[-1]
            input_dim = int(np.shape(input_x)[-1])

            if input_dim * 2 == self.filters:
              
                flag = True
                
                stride = 2
                
                channel = input_dim // 2
                
            else:
              
                flag = False
                
                stride = 1
                
            x = self.split_layer(input_x, filters=self.filters, pool_flag=flag, layer_name='split_layer_'+self.layer_num+'_'+str(i))
            
            x = self.transition_layer(x, filters=self.filters, scope='trans_layer_'+self.layer_num+'_'+str(i))

            if flag is True :
                
                pad_input_x = self.pool(input_x)
                
                pad_input_x = tf.pad(pad_input_x, [[0, 0], [0, 0], [0, 0], [channel, channel]]) # [?, height, width, channel]
                
                
                
            else :
                pad_input_x = input_x
                
            input_x = tf.nn.relu(x + pad_input_x)
        
        return input_x
      
    def call(self, inputs_x):
        
        return self.residual_layer(inputs_x)
      

class DenseNextBlk(tf.keras.Model):
    
        def __init__(self, 
                     
                     filters=1, 
                     
                     pool=tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='same'), 
                     
                     kernel_regularizer=None,
                     
                     kernel_initializer="glorot_uniform",
                     
                     z_dilation_rate=32,
                     
                     num_channels_last_layer = 128):
          
          super().__init__()
          
          self.filters = filters
          
          self.kernel_regularizer = kernel_regularizer
          
          self.kernel_initializer = kernel_initializer
          
          self.conv_bn_3x3 = []
          
          self.z_dilation_rate = z_dilation_rate
          
          num_output_channels = num_channels_last_layer//z_dilation_rate
      
          for i in range(num_output_channels):
              self.conv_bn_3x3.append(ConvBnRl(filters=self.filters, kernel_size=(3,3), strides=(1,1), padding="same" , dilation_rate=(1,1), 
                                  kernel_regularizer = self.kernel_regularizer, kernel_initializer=self.kernel_initializer, conv_flag=True, bnflag=True,  relu=True))
      
        def space_to_depth(self, x, block_size = 2):
          
          if block_size!= 1:
              return tf.space_to_depth(x, block_size=block_size)
          else:
              return x
      
        def create_sampling_indices(self, z_dilation_rate, layer_shapes_dict):
          """
          Based on z_dilation_rate, sample channels.
          Create channel index mapping for each layer in i/p to o/p
          """
          all_layers_sampling_dict = {}
          
          for (k,v) in layer_shapes_dict.items():
              
              channels_to_sample = []
              
              i = 0
              
              while(i < z_dilation_rate):
                  
                  channels_to_sample.append(list(range(i, v[3], z_dilation_rate)))
                  
                  i += 1
                  
              all_layers_sampling_dict[k] = channels_to_sample
          
          return all_layers_sampling_dict
      
        def convolve_on_samples(self, layers_dict, std_vals=[4, 2, 1]):
          
          last_layer = max(layers_dict.keys())
          
          last_layer_shape = tf.shape(layers_dict[last_layer])
          
          num_output_channels = int(last_layer_shape[3])//self.z_dilation_rate
          
          output_channels = []
          
          for i in range(num_output_channels):
              
              roots = [layers_dict[layer_num][:, :, :, i::self.z_dilation_rate] for layer_num in layers_dict.keys()]
              
              roots = [self.space_to_depth(roots[x], block_size=std_vals[x])  for x in range(len(roots))]
              
              root = tf.concat(roots, axis=3)
              
              img_channel = self.conv_bn_3x3[i](root)#, [5, 1, 8, 8])
              
              #print(tf.shape(img_channel))
              
              output_channels.append(img_channel)
              
          return tf.reshape(tf.concat(output_channels, axis=0), [-1, last_layer_shape[1], last_layer_shape[2], num_output_channels])
      
        
        def call(self, layers_dict):
          
          layer_shapes_dict = {}
          
          for (k,v) in layers_dict.items():
              
              layer_shapes_dict[k] = tf.shape(v)
          
          #sampling_dict = self.create_sampling_indices(self.z_dilation_rate, layer_shapes_dict)
          
          #print(sampling_dict[0][0])
          
          output = self.convolve_on_samples(layers_dict)
          
          return output
          
          
        
        
#------------------------------------------------#
          
      
class ZeeConvBlk(tf.keras.Model):
    
    def __init__(self,
                 pool=tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='same'), 
                     
                 kernel_regularizer=None,
                     
                 kernel_initializer="glorot_uniform",
                 
                 dilation_rate = (2,2),
                 
                 gap_mode="x_axis",
                 
                 layers_filters = {0:16, 1:32, 2:64},
                 
                 dimensions_dict = {"dimensions_to_sample":(16, 16)}
                 
                 ):
        
        super().__init__()
        
        self.layers_filters = layers_filters

        self.pool = pool
        
        self.kernel_initializer = kernel_initializer
        
        self.kernel_regularizer = kernel_regularizer
        
        self.dilation_rate = dilation_rate
        
        self.num_layers = len(layers_filters.keys())
        
        self.convolution_blocks = {}
        
        self.dimensions_dict = dimensions_dict
        
        self.gap_mode = gap_mode
        
        
        
        for layer in range(self.num_layers):
            
            self.convolution_blocks[layer] = []
            
            curr_filters = layers_filters[layer]
            
            for x in list(range(self.dimensions_dict["dimensions_to_sample"][0])):
                
                self.convolution_blocks[layer].append(
                        ConvBnRl(filters=curr_filters, kernel_size=(3,3), strides=(1,1), padding="valid" , dilation_rate=self.dilation_rate, 
                                      kernel_regularizer = self.kernel_regularizer, kernel_initializer=self.kernel_initializer, conv_flag=True, bnflag=True,  relu=True))

        return
    
    def _create_upsampling_indices(self, layer_shape):
        
        indices = []
        
        for img_num in range(layer_shape[0]):
            
            img_indces = []
            
            for x_indx in range(layer_shape[1]):
                
                curr_dim_inds = []
                
                for y_indx in range(layer_shape[1]):
                    
                    curr_dim_inds.append([img_num, x_indx, y_indx])
                    
                    curr_dim_inds.append([img_num, x_indx, y_indx])
                    
                img_indces.append(curr_dim_inds)
            
            indices.append(img_indces)
                
        
        return indices
    
    def _upsample_by_replication(self, x, up_factor):
        
        
        gather_indices = self._create_upsampling_indices(x.shape)
        
        return tf.gather_nd(x, gather_indices)
        
        
        
    def set_down_up_sampling(self, layers_dict, downsampling_dict):
        
        """
        Takes a dictionary of each layer output from the primary model architecture
        Accordingly downsamples initial half layers and upsamples last half layers        
        """
        sampled_layers_dict = {}
        
        for (layer_num, x) in layers_dict.items():
            
            #Do maxpooling n number of times, 
            #where n is downsampling value for that layer from downsampling_dict
            
            down_up_strategy = downsampling_dict[layer_num]
            
            for i in range(abs(down_up_strategy)):
                
                if down_up_strategy > 0:
                    x = self.pool(x)
                else:
                    x = self._upsample_by_replication(x, abs(down_up_strategy))
            
            sampled_layers_dict[layer_num] = x
            
        return sampled_layers_dict
    
    def transpose_and_convolve(self, layers_dict, downsampling_dict):
        """
        Now that images x, y and channels are appropriately down/upsampled to the required extent.
        
        Convolution happens with a transposed channel outputs-
            With original image's x becoming channels for us and the channles across various layers
            becoming x for us.
            y remains same.
            
            So a channel output of shape 32, 32, 64 is transformed to 64, 32, 32.
            
            When scaled across all layers, say we have 3 layers each giving output as (32, 32, 32)
            (16, 16, 64) (8, 8, 128)
            
            This is down and upsampled to (16, 16, 32) (16, 16, 64) (16, 16, 128)^
            
            Now the input for convolution would be (32 + 64 +128, 16, 16). Or (224, 16, 16)
            
            First layer output of transpose conv would be (222, 14, 16)-without padding and 
            using a 3*3 conv on all channels with 16 filters
            
            
            ^ Upsampling is done in a memory efficient way. So the input is actually (16, 8, 128),
            however it is handled while convolving
        
        """
        
        output_layers_dict = {}
        
        num_x_pixels = self.dimensions_dict["dimensions_to_sample"][0]#This will become channels for us
        
        for x_idx in range(num_x_pixels):
            
            print("x xhannels:", x_idx)
            
            #For each channel stich the layer outputs together
            
            stitched_image = None
            
            for layer_num in layers_dict.keys():
                
                print("layer_num:", layer_num)
                
                if downsampling_dict[layer_num] < 0:
                    #if upsamplinh then pick the previous x value because of memory management
                    
                    channel = layers_dict[layer_num][:, (x_idx//((abs(downsampling_dict[layer_num]))*2)), :, :]
                        
                    
                else:
                    channel = layers_dict[layer_num][:, x_idx, :, :]
                    
                #print(tf.shape(channel))
                    
                
                channel = tf.reshape(channel, (channel.shape[0], 1, channel.shape[1], channel.shape[2]))
                channel = tf.transpose(channel, [0, 3, 2, 1])#batch remains same, channels become x, y remains same, x becomes channels
                
                #print(tf.shape(channel))
                    
                if stitched_image != None:
                    
                    stitched_image = tf.concat([stitched_image, channel], axis = 1)
                    
                    print("stitched")
                    
                    #print(tf.shape(stitched_image), "stitched image shape")
                    
                else:
                    
                    stitched_image = channel
                    
                    print("stitched first time")
                    
            #Once images are stiched, perform convolution
            print("convolving")
            output_layers_dict[x_idx] = self.convolution_blocks[0][x_idx](stitched_image)
        
        return output_layers_dict
    
    def apply_convolve_layers(self, channels_dict, num_layers):
        
        output_layers_dict = channels_dict
        
        for layer in range(num_layers):  

            channels_dict =  output_layers_dict.copy()       
            
            for (group, imgs) in channels_dict.items():
                
                output_layers_dict[group] = self.convolution_blocks[(layer+1)][group](imgs)
                
        return output_layers_dict
            
        
    
    def call(self, layers_dict, downsampling_dict = {0:1, 1:0, 2:-1}):
        
        print("enter-z conv")
        
        layers_dict_updown = self.set_down_up_sampling(layers_dict, downsampling_dict)
        
        print("updown done")
        
        x = self.transpose_and_convolve(layers_dict_updown, downsampling_dict)
        
        print("transpose done")
        
        x = self.apply_convolve_layers(x, (self.num_layers-1))
        
        print("apply conv on layers")
        
        output = tf.concat([v for (k, v) in x.items()], axis=3)
        
        print("concat done")
        
        if self.gap_mode == "x_axis":
            
            return output
        if self.gap_mode == "channel_axis":
            
            print("transpose")
            
            return tf.transpose(output, [0, 3, 2, 1])
        
            
        