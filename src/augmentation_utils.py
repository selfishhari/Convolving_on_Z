from auto_augment import AutoAugment
import numpy as np
import time, math
import tensorflow as tf


### Horizontal flip
def horizontal_flip(x: tf.Tensor):
    x = tf.image.random_flip_left_right(x)
    return x

	
### Cutout
def replace_slice(input_: tf.Tensor, replacement, begin):
    inp_shape = tf.shape(input_)
    size = tf.shape(replacement)
    padding = tf.stack([begin, inp_shape - (begin + size)], axis=1)
    replacement_pad = tf.pad(replacement, padding)
    mask = tf.pad(tf.ones_like(replacement, dtype=tf.bool), padding)
    return tf.where(mask, replacement_pad, input_)

def cutout(img,train_mean, prob=5,size=8,min_size=5,use_fixed_size=False):
  return tf.cond(tf.random.uniform([], 0, 100) > prob, lambda: img , lambda: get_cutout(img, train_mean, prob,size,min_size,use_fixed_size))



def get_cutout(img,train_mean,prob=50,size=16,min_size=5,use_fixed_size=False):
  
  
  height = tf.shape(img)[0]
  width = tf.shape(img)[1]
  channel = tf.shape(img)[2]
  
  #subtract the mean of train dataset from the image , we will add this back later 
  mean = tf.constant(train_mean, dtype=tf.float32) # (3)
  mean = tf.reshape(mean, [1, 1, 3])
  img_m = img - mean

  #get cutout size and offsets 
  if (use_fixed_size==True):
    s=size
  else:  
    s=tf.random.uniform([], min_size, size, tf.int32) # use a cutout size between 5 and size 

  x1 = tf.random.uniform([], 0, height+1 - s, tf.int32) # get the x offset from top left
  y1 = tf.random.uniform([], 0, width+1 - s, tf.int32) # get the y offset from top left 

  # create the cutout slice and the mask 
  img1 = tf.ones_like(img)  
  
  cut_slice = tf.slice(
  img1,
  [x1, y1, 0],
  [s, s, 3]
     )
  #create mask similar in shape to input image with cutout area having ones and rest of the area padded with zeros 
  mask = tf.image.pad_to_bounding_box(
    [cut_slice],
    x1,
    y1,
    height,
    width
  )
  
  #invert the zeros and ones 
  mask = tf.ones_like(mask ) - mask
  
  #inv_mask = tf.where( tf.equal( -1.0, mask ), 1.0 * tf.ones_like( mask ), mask ) # not needed
  
  #apply cutout on the image , get back a shape of [1,32,32,3] instead of [32,32,3]
  tmp_img = tf.multiply(img_m,mask)

  #add back the mean that we subtracted 
  cut_img = tmp_img[0] + mean
  
  return cut_img

### Rotate 90
def random_rotate_90(x: tf.Tensor):
    x = tf.image.rot90(x, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    return x

def normalize(data:tf.Tensor,mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010]):
    normalize = lambda x: ((x - mean) / std).astype('float16')
    data = data/255
    norm_data = normalize(data)
    return norm_data

### Auto Augment

from PIL import Image
def AutoAug(img: tf.Tensor):
  img = img.numpy()
  autoaug = AutoAugment()
  Auto_aug_im = np.zeros_like(img)
  for i in range(img.shape[0]):
    im = img[i]
    im = Image.fromarray(im)
    im = autoaug(im)
    Auto_aug_im[i] = im
  Auto_aug_im <- tf.convert_to_tensor(Auto_aug_im,dtype = tf.float16)  
  return Auto_aug_im	