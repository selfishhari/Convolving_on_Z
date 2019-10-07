from auto_augment import AutoAugment
import numpy as np
import time, math
import tensorflow as tf


### Horizontal flip
def horizontal_flip(x: tf.Tensor):
    x = tf.image.random_flip_left_right(x)
    return x

	
### Cutout
def replace_slice(input_: tf.Tensor, replacement, begin) -> tf.Tensor:
    inp_shape = tf.shape(input_)
    size = tf.shape(replacement)
    padding = tf.stack([begin, inp_shape - (begin + size)], axis=1)
    replacement_pad = tf.pad(replacement, padding)
#     replacement_pad = tf.cast(replacement_pad, dtype=tf.float16)
    mask = tf.pad(tf.ones_like(replacement, dtype=tf.bool), padding)
    return tf.where(mask, replacement_pad, input_)

def cutout(x: tf.Tensor, h: int=8, w: int=8, c: int = 3) -> tf.Tensor:
    """
    Cutout data augmentation. Randomly cuts a h by w whole in the image, and fill the whole with zeros.
    :param x: Input image.
    :param h: Height of the hole.
    :param w: Width of the hole
    :param c: Number of color channels in the image. Default: 3 (RGB).
    :return: Transformed image.
    """
    shape = tf.shape(x)
    x0 = tf.random.uniform([], 0, shape[0] + 1 - h, dtype=tf.int32)
    y0 = tf.random.uniform([], 0, shape[1] + 1 - w, dtype=tf.int32)
    
    x = replace_slice(x, tf.zeros([h, w, c], dtype = tf.float32), [x0, y0, 0])
#     x = replace_slice(x, tf.constant(1,shape = [h,w,c],dtype = tf.float16)*train_mean, [x0, y0, 0])
    return x

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