# Augmentation

!git clone https://github.com/santuhazra1/Imp_Lib.git /tmp/Imp_Lib
!mv /tmp/Imp_Lib/Auto_Augment/auto_augment.py auto_augment.py 
!rm -r /tmp/Imp_Lib
from auto_augment import AutoAugment
import numpy as np
import time, math
import tensorflow as tf

tf.enable_eager_execution()


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

def cutout(img: tf.Tensor, height, width, channel = 3):
    shape = tf.shape(img)
    x = tf.random.uniform([], 0, shape[0] + 1 - height, dtype=tf.int32)
    img = core.replace_slice(img, tf.zeros([height, width, channel]),)
    return img
`
### Rotate 90
def random_rotate_90(x: tf.Tensor):
    x = tf.image.rot90(x, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    return x

### Auto Augment

from PIL import Image
def AutoAug(img: : tf.Tensor):
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