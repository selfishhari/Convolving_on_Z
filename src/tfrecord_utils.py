import tensorflow as tf
import functools
import data_pipeline
import os
from importlib import reload
reload(data_pipeline)


HEIGHT = 32
WIDTH = 32
DEPTH = 3

CIFAR_FILENAME = 'cifar-10-python.tar.gz'
CIFAR_DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/' + CIFAR_FILENAME
CIFAR_LOCAL_FOLDER = 'cifar-10-batches-py'

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecord(input_files, output_file, mode, preproc_fn):
  """Converts a file to TFRecords."""
  print('Generating %s' % output_file)
  with tf.python_io.TFRecordWriter(output_file) as record_writer:
    for input_file in input_files:
      data_dict = data_pipeline.read_pickle_from_file(input_file)
      data = data_dict[b'data']
      labels = data_dict[b'labels']
      #data = preproc_fn(data, mode)
      
      num_entries_in_batch = len(labels)
      for i in range(num_entries_in_batch):
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image': _bytes_feature(data[i].tobytes()),
                'label': _int64_feature(labels[i])
            }))
        record_writer.write(example.SerializeToString())


class TfDataSetFromRecords(object):
  def __init__(self, path, subset="train"):
    
    self.path = path
    
    self.subset = subset
    
  def get_filenames(self):
    return [self.path]

  def parser(self, serialized_example):
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape([DEPTH * HEIGHT * WIDTH])

    # Reshape from [depth * height * width] to [depth, height, width].
    image = tf.cast(
        tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
        tf.float32)
    label = tf.cast(features['label'], tf.int64)

    # Custom preprocessing.
    #image = self.preprocess(image)

    return image, label

  # Use make batch and edit the code accordingly if we have multiple gpus(then we shall need to edit the train code accordingly)
  def make_batch(self, batch_size):
    """Read the images and labels from 'filenames'."""
    filenames = self.get_filenames()
    # Repeat infinitely.
    dataset = tf.data.TFRecordDataset(filenames).repeat()

    # Parse records.
    dataset = dataset.map(
        self.parser, num_parallel_calls=batch_size)

    # Potentially shuffle records.
    if self.subset == 'train':
      min_queue_examples = int(
          TfDataSetFromRecords.num_examples_per_epoch(self.subset) * 0.4)
      # Ensure that the capacity is sufficiently large to provide good random
      # shuffling.
      dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)

    # Batch it up.
    dataset = dataset.batch(batch_size).prefetch(1)
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()

    return image_batch, label_batch

  def make_dataset(self, batch_size):
    """Read the images and labels from 'filenames'."""
    filenames = self.get_filenames()
    # Repeat infinitely.
    dataset = tf.data.TFRecordDataset(filenames)#.repeat()

    # Parse records. // num_parallel_calls = <number of available cpu threads> or batch_size
    dataset = dataset.map(
        self.parser, num_parallel_calls=batch_size)

    # Potentially shuffle records.
    if self.subset == 'train':
      min_queue_examples = int(
          TfDataSetFromRecords.num_examples_per_epoch(self.subset) * 0.4)
      # Ensure that the capacity is sufficiently large to provide good random
      # shuffling.
      dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)

    # Batch it up.
    #dataset = dataset.batch(batch_size).prefetch(1)

    return dataset
  
  def preprocess(self, image):
    """Preprocess a single image in [height, width, depth] layout."""
    if self.subset == 'train':
      # Pad 4 pixels on each dimension of feature map, done in mini-batch
      image = tf.image.resize_with_crop_or_pad(image, 40, 40)
      image = tf.image.random_crop(image, [HEIGHT, WIDTH, DEPTH])
      image = tf.image.random_flip_left_right(image)
    return image

  @staticmethod
  def num_examples_per_epoch(subset='train'):
    if subset == 'train':
      return 50000
    elif subset == 'eval':
      return 10000
    else:
      raise ValueError('Invalid data subset "%s"' % subset)
