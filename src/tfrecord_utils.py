import tensorflow as tf

# store data in tfrecord
def createDataRecord(out_filename, addrs, labels):
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(out_filename)
    for i in range(len(addrs)):
        # print how many images are saved every 10000 images
        if not i % 10000:
            print('Train data: {}/{}'.format(i, len(addrs)))
            sys.stdout.flush()
        # Load the image
        img = addrs[i]

        label = labels[i]

        if img is None:
            continue

        # Create a feature
        feature = {
            'image_raw': _bytes_feature(img.tostring()),
            'label': _int64_feature(label)
        }
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
        
    writer.close()
    sys.stdout.flush()
    
def tfexample_numpy_image_parser(tfexample: tf.train.Example, h: int, w: int, c: int = 3, dtype=tf.float32):
    feat_dict = {'image': tf.FixedLenFeature([h * w * c], dtype),
                 'label': tf.FixedLenFeature([], tf.int64)}
    feat = tf.parse_single_example(tfexample, features=feat_dict)
    x, y = feat['image'], feat['label']
#     x = tf.cast(x, tf.float32)
#     y = tf.cast(y, tf.int32)
    x = tf.reshape(x, [h, w, c])
    return x, y

def parser_train(tfexample):
  x, y = tfexample_numpy_image_parser(tfexample, img_size, img_size)
  x = random_pad_crop(x, 4)
  x = random_flip(x)
  x = cutout(x, 8, 8)
  return x, y

parser_test = lambda x: tfexample_numpy_image_parser(x, img_size, img_size)


def tfrecord_ds(file_pattern: str, parser, batch_size: int, training: bool = True, shuffle_sz: int = 50000) -> tf.data.Dataset:
    dataset = tf.data.TFRecordDataset(filenames=file_pattern, num_parallel_reads=40)
    dataset = dataset.apply(
        tf.contrib.data.shuffle_and_repeat(shuffle_sz, 1)
    )
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(parser, batch_size, num_parallel_calls=2)
    )
    #dataset = dataset.map(parser, num_parallel_calls=12)
    #dataset = dataset.batch(batch_size=1000)
    dataset = dataset.prefetch(1)
    return dataset


train_input_func = lambda params: tfrecord_ds('train.tfrecords', parser_train, batch_size=params, training=True)
eval_input_func = lambda params: tfrecord_ds('test.tfrecords', parser_test, batch_size=params, training=False)