from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def _parse_image(path):

    load_size = 286
    crop_size = 256
    img = tf.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.resize_images(img, [load_size, load_size])
    img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))
    img = tf.random_crop(img, [crop_size, crop_size, 3])
    img = img * 2 - 1
    return img


def getdata(sess, paths, batch_size, shuffle=True):
    '''
    Arg : session       - tensor flow session
          path          - global path to dataset
          batch_size    - size of the batch

    Return : batch graph
    '''

    prefetch_batch = 2
    num_threads = 16
    buffer_size = 4096
    repeat = -1

    _img_num = len(paths)

    dataset = tf.data.Dataset.from_tensor_slices(paths)

    # The map method takes a map_func argument that describes how each item in the Dataset should be transformed.
    dataset = dataset.map(_parse_image, num_parallel_calls=num_threads)

    # Shuffle
    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    # this transformation combines consecutive elements of this dataset into batches.
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

    # Repeats this dataset count times | repeated indefinitely if -1
    dataset = dataset.repeat(repeat).prefetch(prefetch_batch)

    return dataset.make_one_shot_iterator().get_next()


def batch(sess, dataset):
    '''
    Arg : session - tensor flow session
          dataset - dataset that you have generated with getdata function

    Return : batch
    '''
    return sess.run(dataset)
