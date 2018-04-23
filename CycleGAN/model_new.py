from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import tensorflow as tf
import tensorflow.contrib.slim as slim


class CycleGAN(object):
    """Cycle GAN"""

    def __init__(self, dataset, load_size, crop_size, epoch, batch_size, lr):
        """
        Args:

        """
        # self.dataset = dataset
        # self.load_size = load_size
        # self.crop_size = crop_size
        # self.epoch = epoch
        # self.batch_size = batch_size
        # self.lr = lr

        self.conv = partial(slim.conv2d, activation_fn=None)
        self.deconv = partial(slim.conv2d_transpose, activation_fn=None)
        self.relu = tf.nn.relu
        self.lrelu = partial(tf.nn.leaky_relu, alpha=0.2)
        self.batch_norm = partial(slim.batch_norm, scale=True, decay=0.9, epsilon=1e-5, updates_collections=None)

        self.a_real = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])
        self.b_real = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])
        self.a2b_sample = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])
        self.b2a_sample = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])


    # def graph(self):
    #     self.generator_a2b = partial(self.generator, scope='a2b')
    #     self.generator_b2a = partial(self.generator, scope='b2a')
    #     self.discriminator_a = partial(self.discriminator, scope='a')
    #     self.discriminator_b = partial(self.discriminator, scope='b')

    # def train(self):
    #     pass

    def discriminator(self, img, scope, dim=64, train=True):
        """
        Args:

        """
        bn = partial(self.batch_norm, is_training=train)
        conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)

        with tf.variable_scope(scope + '_discriminator', reuse=tf.AUTO_REUSE):
            net = lrelu(conv(img, dim, 4, 2))
            net = conv_bn_lrelu(net, dim * 2, 4, 2)
            net = conv_bn_lrelu(net, dim * 4, 4, 2)
            net = conv_bn_lrelu(net, dim * 8, 4, 1)
            net = conv(net, 1, 4, 1)

            return net

    def generator(self, img, scope, dim=64, train=True):
        """
        Args:

        """
        bn = partial(batch_norm, is_training=train)
        conv_bn_relu = partial(conv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
        deconv_bn_relu = partial(deconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)

        def _residule_block(x, dim):

            y = conv_bn_relu(x, dim, 3, 1)
            y = bn(conv(y, dim, 3, 1))
            return y + x

        with tf.variable_scope(scope + '_generator', reuse=tf.AUTO_REUSE):
            net = conv_bn_relu(img, dim, 7, 1)
            net = conv_bn_relu(net, dim * 2, 3, 2)
            net = conv_bn_relu(net, dim * 4, 3, 2)

            for i in range(9):
                net = _residule_block(net, dim * 4)

            net = deconv_bn_relu(net, dim * 2, 3, 2)
            net = deconv_bn_relu(net, dim, 3, 2)
            net = conv(net, 3, 7, 1)
            net = tf.nn.tanh(net)

            return net
