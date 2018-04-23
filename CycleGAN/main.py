"""Cycle Gans Implementation"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from functools import partial
from glob import glob

import data
from utils import image_utils as im
from model import models
from utils import utils

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'horse2zebra', 'which dataset to use')
flags.DEFINE_integer('load_size', 286, 'scale images to this size')
flags.DEFINE_integer('crop_size', 256, 'then crop to this size')
flags.DEFINE_integer('epoch', 200, 'number of epoch')
flags.DEFINE_integer('batch_size', 1, 'numer of images in a batch')
flags.DEFINE_float('lr', 0.0002, 'initial learning rate for adam')


def train(dataset, load_size, crop_size, epoch, batch_size, lr):
    """ <--- graph --->"""

    # models
    generator_G = partial(models.generator, scope='G')
    generator_F = partial(models.generator, scope='F')
    discriminator_G = partial(models.discriminator, scope='G')
    discriminator_F = partial(models.discriminator, scope='F')

    # operations
    X_real = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])
    Y_real = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])
    GoX_sample = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])
    FoY_sample = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])

    # Passing images through Generators
    GoX = generator_G(X_real)
    FoY = generator_F(Y_real)

    # Completing Cycle
    GoFoY = generator_G(FoY)
    FoGoX = generator_F(GoX)

    # Ground Truth Logits
    X_logit = discriminator_G(X_real)
    Y_logit = discriminator_F(Y_real)

    # Generated Image Logits
    FoY_logit = discriminator_G(FoY)
    GoX_logit = discriminator_F(GoX)

    # Cycled Image Logits
    FoY_sample_logit = discriminator_G(FoY_sample)
    GoX_sample_logit = discriminator_F(GoX_sample)

    # Generator Loss
    g_loss_GoX = tf.losses.mean_squared_error(GoX_logit, tf.ones_like(GoX_logit))
    g_loss_FoY = tf.losses.mean_squared_error(FoY_logit, tf.ones_like(FoY_logit))

    # Cycle Loss
    cyc_loss_X = tf.losses.absolute_difference(X_real, FoGoX)
    cyc_loss_Y = tf.losses.absolute_difference(Y_real, GoFoY)

    # Generator Net Loss
    g_loss = g_loss_GoX + g_loss_FoY + (cyc_loss_X + cyc_loss_Y) * 10.0

    d_loss_X_real = tf.losses.mean_squared_error(X_logit, tf.ones_like(X_logit))
    d_loss_FoY_sample = tf.losses.mean_squared_error(FoY_sample_logit, tf.zeros_like(FoY_sample_logit))
    d_loss_X = d_loss_X_real + d_loss_FoY_sample

    d_loss_Y_real = tf.losses.mean_squared_error(Y_logit, tf.ones_like(Y_logit))
    d_loss_GoX_sample = tf.losses.mean_squared_error(GoX_sample_logit, tf.zeros_like(GoX_sample_logit))
    d_loss_Y = d_loss_Y_real + d_loss_GoX_sample

    # summaries
    g_summary = utils.summary({g_loss_GoX: 'g_loss_GoX',
                               g_loss_FoY: 'g_loss_FoY',
                               cyc_loss_X: 'cyc_loss_X',
                               cyc_loss_Y: 'cyc_loss_Y'})
    d_summary_X = utils.summary({d_loss_X: 'd_loss_X'})
    d_summary_Y = utils.summary({d_loss_Y: 'd_loss_Y'})

    # optim
    t_var = tf.trainable_variables()
    d_X_var = [var for var in t_var if 'G_discriminator' in var.name]
    d_Y_var = [var for var in t_var if 'F_discriminator' in var.name]
    g_var = [var for var in t_var if 'G_generator' in var.name or 'F_generator' in var.name]

    d_X_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(d_loss_X, var_list=d_X_var)
    d_Y_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(d_loss_Y, var_list=d_Y_var)
    g_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(g_loss, var_list=g_var)

    """ <-- TRAIN [init] -->"""

    # Session Configuration [ADD FALSE IF USING CPU]
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # counter
    it_cnt, update_cnt = utils.counter()

    ''' data '''
    X_img_paths = glob('./datasets/' + dataset + '/trainA/*.jpg')
    Y_img_paths = glob('./datasets/' + dataset + '/trainB/*.jpg')
    X_data_pool = data.ImageData(sess, X_img_paths, batch_size, load_size=load_size, crop_size=crop_size)
    Y_data_pool = data.ImageData(sess, Y_img_paths, batch_size, load_size=load_size, crop_size=crop_size)

    X_test_img_paths = glob('./datasets/' + dataset + '/testA/*.jpg')
    Y_test_img_paths = glob('./datasets/' + dataset + '/testB/*.jpg')
    X_test_pool = data.Imagebatch_sizeData(sess, X_test_img_paths, batch_size, load_size=load_size, crop_size=crop_size)
    Y_test_pool = data.ImageData(sess, Y_test_img_paths, batch_size, load_size=load_size, crop_size=crop_size)

    GoX_pool = utils.ItemPool()
    FoY_pool = utils.ItemPool()

    ''' summary '''
    summary_writer = tf.summary.FileWriter('./outputs/summaries/' + dataset, sess.graph)

    ''' saver '''
    saver = tf.train.Saver(max_to_keep=5)

    ''' restore '''
    ckpt_dir = './outputs/checkpoints/' + dataset
    utils.mkdir(ckpt_dir)
    try:
        utils.load_checkpoint(ckpt_dir, sess)
    except:
        sess.run(tf.global_variables_initializer())

    '''train'''
    try:
        batch_epoch = min(len(X_data_pool), len(Y_data_pool)) // batch_size
        max_it = epoch * batch_epoch
        for it in range(sess.run(it_cnt), max_it):
            sess.run(update_cnt)
            epoch = it // batch_epoch
            it_epoch = it % batch_epoch + 1

            # prepare data
            X_real_ipt = X_data_pool.batch()
            Y_real_ipt = Y_data_pool.batch()
            GoX_opt, FoY_opt = sess.run([GoX, FoY], feed_dict={X_real: X_real_ipt, Y_real: Y_real_ipt})
            GoX_sample_ipt = np.array(GoX_pool(list(GoX_opt)))
            FoY_sample_ipt = np.array(FoY_pool(list(FoY_opt)))

            # train G
            g_summary_opt, _ = sess.run([g_summary, g_train_op], feed_dict={X_real: X_real_ipt, Y_real: Y_real_ipt})
            summary_writer.add_summary(g_summary_opt, it)

            # train D_y
            d_summary_Y_opt, _ = sess.run([d_summary_Y, d_Y_train_op], feed_dict={Y_real: Y_real_ipt, GoX_sample: GoX_sample_ipt})
            summary_writer.add_summary(d_summary_Y_opt, it)

            # train D_x
            d_summary_X_opt, _ = sess.run([d_summary_X, d_X_train_op], feed_dict={X_real: X_real_ipt, FoY_sample: FoY_sample_ipt})
            summary_writer.add_summary(d_summary_X_opt, it)

            # Verbose
            if it % 1 == 0:
                print("Epoch: (%3d) (%5d/%5d)" % (epoch, it_epoch, batch_epoch))

            # save
            if (it + 1) % 1000 == 0:
                save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_epoch, batch_epoch))
                print('Model saved in file: % s' % save_path)

            # sample
            if (it + 1) % 100 == 0:
                X_real_ipt = X_test_pool.batch()
                Y_real_ipt = Y_test_pool.batch()
                [GoX_opt, FoGoX_opt, FoY_opt, GoFoY_opt] = sess.run([GoX, FoGoX, FoY, GoFoY], feed_dict={X_real: X_real_ipt, Y_real: Y_real_ipt})
                sample_opt = np.concatenate((X_real_ipt, GoX_opt, FoGoX_opt, Y_real_ipt, FoX_opt, GoFoY_opt), axis=0)

                save_dir = './outputs/sample_images_while_training/' + dataset
                utils.mkdir(save_dir)
                im.imwrite(im.immerge(sample_opt, 2, 3), '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, it_epoch, batch_epoch))
    except:
        save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_epoch, batch_epoch))
        print('Model saved in file: % s' % save_path)
        sess.close()


def test(dataset, crop_size):
    """ run """
    with tf.Session() as sess:
        X_real = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])
        Y_real = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])

        GoX = models.generator(X_real, 'G')
        FoY = models.generator(Y_real, 'F')
        GoFoY = models.generator(FoY, 'G')
        FoGoX = models.generator(GoX, 'F')

        # retore
        try:
            ckpt_path = utils.load_checkpoint('./outputs/checkpoints/' + dataset, sess)
        except:
            raise Exception('No checkpoint!')

        # test
        X_list = glob('./datasets/' + dataset + '/testA/*.jpg')
        Y_list = glob('./datasets/' + dataset + '/testB/*.jpg')

        X_save_dir = './outputs/test_predictions/' + dataset + '/testA'
        Y_save_dir = './outputs/test_predictions/' + dataset + '/testB'
        utils.mkdir([X_save_dir, Y_save_dir])

        for i in range(len(X_list)):
            X_real_ipt = im.imresize(im.imread(X_list[i]), [crop_size, crop_size])
            X_real_ipt.shape = 1, crop_size, crop_size, 3
            GoX_opt, FoGoX_opt = sess.run([GoX, FoGoX], feed_dict={X_real: X_real_ipt})
            X_img_opt = np.concatenate((X_real_ipt, GoX_opt, FoGoX_opt), axis=0)

            img_name = os.path.basename(X_list[i])
            im.imwrite(im.immerge(X_img_opt, 1, 3), X_save_dir + '/' + img_name)
            print('Save %s' % (X_save_dir + '/' + img_name))

        for i in range(len(Y_list)):
            Y_real_ipt = im.imresize(im.imread(Y_list[i]), [crop_size, crop_size])
            Y_real_ipt.shape = 1, crop_size, crop_size, 3
            FoY_opt, GoFoY_opt = sess.run([FoY, GoFoY], feed_dict={Y_real: Y_real_ipt})
            Y_img_opt = np.concatenate((Y_real_ipt, FoY_opt, GoFoY_opt), axis=0)

            img_name = os.path.basename(Y_list[i])
            im.imwrite(im.immerge(Y_img_opt, 1, 3), Y_save_dir + '/' + img_name)
            print('Save %s' % (Y_save_dir + '/' + img_name))


def main(_):
    """High level pipeline.
    This scripts performs the training for CycleGANs.
    """

    # Start training
    train(FLAGS.dataset, FLAGS.load_size, FLAGS.crop_size, FLAGS.epoch, FLAGS.batch_size, FLAGS.lr)

    # Start testing
    test(FLAGS.dataset, FLAGS.crop_size)


if __name__ == '__main__':
    tf.app.run()
