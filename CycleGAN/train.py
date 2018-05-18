import tensorflow as tf
import Generator as gen
import simple_discriminator as dis
import os
from PIL import Image
import numpy as np
import glob
import io_tools as io
import scipy

# The next function is taken from https://github.com/LynnHo/CycleGAN-Tensorflow-PyTorch/blob/master/image_utils.py
# This function makes sure that the range of the images generated is between 0 and 255.


def _to_range(images, min_value=0.0, max_value=1.0, dtype=None):
    # transform images from [-1.0, 1.0] to [min_value, max_value] of dtype
    assert \
        np.min(images) >= -1.0 - 1e-5 and np.max(images) <= 1.0 + 1e-5 \
        and (images.dtype == np.float32 or images.dtype == np.float64), \
        'The input images should be float64(32) and in the range of [-1.0, 1.0]!'
    if dtype is None:
        dtype = images.dtype
    return ((images + 1.) / 2. * (max_value - min_value) + min_value).astype(dtype)


def training(dataset, epochs, image_shape, batch_size, G_cyc_loss_lambda=10.0, F_cyc_loss_lambda=10.0, learning_rate=0.0002):

    if image_shape == 256:
        no_of_residual_blocks = 9
    elif image_shape == 128:
        no_of_residual_blocks = 6

    # Creating placeholder for images
    X = tf.placeholder(tf.float32, [None, image_shape, image_shape, 3])
    Y = tf.placeholder(tf.float32, [None, image_shape, image_shape, 3])
    GofX = tf.placeholder(tf.float32, [None, image_shape, image_shape, 3])
    FofY = tf.placeholder(tf.float32, [None, image_shape, image_shape, 3])
    #GofFofY = tf.placeholder(tf.float32, [None, image_shape, image_shape, 3])
    #FofGofX = tf.placeholder(tf.float32, [None, image_shape, image_shape, 3])

    """ We will have 2 generators: G and F
        G : X -> Y
        F : Y -> X

        and 2 Discriminators: DX and DY

        DX aims to distinguish between images from {x} & translated images {F(y)}
        DY aims to distinguish between images from {y} & translated images {G(x)}
    """

    # Creating the generators and discriminator networks
    GofX = gen.generator(X, no_of_residual_blocks, scope='G', output_channels=64)
    FofY = gen.generator(Y, no_of_residual_blocks, scope='F', output_channels=64)
    GofFofY = gen.generator(FofY, no_of_residual_blocks, scope='G', output_channels=64)
    FofGofX = gen.generator(GofX, no_of_residual_blocks, scope='F', output_channels=64)

    D_Xlogits = dis.build_gen_discriminator(X, scope='DX')
    D_FofYlogits = dis.build_gen_discriminator(FofY, scope='DX')
    D_Ylogits = dis.build_gen_discriminator(Y, scope='DY')
    D_GofXlogits = dis.build_gen_discriminator(GofX, scope='DY')

    # Setting up losses for generators and discriminators
    """ adv_losses are adversary losses
        cyc_losses are cyclic losses
        real_losses are losses from real images
        fake_losses are from generated images
    """
    # https://arxiv.org/pdf/1611.04076.pdf this paper states that using cross entropy as loss
    # causes the gradient to vanish. To avoid this problem, least square losses are used as suggested by the paper.

    # Adversary and Cycle Losses for G
    G_adv_loss = tf.reduce_mean(tf.squared_difference(D_GofXlogits, tf.ones_like(D_GofXlogits)))
    G_cyc_loss = tf.reduce_mean(tf.abs(GofFofY - Y)) * G_cyc_loss_lambda        # Put lambda for G cyclic loss here
    G_tot_loss = G_adv_loss + G_cyc_loss

    # Adversary and Cycle Losses for F
    F_adv_loss = tf.reduce_mean(tf.squared_difference(D_FofYlogits, tf.ones_like(D_FofYlogits)))
    F_cyc_loss = tf.reduce_mean(tf.abs(FofGofX - X)) * F_cyc_loss_lambda        # Put lambda for F cyclic loss here
    F_tot_loss = F_adv_loss + F_cyc_loss

    # Total Losses for G and F
    GF_tot_loss = G_tot_loss + F_tot_loss

    # Losses for DX
    DX_real_loss = tf.reduce_mean(tf.squared_difference(D_Xlogits, tf.ones_like(D_Xlogits)))
    DX_fake_loss = tf.reduce_mean(tf.squared_difference(D_FofYlogits, tf.zeros_like(D_FofYlogits)))
    DX_tot_loss = (DX_real_loss + DX_fake_loss) / 2

    # Losses for DY
    DY_real_loss = tf.reduce_mean(tf.squared_difference(D_Ylogits, tf.ones_like(D_Ylogits)))
    DY_fake_loss = tf.reduce_mean(tf.squared_difference(D_GofXlogits, tf.zeros_like(D_GofXlogits)))
    DY_tot_loss = (DY_real_loss + DY_fake_loss) / 2

    # Optimization
    # Getting all the variables that belong to the different networks
    # I.e. The weights and biases in G, F, DX and DY
    network_variables = tf.trainable_variables()  # This gets all the variables that will be initialized
    GF_variables = [variables for variables in network_variables if 'G' in variables.name or 'F' in variables.name]
    DX_variables = [variables for variables in network_variables if 'DX' in variables.name]
    DY_variables = [variables for variables in network_variables if 'DY' in variables.name]

    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)  # Put the learning rate here
    GF_train_step = optimizer.minimize(GF_tot_loss, var_list=GF_variables)
    DX_train_step = optimizer.minimize(DX_tot_loss, var_list=DX_variables)
    DY_train_step = optimizer.minimize(DY_tot_loss, var_list=DY_variables)

    # Summary for Tensor Board
    GF_summary = tf.summary.scalar("GF_tot_loss", GF_tot_loss)
    DX_summary = tf.summary.scalar("DX_tot_loss", DX_tot_loss)
    DY_summary = tf.summary.scalar("DY_tot_loss", DY_tot_loss)

    # For saving the model, the max_to_keep parameter saves just 5 models. I did this so that we don't run out of memory.
    saver = tf.train.Saver(max_to_keep=1)

    # Session on GPU
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Obtaining dataset
    # Training data
    """ Need to define getdata"""
    # dataset = 'horse2zebra'
    Xpath = glob.glob('./Datasets/' + dataset + '/trainA/*.jpg')
    Ypath = glob.glob('./Datasets/' + dataset + '/trainB/*.jpg')
    X_data = io.getdata(sess, Xpath, batch_size)     # Need to define getdata
    Y_data = io.getdata(sess, Ypath, batch_size)

    # Test data
    X_test_path = glob.glob('./Datasets/' + dataset + '/testA/*.jpg')
    Y_test_path = glob.glob('./Datasets/' + dataset + '/testB/*.jpg')
    X_test_data = io.getdata(sess, X_test_path, batch_size)     # Need to define getdata
    Y_test_data = io.getdata(sess, Y_test_path, batch_size)     # Need to define getdata

    # Creating a file to write the summaries for tensorboard
    train_summary_writer = tf.summary.FileWriter('./Summary/Train/' + dataset, sess.graph)

    # Initialization if starting from scratch, else restore the variables
    try:

        saver.restore(sess, tf.train.latest_checkpoint("./Checkpoints/" + dataset))
        print('Checkpoints Restored!')
    except:
        init = tf.global_variables_initializer()
        sess.run(init)
    no_of_batches = min(len(Xpath), len(Ypath)) // batch_size
    # Training
    no_of_iterations = 0
    for i in range(1, epochs + 1):
        for j in range(1, no_of_batches + 1):
            no_of_iterations += 1

            X_batch = io.batch(sess, X_data)  # Define batch
            Y_batch = io.batch(sess, Y_data)

            # Creating fake images for the discriminators
            GofXforDis, FofYforDis = sess.run([GofX, FofY], feed_dict={X: X_batch, Y: Y_batch})

            DX_output, DX_vis_summary = sess.run([DX_train_step, DX_summary], feed_dict={X: X_batch, FofY: FofYforDis})

            DY_output, DY_vis_summary = sess.run([DY_train_step, DY_summary], feed_dict={Y: Y_batch, GofX: GofXforDis})

            GF_output, GF_vis_summ = sess.run([GF_train_step, GF_summary], feed_dict={X: X_batch, Y: Y_batch})

            train_summary_writer.add_summary(DX_vis_summary, no_of_iterations)
            train_summary_writer.add_summary(DY_vis_summary, no_of_iterations)
            train_summary_writer.add_summary(GF_vis_summ, no_of_iterations)

            # Creating Checkpoint
            if no_of_iterations % 800 == 0:
                save_path = saver.save(sess, './Checkpoints/' + dataset + '/Epoch_(%d)_(%dof%d).ckpt' % (i, j, no_of_batches))
                print('Model saved in file: % s' % save_path)

            # To see what some of the test images look like after certain number of iterations
            if no_of_iterations % 400 == 0:
                X_test_batch = io.batch(sess, X_test_data)  # Define batch
                Y_test_batch = io.batch(sess, Y_test_data)

                [GofX_sample, FofY_sample, GofFofY_sample, FofGofX_sample] = sess.run([GofX, FofY, GofFofY, FofGofX], feed_dict={X: X_test_batch, Y: Y_test_batch})

                # Saving sample test images
                for l in range(batch_size):

                    new_im_X = np.zeros((image_shape, image_shape * 3, 3))
                    new_im_X[:, :image_shape, :] = np.asarray(X_test_batch[l])
                    new_im_X[:, image_shape:image_shape * 2, :] = np.asarray(GofX_sample[l])
                    new_im_X[:, image_shape * 2:image_shape * 3, :] = np.asarray(FofGofX_sample[l])

                    new_im_Y = np.zeros((image_shape, image_shape * 3, 3))
                    new_im_Y[:, :image_shape, :] = np.asarray(Y_test_batch[l])
                    new_im_Y[:, image_shape:image_shape * 2, :] = np.asarray(FofY_sample[l])
                    new_im_Y[:, image_shape * 2:image_shape * 3, :] = np.asarray(GofFofY_sample[l])

                    scipy.misc.imsave('./Output/Train/' + dataset + '/X' + str(l) + '_Epoch_(%d)_(%dof%d).png' % (i, j, no_of_batches), _to_range(new_im_X, 0, 255, np.uint8))
                    scipy.misc.imsave('./Output/Train/' + dataset + '/Y' + str(l) + '_Epoch_(%d)_(%dof%d).png' % (i, j, no_of_batches), _to_range(new_im_Y, 0, 255, np.uint8))

            print("Epoch: (%3d) Batch Number: (%5d/%5d)" % (i, j, no_of_batches))

    save_path = saver.save(sess, './Checkpoints/' + dataset + '/Epoch_(%d)_(%dof%d).ckpt' % (i, j, no_of_batches))
    print('Model saved in file: % s' % save_path)
    sess.close()

    return
