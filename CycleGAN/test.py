import tensorflow as tf
import Generator as gen
from io_tools import *
import glob
import numpy as np
import scipy.misc

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


def test(dataset_str='horse2zebra', img_width=256, img_height=256):
    """Test and save output images.

    Args:
        dataset_str: Name of the dataset
        X_path, Y_path: Path to data in class X or Y
    """
    image_shape = img_width

    if image_shape == 256:
        no_of_residual_blocks = 9
    elif image_shape == 128:
        no_of_residual_blocks = 6

    # Session on GPU
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # X and Y are for real images.
        X = tf.placeholder(tf.float32, shape=[None, img_width, img_height, 3])
        Y = tf.placeholder(tf.float32, shape=[None, img_width, img_height, 3])

        # Build graph for generator to produce images from real data.
        GofX = gen.generator(X, no_of_residual_blocks, scope='G', output_channels=64)
        FofY = gen.generator(Y, no_of_residual_blocks, scope='F', output_channels=64)
        # Convert transformed images back to original one (cyclic).
        Fof_GofX = gen.generator(GofX, no_of_residual_blocks, scope='F', output_channels=64)
        Gof_FofY = gen.generator(FofY, no_of_residual_blocks, scope='G', output_channels=64)

        saver = tf.train.Saver(None)

        # Restore checkpoint.
        # --------------- Need to implement utils!!!!! ----------------
        try:
            saver.restore(sess, tf.train.latest_checkpoint("./Checkpoints/" + dataset_str))
            print('Checkpoints Restored !')
        except:
            raise Exception('No checkpoint available!')

        # Load data and preprocess (resize and crop).
        X_path_ls = glob.glob('./Datasets/' + dataset_str + '/testA/*.jpg')
        Y_path_ls = glob.glob('./Datasets/' + dataset_str + '/testB/*.jpg')

        batch_size_X = len(X_path_ls)
        batch_size_Y = len(Y_path_ls)

        X_data = getdata(sess, X_path_ls, batch_size_X)
        Y_data = getdata(sess, Y_path_ls, batch_size_Y)

        # Get data into [batch_size, img_width, img_height, channels]
        X_batch = batch(sess, X_data)
        Y_batch = batch(sess, Y_data)

        print('test data :' + dataset_str + '- uploaded!')
        # Feed into test procedure to test and save results.
        X_save_dir = './Output/Test/' + dataset_str + '/testA'
        Y_save_dir = './Output/Test/' + dataset_str + '/testB'
        # utils.mkdir([X_save_dir, Y_save_dir])

        _test_procedure(X_batch, sess, GofX, Fof_GofX, X, X_save_dir, image_shape)
        _test_procedure(Y_batch, sess, FofY, Gof_FofY, Y, Y_save_dir, image_shape)


def _test_procedure(batch, sess, gen_real, gen_cyc, real_placeholder, save_dir, image_shape):
    """Procedure to perform test on a batch of real images and save outputs.
    Args:
        gen_real: Generator that maps real data to fake image.
        gen_cyc: Generator that maps fake image back to original image.
        real_placeholder: Placeholder for real image.
        save_dir: Directory to save output image.
    """
    print('Test Images sent to generator..')
    gen_real_out, gen_cyc_out = sess.run([gen_real, gen_cyc],
                                         feed_dict={real_placeholder: batch})
    print('Images obtatined back generator..')
    for i in range(batch.shape[0]):
        # A single real image in batch.
        real_img = batch[i]
        # Generate fake and cyclic images.

        # Concatenate 3 images into one.
        # out_img = np.concatenate((real_img, gen_real_out, gen_cyc_out), axis=0)
        # # Save result.
        # # --------------- Need the utils file!!! ---------------
        # # Temporarily use i as image name. Should change this.
        # im.imwrite(im.immerge(out_img, 1, 3), save_dir + '/' + str(i))

        # gen_real_out_image = Image.fromarray(gen_real_out, "RGB")
        # gen_cyc_out_image = Image.fromarray(gen_cyc_out, "RGB")

        # new_im = Image.new('RGB', (image_shape * 3, image_shape))
        # new_im.paste(real_img, (0, 0))
        # new_im.paste(gen_real_out_image, (image_shape, 0))
        # new_im.paste(gen_cyc_out_image, (image_shape * 2, 0))

        # new_im.save(save_dir + '(%d).jpg' % (i))

        new_im = np.zeros((image_shape, image_shape * 3, 3))
        new_im[:, :image_shape, :] = np.asarray(real_img)
        new_im[:, image_shape:image_shape * 2, :] = np.asarray(gen_real_out[i])
        new_im[:, image_shape * 2:image_shape * 3, :] = np.asarray(gen_cyc_out[i])

        scipy.misc.imsave(save_dir + 'Image(%d).png' % (i), _to_range(new_im, 0, 255, np.uint8))
        print("Save image.")
