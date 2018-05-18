# Import Library
import tensorflow as tf


# ### If you want to understand the conv2d function and its inputs, go to these pages
# #### https://www.tensorflow.org/api_docs/python/tf/contrib/layers/conv2d
# #### https://stackoverflow.com/questions/34642595/tensorflow-strides-argument
#
# ### If you want to understand the residual blocks used in the generator, go to this page
# #### http://torch.ch/blog/2016/02/04/resnets.html

# #### Functions for Batch Normalization, Residual Bloacks and Generator


# USELESS
# exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
# bnepsilon = 1e-5
# mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
# update_moving_averages = exp_moving_avg.apply([mean, variance])
# m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
# v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
# Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
# return Ybn, update_moving_averages
"""
        # OLD
        # # DEFINING GENERATOR VARIABLES
        # # ````````````````````````````
        # # Define Variables for downsampling
        # D0 = tf.get_variable('D0', dtype=tf.float32, initializer = tf.truncated_normal([7,7,3,output_channels] ,stddev=0.1))
        # DB0 = tf.get_variable('DB0', dtype=tf.float32, initializer = tf.zeros([output_channels]))
        # # First layer with img_width x img_height x 64 dimension. Filter size 7x7
        # D1 = tf.get_variable('D1', dtype=tf.float32, initializer = tf.truncated_normal([3,3,output_channels,output_channels*2] ,stddev=0.1))
        # DB1 = tf.get_variable('DB1', dtype=tf.float32, initializer = tf.zeros([output_channels * 2]))
        # # Second layer with img_width/2 x img_height/2 x 128 dimension. Filter size 3x3
        # D2 = tf.get_variable('D2', dtype=tf.float32, initializer = tf.truncated_normal([3,3,output_channels*2,output_channels*4], stddev=0.1))
        # #Kernel size of 3x3 from "output_channels1" number of input channels. Number of output channels K depends on how many you want in next layer.
        # DB2 = tf.get_variable('DB2', dtype=tf.float32, initializer = tf.zeros([output_channels * 4]))
        # # Second layer with img_width/4 x img_height/4 x 256 dimension. Filter size 3x3

        # # Define Variables for Residual Blocks
        # res_dict = {}
        # for i in range(1,NO_OF_RESIDUAL_BLOCKS+1):
        #     res_dict[i] = {'R1'+str(i) :tf.get_variable('R1'+str(i), dtype=tf.float32, initializer = tf.truncated_normal([3,3,output_channels * 4,output_channels * 4] ,stddev=0.1)),
        #                    'RB1'+str(i) :tf.get_variable('RB1'+str(i), dtype=tf.float32, initializer = tf.Variable(output_channels * 4)/10),
        #                    'R2'+str(i) :tf.get_variable('R2'+str(i), dtype=tf.float32, initializer = tf.Variable(tf.truncated_normal([3,3,K,K] ,stddev=0.1)),
        #                    'RB2'+str(i) :tf.get_variable('RB2'+str(i), dtype=tf.float32, initializer = tf.Variable(tf.ones([J])/10)
        #                       }
        #      # Need to put the right input and output layer numbers


        # # Define Variables for upsampling
        # U1 = tf.get_variable('U1', dtype=tf.float32, initializer = tf.truncated_normal([3,3,number_of_input_channels,L], stddev=0.1))
        # UB1 = tf.get_variable('UB1', dtype=tf.float32, initializer = tf.ones([L])/10)
        # U2 = tf.get_variable('U2', dtype=tf.float32, initializer = tf.truncated_normal([9,9,L,3] ,stddev=0.1))
        # #Here L is the number of output channels from the previous layer and is the number of input channels in this layer
        # UB2 = tf.Variable('UB2', dtype=tf.float32, initializer = tf.ones([3])/10)

        # # DEFINING THE LAYERS
        # # ```````````````````
        # # For Downsampling
        # stride = 2  # if input A x A then output A/2 x A/2
        # # Firt Layer
        # YD1 = tf.nn.conv2d(input, D1, strides=[1, stride, stride, 1], padding='SAME')
        # YD1bn, update_emaYD1 = batchnorm(YD1, tst, iter, DB1, convolutional=True)
        # YD1r = tf.nn.relu(YD1bn)
        # # Second Layer
        # YD2 = tf.nn.conv2d(YD1r, D2, strides=[1, stride, stride, 1], padding='SAME')
        # YD2bn, update_emaYD2 = batchnorm(YD2, tst, iter, DB2, convolutional=True)
        # YD2r = tf.nn.relu(YD2bn)

        # # For Residual Blocks
        # YR = YD2r
        # for i in range(1,NO_OF_RESIDUAL_BLOCKS+1):
        #     YR = residual_block(YR, res_dict[i]['R1'+str(i)], res_dict[i]['R2'+str(i)], res_dict[i]['RB1'+str(i)], res_dict[i]['RB2'+str(i)])

        # # For Upsampling
        # stride = 1/2
        # # Second Last Layer
        # YU1 = tf.nn.conv2d_transpose(YR, U1, strides=[1, stride, stride, 1], padding='SAME')
        # YU1bn, update_emaYU1 = batchnorm(YU1, tst, iter, UB1, convolutional=True)
        # YU1r = tf.nn.relu(YU1bn)
        # # Last Layer
        # YU2 = tf.nn.conv2d_transpose(YU1r, U2, strides=[1, stride, stride, 1], padding='SAME')
        # Y_out = tf.nn.tanh(YU2)

        # return Y_out
"""


def generator(input_imgs, no_of_residual_blocks, scope, output_channels=64):

    # Function for Batch Normalization
    def batchnorm(Ylogits):
        bn = tf.contrib.layers.batch_norm(Ylogits, scale=True, decay=0.9, epsilon=1e-5, updates_collections=None)
        return bn

    # Function for Convolution Layer
    def convolution_layer(input_images, filter_size, stride, o_c=64, padding="VALID", scope_name="convolution"):
        # o_c = Number of output channels/filters
        with tf.variable_scope(scope_name):
            conv = tf.contrib.layers.conv2d(input_images, o_c, filter_size, stride, padding=padding, activation_fn=None,
                                            weights_initializer=tf.truncated_normal_initializer(stddev=0.02))
            return conv

    # Function for deconvolution layer
    def deconvolution_layer(input_images, o_c, filter_size, stride, padding="VALID", scope_name="deconvolution"):
        with tf.variable_scope(scope_name):
            deconv = tf.contrib.layers.conv2d_transpose(input_images, o_c, filter_size, stride, activation_fn=None,
                                                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02))
            return deconv

    # Function for Residual Block
    def residual_block(Y, scope_name="residual_block"):
        with tf.variable_scope(scope_name):
            Y_in = tf.pad(Y, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
            Y_res1 = tf.nn.relu(batchnorm(convolution_layer(Y_in, filter_size=3, stride=1, o_c=output_channels * 4, scope_name="C1")))
            Y_res1 = tf.pad(Y_res1, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
            Y_res2 = batchnorm(convolution_layer(Y_res1, filter_size=3, stride=1, padding="VALID", o_c=output_channels * 4, scope_name="C2"))

            return Y_res2 + Y

    # #### Generator Variables
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

        # Need to pad the images first to get same sized image after first convolution
        input_imgs = tf.pad(input_imgs, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")

        YD0 = tf.nn.relu(batchnorm(convolution_layer(input_imgs, filter_size=7, stride=1, o_c=output_channels, scope_name="D1")))
        YD1 = tf.nn.relu(batchnorm(convolution_layer(YD0, filter_size=3, stride=2, o_c=output_channels * 2, padding="SAME", scope_name="D2")))
        YD2 = tf.nn.relu(batchnorm(convolution_layer(YD1, filter_size=3, stride=2, o_c=output_channels * 4, padding="SAME", scope_name="D3")))

        # For Residual Blocks
        for i in range(1, no_of_residual_blocks + 1):
            Y_res = residual_block(YD2, scope_name="R" + str(i))

        # For Upsampling
        YU1 = tf.nn.relu(batchnorm(deconvolution_layer(Y_res, output_channels * 2, filter_size=3, stride=2, padding="SAME", scope_name="U1")))
        YU2 = tf.nn.relu(batchnorm(deconvolution_layer(YU1, output_channels, filter_size=3, stride=2, padding="SAME", scope_name="U2")))
        Y_out = tf.nn.tanh(convolution_layer(YU2, filter_size=7, stride=1, o_c=3, padding="SAME", scope_name="U3"))

        return Y_out
