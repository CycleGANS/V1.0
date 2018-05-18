import tensorflow as tf


def _conv2d_layer(input_conv, num_filter=64, filter_h=4, filter_w=4, stride_h=1, stride_w=1, stddev=0.02,
                  padding="VALID", name="conv2d", do_norm=True, do_relu=True, relu_alpha=0):
    """Convolution layer for discriminator.
    Supports normalization for image instance and leaky ReLU.

    Note:
        relu_alpha: Slope when x < 0, used in max(x, alpha*x).
    """

    with tf.variable_scope(name):
        conv = tf.contrib.layers.conv2d(input_conv, num_filter, filter_h, stride_h, padding, activation_fn=None,
                                        weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                        biases_initializer=tf.constant_initializer(0.0))

        if do_norm:
            conv = _normalization(conv)

        if do_relu:
            if(relu_alpha == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = _leaky_relu(conv, relu_alpha, "leaky_relu")

        return conv


def _normalization(x):
    """Adapted from hardikbansal's code. Will change it later."""
    with tf.variable_scope("instance_norm"):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset', [x.get_shape()[-1]],
                                 initializer=tf.constant_initializer(0.0))
        out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset

        return out


def _leaky_relu(x, relu_alpha, name="leaky_relu"):
    with tf.variable_scope(name):
        return tf.maximum(x, relu_alpha * x)


def build_gen_discriminator(input_images, num_filters=64, scope="discriminator"):
    """Build model: A simplified discriminator.

    Args:
        input_images: [batch_size, img_width, img_height, img_channel]
            where img_channel refers to layers like R, G, B.
        num_filters: Number of output filters for the very first layer.
            For other layers, the multiplication factors of num_filters depends
            on the strides.
        scope: Change it according to the role of the discriminator.
    Returns:
        layer5: Decision of the input images.
    """

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        filter_size = 4

        layer1 = _conv2d_layer(input_images, num_filters, filter_size, filter_size, 2, 2, 0.02,
                               "SAME", "conv1", do_norm=False, do_relu=True, relu_alpha=0.2)
        layer2 = _conv2d_layer(layer1, num_filters * 2, filter_size, filter_size, 2, 2, 0.02,
                               "SAME", "conv2", do_norm=True, do_relu=True, relu_alpha=0.2)
        layer3 = _conv2d_layer(layer2, num_filters * 4, filter_size, filter_size, 2, 2, 0.02,
                               "SAME", "conv3", do_norm=True, do_relu=True, relu_alpha=0.2)
        layer4 = _conv2d_layer(layer3, num_filters * 8, filter_size, filter_size, 1, 1, 0.02,
                               "SAME", "conv4", do_norm=True, do_relu=True, relu_alpha=0.2)
        layer5 = _conv2d_layer(layer4, 1, filter_size, filter_size, 1, 1, 0.02,
                               "SAME", "conv5", do_norm=False, do_relu=False)

        return layer5
