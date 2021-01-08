import numpy as np
import tensorflow as tf


# dims_of_channels = [3, 5, 10, 15, 20, 25]
def initialize_filter_parameters(channels_of_layers):
    num_of_layers = len(channels_of_layers)
    filter_parameters = {}
    for i in range(1, num_of_layers):
        filter_parameters["w" + str(i)] = tf.get_variable("w" + str(i),
                                                          [3, 3, channels_of_layers[i - 1],
                                                           channels_of_layers[i]],
                                                          initializer=tf.contrib.layers.xavier_initializer())
    return filter_parameters


def create_placeholders(h0, w0, c0, class_size):
    x = tf.placeholder(dtype=tf.float32, shape=[None, h0, w0, c0], name="x")
    y = tf.placeholder(dtype=tf.float32, shape=[None, class_size], name="y")

    return x, y


def one_hot_encoding(y, class_size):
    # 단위행렬
    e = np.eye(class_size)
    # flatten
    y = y.reshape(-1)
    # one hot encoding
    y = e[y].T

    return y


def forward(x, filter_parameters, num_of_layers):
    p = x
    for i in range(1, num_of_layers):
        w = filter_parameters["w" + str(i)]
        z = tf.nn.conv2d(p, w, strides=[1, 1, 1, 1], padding="SAME")
        a = tf.nn.leaky_relu(z)
        p = tf.nn.max_pool(a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    p = tf.contrib.layers.flatten(p)
    a = tf.contrib.layers.fully_connected(p, 100, activation_fn=tf.nn.leaky_relu)
    a = tf.contrib.layers.fully_connected(a, 6, activation_fn=tf.nn.leaky_relu)

    return a


def compute_cost(a7, y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=a7, labels=y))

    return cost
