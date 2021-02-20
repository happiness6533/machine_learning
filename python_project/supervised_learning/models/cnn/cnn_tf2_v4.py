import numpy as np
import tensorflow as tf


# channels_of_layers = [3, 5, 10, 15, 20, 25]
def initialize_filter_parameters(filter_size, channels_of_layers, channels_of_layers_dense):
    num_of_layers = len(channels_of_layers)
    filter_parameters = {}
    dense_parameters = {}
    for i in range(1, num_of_layers):
        filter_parameters["w" + str(i)] = tf.Variable(
            name="w" + str(i),
            shape=(filter_size, filter_size, channels_of_layers[i - 1], channels_of_layers[i]))
        filter_parameters["b" + str(i)] = tf.Variable(0)

    for i in range(1, len(channels_of_layers_dense)):
        dense_parameters["w" + str(i)] = tf.Variable(
            name="w" + str(i),
            shape=(channels_of_layers_dense[i], channels_of_layers_dense[i - 1]))
        filter_parameters["b" + str(i)] = tf.Variable(0)

    return filter_parameters, dense_parameters


def single_convolution_forward(prev_a_slice, single_filter_w, single_filter_b):
    z_element = prev_a_slice * single_filter_w
    z_element = tf.math.reduce_sum(z_element) + single_filter_b

    return z_element


def zero_pad(x, pad_size):
    # x : (m * height * width * channel)
    x_pad = tf.pad(x, tf.constant([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]]))

    return x_pad


def convolution_forward(prev_a, filter_w, filter_b, hparameters):
    # prev_a : (m * prev_a_height * prev_a_width * prev_a_channel)
    # filter_w : (filter_size * filter_size * prev_a_channel * z_channel)
    # filter_b : (1 * 1 * 1 * z_channel)
    # filter 개수 : z_channel
    # hparameters : 사전 : key = "pad_size", "filter_stride"
    (m, prev_a_height, prev_a_width, prev_a_channel) = prev_a.shape
    (filter_size, filter_size, prev_a_channel, z_channel) = filter_w.shape

    pad_size = hparameters["pad_size"]
    prev_a_pad = zero_pad(prev_a, pad_size)

    filter_stride = hparameters["filter_stride"]
    z_height = int((prev_a_height + 2 * pad_size - filter_size) / filter_stride) + 1
    z_width = int((prev_a_width + 2 * pad_size - filter_size) / filter_stride) + 1

    z = tf.zeros((m, z_height, z_width, z_channel))
    for i in range(m):
        for h in range(z_height):
            for w in range(z_width):
                for c in range(z_channel):
                    height_start = h * filter_stride
                    height_end = height_start + filter_size
                    width_start = w * filter_stride
                    width_end = width_start + filter_size

                    prev_a_slice = prev_a_pad[i, height_start:height_end, width_start:width_end, :]
                    z[i, h, w, c] = single_convolution_forward(prev_a_slice, filter_w[:, :, :, c], filter_b[:, :, :, c])

    return z


def pool_forward(a, hparameters, mode="max"):
    (m, a_height, a_width, a_channel) = a.shape

    pool_size = hparameters["pool_size"]
    pool_stride = hparameters["pool_stride"]

    pool_a_height = int((a_height - pool_size) / pool_stride) + 1
    pool_a_width = int((a_width - pool_size) / pool_stride) + 1
    pool_a_channel = a_channel

    pool_a = tf.zeros((m, pool_a_height, pool_a_width, pool_a_channel))
    for i in range(m):
        for h in range(pool_a_height):
            for w in range(pool_a_width):
                for c in range(pool_a_channel):
                    height_start = h * pool_stride
                    height_end = height_start + pool_size
                    width_start = w * pool_stride
                    width_end = width_start + pool_size

                    a_slice = a[i, height_start:height_end, width_start:width_end, c]

                    if mode == "max":
                        pool_a[i, h, w, c] = tf.math.reduce_max(a_slice)
                    elif mode == "average":
                        pool_a[i, h, w, c] = np.mean(a_slice)

    return pool_a


def single_forward(w, b, a, activation):
    z, linear_cache = linear(w, b, a)

    if activation == "relu":
        a = relu(z)
    elif activation == "leaky_relu":
        a = leaky_relu(z)
    elif activation == "sigmoid":
        a = sigmoid(z)
    elif activation == "softmax":
        a = softmax(z)

    return a


def forward(x, filter_parameters, dense_parameters, num_of_layers, hparameters):
    a = x
    for i in range(1, num_of_layers):
        z = convolution_forward(a, filter_parameters["w" + str(i)], filter_parameters["b" + str(i)], hparameters)
        a = relu(z)
        a = pool_forward(a, hparameters, mode="max")
    a = flatten(a)

    for i in range(1, len(dense_parameters) - 1):
        a = single_forward(dense_parameters["w" + str(i)], "b" + str(i), a, activation="relu")
    single_forward(dense_parameters["w" + str(len(dense_parameters) - 1)],
                   "b" + str(len(dense_parameters) - 1), a, activation="softmax")

    return a



def cnn_model(hparameters, x_train, y_train, x_test, y_test, learning_rate, num_of_iteration, mini_batch_size,
              filter_size, channels_of_layers, channels_of_layers_dense):
    costs = []
    num_of_layers = len(channels_of_layers)
    init_dim = x_train.shape[1]
    flatten_dim = init_dim / (2 ** len(channels_of_layers) - 1)
    channels_of_layers_dense.insert(flatten_dim, index=0)
    filter_parameters, dense_parameters = tf_cnn_utils.initialize_filter_parameters(filter_size, channels_of_layers,
                                                                                    channels_of_layers_dense)

    for i in range(num_of_iteration):
        mini_batch_cost = 0
        mini_batches = data_utils.generate_random_mini_batches_for_cnn(x_train, y_train, mini_batch_size)
        for mini_batch in mini_batches:
            with tf.GradientTape() as tape:
                (mini_batch_x, mini_batch_y) = mini_batch
                # 앞으로 ㄱㄱ
                a = tf_cnn_utils.forward(mini_batch_x, filter_parameters, dense_parameters, num_of_layers, hparameters)
                cost = tf_cnn_utils.compute_cost(a, mini_batch_y, cost_function='cross_entropy')

                # 미분계수 계산 후 업데이터
                for k in range(1, len(filter_parameters)):
                    filter_parameters["w" + str(k)].assign_sub(
                        learning_rate * tape.gradient(cost, filter_parameters["w" + str(k)]))
                    filter_parameters["b" + str(k)].assign_sub(
                        learning_rate * tape.gradient(cost, filter_parameters["b" + str(k)]))

                # 미분계수 계산 후 업데이터
                for h in range(1, len(dense_parameters)):
                    dense_parameters["w" + str(h)].assign_sub(
                        learning_rate * tape.gradient(cost, dense_parameters["w" + str(h)]))
                    dense_parameters["b" + str(h)].assign_sub(
                        learning_rate * tape.gradient(cost, dense_parameters["b" + str(h)]))

        if i % 100 == 0:
            print("Cost after iteration %d : %f" % (i, mini_batch_cost))
            costs.append(mini_batch_cost)
    train_accuracy = tf_cnn_utils.predict(filter_parameters, dense_parameters, x_train, y_train, hparameters,
                                          num_of_layers)
    test_accuracy = tf_cnn_utils.predict(filter_parameters, dense_parameters, x_test, y_test, hparameters,
                                         num_of_layers)
    print("train accuracy: {}%".format(train_accuracy))
    print("test accuracy: {}%".format(test_accuracy))

    plt.figure()
    plt.plot(np.squeeze(costs))
    plt.xlabel("dnn")
    plt.ylabel("cost")
    plt.title("dnn")
    plt.show()
