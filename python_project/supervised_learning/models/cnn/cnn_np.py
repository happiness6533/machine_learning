import numpy as np
import matplotlib.pyplot as plt
import python_project.supervised_learning.data.data_utils as data_utils


def initialize_filter(filter_size, num_of_input_channel, num_of_output_channel):
    filter = np.random.randn(filter_size, filter_size, num_of_input_channel, num_of_output_channel) * 0.01

    return filter


def single_convolution_forward(prev_a_slice, single_filter_w, single_filter_b):
    z_element = prev_a_slice * single_filter_w
    z_element = np.sum(z_element) + single_filter_b

    return z_element


def zero_pad(x, pad_size):
    # x : (m * height * width * channel)
    x_pad = np.pad(x, ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'constant', constant_values=0)

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

    z = np.zeros((m, z_height, z_width, z_channel))
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

    cache = prev_a, filter_w, filter_b, hparameters

    return z, cache


def relu(z):
    a = np.maximum(0, z)
    activation_cache = z

    return a, activation_cache


def leaky_relu(z):
    a = np.maximum(0.01 * z, z)
    activation_cache = z

    return a, activation_cache


def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    activation_cache = z

    return a, activation_cache


def softmax(z):
    activation_cache = z
    z = z - np.max(z, axis=0, keepdims=True)
    a = np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)

    return a, activation_cache


def pool_forward(a, hparameters, mode="max"):
    (m, a_height, a_width, a_channel) = a.shape

    pool_size = hparameters["pool_size"]
    pool_stride = hparameters["pool_stride"]

    pool_a_height = int((a_height - pool_size) / pool_stride) + 1
    pool_a_width = int((a_width - pool_size) / pool_stride) + 1
    pool_a_channel = a_channel

    pool_a = np.zeros((m, pool_a_height, pool_a_width, pool_a_channel))
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
                        pool_a[i, h, w, c] = np.max(a_slice)
                    elif mode == "average":
                        pool_a[i, h, w, c] = np.mean(a_slice)

    cache = a, hparameters

    return pool_a, cache


def flatten(a):
    return np.ndarray.flatten(a)


def forward(x, filter_parameters, num_of_layers, hparameters):
    p = x
    for i in range(1, num_of_layers):
        z = convolution_forward(p, filter_parameters["w" + str(i)], filter_parameters["b" + str(i)], hparameters)
        a = leaky_relu(z)
        p = pool_forward(a, hparameters, mode="max")
    p = flatten(p)
    dims_of_layers = [p.shape[2], 16, 16, 6]
    params = dnn_utils.init_params(dims_of_layers=dims_of_layers)
    a = dnn_utils.forward(params, p,
                          activation="relu", last_activation="softmax",
                          num_of_layers=dims_of_layers)

    return a


# 코스트 계산
def compute_cost(a, y, cost_function, params, lam, num_of_layers):
    # optimize: (batch, stochastic, mini batch) gradient descent, momentum, adagrad, RMSProp, adam...
    # cost function: cross entropy, mse...
    # regularize: L2, dropout...
    cost = 0
    if cost_function == 'cross_entropy':
        cost = cross_entropy(a, y)
    elif cost_function == 'mean_square_error':
        cost = mean_square_error(a, y)

    regularize_term = 0
    for i in range(1, num_of_layers):
        w = params["w" + str(i)]
        regularize_term += np.sum(np.square(w))
    regularize_term = (lam / 2) * regularize_term

    cost = cost + regularize_term

    return cost


# 여기서부터 백
def max_pooling_slice_backward(da, a):
    max = np.max(a)
    mask = (a == max)
    da = mask * da

    return da


def average_pooling_slice_backward(d_pool_a_element, pool_size):
    average_d_pool_a_element = d_pool_a_element / (pool_size * pool_size)
    mask = np.ones((pool_size, pool_size))
    d_a_slice = mask * average_d_pool_a_element

    return d_a_slice


def pool_backward(d_pool_a, cache, mode="max"):
    (a, hparameters) = cache

    (m, pool_a_height, pool_a_width, pool_a_channel) = d_pool_a.shape
    (m, a_height, a_width, a_channel) = a.shape

    pool_size = hparameters["pool_size"]
    pool_stride = hparameters["pool_stride"]

    da = np.zeros(a.shape)
    for i in range(m):
        for h in range(pool_a_height):
            for w in range(pool_a_width):
                for c in range(pool_a_channel):
                    height_start = h * pool_stride
                    height_end = height_start + pool_size
                    width_start = w * pool_stride
                    width_end = width_start + pool_size

                    if mode == "max":
                        a_slice = a[i, height_start:height_end, width_start:width_end, c]
                        d_a_slice = max_pooling_slice_backward(d_pool_a[i, h, w, c], a_slice)
                        da[i, height_start:height_end, width_start:width_end, c] += d_a_slice
                    elif mode == "average":
                        d_a_slice = average_pooling_slice_backward(d_pool_a[i, h, w, c], pool_size)
                        da[i, height_start:height_end, width_start:width_end, c] += d_a_slice

    return da


def cross_entropy_gradient(a, y):
    m = y.shape[1]
    da = -(y / a) / m

    return da


def mean_square_error_gradient(a, y):
    m = y.shape[1]
    da = (a - y) / m

    return da


def leaky_relu_backward(da, activation_cache):
    z = activation_cache

    dz = np.array(da, copy=True)
    dz[z < 0] = 0.01

    return dz


def convolution_backward(dz, cache):
    (m, z_height, z_width, z_channel) = dz.shape
    (prev_a, filter_w, filter_b, hparameters) = cache
    pad_size = hparameters["pad_size"]
    prev_a_pad = zero_pad(prev_a, pad_size)
    (m, prev_a_height, prev_a_width, prev_a_channel) = prev_a.shape
    (filter_size, filter_size, prev_a_channel, z_channel) = filter_w.shape

    filter_stride = hparameters["filter_stride"]

    d_prev_a = np.zeros(prev_a.shape)
    d_prev_a_pad = np.zeros(prev_a_pad.shape)
    d_filter_w = np.zeros(filter_w.shape)
    d_filter_b = np.zeros(filter_b.shape)
    for i in range(m):
        for h in range(z_height):
            for w in range(z_width):
                for c in range(z_channel):
                    height_start = h * filter_stride
                    height_end = height_start + pad_size
                    width_start = w * filter_stride
                    width_end = width_start + pad_size

                    prev_a_slice = prev_a_pad[i, height_start:height_end, width_start:width_end, :]

                    # W = 가로 * 세로 * 현재 채널 * 필터의 개수
                    # W[:, :, :, c] = 가로 * 세로 * 현재 채널 * 현재 필터
                    d_filter_w[:, :, :, c] += dz[i, h, w, c] * prev_a_slice
                    d_filter_b[:, :, :, c] += dz[i, h, w, c]
                    d_prev_a_pad[i, height_start:height_end, width_start:width_end, :] += dz[i, h, w, c] * filter_w[:,
                                                                                                           :, :, c]

        d_prev_a[i, :, :, :] = d_prev_a_pad[i, pad_size:-pad_size, pad_size:-pad_size, :]

    return d_prev_a, d_filter_w, d_filter_b


def cross_entropy(a, y):
    # 베르누이 확률분포
    m = y.shape[1]
    cost = np.sum(-(y * np.log(a))) / m

    return cost


def mean_square_error(a, y):
    # 가우시안 확률분포
    m = y.shape[1]
    cost = np.sum(np.square(a - y)) / (2 * m)

    return cost


def backward(a, y, forward_cache, cost_function, activation, last_activation, num_of_layers):
    grads = {}

    if cost_function == 'cross_entropy':
        da = cross_entropy_gradient(a, y)
    elif cost_function == 'mean_square_error':
        da = mean_square_error_gradient(a, y)

    for i in reversed(range(1, num_of_layers)):
        if i == num_of_layers - 1:
            grads["dw" + str(i)], grads["db" + str(i)], grads["da" + str(i - 1)] = single_backward(da,
                                                                                                   forward_cache[i - 1],
                                                                                                   last_activation)
        else:
            grads["dw" + str(i)], grads["db" + str(i)], grads["da" + str(i - 1)] = single_backward(grads["da" + str(i)],
                                                                                                   forward_cache[i - 1],
                                                                                                   activation)

    return grads


def forward_and_backward():
    pass


def update_parameters():
    pass


def predict(params, x_test, y_test, activation, last_activation, num_of_layers):
    a, _ = forward(params, x_test, activation, last_activation, num_of_layers)
    prediction = np.zeros(a.shape)
    prediction[a >= 0.75] = 1
    accuracy = np.mean(np.all(prediction == y_test, axis=0, keepdims=True)) * 100

    return accuracy


def cnn_model(x_train, y_train, x_test, y_test,
              activation, last_activation,
              learning_rate, cost_function, lam,
              batch_size, num_of_iterations,
              dims_of_layers,
              filter_size, num_of_input_channel, num_of_output_channel):
    params = cnn_utils.initialize_filter(filter_size, num_of_input_channel, num_of_output_channel)
    num_of_layers = len(dims_of_layers)

    costs = []
    hparameter = {
        'ksize': [1, 2, 2, 1], 'strides': [1, 2, 2, 1], 'padding': 'same'
    }

    mini_batches = data_utils.generate_random_mini_batches(x_train, y_train, batch_size)
    # 1 epoch
    for index, mini_batch in enumerate(mini_batches):
        mini_batch_x, mini_batch_y = mini_batch
        for j in range(0, num_of_iterations):
            grads, cost = cnn_utils.forward_and_backward(params, mini_batch_x, mini_batch_y,
                                                         activation, last_activation,
                                                         cost_function, lam,
                                                         num_of_layers)
            params = cnn_utils.update_parameters(len(mini_batch_x), params, grads, learning_rate, lam, num_of_layers)
            if j % 100 == 0:
                print("%dth mini batch, cost after iteration %d: %f" % (index + 1, j + 1, cost))
                costs.append(cost)
    train_accuracy = cnn_utils.predict(params, x_train, y_train, activation, last_activation, num_of_layers)
    test_accuracy = cnn_utils.predict(params, x_test, y_test, activation, last_activation, num_of_layers)
    print("train accuracy: {}%".format(train_accuracy))
    print("test accuracy: {}%".format(test_accuracy))

    plt.figure()
    plt.plot(np.squeeze(costs))
    plt.xlabel("dnn")
    plt.ylabel("cost")
    plt.title("dnn")
    plt.show()

    return params


x_train, y_train, x_test, y_test, input_dim, output_dim = data_utils.load_sign_data_sets()
x_train, y_train = data_utils.flatten(x_train, y_train)
x_test, y_test = data_utils.flatten(x_test, y_test)
x_train = data_utils.centralize_x(x_train)
x_test = data_utils.centralize_x(x_test)
y_train = data_utils.one_hot_encoding_y(y_train, output_dim)
y_test = data_utils.one_hot_encoding_y(y_test, output_dim)

params = cnn_model(x_train, y_train, x_test, y_test,
                   activation="relu", last_activation="softmax",
                   learning_rate=0.001, cost_function='cross_entropy', lam=0.1,
                   batch_size=108, num_of_iterations=2000,
                   dims_of_layers=[input_dim, 16, 16, output_dim])
