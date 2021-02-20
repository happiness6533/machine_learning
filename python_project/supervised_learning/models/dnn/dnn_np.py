import numpy as np


def init_params(dims_of_layers):
    # w 초기화: 가우시안, Xavier initialization, He initialization...
    # b 초기화: 0
    num_of_layers = len(dims_of_layers)
    params = {}
    for i in range(1, num_of_layers):
        params["w" + str(i)] = np.random.randn(dims_of_layers[i - 1], dims_of_layers[i]) * 0.01
        params["b" + str(i)] = np.zeros((dims_of_layers[i], 1))

    return params


def linear(w, b, a):
    z = np.matmul(w.T, a) + b
    linear_cache = w, b, a

    return z, linear_cache


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


def single_forward(w, b, a, activation):
    z, linear_cache = linear(w, b, a)

    if activation == "relu":
        a, activation_cache = relu(z)
    elif activation == "leaky_relu":
        a, activation_cache = leaky_relu(z)
    elif activation == "sigmoid":
        a, activation_cache = sigmoid(z)
    elif activation == "softmax":
        a, activation_cache = softmax(z)

    linear_activation_cache = linear_cache, activation_cache

    return a, linear_activation_cache


def forward(params, x, activation, last_activation, num_of_layers):
    a = x
    forward_cache = []
    for i in range(1, num_of_layers):
        if i != num_of_layers - 1:
            a, linear_activation_cache = single_forward(params['w' + str(i)], params['b' + str(i)], a, activation)
        else:
            a, linear_activation_cache = single_forward(params['w' + str(i)], params['b' + str(i)], a, last_activation)
        forward_cache.append(linear_activation_cache)

    return a, forward_cache


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


def cross_entropy_gradient(a, y):
    m = y.shape[1]
    da = -(y / a) / m

    return da


def mean_square_error_gradient(a, y):
    m = y.shape[1]
    da = (a - y) / m

    return da


def relu_gradient(da, activation_cache):
    z = activation_cache

    dz = np.ones(z.shape)
    dz[z < 0] = 0
    dz = da * dz

    return dz


def leaky_relu_gradient(da, activation_cache):
    z = activation_cache

    dz = np.ones(da.shape)
    dz[z < 0] = 0.01
    dz = da * dz

    return dz


def sigmoid_gradient(da, activation_cache):
    z = activation_cache
    a = 1 / (1 + np.exp(-z))

    dz = da * a * (1 - a)

    return dz


def softmax_gradient(da, activation_cache):
    z = activation_cache
    z = z - np.max(z, axis=0, keepdims=True)
    a = np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)

    dz = np.zeros(da.shape)
    (r, m) = da.shape
    for k in range(m):
        middle_matrix = np.zeros((r, r))
        for i in range(r):
            for j in range(r):
                if i == j:
                    middle_matrix[i, j] = a[i, k] * (1 - a[i, k])
                else:
                    middle_matrix[i, j] = -(a[j, k] * a[i, k])
        dz[:, k] = np.matmul(middle_matrix, da[:, k])

    return dz


def linear_gradient(dz, linear_cache):
    w, b, a = linear_cache
    dw = np.matmul(a, dz.T)
    db = np.sum(dz, axis=1, keepdims=True)
    da = np.matmul(w, dz)

    return dw, db, da


def single_backward(da, linear_activation_cache, activation):
    linear_cache, activation_cache = linear_activation_cache

    if activation == "sigmoid":
        dz = sigmoid_gradient(da, activation_cache)
    elif activation == "relu":
        dz = relu_gradient(da, activation_cache)
    elif activation == "leaky_relu":
        dz = leaky_relu_gradient(da, activation_cache)
    elif activation == "softmax":
        dz = softmax_gradient(da, activation_cache)

    dw, db, prev_da = linear_gradient(dz, linear_cache)

    return dw, db, prev_da


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


def forward_and_backward(params, x, y, activation, last_activation, cost_function, lam, num_of_layers):
    a, forward_cache = forward(params, x, activation, last_activation, num_of_layers)
    cost = compute_cost(a, y, cost_function, params, lam, num_of_layers)
    grads = backward(a, y, forward_cache, cost_function, activation, last_activation, num_of_layers)

    return grads, cost


def gradient_clip(grad, limit):
    if np.linalg.norm(grad) >= limit:
        grad = limit * (grad / np.linalg.norm(grad))

    return grad


def update_parameters(m, params, grads, learning_rate, lam, num_of_layers):
    for i in range(1, num_of_layers):
        params["w" + str(i)] = params["w" + str(i)] - learning_rate * gradient_clip(grads["dw" + str(i)], 1)
        params["b" + str(i)] = params["b" + str(i)] - learning_rate * gradient_clip(grads["db" + str(i)], 1)

    return params


def predict(params, x_test, y_test, activation, last_activation, num_of_layers):
    a, _ = forward(params, x_test, activation, last_activation, num_of_layers)
    prediction = np.zeros(a.shape)
    prediction[a >= 0.75] = 1
    accuracy = np.mean(np.all(prediction == y_test, axis=0, keepdims=True)) * 100

    return accuracy


def dnn_model(x_train, y_train, x_test, y_test,
              activation, last_activation,
              learning_rate, cost_function, lam,
              batch_size, num_of_iterations,
              dims_of_layers):
    params = dnn_utils.init_params(dims_of_layers)
    num_of_layers = len(dims_of_layers)

    costs = []

    mini_batches = data_utils.generate_random_mini_batches(x_train, y_train, batch_size)
    # 1 epoch
    for index, mini_batch in enumerate(mini_batches):
        mini_batch_x, mini_batch_y = mini_batch
        for j in range(0, num_of_iterations):
            grads, cost = dnn_utils.forward_and_backward(params, mini_batch_x, mini_batch_y,
                                                         activation, last_activation,
                                                         cost_function, lam,
                                                         num_of_layers)
            params = dnn_utils.update_parameters(len(mini_batch_x), params, grads, learning_rate, lam, num_of_layers)
            if j % 100 == 0:
                print("%dth mini batch, cost after iteration %d: %f" % (index + 1, j + 1, cost))
                costs.append(cost)
    train_accuracy = dnn_utils.predict(params, x_train, y_train, activation, last_activation, num_of_layers)
    test_accuracy = dnn_utils.predict(params, x_test, y_test, activation, last_activation, num_of_layers)
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

params = dnn_model(x_train, y_train, x_test, y_test,
                   activation="relu", last_activation="softmax",
                   learning_rate=0.001, cost_function='cross_entropy', lam=0.1,
                   batch_size=108, num_of_iterations=2000,
                   dims_of_layers=[input_dim, 32, 32, output_dim])
