import tensorflow as tf
import numpy as np

def initialize_parameters(dims_of_layers):
    # w 초기화: 가우시안, Xavier initialization, He initialization...
    # b 초기화: 0
    num_of_layers = len(dims_of_layers)
    params = {}
    for i in range(1, num_of_layers):
        params["w" + str(i)] = tf.Variable(
            initial_value=np.random.randn(dims_of_layers[i], dims_of_layers[i - 1]) * 0.01)
        params["b" + str(i)] = tf.zeros(shape=(dims_of_layers[i], 1))

    return params


def linear_forward(w, b, a):
    z = tf.linalg.matmul(w, a) + b
    linear_cache = w, b, a

    return z, linear_cache

def linear_activation_forward(w, b, a, activation):
    z, linear_cache = linear_forward(w, b, a)

    if activation == "sigmoid":
        next_a, activation_cache = tf.nn.sigmoid(z)
    elif activation == "relu":
        next_a, activation_cache = tf.nn.relu(z)
    elif activation == "leaky_relu":
        next_a, activation_cache = tf.nn.leaky_relu(z)
    elif activation == "softmax":
        next_a, activation_cache = tf.nn.softmax(z)

    linear_activation_cache = (linear_cache, activation_cache)

    return next_a, linear_activation_cache

def forward(x, params, num_of_layers, activation, last_activation):
    forward_cache = []

    a = x
    for i in range(1, num_of_layers - 1):
        next_a, linear_activation_cache = linear_activation_forward(params['w' + str(i)],
                                                                    params['b' + str(i)], a,
                                                                    activation=activation)
        a = next_a
        forward_cache.append(linear_activation_cache)

    next_a, linear_activation_cache = linear_activation_forward(params['w' + str(num_of_layers - 1)],
                                                                params['b' + str(num_of_layers - 1)], a,
                                                                activation=last_activation)
    forward_cache.append(linear_activation_cache)

    return next_a, forward_cache


def compute_cost(a, y, params, num_of_layers, cost_function, lam):
    # optimize: (batch, stochastic, mini batch) gradient descent, momentum, adagrad, RMSProp, adam...
    # cost function: cross entropy LS...
    # regularize: L2, dropout...
    cost = 0
    if cost_function == 'cross_entropy':
        cost = tf.nn.softmax_cross_entropy_with_logits(y, a)
    elif cost_function == 'least_square_error':
        mse = tf.keras.losses.MeanSquaredError()
        cost = mse(y, a)

    regularize_term = 0
    for i in range(1, num_of_layers):
        w = params["w" + str(i)]
        regularize_term += tf.nn.l2_loss(w)
    regularize_term = lam * regularize_term

    cost = cost + regularize_term

    return cost
