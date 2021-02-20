import numpy as np


def init_params(dim):
    params = {"w": np.random.randn(1, dim) * 0.01,
              "b": 0}

    return params


def linear(w, b, x):
    z = np.matmul(w, x) + b

    return z


def sigmoid(z):
    a = 1 / (1 + np.exp(-z))

    return a


def single_forward(w, b, x):
    z = linear(w, b, x)
    a = sigmoid(z)

    return a


def forward_and_backward(w, b, x, y):
    m = x.shape[1]

    a = single_forward(w, b, x)
    cost = np.sum(-(y * np.log(a) + (1 - y) * np.log(1 - a))) / m
    dw = np.matmul((a - y), x.T) / m
    db = np.sum(a - y) / m

    grads = {"dw": dw,
             "db": db}

    return grads, cost


def predict(params, x, y):
    w = params["w"]
    b = params["b"]
    a = single_forward(w, b, x)

    prediction = np.zeros(a.shape)
    prediction[a >= 0.9] = 1

    accuracy = (1 - np.mean(np.abs(prediction - y))) * 100

    return accuracy

import matplotlib.pyplot as plt

def logistic_regression_model(x_train, y_train, x_test, y_test, learning_rate, num_of_iterations):
    dim = x_train.shape[0]
    params = logistic_regression_util.init_params(dim)

    costs = []
    for i in range(num_of_iterations):
        grads, cost = logistic_regression_util.forward_and_backward(params["w"], params["b"], x_train, y_train)
        params["w"] = params["w"] - learning_rate * grads["dw"]
        params["b"] = params["b"] - learning_rate * grads["db"]
        if i % 100 == 0:
            costs.append(cost)
            print("cost after iteration %d: %f" % (i, cost))

    train_accuracy = logistic_regression_util.predict(params, x_train, y_train)
    test_accuracy = logistic_regression_util.predict(params, x_test, y_test)
    print("train accuracy: {}%".format(train_accuracy))
    print("test accuracy: {}%".format(test_accuracy))

    plt.figure()
    plt.plot(costs)
    plt.xlabel("num of iterations")
    plt.ylabel("cost")
    plt.title("logistic regression")
    plt.show()

    return params


x_train, y_train, x_test, y_test, input_dim, output_di = data_utils.load_sign_data_sets()
x_train, y_train = data_utils.flatten(x_train, y_train)
x_test, y_test = data_utils.flatten(x_test, y_test)
x_train = data_utils.centralize_x(x_train)
x_test = data_utils.centralize_x(x_test)
y_train = data_utils.one_hot_encoding_y(y_train)
y_test = data_utils.one_hot_encoding_y(y_test)
learning_rate = 0.005
num_of_iterations = 2000
logistic_regression_model(x_train, y_train, x_test, y_test, learning_rate, num_of_iterations)
