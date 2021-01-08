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
