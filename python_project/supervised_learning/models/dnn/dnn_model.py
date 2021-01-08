import numpy as np
import matplotlib.pyplot as plt
import supervised_learning.models.dnn.dnn_utils as dnn_utils
import supervised_learning.models.data.data_utils as data_utils


def dnn_model(x_train, y_train, x_test, y_test,
              activation, last_activation,
              learning_rate, cost_function, lam,
              batch_size, num_of_iterations,
              dims_of_layers):
    params = dnn_utils.init_params(dims_of_layers)
    num_of_layers = len(dims_of_layers)

    costs = []

    mini_batches = data_utils.generate_random_mini_batches(x_train, y_train, batch_size)
    for index, mini_batch in enumerate(mini_batches):
        mini_batch_x, mini_batch_y = mini_batch
        for j in range(0, num_of_iterations):
            grads, cost = dnn_utils.forward_and_backward(params, mini_batch_x, mini_batch_y,
                                                         activation, last_activation,
                                                         cost_function, lam,
                                                         num_of_layers)
            params = dnn_utils.update_parameters(params, grads, learning_rate, lam, num_of_layers)
            if j == 0 or j % 100 == 99:
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


x_train, y_train, x_test, y_test, input_dim, output_dim = data_utils.load_sign_dataset()

x_train, y_train = data_utils.flatten_and_reshape(x_train, y_train)
x_test, y_test = data_utils.flatten_and_reshape(x_test, y_test)

x_train = data_utils.centralize(x_train)
x_test = data_utils.centralize(x_test)

params = dnn_model(x_train, y_train, x_test, y_test,
                   activation="relu", last_activation="softmax",
                   learning_rate=0.005, cost_function='cross_entropy', lam=0.1,
                   batch_size=1, num_of_iterations=100,
                   dims_of_layers=[input_dim, 32, output_dim])
