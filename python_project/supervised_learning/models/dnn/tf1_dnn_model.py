import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import supervised_learning_ml_training_server.utils.tf_dnn_utils as tf_dnn_utils
import data.data_utils as data_utils


def dnn_model(x_train, y_train, x_test, y_test, learning_rate=0.001, num_of_iterations=1500, batch_size=32):
    costs = []
    x, y = tf_dnn_utils.create_placeholders(dims_of_layers[0], dims_of_layers[-1])
    params = tf_dnn_utils.initialize_parameters(dims_of_layers)
    a = tf_dnn_utils.forward_propagation(x, params, len(dims_of_layers))
    cost = tf_dnn_utils.compute_cost(a, y,,
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        mini_batches = data_utils.generate_random_mini_batches(x_train, y_train, batch_size)
        for mini_batch in mini_batches:
            mini_batch_x, mini_batch_y = mini_batch
            for j in range(num_of_iterations):
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={x: mini_batch_x, y: mini_batch_y})
                if j % 100 == 0:
                    print("minibatch_cost after iteration %d: %f" % (j, minibatch_cost))
                    costs.append(minibatch_cost)
                    # print(a.eval(feed_dict={x: mini_batch_x, y: mini_batch_y}))

        train_accuracy = tf_dnn_utils.predict(len(dims_of_layers), x_train, params,,
        test_accuracy = tf_dnn_utils.predict(len(dims_of_layers), x_test, params,,
        print("Train Accuracy:", train_accuracy.eval({x: x_train, y: y_train}))
        print("Test Accuracy:", test_accuracy.eval({x: x_test, y: y_test}))

        plt.figure()
        plt.plot(np.squeeze(costs))
        plt.xlabel("dnn")
        plt.ylabel("cost")
        plt.title("machine learning2")
        plt.show()

        return params


x_train, y_train, x_test, y_test, input_dim, output_dim = data_utils.load_sign_dataset()

x_train, y_train = data_utils.flatten_and_reshape(x_train, y_train)
x_test, y_test = data_utils.flatten_and_reshape(x_test, y_test)

x_train = data_utils.centralize(x_train)
x_test = data_utils.centralize(x_test)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

dims_of_layers = [input_dim, 32, output_dim]
params = dnn_model(x_train, y_train, x_test, y_test, learning_rate=0.001, num_of_iterations=1500, batch_size=108)
