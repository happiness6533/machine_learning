import matplotlib.pyplot as plt
import tensorflow as tf
import supervised_learning_ml_training_server.utils.tf_cnn_utils as tf_cnn_utils
import data.data_utils as data_utils


def cnn_model(x_train, y_train, x_test, y_test, learning_rate, num_of_iteration, mini_batch_size, dims_of_channels):
    costs = []
    num_of_layers = len(dims_of_channels)
    (m, h0, w0, c0) = x_train.shape
    (m, class_size) = y_train.shape

    tf.reset_default_graph()

    x, y = tf_cnn_utils.create_placeholders(h0, w0, c0, class_size)
    filter_parameters = tf_cnn_utils.initialize_filter_parameters(dims_of_channels)
    a = tf_cnn_utils.forward(x, filter_parameters, num_of_layers)
    cost = tf_cnn_utils.compute_cost(a, y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    correctPrediction = tf.equal(tf.argmax(a, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correctPrediction, dtype=tf.float32))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


    sess = tf.Session()
    update = input("new vs update : ")
    if update == "new":
        sess.run(init)
    else:
        # 세션에 텐서 로드
        saver.restore(sess=sess, save_path="trained/trainedModel.ckpt")

    for i in range(num_of_iteration):
        mini_batch_cost = 0
        mini_batches = data_utils.generate_random_mini_batches_for_cnn(x_train, y_train, mini_batch_size)
        for mini_batch in mini_batches:
            (mini_batch_x, mini_batch_y) = mini_batch
            _, mini_batch_cost = sess.run([optimizer, cost], feed_dict={x: mini_batch_x, y: mini_batch_y})
        costs.append(mini_batch_cost)
        if i % 100 == 0:
            print("Cost after iteration %d : %f" % (i, mini_batch_cost))

    # 정확도
    train_accuracy = sess.run(accuracy, feed_dict={x: x_train, y: y_train})
    test_accuracy = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
    print("Train Accuracy : ", train_accuracy)
    print("Test Accuracy : ", test_accuracy)

    # 코스트 변화
    plt.plot(costs)
    plt.xlabel("iteration")
    plt.ylabel("costs")
    plt.title("cnn")
    plt.show()

    saver.save(sess=sess, save_path="trained/trainedModel.ckpt")

    sess.close()


x_train, y_train, x_test, y_test, input_dim, output_dim = data_utils.load_sign_dataset()

x_train = data_utils.centralize(x_train)
x_test = data_utils.centralize(x_test)

# 트레이닝
cnn_model(x_train, y_train, x_test, y_test, learning_rate=0.005, num_of_iteration=1000, mini_batch_size=64, dims_of_channels=[3, 5, 10, 15, 20, 25])
