import tensorflow as tf


def create_placeholders(input_dim, out_dim):
    x = tf.placeholder(dtype=tf.float32, shape=[input_dim, None], name="x")
    y = tf.placeholder(dtype=tf.float32, shape=[out_dim, None], name="y")

    return x, y


def initialize_parameters(dims_of_layers):
    params = {}
    for i in range(1, len(dims_of_layers)):
        params['w' + str(i)] = tf.get_variable('w' + str(i), [dims_of_layers[i], dims_of_layers[i - 1]],
                                               initializer=tf.keras.initializers.he_uniform())
        params['b' + str(i)] = tf.get_variable("b" + str(i), [dims_of_layers[i], 1], initializer=tf.zeros_initializer())

    return params


def linear_forward(w, b, x):
    z = tf.add(tf.linalg.matmul(w, x), b)
    return z


def forward_propagation(x, params, num_of_layers):
    a = x
    for i in range(1, num_of_layers):
        if i == num_of_layers - 1:
            w = params['w' + str(i)]
            b = params['b' + str(i)]
            z = linear_forward(w, b, a)
            a = tf.nn.softmax(z, axis=0)
        else:
            w = params['w' + str(i)]
            b = params['b' + str(i)]
            z = linear_forward(w, b, a)
            a = tf.nn.leaky_relu(z)

    return a


# 예제6 # 정규화 할 것
def compute_cost(a, y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=a, labels=y))
    return cost


def predict(x, params, num_of_layers, y):
    a = forward_propagation(x, params, num_of_layers)
    correct_prediction = tf.equal(tf.argmax(a), tf.argmax(y))

    return correct_prediction


# 예제7
def one_hot_matrix(labels, C):
    # labels : 주어진 행벡터 / C : 클래스의 개수 = depth
    C = tf.constant(C, name="C")

    # labels >> depth = c >> axis=0 방향 >> 원핫인코딩
    one_hot_matrix = tf.one_hot_encoder(labels, C, axis=0)

    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()

    return one_hot


# 예제8
def ones(shape):
    ones = tf.ones(shape, name="ones")

    sess = tf.Session()
    ones = sess.run(ones)
    sess.close()

    return ones
