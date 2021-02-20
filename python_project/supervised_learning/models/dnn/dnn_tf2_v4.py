import tensorflow as tf
import python_project.supervised_learning.data.data_utils as data_utils


# 1. 데이터
# mnist 데이터
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[:].reshape(60000, 784).astype('float32') / 255
x_test = x_test[:].reshape(10000, 784).astype('float32') / 255
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.shuffle(buffer_size=1000).batch(100)
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_data = test_data.shuffle(buffer_size=1000).batch(10000)

# 실제 데이터
# x_train, y_train, x_test, y_test, input_dim, output_dim = data_utils.load_sign_data_sets()
# x_train = x_train.reshape(1080, 64 * 64 * 3).astype('float32') / 255
# x_test = x_test.reshape(120, 64 * 64 * 3).astype('float32') / 255
# train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# train_data = train_data.shuffle(buffer_size=108).batch(108)
# test_data = test_data.shuffle(buffer_size=120).batch(120)


# 2. 레이어 생성
# 레이어 생성1
class CustomDense1(tf.keras.layers.Layer):
    def __init__(self, input_dim, units, activation):
        super(CustomDense1, self).__init__()
        self.w = tf.Variable(initial_value=tf.random.normal(shape=(input_dim, units), mean=0., stddev=1),
                             trainable=True)
        self.b = tf.Variable(initial_value=tf.zeros(shape=(units,)),
                             trainable=True)
        self.activation = tf.keras.activations.get(activation)

    def call(self, inputs):
        return self.activation(tf.linalg.matmul(inputs, self.w) + self.b)


custom_dense1 = CustomDense1(input_dim=2, units=1, activation='sigmoid')
assert custom_dense1.weights == [custom_dense1.w, custom_dense1.b]
assert custom_dense1.trainable_weights == [custom_dense1.w, custom_dense1.b]
assert custom_dense1.non_trainable_weights == []


# 레이어 생성2(build 메소드)
class CustomDense2(tf.keras.layers.Layer):
    def __init__(self, input_dim, units, activation):
        super(CustomDense2, self).__init__()
        self.input_dim = input_dim
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self):
        self.w = self.add_weight(shape=(self.input_dim, self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        return self.activation(tf.linalg.matmul(inputs, self.w) + self.b)


# 레이어 생성3(call 메소드 training)
class CustomDropout(tf.keras.layers.Layer):
    def __init__(self, rate):
        super(CustomDropout, self).__init__()
        self.rate = rate

    def call(self, inputs, training=None):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate)
        return inputs


# 레이어 생성4(레이어 조합)
class CustomForward(tf.keras.layers.Layer):
    def __init__(self, input_dim, units, activation):
        super(CustomForward, self).__init__()
        self.custom_dense1 = CustomDense1(input_dim, units, activation)
        self.custom_dropout = CustomDropout(0.1)

    def call(self, inputs, training=None):
        a = self.custom_dense1(inputs)
        return self.custom_dropout(a, training=training)


# 3. 모델 생성
class CustomModel1(tf.keras.models.Model):
    def __init__(self):
        super(CustomModel1, self).__init__()
        self.hidden_layer1 = CustomForward(input_dim=784, units=64, activation='relu')
        self.hidden_layer2 = CustomForward(input_dim=64, units=64, activation='relu')
        self.hidden_layer3 = CustomForward(input_dim=64, units=32, activation='relu')
        self.output_layer = CustomForward(input_dim=32, units=10, activation='softmax')

    def call(self, inputs, training=None):
        hidden1 = self.hidden_layer1(inputs, training=training)
        hidden2 = self.hidden_layer2(hidden1, training=training)
        hidden3 = self.hidden_layer3(hidden2, training=training)
        return self.output_layer(hidden3, training=training)


custom_model1 = CustomModel1()

# 4. 모델 로스/옵티마이저/메트릭
loss = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


# 5. 모델 트레이닝/평가
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = custom_model1(x, training=True)
        loss_value = loss(y, logits)
    gradients = tape.gradient(loss_value, custom_model1.trainable_weights)
    optimizer.apply_gradients(zip(gradients, custom_model1.trainable_weights))
    accuracy.update_state(y, logits)

    return loss_value


def test_step(x, y):
    accuracy.reset_states()
    logits = custom_model1(x, training=False)
    accuracy.update_state(y, logits)


for epoch in range(10):
    print('%d번째 epoch' % (epoch + 1))
    for index, (x, y) in enumerate(train_data):
        loss_value = train_step(x, y)
        if index % 100 == 0:
            print('%d 단계 / loss_value: %f / accuracy: %f' % (index, float(loss_value), float(accuracy.result())))
    for step, (x, y) in enumerate(test_data):
        test_step(x, y)
        print('test accuracy: %f' % (float(accuracy.result())))
