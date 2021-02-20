import tensorflow as tf
import python_project.supervised_learning.data.data_utils as data_utils

# 1. 데이터
# mnist 데이터
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[:].astype('float32') / 255
x_test = x_test[:].astype('float32') / 255

# 실제 데이터
# x_train, y_train, x_test, y_test, input_dim, output_dim = data_utils.load_sign_data_sets()
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255

# 2. 모델 생성
input = tf.keras.Input(shape=(28, 28,))
input_flatten = tf.keras.layers.Flatten(input_shape=(28, 28))(input)
hidden1 = tf.keras.layers.Dense(units=32, activation='relu')(input_flatten)
hidden2 = tf.keras.layers.Dense(units=32, activation='relu')(hidden1)
hidden3 = tf.keras.layers.Dense(units=32, activation='relu')(hidden1 + hidden2)
output = tf.keras.layers.Dense(units=10, activation='softmax')(hidden3)
model = tf.keras.Model(input, output)

# 3. 모델 트레이닝
model.compile(optimizer="adam",
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=100, epochs=10)

# 4. 모델 평가
model.evaluate(x_train, y_train)
model.evaluate(x_test, y_test)

# 5. 모델 배포
