import tensorflow as tf

# 텐서플로우2
# 1. 텐서플로우는 미분을 도와준다
# 2. 즉시 실행과 컴파일의 2가지 모드가 있다
# 3. 케라스를 딥러닝의 API로 사용한다
# 우선 대부분의 모델은 케라스를 사용해서 빌드한다
# 하위 api는 커스텀할때 사용한다
# 아래로 상속되므로 상위의 함수는 아래에서 호출할 수 있다
# tf.keras.modules.variables(), tf.keras.modules.trainable_variables()
# tf.keras.layers > Dense, Conv, Conv2d, RNN, LSTM / __call__ build() add_weight() add_loss() >  call()
# tf.keras.network > layers, summary save
# tf.keras.Model > compile, fit, evaluate
# tf.keras.Sequential > add, input
# tf.keras.layers > Dense, Conv, Conv2d, RNN, LSTM

print(tf.__version__)

# 상수
c = tf.constant([[1, 2],
                 [3, 4]])
# print(c.shape)
# print(c.dtype)
# print(tf.ones(shape=(2, 2)))
# print(tf.zeros(shape=(2, 2)))
# print(tf.random.normal(shape=(2, 2), mean=0., stddev=1.))
# print(tf.random.uniform(shape=(2, 2), minval=0, maxval=10, dtype='int32'))

# 미지수
# x = tf.Variable(tf.random.normal(shape=(2, 2), mean=0., stddev=1.))
# # print(x)
# x.assign(tf.zeros(shape=(2, 2)))
# x.assign_add(tf.ones(shape=(2, 2)))
# x.assign_sub(tf.ones(shape=(2, 2)))
# x.assign_add(tf.ones(shape=(2, 2)))
# print(x)

# 미분계수 구하기1
c1 = tf.random.uniform(shape=(2, 2), minval=1, maxval=4, dtype='float32')
# with tf.GradientTape() as tape:
#     tape.watch(c1)
#     c2 = tf.random.uniform(shape=(2, 2), minval=1, maxval=4, dtype='float32')
#     c3 = tf.square(c1) + tf.square(c2)
#     dc3_dc1 = tape.gradient(c3, c1)
#     print(dc3_dc1)

# 미분계수 구하기2(변수는 워치를 자동으로 실행)
# with tf.GradientTape() as tape:
#     y = tf.square(x) + tf.square(c1)
#     dy_dx = tape.gradient(y, x)
#     print(dy_dx)

# 2차 미분계수 구하기
# with tf.GradientTape() as outer_tape:
#     with tf.GradientTape() as inner_tape:
#         y = tf.square(x) + tf.square(c1)
#         dy_dx = inner_tape.gradient(y, x)
#     d2y_dx2 = outer_tape.gradient(dy_dx, x)
#     print(d2y_dx2)

x = tf.Variable(1.)
y = tf.Variable(1.)
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        z = tf.math.pow(x, 3) * tf.math.pow(y, 3)
        + 2 * tf.square(x) * tf.square(y)
        + 3 * x * y + 4
        dz_dx = inner_tape.gradient(z, x)
        print(dz_dx)
    dz_dxy = outer_tape.gradient(dz_dx, y)
    print(dz_dxy)
