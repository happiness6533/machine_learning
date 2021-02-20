class ActivityRegularization(Layer):
    """활성 희소 정규화 손실(activity sparsity regularization loss)을 생성하는 Layer 입니다"""
    def __init__(self, rate=1e-2):
        super(ActivityRegularization, self).__init__()
        self.rate = rate

    def call(self, inputs):
        # 입력값에 기반하는
        # `add_loss`를 사용해서 정규화 손실을 생성합니다
        self.add_loss(self.rate * tf.reduce_sum(tf.square(inputs)))
        return inputs

class SparseMLP(Layer):
    """희소 정규화 손실을 가지는 선형 계층을 쌓아올린 Layer 입니다"""

    def __init__(self, output_dim):
        super(SparseMLP, self).__init__()
        self.dense_1 = layers.Dense(32, activation=tf.nn.relu)
        self.regularization = ActivityRegularization(1e-2)
        self.dense_2 = layers.Dense(output_dim)

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.regularization(x)
        return self.dense_2(x)


mlp = SparseMLP(1)
print(mlp.losses)  # float32 자료형의 단일 스칼라값을 가지는 리스트 입니다

# 데이터셋을 준비합니다
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    (x_train.reshape(60000, 784).astype('float32') / 255, y_train))
dataset = dataset.shuffle(buffer_size=1024).batch(64)

# 새로운 MLP를 만듭니다
mlp = SparseMLP(10)

# Loss와 Optimizer를 만듭니다
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

for step, (x, y) in enumerate(dataset):
    with tf.GradientTape() as tape:

        # 순방향 전파를 수행합니다
        logits = mlp(x)

        # 현재 배치에 대한 외부의 손실값을 계산합니다
        loss = loss_fn(y, logits)

        # 순방향 전파시 생성된 손실값을 더해줍니다
        loss += sum(mlp.losses)

        # 해당 손실에 대한 가중치의 경사도를 계산합니다
        gradients = tape.gradient(loss, mlp.trainable_weights)

    # 모델의 가중치를 갱신합니다
    optimizer.apply_gradients(zip(gradients, mlp.trainable_weights))

    # 로그를 출력합니다
    if step % 100 == 0:
        print(step, float(loss))
