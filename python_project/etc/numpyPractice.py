import numpy as np


# 벡터(2, 1)
v1 = np.array([[1],
               [2]])

# 랜덤
v2 = np.random.rand(2, 1)

# 가우시안 랜덤
v3 = np.random.randn(2, 1)

# 행렬(2, 2)
m1 = np.array([[1, 2],
               [3, 4]])

# 텐서(2, 2, 2)
t = np.array([[[1, 2], [3, 4]],
              [[5, 6], [7, 8]]])

# 행렬곱 : 차원을 고려해서 사용한다
result1 = np.matmul(v1.T, v2)

# 벡터의 내적 : 차원을 고려해서 사용한다
result2 = np.dot(v1.T, v2)

# element-wise 곱
result3 = np.multiply(v1, v2)
result4 = v1 * v2


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))

    return s


def sigmoidDerivative(x):
    s = 1 / (1 + np.exp(-x))
    ds = s * (1 - s)

    return ds


def imageToVector(image):
    v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)

    return v


def normalizeRows(x):
    # 각 행의 노말라이즈 상수로 이루어진 벡터 생성
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    xNorm = x / norm

    return xNorm


def softmax(x):
    xExp = np.exp(x)
    xSum = np.sum(xExp, axis=1, keepdims=True)
    s = xExp / xSum

    return s


def loss(yhat, y):
    loss = np.dot((y - yhat), (y - yhat))

    return loss
