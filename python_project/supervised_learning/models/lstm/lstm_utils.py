import numpy as np
import rnnAndLstmProject.rnnAndLstmUtils as rnnAndLstmUtils


def single_rnn_forward(aPrev, xt, parameters):
    # 활성벡터의 차원 != 입력벡터 사전의 차원 != 예측벡터 사전의 차원
    # 활성벡터의 차원 : 엔지니어 마음대로
    # 입력벡터 사전의 차원 : 입력하는 자료의 사전에 의해 정해진다
    # 예측벡터 사전의 차원 : 예측하고자 하는 자료의 사전에 의해 정해진다

    # waa : 활성벡터 >> 활성벡터 파라미터 : (활성벡터의 차원 * 활성벡터의 차원)
    # aPrev : 활성벡터 : (활성벡터의 차원 * m)
    # wax : 입력벡터 >> 활성벡터 파라미터 : (활성벡터의 차원 * 입력벡터 사전의 차원)
    # xt : 입력벡터 : (입력벡터 사전의 차원 * m)
    # ba : 활성벡터 바이어스 : (활성벡터의 차원 * 1)
    # >> tanh >> at 생성 : (활성벡터의 차원 * m)

    # wya : 예측벡터 파라미터 : (예측벡터 사전의 차원 * 활성벡터의 차원)
    # at : 활성벡터 : (활성벡터의 차원 * m)
    # by : 예측벡터 바이어스 : (예측벡터 사전의 차원 * 1)
    # >> softmax >> yt : 예측벡터 생성 : (예측벡터 사전의 차원 * m)
    Wax = parameters["Wax"]
    ba = parameters["ba"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    by = parameters["by"]

    # 활성벡터
    at = np.tanh(np.matmul(Wax, xt) + np.matmul(Waa, aPrev) + ba)

    # 예측벡터
    yt = rnnAndLstmUtils.softmax(np.matmul(Wya, at) + by)

    # 저장소
    cache = (at, aPrev, xt, parameters)

    return at, yt, cache


# rnn forward
def rnn_forward(a0, x, parameters):
    # 활성벡터의 차원 != 입력벡터 사전의 차원 != 예측벡터 사전의 차원
    # 활성벡터의 차원 : 엔지니어 마음대로
    # 입력벡터 사전의 차원 : 입력하는 자료의 사전에 의해 정해진다
    # 예측벡터 사전의 차원 : 예측하고자 하는 자료의 사전에 의해 정해진다

    # waa : 활성벡터 >> 활성벡터 파라미터 : (활성벡터의 차원 * 활성벡터의 차원) : 모든 t에서 공유한다!
    # a_prev : 활성벡터 : (활성벡터의 차원 * m * t)
    # wax : 입력벡터 >> 활성벡터 파라미터 : (활성벡터의 차원 * 입력벡터 사전의 차원) : 모든 t에서 공유한다!
    # xt : 입력벡터 : (입력벡터 사전의 차원 * m * t)
    # ba : 활성벡터 바이어스 : (활성벡터의 차원 * 1) : 모든 t에서 공유한다!
    # >> tanh >> at 생성 : (활성벡터의 차원 * m * t)

    # wya : 예측벡터 파라미터 : (예측벡터 사전의 차원 * 활성벡터의 차원) : 모든 t에서 공유한다!
    # at : 활성벡터 : (활성벡터의 차원 * m * t)
    # by : 예측벡터 바이어스 : (예측벡터 사전의 차원 * 1) : 모든 t에서 공유한다!
    # >> softmax >> yt : 예측벡터 생성 : (예측벡터 사전의 차원 * m * t)
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape

    # aStore, yStore 초기화
    aStore = np.zeros((n_a, m, T_x))
    yStore = np.zeros((n_y, m, T_x))

    # 저장소 생성
    caches = []

    # a0 : 가상의 활성벡터 초기화 : (활성벡터의 차원 * m)
    a_prev = a0
    for t in range(T_x):
        # 활성벡터, 예측벡터 계산
        at, yt, cache = single_rnn_forward(a_prev, x[:, :, t], parameters)

        # 업데이트
        aStore[:, :, t] = at
        yStore[:, :, t] = yt

        # 저장
        caches.append(cache)

        # 다음 레이어로 활성벡터 전송
        a_prev = at

    caches = (caches, x)

    return aStore, yStore, caches


def single_lstm_forward(aPrev, mPrev, xt, parameters):
    # 활성벡터의 차원 != 입력벡터 사전의 차원 != 예측벡터 사전의 차원
    # 활성벡터의 차원 : 엔지니어 마음대로
    # 입력벡터 사전의 차원 : 입력하는 자료의 사전에 의해 정해진다
    # 예측벡터 사전의 차원 : 예측하고자 하는 자료의 사전에 의해 정해진다

    # wf : 전 활성, 입력벡터 >> forget gate 벡터 파라미터 : (활성벡터의 차원 * (활성벡터의 차원 + 입력벡터 사전의 차원))
    # a_prev : 전 활성벡터 : (활성벡터의 차원 * m)
    # xt : 입력벡터 : (입력벡터 사전의 차원 * m)
    # bf : forget gate 벡터 바이어스 : (활성벡터의 차원 * 1)
    # >> sigmoid >> forget gate 벡터 생성 : (활성벡터의 차원 * m)

    # wu : 전 활성, 입력벡터 >> update gate 벡터 파라미터 : (활성벡터의 차원 * (활성벡터의 차원 + 입력벡터 사전의 차원))
    # a_prev : 전 활성벡터 : (활성벡터의 차원 * m)
    # xt : 입력벡터 : (입력벡터 사전의 차원 * m)
    # bu : update gate 벡터 바이어스 : (활성벡터의 차원 * 1)
    # >> sigmoid >> update gate 벡터 생성 : (활성벡터의 차원 * m)

    # wcm : 전 활성, 입력벡터 >> 후보 메모리벡터 파라미터 : (활성벡터의 차원 * (활성벡터의 차원 + 입력벡터 사전의 차원))
    # a_prev : 전 활성벡터 : (활성벡터의 차원 * m)
    # xt : 입력벡터 : (입력벡터 사전의 차원 * m)
    # bcm : 후보 메모리벡터 바이어스 : (활성벡터의 차원 * 1)
    # >> tanh >> 후보 메모리벡터 생성 : (활성벡터의 차원 * m)

    # mt = 메모리벡터 = forget gate 벡터 * 전 메모리벡터 + update gate 벡터 * 후보 메모리벡터

    # wca : 전 활성, 입력벡터 >> 후보 활성벡터 파라미터 : (활성벡터의 차원 * (활성벡터의 차원 + 입력벡터 사전의 차원))
    # a_prev : 전 활성벡터 : (활성벡터의 차원 * m)
    # xt : 입력벡터 : (입력벡터 사전의 차원 * m)
    # bca : 후보 활성벡터 바이어스 : (활성벡터의 차원 * 1)
    # >> sigmoid >> 후보 활성벡터 생성 : (활성벡터의 차원 * m)

    # at = 활성벡터 = 후보 활성벡터 * tanh(mt)

    # wy : 예측벡터 파라미터 : (예측벡터 사전의 차원 * 활성벡터의 차원)
    # at : 활성벡터 : (활성벡터의 차원 * m)
    # by : 예측벡터 바이어스 : (예측벡터 사전의 차원 * 1)
    # >> softmax >> yt : 예측벡터 생성 : (예측벡터 사전의 차원 * m)
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wu = parameters["Wu"]
    bu = parameters["bu"]
    Wcm = parameters["Wcm"]
    bcm = parameters["bcm"]
    Wca = parameters["Wca"]
    bca = parameters["bca"]
    Wy = parameters["Wy"]
    by = parameters["by"]

    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # a_prev / xt 쌓기
    concat = np.zeros(((n_a + n_x), m))
    concat[: n_a, :] = aPrev
    concat[n_a:, :] = xt

    # forget gate
    fg = rnnAndLstmUtils.sigmoid(np.matmul(Wf, concat) + bf)

    # update gate
    ug = rnnAndLstmUtils.sigmoid(np.matmul(Wu, concat) + bu)

    # candidate memory
    cm = np.tanh(np.matmul(Wcm, concat) + bcm)

    # memory
    mt = fg * mPrev + ug * cm

    # candidate activation
    ca = rnnAndLstmUtils.sigmoid(np.matmul(Wca, concat) + bca)

    # at
    at = ca * np.tanh(mt)

    # yt
    yt = rnnAndLstmUtils.softmax(np.matmul(Wy, at) + by)

    # 저장
    cache = (at, mt, aPrev, mPrev, fg, ug, cm, ca, xt, parameters)

    return at, mt, yt, cache


# lstm forward
def lstm_forward(x, a0, parameters):
    # 활성벡터의 차원 != 입력벡터 사전의 차원 != 예측벡터 사전의 차원
    # 활성벡터의 차원 : 엔지니어 마음대로
    # 입력벡터 사전의 차원 : 입력하는 자료의 사전에 의해 정해진다
    # 예측벡터 사전의 차원 : 예측하고자 하는 자료의 사전에 의해 정해진다

    # wf : 전 활성, 입력벡터 >> forget gate 벡터 파라미터 : (활성벡터의 차원 * (활성벡터의 차원 + 입력벡터 사전의 차원)) : 모든 t에서 공유한다!
    # a_prev : 전 활성벡터 : (활성벡터의 차원 * m * t)
    # xt : 입력벡터 : (입력벡터 사전의 차원 * m * t)
    # bf : forget gate 벡터 바이어스 : (활성벡터의 차원 * 1) : 모든 t에서 공유한다
    # >> sigmoid >> forget gate 벡터 생성 : (활성벡터의 차원 * m * t)

    # wu : 전 활성, 입력벡터 >> update gate 벡터 파라미터 : (활성벡터의 차원 * (활성벡터의 차원 + 입력벡터 사전의 차원)) : 모든 t에서 공유한다!
    # a_prev : 전 활성벡터 : (활성벡터의 차원 * m * t)
    # xt : 입력벡터 : (입력벡터 사전의 차원 * m * t)
    # bu : update gate 벡터 바이어스 : (활성벡터의 차원 * 1) : 모든 t에서 공유한다!
    # >> sigmoid >> update gate 벡터 생성 : (활성벡터의 차원 * m * t)

    # wcm : 전 활성, 입력벡터 >> 후보 메모리벡터 파라미터 : (활성벡터의 차원 * (활성벡터의 차원 + 입력벡터 사전의 차원)) : 모든 t에서 공유한다!
    # a_prev : 전 활성벡터 : (활성벡터의 차원 * m * t)
    # xt : 입력벡터 : (입력벡터 사전의 차원 * m * t)
    # bcm : 후보 메모리벡터 바이어스 : (활성벡터의 차원 * 1) : 모든 t에서 공유한다!
    # >> tanh >> 후보 메모리벡터 생성 : (활성벡터의 차원 * m * t)

    # mt = 메모리벡터 = forget gate 벡터 * 전 메모리벡터 + update gate 벡터 * 후보 메모리벡터

    # wca : 전 활성, 입력벡터 >> 후보 활성벡터 파라미터 : (활성벡터의 차원 * (활성벡터의 차원 + 입력벡터 사전의 차원)) : 모든 t에서 공유한다!
    # a_prev : 전 활성벡터 : (활성벡터의 차원 * m * t)
    # xt : 입력벡터 : (입력벡터 사전의 차원 * m * t)
    # bca : 후보 활성벡터 바이어스 : (활성벡터의 차원 * 1) : 모든 t에서 공유한다!
    # >> sigmoid >> 후보 활성벡터 생성 : (활성벡터의 차원 * m * t)

    # at = 활성벡터 = 후보 활성벡터 * tanh(mt)

    # wy : 예측벡터 파라미터 : (예측벡터 사전의 차원 * 활성벡터의 차원) : 모든 t에서 공유한다!
    # at : 활성벡터 : (활성벡터의 차원 * m * t)
    # by : 예측벡터 바이어스 : (예측벡터 사전의 차원 * 1) : 모든 t에서 공유한다!
    # >> softmax >> yt : 예측벡터 생성 : (예측벡터 사전의 차원 * m * t)
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wy"].shape

    # aStore, mStore, yStore 초기화
    aStore = np.zeros((n_a, m, T_x))
    mStore = np.zeros((n_a, m, T_x))
    yStore = np.zeros((n_y, m, T_x))

    # 저장소 생성
    caches = []

    # a0 : 가상의 활성벡터 초기화 : (활성벡터의 차원 * m)
    # m0 : 가상의 0 메모리벡터 초기화 : (메모리벡터의 차원 * m)
    a_prev = a0
    m_prev = np.zeros(a_prev.shape)
    for t in range(T_x):
        # 계산
        at, mt, yt, cache = single_lstm_forward(a_prev, m_prev, x[:, :, t], parameters)

        # 업데이트
        aStore[:, :, t] = at
        mStore[:, :, t] = mt
        yStore[:, :, t] = yt

        # 저장
        caches.append(cache)

        # 다음 레이어로 활성, 메모리벡터 전송
        a_prev = at
        m_prev = mt

    caches = (caches, x)

    return aStore, yStore, mStore, caches


# 여기서부터는 나중에 시간나면 해보자
# single rnn backward
def rnn_cell_backward(da_next, cache):
    """
    Implements the backward pass for the RNN-cell (single time-step).

    Arguments:
    da_next -- Gradient of loss with respect to next hidden state
    cache -- python dictionary containing useful values (output of rnn_cell_forward())

    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradients of input data, of shape (n_x, m)
                        da_prev -- Gradients of previous hidden state, of shape (n_a, m)
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dba -- Gradients of bias vector, of shape (n_a, 1)
    """
    # Retrieve values from cache
    (a_next, a_prev, xt, parameters) = cache

    # Retrieve values from parameters
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    ### START CODE HERE ###
    # compute the gradient of tanh with respect to a_next (≈1 line)
    dtanh = None

    # compute the gradient of the loss with respect to Wax (≈2 lines)
    dxt = None
    dWax = None

    # compute the gradient with respect to Waa (≈2 lines)
    da_prev = None
    dWaa = None

    # compute the gradient with respect to b (≈1 line)
    dba = None

    ### END CODE HERE ###

    # Store the gradients in a python dictionary
    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}

    return gradients


# rnn backward
def rnn_backward(da, caches):
    """
    Implement the backward pass for a RNN over an entire sequence of input data.

    Arguments:
    da -- Upstream gradients of all hidden states, of shape (n_a, m, T_x)
    caches -- tuple containing information from the forward pass (rnn_forward)

    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradient w.r.t. the input data, numpy-array of shape (n_x, m, T_x)
                        da0 -- Gradient w.r.t the initial hidden state, numpy-array of shape (n_a, m)
                        dWax -- Gradient w.r.t the input's weight matrix, numpy-array of shape (n_a, n_x)
                        dWaa -- Gradient w.r.t the hidden state's weight matrix, numpy-arrayof shape (n_a, n_a)
                        dba -- Gradient w.r.t the bias, of shape (n_a, 1)
    """

    ### START CODE HERE ###

    # Retrieve values from the first cache (t=1) of caches (≈2 lines)
    (caches, x) = None
    (a1, a0, x1, parameters) = None

    # Retrieve dimensions from da's and x1's shapes (≈2 lines)
    n_a, m, T_x = None
    n_x, m = None

    # initialize the gradients with the right sizes (≈6 lines)
    dx = None
    dWax = None
    dWaa = None
    dba = None
    da0 = None
    da_prevt = None

    # Loop through all the time steps
    for t in reversed(range(None)):
        # Compute gradients at time step t. Choose wisely the "da_next" and the "cache" to use in the backward propagation step. (≈1 line)
        gradients = None
        # Retrieve derivatives from gradients (≈ 1 line)
        dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients[
            "dWaa"], gradients["dba"]
        # Increment global derivatives w.r.t parameters by adding their derivative at time-step t (≈4 lines)
        dx[:, :, t] = None
        dWax += None
        dWaa += None
        dba += None

    # Set da0 to the gradient of a which has been backpropagated through all time-steps (≈1 line)
    da0 = None
    ### END CODE HERE ###

    # Store the gradients in a python dictionary
    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa, "dba": dba}

    return gradients


# single lstm backward
def lstm_cell_backward(da_next, dc_next, cache):
    """
    Implement the backward pass for the LSTM-cell (single time-step).

    Arguments:
    da_next -- Gradients of next hidden state, of shape (n_a, m)
    dc_next -- Gradients of next cell state, of shape (n_a, m)
    cache -- cache storing information from the forward pass

    Returns:
    gradients -- python dictionary containing:
                        dxt -- Gradient of input data at time-step t, of shape (n_x, m)
                        da_prev -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dc_prev -- Gradient w.r.t. the previous memory state, of shape (n_a, m, T_x)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the output gate, of shape (n_a, 1)
    """

    # Retrieve information from "cache"
    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache

    ### START CODE HERE ###
    # Retrieve dimensions from xt's and a_next's shape (≈2 lines)
    n_x, m = None
    n_a, m = None

    # Compute gates related derivatives, you can find their values can be found by looking carefully at equations (7) to (10) (≈4 lines)
    dot = None
    dcct = None
    dit = None
    dft = None

    # Code equations (7) to (10) (≈4 lines)
    dit = None
    dft = None
    dot = None
    dcct = None

    # Compute parameters related derivatives. Use equations (11)-(14) (≈8 lines)
    dWf = None
    dWi = None
    dWc = None
    dWo = None
    dbf = None
    dbi = None
    dbc = None
    dbo = None

    # Compute derivatives w.r.t previous hidden state, previous memory state and input. Use equations (15)-(17). (≈3 lines)
    da_prev = None
    dc_prev = None
    dxt = None
    ### END CODE HERE ###

    # Save gradients in dictionary
    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                 "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

    return gradients


# lstm backward
def lstm_backward(da, caches):
    """
    Implement the backward pass for the RNN with LSTM-cell (over a whole sequence).

    Arguments:
    da -- Gradients w.r.t the hidden states, numpy-array of shape (n_a, m, T_x)
    dc -- Gradients w.r.t the memory states, numpy-array of shape (n_a, m, T_x)
    caches -- cache storing information from the forward pass (lstm_forward)

    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradient of inputs, of shape (n_x, m, T_x)
                        da0 -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the save gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the save gate, of shape (n_a, 1)
    """

    # Retrieve values from the first cache (t=1) of caches.
    (caches, x) = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]

    ### START CODE HERE ###
    # Retrieve dimensions from da's and x1's shapes (≈2 lines)
    n_a, m, T_x = None
    n_x, m = None

    # initialize the gradients with the right sizes (≈12 lines)
    dx = None
    da0 = None
    da_prevt = None
    dc_prevt = None
    dWf = None
    dWi = None
    dWc = None
    dWo = None
    dbf = None
    dbi = None
    dbc = None
    dbo = None

    # loop back over the whole sequence
    for t in reversed(range(None)):
        # Compute all gradients using lstm_cell_backward
        gradients = None
        # Store or add the gradient to the parameters' previous step's gradient
        dx[:, :, t] = None
        dWf = None
        dWi = None
        dWc = None
        dWo = None
        dbf = None
        dbi = None
        dbc = None
        dbo = None
    # Set the first activation's gradient to the backpropagated gradient da_prev.
    da0 = None

    ### END CODE HERE ###

    # Store the gradients in a python dictionary
    gradients = {"dx": dx, "da0": da0, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                 "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

    return gradients
