import time
import collections
import threading
import numpy as np
import keras.models as models
import keras.layers as layers
import keras.optimizers as optimizers
import keras.backend as K
import gym
import skimage.transform as transform
import skimage.color as color


class A3cGlobalAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.learning_rate = 0.005
        self.critic_network, self.actor_network = self.build_networks()
        self.critic_optimizer = self.build_critic_optimizer()
        self.actor_optimizer = self.build_actor_optimizer()
        self.discount_factor = 0.99

        self.threads = 8
        # self.no_op_steps = 30

    def build_networks(self):
        input = layers.Input(shape=self.state_size, batch_shape=None)
        hidden = layers.Conv2D(filters=16, kernel_size=(8, 8), strides=(2, 2), padding='same')(input)
        hidden = layers.LeakyReLU()(hidden)
        hidden = layers.Conv2D(filters=16, kernel_size=(8, 8), strides=(2, 2), padding='same')(hidden)
        hidden = layers.LeakyReLU()(hidden)
        hidden = layers.Conv2D(filters=16, kernel_size=(8, 8), strides=(2, 2), padding='same')(hidden)
        hidden = layers.LeakyReLU()(hidden)

        # 네트워크 분리
        value = layers.Dense(units=1, activation='softmax', kernel_initializer='he_uniform')(hidden)
        policy = layers.Dense(units=self.action_size, activation='linear', kernel_initializer='he_uniform')(hidden)

        critic_network = models.Model(inputs=input, outputs=value)
        actor_network = models.Model(inputs=input, outputs=policy)

        critic_network._make_predict_function()
        actor_network._make_predict_function()

        critic_network.summary()
        actor_network.summary()

        return critic_network, actor_network

    def build_critic_optimizer(self):
        pass

    def build_actor_optimizer(self):
        one_hot_action = K.placeholder(shape=[None, self.action_size])
        advantage = K.placeholder(shape=[None, 1])
        cross_entropy = -K.log(K.sum(self.actor_network.output * one_hot_action, axis=1, keepdims=True)) * advantage
        entropy = K.sum(self.actor_network.output * K.log(self.actor_network.output), axis=1, keepdims=True)

        loss = cross_entropy + entropy
        optimizer = optimizers.Adam(lr=self.learning_rate).get_updates(loss=loss,
                                                                       params=self.actor_network.trainable_weights)
        actor_optimizer = K.function(inputs=[self.actor_network.input, one_hot_action, advantage], outputs=[],
                                     updates=optimizer)
        return actor_optimizer

    def learn(self):
        local_agents = []
        for i in range(self.threads):
            local_agents.append(A3cLocalAgent(self.action_size, self.state_size,
                                              [self.actor, self.critic], self.sess,
                                              self.optimizer, self.discount_factor,
                                              [self.summary_op, self.summary_placeholders,
                                               self.update_ops, self.summary_writer]))

        for local_agent in local_agents:
            time.sleep(1)
            local_agent.learn()

        while True:
            time.sleep(600)
            self.actor_network.save_weights('./save/actor_network.h5')


class A3cLocalAgent(A3cGlobalAgent, threading.Thread):
    def __init__(self, env_name):
        threading.Thread.__init__(self)
        self.global_agent = global_agent
        self.memory_limit = 20
        self.memory = collections.deque(maxlen=self.memory_limit)

        self.env_name = env_name

    # run 메소드 오버라이드
    def run(self):
        env = gym.make(self.env_name)
        env.render()
        step = 0

        while True:
            done = False
            dead = False

            observe = env.reset()
            next_observe = observe
            # 행동하지 않고 관찰만 하는 초반 시기 : 30초
            # 액션 = 1: 정지, 2: 왼쪽, 3: 오른쪽

            for
                state = self.pre_processing(next_observe, observe)
            history = np.stack((state, state, state, state), axis=2)
            history = np.reshape([history], (1, 84, 84, 4))
            while not done:
                action, policy = self.get(history)
                if action == 0:

    def memorize(self, state, action, reward, nest_state, done):
        self.memory.append([state, action, reward, nest_state, done])
        pass

    def calculate_every_gt(self, rewards, done):
        k_step_every_gt = np.zeros_like(rewards)
        gt = 0
        # k_step 진행 후 게임이 끝나지 않았을 때의 마지막 gt
        if not done:
            gt = self.critic_network.predict(np.float32(self.states[-1] / 255.))[0]

        for t in reversed(range(0, len(rewards))):
            gt = gt * self.discount_factor + rewards[t]
            k_step_every_gt[t] = gt
        return k_step_every_gt

    def build_local_network(self):
        input = layers.Input(shape=self.state_size, batch_shape=None)
        hidden = layers.Conv2D(filters=16, kernel_size=(8, 8), strides=(2, 2), padding='same')(input)
        hidden = layers.LeakyReLU()(hidden)
        hidden = layers.Conv2D(filters=16, kernel_size=(8, 8), strides=(2, 2), padding='same')(hidden)
        hidden = layers.LeakyReLU()(hidden)
        hidden = layers.Conv2D(filters=16, kernel_size=(8, 8), strides=(2, 2), padding='same')(hidden)
        hidden = layers.LeakyReLU()(hidden)

        # 네트워크 분리
        value = layers.Dense(units=1, activation='softmax', kernel_initializer='he_uniform')(hidden)
        policy = layers.Dense(units=self.action_size, activation='linear', kernel_initializer='he_uniform')(hidden)

        local_critic_network = models.Model(inputs=input, outputs=value)
        local_actor_network = models.Model(inputs=input, outputs=policy)

        local_critic_network._make_predict_function()
        local_actor_network._make_predict_function()

        local_critic_network.summary()
        local_actor_network.summary()

        return local_critic_network, local_actor_network

    def copy_network(self):
        self.

    def learn(self):
        every_gt = self.calculate_every_gt()

        batch_size = self.memory_limit
        states = np.zeros(shape=(batch_size, 84, 84, 4))
        for i in range(batch_size):

    @staticmethod
    # state를 흑백 화면으로 전처리
    def pre_processing(next_observe, observe):
        processed_observe = np.maximum(next_observe, observe)
        processed_observe = np.uint8(
            transform.resize(color.rgb2gray(processed_observe), (84, 84), mode='constant') * 255)
        return processed_observe


if __name__ == "__main__":
    state_size = (84, 84, 4)
    action_size = 3
    global_agent = A3cGlobalAgent(state_size, action_size)
    global_agent.learn()
