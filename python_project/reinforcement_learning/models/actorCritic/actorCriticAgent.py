import copy
import numpy as np
import matplotlib.pyplot as plt
import keras.models as models
import keras.layers as layers
import keras.optimizers as optimizers
import keras.backend as K
import gym


class ActorCriticAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.learning_rate = 0.001
        self.critic_network = self.build_critic_network()
        self.actor_network = self.build_actor_network()
        self.actor_optimizer = self.build_actor_optimizer()
        self.discount_factor = 0.99

        self.load_networks = True
        if self.load_networks:
            self.critic_network.load_weights('./save/critic_network.h5')
            self.actor_network.load_weights('./save/actor_network.h5')

    def build_critic_network(self):
        critic_network = models.Sequential()
        critic_network.add(
            layers.Dense(units=16, activation='linear', kernel_initializer='he_uniform', input_dim=self.state_size))
        critic_network.add(layers.LeakyReLU())
        critic_network.add(layers.Dense(units=16, activation='linear', kernel_initializer='he_uniform'))
        critic_network.add(layers.LeakyReLU())
        critic_network.add(layers.Dense(units=16, activation='linear', kernel_initializer='he_uniform'))
        critic_network.add(layers.LeakyReLU())
        critic_network.add(layers.Dense(units=1, activation='linear', kernel_initializer='he_uniform'))
        critic_network.summary()
        critic_network.compile(optimizer=optimizers.Adam(lr=self.learning_rate), loss='mse')
        return critic_network

    def build_actor_network(self):
        actor_network = models.Sequential()
        actor_network.add(
            layers.Dense(units=16, activation='linear', kernel_initializer='he_uniform', input_dim=self.state_size))
        actor_network.add(layers.LeakyReLU())
        actor_network.add(layers.Dense(units=16, activation='linear', kernel_initializer='he_uniform'))
        actor_network.add(layers.LeakyReLU())
        actor_network.add(layers.Dense(units=16, activation='linear', kernel_initializer='he_uniform'))
        actor_network.add(layers.LeakyReLU())
        actor_network.add(layers.Dense(units=self.action_size, activation='softmax', kernel_initializer='he_uniform'))
        actor_network.summary()
        return actor_network

    def build_actor_optimizer(self):
        one_hot_action = K.placeholder(shape=[None, self.action_size])
        advantage = K.placeholder(shape=[None, 1])
        loss = -K.log(K.sum(self.actor_network.output * one_hot_action, axis=1, keepdims=True)) * advantage
        optimizer = optimizers.Adam(lr=self.learning_rate).get_updates(loss=loss,
                                                                       params=self.actor_network.trainable_weights)
        actor_optimizer = K.function(inputs=[self.actor_network.input, one_hot_action, advantage], outputs=[],
                                     updates=optimizer)
        return actor_optimizer

    def get_action(self, state):
        policy = self.actor_network.predict(test_input=state, batch_size=1, verbose=0)[0]
        # size를 사용하면 배열이 리턴된다
        action = np.random.choice(a=self.action_size, size=None, p=policy)
        return action

    def learn(self, state, action, reward, next_state, done):
        value = self.critic_network.predict(state)[0][0]
        next_value = self.critic_network.predict(next_state)[0][0]
        one_hot_action = np.zeros([1, self.action_size])
        one_hot_action[0][action] = 1

        if not done:
            target_value = reward + self.discount_factor * next_value
            advantage = reward + self.discount_factor * next_value - value
        else:
            target_value = reward
            advantage = reward - value
        target_value = np.reshape(target_value, [1, 1])
        self.critic_network.fit(x=state, y=target_value, batch_size=1, epochs=1, verbose=0)
        self.actor_optimizer([state, one_hot_action, advantage])



if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = ActorCriticAgent(state_size, action_size)
    scores = []

    for episode in range(1000):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        score = 0

        done = False
        while not done:
            env.render()
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            if score < 499 and done:
                reward = -100
            score += reward
            agent.learn(state, action, reward, next_state, done)
            state = copy.deepcopy(next_state)

        if score != 500:
            score += 100
        scores.append(score)
        print("episode : " + str(episode + 1) + " epsilon : " + " score : " + str(score))

        if len(scores) >= 10 and np.mean(scores[-10:]) >= 499:
            plt.plot(np.squeeze(scores))
            plt.ylabel('scores')
            plt.xlabel('episodes')
            plt.show()
            agent.actor_network.save_weights('./save/actor_network.h5')
            agent.critic_network.save_weights('./save/critic_network.h5')
            break
