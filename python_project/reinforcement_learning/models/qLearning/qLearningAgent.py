import copy
import collections
import numpy as np
import reinforcement_learning_ml_training_server.qLearning.environment as environment


class QLearningAgent:
    def __init__(self):
        self.actions = [0, 1, 2, 3]
        self.action_size = len(self.actions)
        self.epsilon = 0.999
        self.epsilon_discount_factor = 0.999
        self.epsilon_min = 0.001

        self.learning_rate = 0.001
        self.q_table = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        self.discount_factor = 0.99

    @staticmethod
    def arg_max(every_q):
        max_indexes = []
        max_q = every_q[0]
        for index, q in enumerate(every_q):
            if q > max_q:
                max_q = q
                max_indexes.clear()
                max_indexes.append(index)
            elif q == max_q:
                max_indexes.append(index)
        return np.random.choice(max_indexes)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_size)
        else:
            every_q = self.q_table[state]
            action = self.arg_max(every_q)
        return action

    def learn(self, state, action, reward, next_state, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_discount_factor

        if not done:
            q = self.q_table[state][action]
            target_q = np.amax(self.q_table[next_state])
            q = q + self.learning_rate * (reward + self.discount_factor * target_q - q)
        else:
            q = reward
        self.q_table[state][action] = q


if __name__ == '__main__':
    env = environment.Env()
    agent = QLearningAgent()

    for episode in range(100):
        state = env.reset()
        state = str(state)

        done = False
        while not done:
            env.render()
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = str(next_state)
            agent.learn(state, action, reward, next_state, done)
            state = copy.deepcopy(next_state)
            env.print_value_all(agent.q_table)
        print("episode : " + str(episode + 1) + ", epsilon : " + str(agent.epsilon))
