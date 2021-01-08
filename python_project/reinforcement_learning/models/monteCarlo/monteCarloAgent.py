import collections
import numpy as np
import reinforcementProject.monteCarlo.environment as environment

class MonteCarloAgent:
    def __init__(self):
        self.actions = [0, 1, 2, 3]
        self.action_size = len(self.actions)
        self.epsilon = 0.999
        self.epsilon_discount_factor = 0.999
        self.epsilon_min = 0.001

        self.learning_rate = 0.001
        self.q_table = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        self.discount_factor = 0.99
