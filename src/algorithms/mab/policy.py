import numpy as np

class Policy(object):
    def __str__(self):
        return "generic policy"

    def choose(self, agent):
        return 0

class StaticEpsilonGreedy(Policy):
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def __str__(self):
        return f'Static \u03B5-greedy (\u03B5={self.epsilon})'

    def choose(self, agent):
        if np.random.random() < self.epsilon:
            return np.random.choice(len(agent.value_estimates))

        else:
            action = np.argmax(agent.value_estimates)
            check = np.where(agent.value_estimates == agent.value_estimates[action])[0]

            if len(check) == 1: # True
                return action
            else:
                return np.random.choice(check)

class DecayEpsilonGreedy(Policy):
    def __init__(self, eps_linear):
        self.eps_linear = eps_linear
        self.max_eps = 1.0
        self.min_eps = 0.01
        self.epsilon = self.max_eps

    def __str__(self):
        if self.eps_linear:
            return f'Decay \u03B5-greedy (Decay=linear)'
        else:
            return f'Decay \u03B5-greedy (Decay=exponential)'

    def choose(self, agent):
        self.epsilon -= 1e-5

        if np.random.random() < self.epsilon:
            return np.random.choice(len(agent.value_estimates))

        else:
            action = np.argmax(agent.value_estimates)
            check = np.where(agent.value_estimates == agent.value_estimates[action])[0]

            if len(check) == 1:
                return action
            else:
                return np.random.choice(check)

class UCB(Policy):
    def __init__(self, c):
        self.c = c

    def __str__(self):
        return f'UCB (c={self.c})'

    def choose(self, agent):
        exploration = np.log(agent.t+1) / agent.action_attempts
        exploration[np.isnan(exploration)] = 0
        exploration = np.power(exploration, 1/self.c)

        q = agent.value_estimates + exploration
        action = np.argmax(q)
        check = np.where(q == q[action])[0]

        if len(check) == 1:
            return action
        else:
            return np.random.choice(check)

class Softmax(Policy):
    def __str__(self):
        return 'SoftMax'

    def choose(self, agent):
        a = agent.value_estimates
        pi = np.exp(a) / np.sum(np.exp(a))
        cdf = np.cumsum(pi)
        s = np.random.random()
        return np.where(s < cdf)[0][0]