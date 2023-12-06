from .agent import Agent, GradientAgent, BetaAgent
from .bandit import GaussianBandit, BinomialBandit, BernoulliBandit
from .policy import StaticEpsilonGreedy, UCB, Softmax

'''
from envs.env import CellEnv
from algorithms.mab.mab import MAB
from algorithms.mab.agent import *
from algorithms.mab.policy import *
from algorithms.mab.bandit import *
env = CellEnv(reward_mode="local", max_time=50)
bandit = GaussianBandit(9)
policy = StaticEpsilonGreedy()
mab = MAB(env, Agent, bandit, policy, is_train=True)
mab.train()


policy = DecayEpsilonGreedy(eps_linear=False)
policy = UCB(2)
policy = Softmax

print(mab.alpha_agent._value_estimates)
print(mab.alpha_agent.action_attempts)

print(mab.beta_agent._value_estimates)
print(mab.beta_agent.action_attempts)

print(mab.delta_agent._value_estimates)
print(mab.delta_agent.action_attempts)

'''