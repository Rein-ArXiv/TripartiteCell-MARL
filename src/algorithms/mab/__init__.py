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



policy = UCB(1)
bandit = GaussianBandit(9)
mab = MAB(env, Agent, bandit, policy, is_train=True, gamma=0.95)
mab.train(max_epi=2000)


print(mab.alpha_agent._value_estimates)
print(mab.alpha_agent.action_attempts)
print(mab.alpha_agent._value_estimates.argmax())

print(mab.beta_agent._value_estimates)
print(mab.beta_agent.action_attempts)
print(mab.beta_agent._value_estimates.argmax())

print(mab.delta_agent._value_estimates)
print(mab.delta_agent.action_attempts)
print(mab.delta_agent._value_estimates.argmax())


policy = Softmax()
policy = StaticEpsilonGreedy(0.5)

policy = UCB(1)

policy = DecayEpsilonGreedy(eps_linear=False)
policy = Softmax


'''