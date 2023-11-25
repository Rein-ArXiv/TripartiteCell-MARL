import argparse
from cell_env import Cellenv
from agent import DQNAgent

parser = argparse.ArgumentParser(description='Tripartite Cell parser')

if __name__ == "__main__":
    env = Cellenv(max_time=100, islet_num=20)
    agent = DQNAgent(env, log_prefix="1_islet_1000_epi")
    agent.test(param_log_prefix="1_islet_1000_epi", share=True, param_location=None, test_log_prefix="1_islet_1000_epi", log_location=None, max_epi=1000)