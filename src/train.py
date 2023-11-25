import argparse
from cell_env import Cellenv
from agent import DQNAgent

parser = argparse.ArgumentParser(description='Tripartite Cell parser')

if __name__ == "__main__":
    env = Cellenv(max_time=200, islet_num=20)
    agent = DQNAgent(env, log_prefix="20_clip4")
    agent.train(max_frame=int(7e6), max_epi=1000)
    