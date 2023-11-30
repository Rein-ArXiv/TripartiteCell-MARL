import os, sys
sys.path.append(os.pardir)

import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch

from envs.env import CellEnv
from algorithms.iql.agent import IQL

parser = argparse.ArgumentParser(description="Q value analysis")
parser.add_argument('')

if __name__=="__main__":

	env = CellEnv()