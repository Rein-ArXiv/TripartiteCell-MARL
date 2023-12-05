import os
import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, env, is_train=None, cuda=False, cuda_num=None):
        self.env = env
