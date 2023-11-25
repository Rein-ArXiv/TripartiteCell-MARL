from env import CellEnv
from gymnasium.envs.registration import register

register(
	id = "CellEnv-v0",
	entry_point = "env:CellEnv"
	)