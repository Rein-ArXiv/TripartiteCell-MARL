# TripartiteCell-MARL

A Multi-Agent Reinforcement Learning (MARL) project for glucose homeostasis modeling using tripartite cell dynamics.

## Overview

This project implements reinforcement learning algorithms to model glucose homeostasis through the interaction of three types of pancreatic cells: alpha cells (glucagon-secreting), beta cells (insulin-secreting), and delta cells (somatostatin-secreting). The system uses multi-agent reinforcement learning to simulate and optimize glucose regulation in pancreatic islets.

## Project Structure

```
TripartiteCell-MARL/
├── src/
│   ├── algorithms/
│   │   ├── iql/               # Independent Q-Learning implementation
│   │   │   ├── agent.py       # IQL agent implementation
│   │   │   ├── memory.py      # Replay buffer
│   │   │   └── model.py       # Neural network models (DDQN)
│   │   └── mab/               # Multi-Armed Bandit implementation
│   │       ├── agent.py       # MAB agents
│   │       ├── bandit.py      # Bandit environments
│   │       ├── mab.py         # MAB main implementation
│   │       └── policy.py      # MAB policies (ε-greedy, UCB, Softmax)
│   ├── analysis/              # Experimental analysis tools
│   │   ├── analysis_main.py   # Main analysis script
│   │   └── experiments.py     # Experiment utilities
│   ├── envs/
│   │   └── env.py             # CellEnv - Gymnasium environment for cell simulation
│   ├── config.py              # Configuration settings
│   ├── main.py                # Main training/testing script
│   └── util.py                # Utility functions
├── parameters/                # Model parameters storage
├── image/                     # Generated plots and visualizations
├── papers/                    # Research papers and references
└── Reinforcement learning of glucose homeostasis.hwpx
```

## Features

### Algorithms
- **Independent Q-Learning (IQL)**: Deep Double Q-Network (DDQN) implementation for multi-agent learning
- **Multi-Armed Bandit (MAB)**: Various bandit algorithms with different policies

### Environment
- **CellEnv**: Custom Gymnasium environment simulating pancreatic islet behavior
- Configurable number of islets (default: 20)
- Adjustable episode duration (default: 200 minutes)
- Multiple reward modes: local (individual hormone secretion) or global (total hormone secretion)

### Cell Types
- **Alpha cells**: Glucagon secretion for glucose elevation
- **Beta cells**: Insulin secretion for glucose reduction  
- **Delta cells**: Somatostatin secretion for hormone regulation

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TripartiteCell-MARL
```

2. Install required dependencies:
```bash
pip install torch numpy matplotlib gymnasium wandb pymc tqdm
```

## Usage

### Training

Train a DDQN model:
```bash
python src/main.py --model DDQN --train --islet_num 20 --max_time 200 --reward_mode local
```

Train with specific hyperparameters:
```bash
python src/main.py --model DDQN --train \
    --batch_size 256 \
    --learning_rate 1e-4 \
    --gamma 0.95 \
    --max_epi 1000 \
    --memory_cap 200000
```

### Testing

Test a trained model:
```bash
python src/main.py --model DDQN --param_loc ../parameters/iql --plot --action_view
```

Test with fixed glucose level:
```bash
python src/main.py --model DDQN --glu_fix --glu_level 5.0 --plot
```

### Multi-Armed Bandit

Run MAB experiments:
```bash
python src/main.py --model MAB --islet_num 20 --max_time 200
```

## Command Line Arguments

### Environment Parameters
- `--islet_num`: Number of environment islets (default: 20)
- `--max_time`: Maximum time per episode in minutes (default: 200)
- `--reward_mode`: Reward mode - 'local' or 'global'

### Model Selection
- `--model`: Choose algorithm - 'DDQN' or 'MAB'

### DDQN Training Parameters
- `--train`: Enable training mode
- `--batch_size`: Training batch size (default: 256)
- `--learning_rate`: Optimizer learning rate (default: 1e-4)
- `--gamma`: Reward discount factor (default: 0.95)
- `--max_epi`: Maximum training episodes (default: 1000)
- `--memory_cap`: Replay buffer capacity (default: 200,000)
- `--eps_linear`: Use linear epsilon decay
- `--eps_decay`: Epsilon decay rate

### Testing Parameters
- `--glu_fix`: Use fixed initial glucose level
- `--glu_level`: Initial glucose level value
- `--plot`: Enable result plotting
- `--plot_dir`: Plot save directory (default: ../image)
- `--action_view`: Display actions in terminal
- `--action_share`: Enable action sharing analysis

### Hardware
- `--cuda`: Enable CUDA acceleration
- `--cuda_num`: CUDA device number (default: 0)

## Research Context

This project models glucose homeostasis as a multi-agent reinforcement learning problem, where each pancreatic cell type acts as an independent agent learning optimal hormone secretion policies. The approach aims to understand and optimize glucose regulation mechanisms through computational modeling.

## Output

- **Training**: Model parameters saved to `parameters/` directory
- **Testing**: Plots and visualizations saved to `image/` directory
- **Analysis**: Experimental results and metrics

## Dependencies

- PyTorch
- NumPy
- Matplotlib
- Gymnasium
- Weights & Biases (wandb)
- PyMC
- tqdm

## License

This project is part of research on reinforcement learning applications in glucose homeostasis modeling.