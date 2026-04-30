# config.py
import torch
import os

# GENERAL SETTINGS 
ENV_NAMES = ["BipedalWalker-v3", "Pusher-v5"] 
DEVICE = torch.device("mps")  

# TRAINING HYPERPARAMETERS
MAX_TIMESTEPS = 300000        
START_TIMESTEPS = 5000
BATCH_SIZE = 256

# EVALUATION SETTINGS 
NUM_SEEDS = 3            
EVAL_FREQ = 10000        
EVAL_EPISODES = 3        
EVAL_STEPS = MAX_TIMESTEPS // EVAL_FREQ 

# EXPERIMENTS CONFIGURATION ---

# 1. Sample Efficiency Comparison
EXP_SAMPLE_EFFICIENCY = {
    "SAC Baseline": {"N": 2, "M": 2, "G": 1, "color": "red"},
    "REDQ Standard": {"N": 10, "M": 2, "G": 20, "color": "blue"}
}

# 2. Ablation Study: UTD Ratio (G) Effect
EXP_ABLATION_G = {
    "REDQ (G=1)": {"N": 10, "M": 2, "G": 1, "color": "green"},
    "REDQ (G=10)": {"N": 10, "M": 2, "G": 10, "color": "orange"},
    "REDQ Standard (G=20)": {"N": 10, "M": 2, "G": 20, "color": "blue"}
}

# 3. Robustness and Stability: The Ensemble (N)
EXP_ROBUSTNESS_N = {
    "REDQ (N=2) Unstable": {"N": 2, "M": 2, "G": 20, "color": "purple"},
    "REDQ Standard (N=10)": {"N": 10, "M": 2, "G": 20, "color": "blue"}
}

# 4. Overestimation Analysis: The Subset (M)
EXP_OVERESTIMATION_M = {
    "REDQ Standard (M=2)": {"N": 10, "M": 2, "G": 20, "color": "blue"},
    "REDQ (M=5) Conservative": {"N": 10, "M": 5, "G": 20, "color": "brown"}
}


# PATHS
os.makedirs("videos", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)