from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

def record_agent_video(env_name, agent, algo_name, exp_name, seed=0):
    """Genera y guarda un vídeo del agente interactuando con el entorno."""
    video_env = gym.make(env_name, render_mode="rgb_array")
    
    # Añadimos exp_name a la ruta de la carpeta
    safe_algo_name = algo_name.replace(" ", "_")
    folder_path = f"videos/{exp_name}_{env_name}_{safe_algo_name}"
    
    video_env = RecordVideo(video_env, video_folder=folder_path, episode_trigger=lambda x: True)
    
    state, _ = video_env.reset(seed=seed)
    done = False
    
    while not done:
        action = agent.select_action(state)
        state, _, terminated, truncated, _ = video_env.step(action)
        done = terminated or truncated
        
    video_env.close()
    print(f"Video saved in: {folder_path}")


def plot_experiment_results(returns_dict, biases_dict, exp_config, env_name, max_steps, eval_freq, num_seeds, exp_name):
    """Genera, muestra y guarda las gráficas comparativas del paper."""
    timesteps_axis = np.arange(0, max_steps, eval_freq)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for algo_name, config in exp_config.items():
        color = config["color"]
        
        returns = returns_dict[algo_name]
        biases = biases_dict[algo_name]
        
        mean_ret, std_ret = returns.mean(axis=0), returns.std(axis=0)
        mean_bias, std_bias = biases.mean(axis=0), biases.std(axis=0)
        
        ax1.plot(timesteps_axis, mean_ret, label=algo_name, color=color, linewidth=2)
        ax1.fill_between(timesteps_axis, mean_ret - std_ret, mean_ret + std_ret, color=color, alpha=0.2)
        
        ax2.plot(timesteps_axis, mean_bias, label=algo_name, color=color, linewidth=2)
        ax2.fill_between(timesteps_axis, mean_bias - std_bias, mean_bias + std_bias, color=color, alpha=0.2)

    ax1.set_title(f"[{exp_name}] Return ({num_seeds} Seeds) - {env_name}")
    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Return")
    if "BipedalWalker" in env_name: 
        ax1.axhline(y=200, color='green', linestyle='--', alpha=0.5, label='Umbral "Resuelto"')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title(f"[{exp_name}] Bias - {env_name}")
    ax2.set_xlabel("Timesteps")
    ax2.set_ylabel("Bias (Q_pred - Real Return)")
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.8)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    # Añadimos exp_name al nombre del PNG
    filename = f"results/{exp_name}_{env_name}_plot.png"
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"Plots saved as: {filename}")