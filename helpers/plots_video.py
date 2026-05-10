import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
sns.set_theme(style="darkgrid", context="paper", font_scale=1.4)
os.makedirs('figures_report', exist_ok=True)

def load_norm(path, algo_name=None):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    df = pd.read_csv(path)
    rename_map = {
        'Timesteps':   'Timestep',
        'Mean_Return': 'Return',
        'Mean_Bias':   'Bias',
        'Std_Return':  'Std_Return',
        'Std_Bias':    'Std_Bias',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if algo_name is not None:
        df['Algorithm'] = algo_name
    return df


def plot_final_report(df_list, y_col, title, filename, palette, xlim=None):
    plt.figure(figsize=(10, 6))
    valid = [d for d in df_list if d is not None]
    if not valid:
        print(f"No data to plot for '{filename}' – skipping.")
        return

    full_df = pd.concat(valid)

    for algo in full_df['Algorithm'].unique():
        subset = full_df[full_df['Algorithm'] == algo].copy()
        color  = palette.get(algo) if isinstance(palette, dict) else None

        if f'Std_{y_col}' in subset.columns:
            subset = subset.sort_values('Timestep')
            subset = subset.groupby('Timestep', as_index=False).mean(numeric_only=True)
            plt.plot(subset['Timestep'], subset[y_col],
                     label=algo, color=color, linewidth=2.5)
            plt.fill_between(
                subset['Timestep'],
                subset[y_col] - subset[f'Std_{y_col}'],
                subset[y_col] + subset[f'Std_{y_col}'],
                color=color, alpha=0.15
            )
        else:
            sns.lineplot(data=subset, x='Timestep', y=y_col,
                         label=algo, errorbar='sd',
                         linewidth=2.5, color=color)

    plt.title(title, pad=20, fontweight='bold', fontsize=16)
    plt.xlabel('Environment Timesteps', fontsize=14)
    plt.ylabel(f'Average {y_col}', fontsize=14)
    plt.legend(loc='best')
    if xlim is not None:
        plt.xlim(0, xlim)
    plt.tight_layout()
    plt.savefig(f'figures_report/{filename}', dpi=300)
    plt.close()
    print(f"Generated: {filename}")


def record_agent_video(env_name, agent, algo_name, exp_name, seed=0):
    video_env = gym.make(env_name, render_mode="rgb_array")

    safe_algo_name = algo_name.replace(" ", "_")
    folder_path    = f"videos/{exp_name}_{env_name}_{safe_algo_name}"

    video_env = RecordVideo(video_env, video_folder=folder_path,
                            episode_trigger=lambda x: True)

    state, _ = video_env.reset(seed=seed)
    done = False

    while not done:
        action = agent.select_action(state)
        state, _, terminated, truncated, _ = video_env.step(action)
        done = terminated or truncated

    video_env.close()
    print(f"Video saved in: {folder_path}")


def plot_experiment_results(returns_dict, biases_dict, exp_config,
                             env_name, max_steps, eval_freq,
                             num_seeds, exp_name):

    timesteps_axis = np.arange(0, max_steps, eval_freq)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for algo_name, config in exp_config.items():
        color = config["color"]

        returns = returns_dict[algo_name]
        biases  = biases_dict[algo_name]

        mean_ret,  std_ret  = returns.mean(axis=0), returns.std(axis=0)
        mean_bias, std_bias = biases.mean(axis=0),  biases.std(axis=0)

        ax1.plot(timesteps_axis, mean_ret, label=algo_name, color=color, linewidth=2)
        ax1.fill_between(timesteps_axis, mean_ret - std_ret,
                         mean_ret + std_ret, color=color, alpha=0.2)

        ax2.plot(timesteps_axis, mean_bias, label=algo_name, color=color, linewidth=2)
        ax2.fill_between(timesteps_axis, mean_bias - std_bias,
                         mean_bias + std_bias, color=color, alpha=0.2)

    ax1.set_title(f"[{exp_name}] Return ({num_seeds} Seeds) - {env_name}")
    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Return")
    if "BipedalWalker" in env_name:
        ax1.axhline(y=200, color='green', linestyle='--', alpha=0.5,
                    label='Solved threshold')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title(f"[{exp_name}] Bias - {env_name}")
    ax2.set_xlabel("Timesteps")
    ax2.set_ylabel("Bias (Q_pred - Real Return)")
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.8)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    filename = f"results/{exp_name}_{env_name}_plot.png"
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"Plots saved as: {filename}")