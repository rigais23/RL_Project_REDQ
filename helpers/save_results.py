import pandas as pd
import os
import numpy as np


def save_raw_logs(returns_dict, times_dict, biases_dict, exp_config, env_name, max_steps, eval_freq, exp_name):
    os.makedirs("logs", exist_ok=True)
    
    timesteps_axis = np.arange(0, max_steps, eval_freq)
    
    for algo in exp_config.keys():
        safe_algo = algo.replace(" ", "_")
        df_logs = pd.DataFrame({
            'Timesteps': timesteps_axis,
            'Mean_Return': returns_dict[algo].mean(axis=0),
            'Std_Return': returns_dict[algo].std(axis=0),
            'Mean_Bias': biases_dict[algo].mean(axis=0),
            'Std_Bias': biases_dict[algo].std(axis=0)
        })
        filename = f"logs/{exp_name}_{env_name}_{safe_algo}_metrics.csv"
        df_logs.to_csv(filename, index=False)
        print(f"Saved: {filename}")

    times_data = []
    for algo in exp_config.keys():
        avg_time_sec = np.mean(times_dict[algo])
        std_time_sec = np.std(times_dict[algo])
        times_data.append({
            'Algorithm': algo,
            'Avg_Time_Minutes': avg_time_sec / 60,
            'Std_Time_Minutes': std_time_sec / 60
        })
        
    df_times = pd.DataFrame(times_data)
    times_filename = f"logs/{exp_name}_{env_name}_execution_times.csv"
    df_times.to_csv(times_filename, index=False)
    print(f"Saved: {times_filename}")