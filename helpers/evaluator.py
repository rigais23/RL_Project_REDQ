import torch
import gymnasium as gym 

def evaluate_policy_and_bias(env_name, agent, eval_episodes):
    eval_env = gym.make(env_name)
    avg_reward = 0.0
    avg_bias = 0.0
    
    for _ in range(eval_episodes):
        state, _ = eval_env.reset()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            action_tensor, _ = agent.actor.sample(state_tensor)
            all_q_values = agent.critic(state_tensor, action_tensor)
            q_pred = all_q_values.mean().item()

        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
        avg_reward += episode_reward
        avg_bias += (q_pred - episode_reward)
        
    eval_env.close()
    return avg_reward / eval_episodes, avg_bias / eval_episodes