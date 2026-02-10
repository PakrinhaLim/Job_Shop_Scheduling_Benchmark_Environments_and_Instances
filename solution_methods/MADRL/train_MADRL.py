import os
import sys
import logging
import torch
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from solution_methods.helper_functions import load_parameters
from solution_methods.MADRL.src.madrl_env import MADRL_Env
from solution_methods.MADRL.src.ppo_agent import PPOAgent, Memory

logging.basicConfig(level=logging.INFO)

def train_madrl():
    # Load config from MADRL.toml
    param_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../configs/MADRL.toml"))
    parameters = load_parameters(param_file)
    
    # Initialize Environment
    env = MADRL_Env(parameters, mode='train')
    
    # Initialize Agent
    obs_dim = env.obs_dim
    action_dim = env.max_queue_size
    
    madrl_params = parameters['madrl']
    agent = PPOAgent(
        obs_dim, 
        action_dim, 
        lr=madrl_params['lr'], 
        gamma=madrl_params['gamma'], 
        eps_clip=madrl_params['eps_clip'], 
        K_epochs=madrl_params['k_epochs']
    )
    
    memory = Memory()
    
    train_params = parameters['train_params']
    max_episodes = train_params['max_episodes']
    max_timesteps = train_params['max_timesteps']
    update_timestep = train_params['update_timestep']
    
    timestep = 0
    
    logging.info(f"Starting training on {parameters['train']['problem_instance']}")
    logging.info(f"Agents: {env.num_machines}, Obs Dim: {obs_dim}, Action Dim: {action_dim}")

    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        current_ep_reward = 0
        
        for t in range(max_timesteps):
            timestep += 1
            
            # Select action for each machine
            actions = {}
            for machine_id in range(env.num_machines):
                machine_obs = state[machine_id]
                # Check if machine has any task in queue (obs not all zero)
                if np.sum(np.abs(machine_obs)) > 0:
                    action = agent.select_action(machine_obs, memory)
                    actions[machine_id] = action
            
            # Step environment
            next_state, rewards, dones, _ = env.step(actions)
            
            # Saving reward and is_terminals
            # Since we have one memory for shared agent, we push rewards for each active machine
            # NOTE: simplified mapping. In robust impl, we need to map memory index to machine
            # Here we assume sequential pushes match sequential appends in memory
            for machine_id in actions.keys():
                memory.rewards.append(rewards[machine_id])
                memory.is_terminals.append(dones[machine_id])
                current_ep_reward += rewards[machine_id]
            
            state = next_state
            
            # Update
            if timestep % update_timestep == 0:
                logging.info(f"Updating PPO agent at timestep {timestep}")
                agent.update(memory)
                memory.clear_memory()
                
            if all(dones.values()):
                break
                
        logging.info(f"Episode {i_episode} finished. Total Reward: {current_ep_reward}")
        
    # Save model
    save_path = "results/MADRL"
    os.makedirs(save_path, exist_ok=True)
    torch.save(agent.policy.state_dict(), os.path.join(save_path, "ppo_madrl.pth"))
    logging.info(f"Model saved to {save_path}")

if __name__ == '__main__':
    train_madrl()
