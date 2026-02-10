import os
import sys
import logging
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from solution_methods.helper_functions import load_parameters
from solution_methods.MADRL.src.madrl_env import MADRL_Env
from solution_methods.MADRL.src.ppo_agent import PPOAgent, Memory
from visualization import gantt_chart

logging.basicConfig(level=logging.INFO)

def run_madrl():
    param_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../configs/MADRL.toml"))
    parameters = load_parameters(param_file)
    
    env = MADRL_Env(parameters, mode='test')
    obs_dim = env.obs_dim
    action_dim = env.max_queue_size
    
    agent = PPOAgent(obs_dim, action_dim)
    
    # Load model
    model_path = "results/MADRL/ppo_madrl.pth"
    if os.path.exists(model_path):
        agent.policy.load_state_dict(torch.load(model_path))
        logging.info("Loaded trained model.")
    else:
        logging.warning("No trained model found, using random weights.")
    
    state = env.reset()
    memory = Memory() # Dummy memory
    
    done = False
    steps = 0
    while not done and steps < 1000:
        actions = {}
        for machine_id in range(env.num_machines):
            machine_obs = state[machine_id]
            if np.sum(np.abs(machine_obs)) > 0:
                # Greedy action for inference
                state_tensor = torch.FloatTensor(machine_obs).unsqueeze(0)
                mask = (torch.sum(torch.abs(state_tensor), dim=2) > 0).float()
                probs = agent.policy.get_action_prob(state_tensor, mask)
                action = torch.argmax(probs).item()
                actions[machine_id] = action
        
        state, rewards, dones, _ = env.step(actions)
        done = all(dones.values())
        steps += 1
        
    logging.info(f"Finished. Makespan: {env.jobShopEnv.makespan}")
    
    # Plot Gantt Chart
    logging.info("Generating Gantt chart...")
    plt = gantt_chart.plot(env.jobShopEnv)
    plt.show()

if __name__ == '__main__':
    run_madrl()
