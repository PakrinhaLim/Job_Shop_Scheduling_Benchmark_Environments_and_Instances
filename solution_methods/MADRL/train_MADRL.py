
import argparse
import logging
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# Setup path before project imports
base_path = Path(__file__).resolve().parents[2]
sys.path.append(str(base_path))

from solution_methods.DANIEL.src.common_utils import setup_seed, strToSuffix
from solution_methods.DANIEL.src.data_utils import CaseGenerator, SD2_instance_generator, load_data_from_files
from solution_methods.MADRL.src.madrl_env import MADRL_FJSPEnv
from solution_methods.MADRL.network.ppo_madrl import PPO_initialize, Memory
from solution_methods.MADRL.utils import get_objective_folder
from solution_methods.helper_functions import load_parameters, initialize_device

PARAM_FILE = str(base_path) + "/configs/MADRL.toml"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def sample_action_madrl(pi):
    # pi: [sz_b, J, M]
    # Return actions: [sz_b, M]
    # Return log_probs: [sz_b, M]
    
    sz_b, J, M = pi.size()
    
    # Check for NaNs (all masked)
    # If sum(pi, dim=1) is NaN or 0 (all -inf led to 0 after softmask? No softmax handles -inf -> 0? No, all -inf -> NaN)
    # Actually F.softmax with all -inf produces NaNs.
    
    # Detect invalid machines
    invalid_mask = torch.isnan(pi).any(dim=1) # [sz_b, M]
    
    # Replace NaNs with uniform to allow sampling (action will be ignored later)
    # Or just mask pi
    pi[torch.isnan(pi)] = 1.0/J
    
    # Sample
    dist = torch.distributions.Categorical(pi.permute(0, 2, 1)) # [sz_b, M, J]
    actions = dist.sample() # [sz_b, M]
    log_probs = dist.log_prob(actions) # [sz_b, M]
    
    # Apply invalid mask
    actions[invalid_mask] = -1
    log_probs[invalid_mask] = 0.0
    
    return actions, log_probs

class Trainer:
    def __init__(self, config, device):
        self.n_j = config["env"]["n_j"]
        self.n_m = config["env"]["n_m"]
        self.low = config["env"]["low"]
        self.high = config["env"]["high"]
        self.config = config
        self.max_updates = config["PPO_Algorithm"]["max_updates"]
        self.reset_env_timestep = config["training"]["reset_env_timestep"]
        self.validate_timestep = config["training"]["validate_timestep"]
        self.num_envs = config["PPO_Algorithm"]["num_envs"]
        self.device = device
        
        # Data Setup
        self.data_source = config["data"]["source"] # SD2
        self.op_per_job_min = int(0.8 * self.n_m)
        self.op_per_job_max = int(1.2 * self.n_m)
        
        self.script_dir = Path(__file__).resolve().parent
        if not os.path.exists(f"{self.script_dir}/save/{self.data_source}"):
            os.makedirs(f"{self.script_dir}/save/{self.data_source}")
        if not os.path.exists(f"{self.script_dir}/train_log/{self.data_source}"):
             os.makedirs(f"{self.script_dir}/train_log/{self.data_source}")

        if device.type == "cuda":
            torch.set_default_dtype(torch.float32)
            torch.set_default_device("cuda")
        else:
            torch.set_default_dtype(torch.float32)
            torch.set_default_device("cpu")

        self.data_name = f'{self.n_j}x{self.n_m}{strToSuffix(config["data"]["suffix"])}'
        self.model_name = f'{self.data_name}{strToSuffix(config["model"]["suffix"])}'
        
        self.seed_train = config["seed"]["seed_train"]
        setup_seed(self.seed_train)

        # Env
        self.env = MADRL_FJSPEnv(self.n_j, self.n_m, device, config=self.config)
        
        self.ppo = PPO_initialize(config)
        self.memory = Memory(
            gamma=config["PPO_Algorithm"]["gamma"],
            gae_lambda=config["PPO_Algorithm"]["gae_lambda"],
        )

    def train(self):
        setup_seed(self.seed_train)
        self.log = []
        self.train_st = time.time()
        
        print("Starting MADRL Training...")

        for i_update in tqdm(range(self.max_updates), file=sys.stdout, desc="progress"):
            
            # Reset / Resample Data
            if i_update % self.reset_env_timestep == 0:
                dataset_job_length, dataset_op_pt = self.sample_training_instances()
                state = self.env.set_initial_data(dataset_job_length, dataset_op_pt)
            else:
                state = self.env.reset()
                
            ep_rewards = -deepcopy(self.env.init_quality)
            
            while True:
                self.memory.push(state)
                with torch.no_grad():
                    pi_envs, val_envs = self.ppo.policy_old(
                        fea_j=state.fea_j_tensor,
                        op_mask=state.op_mask_tensor,
                        candidate=state.candidate_tensor,
                        fea_m=state.fea_m_tensor,
                        mch_mask=state.mch_mask_tensor,
                        comp_idx=state.comp_idx_tensor,
                        dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                        fea_pairs=state.fea_pairs_tensor,
                    )
                
                # Sample Actions (MADRL)
                # pi_envs: [sz_b, J, M]
                actions, log_probs = sample_action_madrl(pi_envs)
                
                # Step
                state, reward, done = self.env.step_madrl(actions.cpu().numpy())
                
                ep_rewards += reward
                reward_tensor = torch.from_numpy(reward).to(self.device)
                
                self.memory.done_seq.append(torch.from_numpy(done).to(self.device))
                self.memory.reward_seq.append(reward_tensor)
                self.memory.action_seq.append(actions)
                self.memory.log_probs.append(log_probs)
                self.memory.val_seq.append(val_envs.squeeze(1))
                
                if done.all():
                    break
            
            # Update PPO
            loss, v_loss = self.ppo.update(self.memory)
            self.memory.clear_memory()
            
            # Update rewards and log energy info
            mean_rewards = np.mean(ep_rewards)
            mean_energy = np.mean(self.env.energy_consumed) # Energy consumed in the last rollout
            self.log.append([i_update, mean_rewards, mean_energy])
            
            if (i_update + 1) % 10 == 0:
                 tqdm.write(f"Episode {i_update+1}: Reward {mean_rewards:.2f} Energy {mean_energy:.2f} Loss {loss:.4f}")
                 self.save_model()

        self.save_training_log()

    def sample_training_instances(self):
        # Uses DANIEL's data generation logic
        # Either CaseGenerator (SD1) or SD2_instance_generator
        
        # Simplified for SD2 as default in config
        prepare_JobLength = [
            random.randint(self.op_per_job_min, self.op_per_job_max)
            for _ in range(self.n_j)
        ]
        dataset_JobLength = []
        dataset_OpPT = []
        
        for _ in range(self.num_envs):
            # Using SD2 generator as configured
            JobLength, OpPT, _ = SD2_instance_generator(config=self.config)
            dataset_JobLength.append(JobLength)
            dataset_OpPT.append(OpPT)
            
        return dataset_JobLength, dataset_OpPT

    def save_model(self):
        obj_folder = get_objective_folder(self.config)
        save_path = f"{self.script_dir}/save/{obj_folder}/{self.data_source}"
        os.makedirs(save_path, exist_ok=True)
        torch.save(
            self.ppo.policy.state_dict(),
            f"{save_path}/{self.model_name}.pth",
        )

    def save_training_log(self):
        with open(f"{self.script_dir}/train_log/{self.data_source}/reward_{self.model_name}.txt", "w") as f:
            f.write(str(self.log))

def main(param_file: str = PARAM_FILE):
    try:
        parameters = load_parameters(param_file)
    except FileNotFoundError:
        logging.error(f"Parameter file {param_file} not found.")
        return
    except Exception as e:
        logging.error(f"Error loading parameters: {e}")
        return

    device = initialize_device(parameters, method="DANIEL") # reuse DANIEL method name for consistency
    trainer = Trainer(parameters, device)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train MADRL")
    parser.add_argument("config_file", metavar="-f", type=str, nargs="?", default=PARAM_FILE)
    args = parser.parse_args()
    main(param_file=args.config_file)
