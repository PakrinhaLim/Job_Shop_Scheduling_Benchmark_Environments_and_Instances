import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
base_path = Path(__file__).resolve().parents[2]
sys.path.append(str(base_path))

from solution_methods.helper_functions import load_job_shop_env, load_parameters, initialize_device, set_seeds
from solution_methods.MADRL.src.env_test_madrl import MADRL_FJSPEnv_test
from solution_methods.MADRL.network.ppo_madrl import PPO_initialize
from solution_methods.MADRL.train_MADRL import sample_action_madrl

def evaluate_model(model_path, instance_path, parameters):
    # Set up device
    device = initialize_device(parameters, method="DANIEL")
    parameters['test_parameters']['problem_instance'] = instance_path
    
    # Load env
    jobShopEnv = load_job_shop_env(instance_path)
    env_test = MADRL_FJSPEnv_test(jobShopEnv, parameters)
    
    # Initialize PPO and load policy
    ppo = PPO_initialize(parameters)
    policy_state = torch.load(model_path, map_location=device, weights_only=True)
    ppo.policy.load_state_dict(policy_state)
    ppo.policy.eval()
    
    state = env_test.state
    while True:
        with torch.no_grad():
            pi, _ = ppo.policy(
                fea_j=state.fea_j_tensor,
                op_mask=state.op_mask_tensor,
                candidate=state.candidate_tensor,
                fea_m=state.fea_m_tensor,
                mch_mask=state.mch_mask_tensor,
                comp_idx=state.comp_idx_tensor,
                dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                fea_pairs=state.fea_pairs_tensor,
            )
        
        if parameters["test_parameters"]["sample"]:
            actions, _ = sample_action_madrl(pi)
        else:
            pi_m = pi.permute(0, 2, 1)
            actions = torch.argmax(pi_m, dim=2)
            mask = state.dynamic_pair_mask_tensor
            actions[mask.all(dim=1)] = -1
            
        state, _, done = env_test.step_madrl(actions.cpu().numpy())
        if done.all():
            break
            
    return env_test.JSP_instance.makespan, np.sum(env_test.energy_consumed)

def main():
    # Configuration
    PARAM_FILE = str(base_path / "configs/MADRL.toml")
    parameters = load_parameters(PARAM_FILE)
    
    # Define models to compare
    models = {
        "Baseline (Makespan-Only)": str(base_path / "solution_methods/DANIEL/save/SD2/20x10+mix.pth"),
        "Green-MADRL (Beta=0.0001)": str(base_path / "solution_methods/MADRL/save/makespan_energy/SD2/20x10+mix.pth")
    }
    
    # Define instances to test
    instances = [
        "/fjsp/brandimarte/Mk10.fjs",
        "/fjsp/hurink/vdata/la40.fjs"
    ]
    
    results = []
    
    for inst in instances:
        print(f"\nEvaluating Instance: {inst}")
        row = {"Instance": inst}
        for name, path in models.items():
            if not os.path.exists(path):
                print(f"  [!] Model {name} not found at {path}. Skipping.")
                continue
            
            try:
                m, e = evaluate_model(path, inst, parameters)
                row[f"{name}_Makespan"] = m
                row[f"{name}_Energy"] = round(e, 2)
                print(f"  {name}: Makespan={m}, Energy={e:.2f}")
            except Exception as ex:
                print(f"  [!] Error evaluating {name}: {ex}")
                
        results.append(row)
    
    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*50)
        print("Comparison Results Summary")
        print("="*50)
        print(df)
        df.to_csv(str(base_path / "madrl_comparison_results.csv"), index=False)
        print(f"\nResults saved to {str(base_path / 'madrl_comparison_results.csv')}")

if __name__ == "__main__":
    main()
