
import sys
import os
from pathlib import Path

# Add project root to path
base_path = Path(__file__).resolve().parents[2]
sys.path.append(str(base_path))

import torch

try:
    from solution_methods.helper_functions import load_job_shop_env, load_parameters, initialize_device, set_seeds
    from solution_methods.MADRL.src.env_test_madrl import MADRL_FJSPEnv_test
    from solution_methods.MADRL.network.ppo_madrl import PPO_initialize
    print("Imports successful")
    
    PARAM_FILE = "configs/MADRL.toml"
    parameters = load_parameters(PARAM_FILE)
    print("Parameters loaded")
    
    device = initialize_device(parameters, method="DANIEL")
    print(f"Device initialized: {device}")
    
    jobShopEnv = load_job_shop_env(parameters["test_parameters"]["problem_instance"])
    print("Env loaded")
    
    env_test = MADRL_FJSPEnv_test(jobShopEnv, parameters)
    print("MADRL Env initialized")
    
    ppo = PPO_initialize(parameters)
    print("PPO initialized")

    # load trained policy
    model_source = parameters['model']['source']
    trained_policy_name = parameters['test_parameters']['trained_policy']
    trained_policy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save", model_source, f"{trained_policy_name}.pth")
    print(f"Loading policy from: {trained_policy_path}")

    if not os.path.exists(trained_policy_path):
        print(f"Policy file NOT FOUND at {trained_policy_path}")
    else:
        try:
            print("Attempting torch.load...")
            policy = torch.load(trained_policy_path, map_location=device, weights_only=False)
            print("torch.load successful")
            ppo.policy.load_state_dict(policy)
            ppo.policy.eval()
            print("Policy loaded successfully")
        except Exception as e2:
            print(f"FAILED during torch.load: {e2}")
            import traceback
            traceback.print_exc()
            raise e2
    
    # Test state
    state = env_test.state
    print("Initial state retrieved")

except Exception as e:
    import traceback
    print("An error occurred:")
    traceback.print_exc()
    sys.exit(1)
