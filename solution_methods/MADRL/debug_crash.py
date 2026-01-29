
import sys
import os
from pathlib import Path

print("1. Starting debug_madrl_crash.py")
sys.stdout.flush()

# Add project root to path
base_path = Path(__file__).resolve().parents[2]
sys.path.append(str(base_path))
print(f"2. Base path added: {base_path}")
sys.stdout.flush()

try:
    print("3. Importing torch...")
    sys.stdout.flush()
    import torch
    print(f"   Torch version: {torch.__version__}")
    
    print("4. Importing numpy...")
    sys.stdout.flush()
    import numpy as np
    print(f"   Numpy version: {np.__version__}")
    
    print("5. Importing matplotlib...")
    sys.stdout.flush()
    import matplotlib
    matplotlib.use('Agg') # Use non-interactive backend
    import matplotlib.pyplot as plt
    print("   Matplotlib imported")
    
    print("6. Importing solution_methods.helper_functions...")
    sys.stdout.flush()
    from solution_methods.helper_functions import load_job_shop_env, load_parameters, initialize_device, set_seeds
    print("   Helper functions imported")
    
    print("7. Importing MADRL_FJSPEnv_test...")
    sys.stdout.flush()
    from solution_methods.MADRL.src.env_test_madrl import MADRL_FJSPEnv_test
    print("   MADRL Env Test imported")
    
    print("8. Importing PPO_initialize...")
    sys.stdout.flush()
    from solution_methods.MADRL.network.ppo_madrl import PPO_initialize
    print("   PPO Initialize imported")
    
    print("9. All essential imports successful")
    sys.stdout.flush()

except Exception as e:
    print(f"!!! Error during imports: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("10. Finished debug_madrl_crash.py")
sys.stdout.flush()
