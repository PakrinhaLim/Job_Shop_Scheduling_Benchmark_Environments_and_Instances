
import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())

from solution_methods.helper_functions import load_job_shop_env

instance_path = "/fjsp/barnes/mt10c1.fjs"
print(f"Loading instance: {instance_path}")


try:
    js_env = load_job_shop_env(instance_path)
    print(f"Loaded instance")
    print(f"Jobs (len): {len(js_env.jobs)}")
    print(f"Machines (len): {len(js_env.machines)}")
    print(f"nr_of_jobs (prop): {js_env.nr_of_jobs}")
    print(f"nr_of_machines (prop): {js_env.nr_of_machines}")

    print("Finished inspection HEAD.")
    sys.exit(0)

except Exception as e:
    print(f"Exception: {e}")
