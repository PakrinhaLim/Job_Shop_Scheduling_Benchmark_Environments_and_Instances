import os
import sys
import subprocess
import re
import shutil
import argparse

# Configuration
INSTANCE_PATH = "/fjsp/brandimarte/Mk01.fjs"
RESULTS_DIR = "benchmark_results"
TEMP_CONFIG_DIR = "temp_configs"

METHODS = {
    "FJSP_DRL": {
        "script": "solution_methods/FJSP_DRL/run_FJSP_DRL.py",
        "config": "configs/FJSP_DRL.toml",
        "arg_style": "positional",
        "updates": [
            ('problem_instance = ".*"', f'problem_instance = "{INSTANCE_PATH}"'),
            ('save_results = .*', 'save_results = false'),
            ('show_gantt = .*', 'show_gantt = false'),
            ('save_gantt = .*', 'save_gantt = false'),
            ('show_precedences = .*', 'show_precedences = false'),
            ('trained_policy = ".*"', 'trained_policy = "/saved_models/train_20240314_192906/song_10_5.pt"'),
            ('device = "cuda"', 'device = "cpu"'),
        ]
    },
    "DANIEL": {
        "script": "solution_methods/DANIEL/run_DANIEL.py",
        "config": "configs/DANIEL.toml",
        "arg_style": "positional",
        "updates": [
            ('problem_instance = ".*"', f'problem_instance = "{INSTANCE_PATH}"'),
            ('save_results = .*', 'save_results = false'),
            ('show_gantt = .*', 'show_gantt = false'),
            ('save_gantt = .*', 'save_gantt = false'),
            ('show_precedences = .*', 'show_precedences = false'),
            ('name = "cuda"', 'name = "cpu"'),
            ('n_j = .*', 'n_j = 10'),
            ('n_m = .*', 'n_m = 6'),
            ('n_op = .*', 'n_op = 100'),
        ]
    },
    "L2D": {
        "script": "solution_methods/L2D/run_L2D.py",
        "config": "configs/L2D.toml",
        "arg_style": "positional",
        "updates": [
            ('problem_instance = ".*"', f'problem_instance = "{INSTANCE_PATH}"'),
            ('save_results = .*', 'save_results = false'),
            ('show_gantt = .*', 'show_gantt = false'),
            ('save_gantt = .*', 'save_gantt = false'),
            ('show_precedences = .*', 'show_precedences = false')
        ]
    },
    "CP_SAT": {
        "script": "solution_methods/CP_SAT/run_CP_SAT.py",
        "config": "configs/cp_sat.toml",
        "arg_style": "flag",
        "updates": [
            ('problem_instance = ".*"', f'problem_instance = "{INSTANCE_PATH}"'),
            ('model = ".*"', 'model = "fjsp"'),
            ('save_results = .*', 'save_results = false'),
            ('show_gantt = .*', 'show_gantt = false'),
            ('save_gantt = .*', 'save_gantt = false'),
            ('show_precedences = .*', 'show_precedences = false')
        ]
    },
    "GA": {
        "script": "solution_methods/GA/run_GA.py",
        "config": "configs/GA.toml",
        "arg_style": "flag",
        "updates": [
            ('problem_instance = ".*"', f'problem_instance = "{INSTANCE_PATH}"'),
            ('save_results = .*', 'save_results = false'),
            ('show_gantt = .*', 'show_gantt = false'),
            ('save_gantt = .*', 'save_gantt = false'),
            ('show_precedences = .*', 'show_precedences = false')
        ]
    },
    "MILP": {
        "script": "solution_methods/MILP/run_MILP.py",
        "config": "configs/milp.toml",
        "arg_style": "flag",
        "updates": [
            ('problem_instance = ".*"', f'problem_instance = "{INSTANCE_PATH}"'),
            ('save_results = .*', 'save_results = false'),
            ('show_gantt = .*', 'show_gantt = false'),
            ('save_gantt = .*', 'save_gantt = false'),
            ('show_precedences = .*', 'show_precedences = false'),
            ('time_limit = .*', 'time_limit = 60') # Limit time for benchmark speed
        ]
    },
    "dispatching_rules": {
        "script": "solution_methods/dispatching_rules/run_dispatching_rules.py",
        "config": "configs/dispatching_rules.toml",
        "arg_style": "flag",
        "updates": [
            ('problem_instance = ".*"', f'problem_instance = "{INSTANCE_PATH}"'),
            ('online_arrivals = .*', 'online_arrivals = false'),
            ('save_results = .*', 'save_results = false'),
            ('show_gantt = .*', 'show_gantt = false'),
            ('save_gantt = .*', 'save_gantt = false'),
            ('show_precedences = .*', 'show_precedences = false')
        ]
    }
}

def create_temp_config(method_name, config_info):
    if not os.path.exists(TEMP_CONFIG_DIR):
        os.makedirs(TEMP_CONFIG_DIR)
        
    original_config_path = config_info['config']
    with open(original_config_path, 'r') as f:
        content = f.read()
        
    for pattern, replacement in config_info['updates']:
        content = re.sub(pattern, replacement, content)
        
    temp_config_path = os.path.join(TEMP_CONFIG_DIR, f"{method_name}.toml")
    with open(temp_config_path, 'w') as f:
        f.write(content)
        
    return temp_config_path

def run_method(method_name, info):
    print(f"Running {method_name}...")
    temp_config_path = os.path.join(TEMP_CONFIG_DIR, f"{method_name}.toml")
    
    # Create the config file
    if not os.path.exists(TEMP_CONFIG_DIR):
        os.makedirs(TEMP_CONFIG_DIR)
    
    with open(info['config'], 'r') as f:
        content = f.read()
        
    for pattern, replacement in info['updates']:
        content = re.sub(pattern, replacement, content)
        
    with open(temp_config_path, 'w') as f:
        f.write(content)

    # Run the script
    if info.get('arg_style') == 'flag':
        cmd = [sys.executable, info['script'], '-f', temp_config_path]
    else:
        cmd = [sys.executable, info['script'], temp_config_path]
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd() + os.pathsep + env.get('PYTHONPATH', '')
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env) # 5 min timeout
        
        if result.returncode != 0:
            print(f"Error running {method_name}: {result.stderr}")
            if "not compiled with CUDA enabled" in result.stderr:
                print(f"Hint: check config for {method_name} and ensure cpu is used if cuda is not available.")
            return "Failed"
            
        # Parse output for Makespan
        # Look for "Makespan: X", "Solution: X", "Objective: X", "objective = X"
        # Use findall to get all matches, and take the last one as it's likely the best found so far
        matches = re.findall(r"(?:Makespan|Solution|Objective|objective)(?:\s*[:=]\s*|\s+)(\d+(\.\d+)?)", result.stderr + result.stdout, re.IGNORECASE)
        if matches:
            return float(matches[-1][0])
        else:
            print(f"Could not find Makespan in output for {method_name}")
            print(f"Output snippet: {result.stdout[-200:]} {result.stderr[-200:]}")
            return "No Output"
            
    except subprocess.TimeoutExpired:
        print(f"Timed out running {method_name}")
        return "Timeout"
    except Exception as e:
        print(f"Exception running {method_name}: {e}")
        return "Error"



import csv
import glob
import time

def main():
    parser = argparse.ArgumentParser(description="Run Benchmark Comparison")
    parser.add_argument("--all", action="store_true", help="Run on all found FJSP instances")
    parser.add_argument("--instance", type=str, default="/fjsp/brandimarte/Mk01.fjs", help="Specific instance to run (default: Mk01)")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of instances to run (0 for no limit)")
    parser.add_argument("--output", type=str, default="benchmark_results_all.csv", help="Output CSV file for results")
    args = parser.parse_args()

    # Discover instances
    instances = []
    if args.all:
        # Search for all .fjs files in data/fjsp recursively
        # Note: glob might return absolute paths, need to convert to format expected by configs (relative to project root or accessible)
        # The configs expect paths starting with /fjsp/... which seems to map effectively if we run from project root?
        # Actually, let's check how load_job_shop_env works. It typically takes paths relative to 'data' or absolute.
        # The current defaults uses "/fjsp/brandimarte/Mk01.fjs" which implies relative to some data root or handled by parser.
        # Let's assume we need to pass paths that start with "/fjsp/..." or absolute paths.
        
        data_dir = os.path.join(os.getcwd(), "data")
        fjsp_dir = os.path.join(data_dir, "fjsp")
        
        print(f"Searching for instances in {fjsp_dir}...")
        for root, dirs, files in os.walk(fjsp_dir):
            for file in files:
                if file.endswith(".fjs"):
                    # Create path relative to 'data' folder, e.g. /fjsp/brandimarte/Mk01.fjs
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, data_dir)
                    # Ensure it starts with / for consistency with current config usage if needed, 
                    # though helper_functions.py might handle it. safe bet is ensuring it looks like current default.
                    formatted_path = "/" + rel_path.replace(os.sep, "/")
                    instances.append(formatted_path)
        
        print(f"Found {len(instances)} instances.")
        if args.limit > 0:
            instances = instances[:args.limit]
            print(f"Limiting to first {args.limit} instances.")
            
    else:
        instances = [args.instance]

    # Initialize CSV result file
    with open(args.output, 'w', newline='') as csvfile:
        fieldnames = ['Instance', 'Method', 'Makespan', 'Status', 'Time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for instance_path in instances:
            print(f"\nProcessing instance: {instance_path}")
            
            # Update METHODS with the current instance
            for method in METHODS:
                # We need to update the regex replacement for problem_instance in each method
                # This is a bit tricky because the tuples are immutable in the dictionary. 
                # We need to reconstruct the updates list or modify the logic in run_method to take instance as arg.
                pass 
                # Actually, run_method uses the list from METHODS. 
                # Let's Modify run_method to accept the instance path request dynamically 
                # OR update the global dictionary. Updating global dict is easier but less clean. 
                # Better: Create a copy of info and inject the instance path.
            
            results = {}
            for method, info in METHODS.items():
                start_time = time.time()
                
                # Create a deep copy of updates to modify the instance path
                # info['updates'] is a list of tuples.
                updates_copy = []
                for pattern, replacement in info['updates']:
                    if 'problem_instance' in pattern:
                        updates_copy.append(('problem_instance = ".*"', f'problem_instance = "{instance_path}"'))
                    else:
                        updates_copy.append((pattern, replacement))
                
                # Create a temporary info dict
                temp_info = info.copy()
                temp_info['updates'] = updates_copy
                
                makespan = run_method(method, temp_info)
                duration = time.time() - start_time
                
                status = "Success" if isinstance(makespan, float) else makespan
                
                print(f"  {method}: {makespan}")
                
                writer.writerow({
                    'Instance': instance_path,
                    'Method': method,
                    'Makespan': makespan,
                    'Status': status,
                    'Time': f"{duration:.2f}"
                })
                csvfile.flush() # Ensure data is written immediately

    # Cleanup
    if os.path.exists(TEMP_CONFIG_DIR):
        shutil.rmtree(TEMP_CONFIG_DIR)

if __name__ == "__main__":
    main()
