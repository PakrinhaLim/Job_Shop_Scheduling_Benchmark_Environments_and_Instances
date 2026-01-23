import os
import sys
import subprocess
import re
import shutil

# Configuration
INSTANCE_PATH = "/fjsp/brandimarte/Mk01.fjs"
RESULTS_DIR = "benchmark_results"
TEMP_CONFIG_DIR = "temp_configs"

METHODS = {
    "FJSP_DRL": {
        "script": "solution_methods/FJSP_DRL/run_FJSP_DRL.py",
        "config": "configs/FJSP_DRL.toml",
        "updates": [
            ('problem_instance = ".*"', f'problem_instance = "{INSTANCE_PATH}"'),
            ('save_results = .*', 'save_results = false'),
            ('show_gantt = .*', 'show_gantt = false'),
            ('save_gantt = .*', 'save_gantt = false'),
            ('show_precedences = .*', 'show_precedences = false')
        ]
    },
    "DANIEL": {
        "script": "solution_methods/DANIEL/run_DANIEL.py",
        "config": "configs/DANIEL.toml",
        "updates": [
            ('problem_instance = ".*"', f'problem_instance = "{INSTANCE_PATH}"'),
            ('save_results = .*', 'save_results = false'),
            ('show_gantt = .*', 'show_gantt = false'),
            ('save_gantt = .*', 'save_gantt = false'),
            ('show_precedences = .*', 'show_precedences = false')
        ]
    },
    "L2D": {
        "script": "solution_methods/L2D/run_L2D.py",
        "config": "configs/L2D.toml",
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
        
    return temp_temp_config_path

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
    cmd = [sys.executable, info['script'], temp_config_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300) # 5 min timeout
        
        if result.returncode != 0:
            print(f"Error running {method_name}: {result.stderr}")
            return "Failed"
            
        # Parse output for Makespan
        match = re.search(r"Makespan: (\d+(\.\d+)?)", result.stderr + result.stdout)
        if match:
            return float(match.group(1))
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

def main():
    results = {}
    
    for method, info in METHODS.items():
        makespan = run_method(method, info)
        results[method] = makespan
        
    print("\n\nBenchmark Results:")
    print("| Method | Makespan |")
    print("|--------|----------|")
    for method, makespan in results.items():
        print(f"| {method} | {makespan} |")
        
    # Cleanup
    if os.path.exists(TEMP_CONFIG_DIR):
        shutil.rmtree(TEMP_CONFIG_DIR)

if __name__ == "__main__":
    main()
