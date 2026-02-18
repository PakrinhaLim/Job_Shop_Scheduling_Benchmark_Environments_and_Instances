import json
import numpy as np
import argparse

def calculate_energy(json_path, busy_power, idle_power):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    schedule = data['schedule']
    
    # Track machine free times for idle energy calculation
    # machine_free_times[machine_id] = finish time of last operation on that machine
    machine_free_times = {}
    total_energy = 0.0
    
    # Process all tasks
    all_tasks = []
    for job in schedule:
        for task in job['tasks']:
            all_tasks.append(task)
            
    # Sort tasks by start time to process chronologically per machine
    all_tasks.sort(key=lambda x: x['start'])
    
    for task in all_tasks:
        m_id = task['machine']
        duration = task['duration']
        start = task['start']
        end = task['end']
        
        p_busy = busy_power[m_id]
        p_idle = idle_power[m_id]
        
        # Idle time since last task on this machine
        last_free = machine_free_times.get(m_id, 0)
        idle_time = max(0, start - last_free)
        
        # Energy = BusyEnergy + IdleEnergy
        task_energy = p_busy * duration + p_idle * idle_time
        total_energy += task_energy
        
        # Update machine free time
        machine_free_times[m_id] = end
        
    return total_energy

if __name__ == "__main__":
    # Example power values (matching your MADRL.toml for 15 machines)
    busy = [15.0, 20.0, 12.0, 18.0, 25.0, 14.0, 22.0, 16.0, 19.0, 21.0, 17.5, 23.0, 13.5, 16.5, 20.5]
    idle = [2.0, 3.5, 1.8, 2.5, 4.0, 2.2, 3.1, 2.6, 2.9, 3.3, 2.4, 3.8, 1.9, 2.7, 3.0]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to MADRL_results.json")
    args = parser.parse_args()
    
    energy = calculate_energy(args.file, busy, idle)
    print(f"File: {args.file}")
    print(f"Energy Consumed: {energy:.2f}")
