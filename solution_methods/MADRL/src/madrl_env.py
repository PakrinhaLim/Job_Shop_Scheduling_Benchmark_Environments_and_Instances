import numpy as np
import copy
from solution_methods.helper_functions import load_job_shop_env

import os
import random

class MADRL_Env:
    def __init__(self, parameters, mode='train'):
        self.parameters = parameters
        self.instance_path = parameters[mode]['problem_instance']
        
        self.is_batch = False
        self.instance_files = []
        
        # Check if path is directory (for batch training)
        # We need absolute path for os.path.isdir check, assuming instance_path might be relative or absolute
        # Ideally, helper functions handle this, but let's be robust
        
        # Resolve path similar to helper functions
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) # madrl/src/../.. -> solution_methods -> root
        # Actually helper usually handles resolution. Let's assume absolute or project relative.
        # But load_job_shop_env handles the reading.
        # Let's try to resolve it if it starts with /
        if self.instance_path.startswith("/") or self.instance_path.startswith("\\"):
             full_path = os.path.join(project_root, self.instance_path.lstrip("/\\"))
        else:
             full_path = self.instance_path
             
        if os.path.isdir(full_path):
            self.is_batch = True
            self.instance_files = [os.path.join(full_path, f) for f in os.listdir(full_path) if f.endswith('.fjs')]
            # Load first one to initialize dims
            self.current_instance = self.instance_files[0]
            self.jobShopEnv = load_job_shop_env(self.current_instance)
        else:
            self.jobShopEnv = load_job_shop_env(self.instance_path)

        self.num_machines = self.jobShopEnv.nr_of_machines
        self.num_jobs = self.jobShopEnv.nr_of_jobs
        
        # Define observation and action spaces generally
        # Observation: [Processing Time, Job Due Date (normalized), Operation Progress, Machine Workload] per job in queue
        self.obs_dim = 4 
        # Action: Select a job from the queue (max queue size assumption or masking)
        # For batch training, max_queue often fixed or max of batch. 
        # CAUTION: If different instances have different job counts, max_queue_size changes.
        # For this dataset (1005), all are 10 jobs.
        self.max_queue_size = self.num_jobs 
        
    def reset(self):
        if self.is_batch:
            # Pick random instance
            self.current_instance = random.choice(self.instance_files)
            self.jobShopEnv = load_job_shop_env(self.current_instance)
            
        self.jobShopEnv.reset()
        return self._get_observations()

    def step(self, actions):
        """
        actions: dict {machine_id: job_index_in_queue}
        """
        # Execute actions
        # In a real sync step, we would need to simulate time until the next decision point.
        # For simplicity in this discrete event simulation:
        # We process the chosen operations.
        
        rewards = {m_id: 0 for m_id in range(self.num_machines)}
        dones = {m_id: False for m_id in range(self.num_machines)}
        
        # Sort actions by machine availability (heuristic: process earliest avail machine first)
        # But in standard step, we might just try to schedule all.
        
        try:
            prev_makespan = self.jobShopEnv.makespan
        except ValueError:
            prev_makespan = 0
        
        scheduled_count = 0
        scheduled_ops_this_step = set()
        
        # Sort actions by machine availability (heuristic: process earliest avail machine first)
        # We need a stable order to resolve conflicts
        
        for machine_id, queue_idx in actions.items():
            machine = self.jobShopEnv.get_machine(machine_id)
            # Get operations that are ready to be processed on this machine
            ready_ops = self._get_ready_ops_for_machine(machine)
            
            if len(ready_ops) > 0 and queue_idx < len(ready_ops):
                op_to_schedule = ready_ops[queue_idx]
                
                # Check if already scheduled this step (Race condition prevention)
                if op_to_schedule.operation_id in scheduled_ops_this_step:
                    continue
                
                # Check if the operation is already scheduled in the env (from previous steps)
                # MADRL_Env should only show available ops, but double check
                if op_to_schedule.scheduling_information:
                    continue

                # Schedule it
                # We need to find the specific operation instance
                duration = op_to_schedule.processing_times[machine_id]
                try:
                    self.jobShopEnv.schedule_operation_with_backfilling(op_to_schedule, machine_id, duration)
                    scheduled_count += 1
                    scheduled_ops_this_step.add(op_to_schedule.operation_id)
                except ValueError:
                    # Could happen if constraints were violated, though _get_ready_ops should prevent this
                    pass
        
        self.jobShopEnv.update_operations_available_for_scheduling()
        
        # Calculate Reward (Global reward: - Change in Makespan, or +1 for doing work)
        # A simple dense reward: -1 per step per agent until done (to minimize time) 
        # OR improvement in makespan.
        
        try:
            current_makespan = self.jobShopEnv.makespan
        except ValueError:
            current_makespan = 0
        
        # Check if done
        all_done = len(self.jobShopEnv.operations_to_be_scheduled) == 0
        
        obs = self._get_observations()
        
        common_reward = 0
        if all_done:
            # Big positive reward for finishing or negative for final makespan
            common_reward = 1000.0 / current_makespan 
        else:
             # Small penalty for time passing
            common_reward = -0.1

        for m_id in range(self.num_machines):
            rewards[m_id] = common_reward
            dones[m_id] = all_done
            
        return obs, rewards, dones, {}

    def _get_observations(self):
        observations = {}
        for machine_id in range(self.num_machines):
            machine = self.jobShopEnv.get_machine(machine_id)
            ready_ops = self._get_ready_ops_for_machine(machine)
            
            # Construct feature matrix [MaxQueueSize, ObsDim]
            obs_matrix = np.zeros((self.max_queue_size, self.obs_dim))
            
            for i, op in enumerate(ready_ops):
                if i >= self.max_queue_size: break
                
                # Feat 1: Processing Time
                obs_matrix[i, 0] = op.processing_times[machine_id]
                # Feat 2: Remaining Ops in Job
                obs_matrix[i, 1] = self.jobShopEnv.get_job(op.job_id).nr_of_ops - op.operation_id
                # Feat 3: Job Priority/Weight (placeholder, 1.0)
                obs_matrix[i, 2] = 1.0 
                # Feat 4: Machine Id (normalized)
                obs_matrix[i, 3] = machine_id / self.num_machines
                
            observations[machine_id] = obs_matrix
            
        return observations

    def _get_ready_ops_for_machine(self, machine):
        # Filter operations available for scheduling that can be processed on this machine
        # AND are the specific option for this machine (for FJSP)
        
        potential_ops = self.jobShopEnv.operations_available_for_scheduling
        ready_ops = []
        for op in potential_ops:
            # Check if this machine is a valid option for this operation
            if machine.machine_id in op.processing_times:
                 ready_ops.append(op)
        return ready_ops
