
import numpy as np
from solution_methods.MADRL.src.madrl_env import MADRL_FJSPEnv
from solution_methods.helper_functions import initialize_device

class MADRL_FJSPEnv_test(MADRL_FJSPEnv):
    def __init__(self, JobShop_module, parameters):
        n_j = JobShop_module.nr_of_jobs
        n_m = JobShop_module.nr_of_machines
        device = initialize_device(parameters, method="DANIEL") # Reuse DANIEL method name for consistency
        super().__init__(n_j=n_j, n_m=n_m, device=device, config=parameters)

        # Assign values to the job_length_list and op_pt_list
        job_length_list = np.asarray([[job.nr_of_ops for job in JobShop_module.jobs]])
        op_pt_list = np.zeros((1, JobShop_module.nr_of_operations, n_m), dtype=np.int32)

        for job in JobShop_module.jobs:
            for operation in job.operations:
                for machine, pt in operation.processing_times.items():
                    op_pt_list[0, operation.operation_id, machine] = pt

        super(MADRL_FJSPEnv_test, self).set_initial_data(job_length_list, op_pt_list)

        self.JSP_instance = JobShop_module

    def step_madrl(self, actions):
        # actions: [1, M] (batch size is 1 for test)
        actions = actions[0] # [M]
        
        # We need to apply actions sequentially as in madrl_env.py
        # But here we also need to update self.JSP_instance
        
        # Get machine order (same as in env, but here we can just iterate 0..M-1 because env does it internally?)
        # Actually madrl_env.step_madrl does shuffling. ideally we want to replicate the same logic.
        # But super().step_madrl() will do the internal state updates.
        # We need to capture the external JSP_instance updates.
        
        # Since super().step_madrl() iterates internally, we can't easily inject the JSP_instance update inside the loop
        # UNLESS we override step_madrl completely or _step_core.
        # Let's override _step_core in this subclass.
        
        return super(MADRL_FJSPEnv_test, self).step_madrl(actions.reshape(1, -1))

    def _step_core(self, chosen_job, m, active_envs):
        # Call parent's _step_core to update tensor states
        # The parent _step_core handles internal state updates.
        
        # We need to execute the scheduling on JSP_instance for each active environment
        op_indices = self.candidate[active_envs, chosen_job]

        for i, env_idx in enumerate(active_envs):
            job_idx = chosen_job[i]
            op_idx = op_indices[i]

            if job_idx == -1:
                continue

            # --- Update JSP_instance ---
            operation = self.JSP_instance.operations[op_idx]
            # processing time is in true_op_pt[env_idx, op_idx, m]
            duration = self.true_op_pt[env_idx, op_idx, m].item()

            self.JSP_instance.schedule_operation_on_machine(operation, m, duration)
            self.JSP_instance.get_job(operation.job_id).scheduled_operations.append(operation)
            # ---------------------------

        # Call parent core (updates dynamic_pair_mask, etc.)
        super()._step_core(chosen_job, m, active_envs)

    def reset(self):
        self.JSP_instance.reset()
        return super(MADRL_FJSPEnv_test, self).reset()
