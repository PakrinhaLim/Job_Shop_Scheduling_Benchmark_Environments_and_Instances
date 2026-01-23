
import copy
import sys
from dataclasses import dataclass

import numpy as np
import numpy.ma as ma
import torch

# Inherit or copy EnvState. copying for simplicity and self-containment
@dataclass
class EnvState():
    """
    state definition
    """
    device : torch.device
    fea_j_tensor: torch.Tensor = None
    op_mask_tensor: torch.Tensor = None
    fea_m_tensor: torch.Tensor = None
    mch_mask_tensor: torch.Tensor = None
    dynamic_pair_mask_tensor: torch.Tensor = None
    comp_idx_tensor: torch.Tensor = None
    candidate_tensor: torch.Tensor = None
    fea_pairs_tensor: torch.Tensor = None


    def update(
        self,
        fea_j,
        op_mask,
        fea_m,
        mch_mask,
        dynamic_pair_mask,
        comp_idx,
        candidate,
        fea_pairs,
    ):
        device = self.device
        self.fea_j_tensor = torch.from_numpy(np.copy(fea_j)).float().to(device)
        self.fea_m_tensor = torch.from_numpy(np.copy(fea_m)).float().to(device)
        self.fea_pairs_tensor = torch.from_numpy(np.copy(fea_pairs)).float().to(device)

        self.op_mask_tensor = torch.from_numpy(np.copy(op_mask)).to(device)
        self.candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
        self.mch_mask_tensor = torch.from_numpy(np.copy(mch_mask)).float().to(device)
        self.comp_idx_tensor = torch.from_numpy(np.copy(comp_idx)).to(device)
        self.dynamic_pair_mask_tensor = torch.from_numpy(np.copy(dynamic_pair_mask)).to(
            device
        )

# Use absolute import for DANIEL src if needed, but since I'm copying most logic,
# I will just reimplement the class to avoid import issues and allowing modifications.

class MADRL_FJSPEnv:
    def __init__(self, n_j, n_m, device):
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.old_state = EnvState(device)
        self.device = device

        self.op_fea_dim = 10
        self.mch_fea_dim = 8

    def set_static_properties(self):
        self.multi_env_mch_diag = np.tile(
            np.expand_dims(np.eye(self.number_of_machines, dtype=bool), axis=0),
            (self.number_of_envs, 1, 1),
        )

        self.env_idxs = np.arange(self.number_of_envs)
        self.env_job_idx = self.env_idxs.repeat(self.number_of_jobs).reshape(
            self.number_of_envs, self.number_of_jobs
        )
        self.op_idx = np.arange(self.number_of_ops)[np.newaxis, :]

    def set_initial_data(self, job_length_list, op_pt_list):
        self.number_of_envs = len(job_length_list)
        self.job_length = np.array(job_length_list)
        self.op_pt = np.array(op_pt_list)
        self.number_of_ops = self.op_pt.shape[1]
        self.number_of_machines = op_pt_list[0].shape[1]
        self.number_of_jobs = job_length_list[0].shape[0]

        self.set_static_properties()

        self.pt_lower_bound = np.min(self.op_pt)
        self.pt_upper_bound = np.max(self.op_pt)
        self.true_op_pt = np.copy(self.op_pt)

        self.op_pt = (self.op_pt - self.pt_lower_bound) / (
            self.pt_upper_bound - self.pt_lower_bound + 1e-8
        )

        self.process_relation = self.op_pt != 0
        self.reverse_process_relation = ~self.process_relation

        self.compatible_op = np.sum(self.process_relation, 2)
        self.compatible_mch = np.sum(self.process_relation, 1)

        self.unmasked_op_pt = np.copy(self.op_pt)

        head_op_id = np.zeros((self.number_of_envs, 1))

        self.job_first_op_id = np.concatenate(
            [head_op_id, np.cumsum(self.job_length, axis=1)[:, :-1]], axis=1
        ).astype("int")
        self.job_last_op_id = self.job_first_op_id + self.job_length - 1

        self.initial_vars()
        self.init_op_mask()
        self.op_pt = ma.array(self.op_pt, mask=self.reverse_process_relation)

        self.op_mean_pt = np.mean(self.op_pt, axis=2).data
        self.op_min_pt = np.min(self.op_pt, axis=-1).data
        self.op_max_pt = np.max(self.op_pt, axis=-1).data
        self.pt_span = self.op_max_pt - self.op_min_pt
        self.mch_min_pt = np.max(self.op_pt, axis=1).data
        self.mch_max_pt = np.max(self.op_pt, axis=1)

        self.op_ct_lb = copy.deepcopy(self.op_min_pt)
        for k in range(self.number_of_envs):
            for i in range(self.number_of_jobs):
                self.op_ct_lb[k][
                    self.job_first_op_id[k][i] : self.job_last_op_id[k][i] + 1
                ] = np.cumsum(
                    self.op_ct_lb[k][
                        self.job_first_op_id[k][i] : self.job_last_op_id[k][i] + 1
                    ]
                )

        self.op_match_job_left_op_nums = np.array(
            [
                np.repeat(self.job_length[k], repeats=self.job_length[k])
                for k in range(self.number_of_envs)
            ]
        )
        self.job_remain_work = []
        for k in range(self.number_of_envs):
            self.job_remain_work.append(
                [
                    np.sum(
                        self.op_mean_pt[k][
                            self.job_first_op_id[k][i] : self.job_last_op_id[k][i] + 1
                        ]
                    )
                    for i in range(self.number_of_jobs)
                ]
            )

        self.op_match_job_remain_work = np.array(
            [
                np.repeat(self.job_remain_work[k], repeats=self.job_length[k])
                for k in range(self.number_of_envs)
            ]
        )

        self.construct_op_features()
        self.init_quality = np.max(self.op_ct_lb, axis=1)
        self.max_endTime = self.init_quality

        self.mch_available_op_nums = np.copy(self.compatible_mch)
        self.mch_current_available_op_nums = np.copy(self.compatible_mch)
        self.candidate_pt = np.array(
            [
                self.unmasked_op_pt[k][self.candidate[k]]
                for k in range(self.number_of_envs)
            ]
        )

        self.dynamic_pair_mask = self.candidate_pt == 0
        self.candidate_process_relation = np.copy(self.dynamic_pair_mask)
        self.mch_current_available_jc_nums = np.sum(
            ~self.candidate_process_relation, axis=1
        )

        self.mch_mean_pt = np.mean(self.op_pt, axis=1).filled(0)
        self.comp_idx = self.logic_operator(x=~self.dynamic_pair_mask)
        self.init_mch_mask()
        self.construct_mch_features()
        self.construct_pair_features()

        self.old_state.update(
            self.fea_j,
            self.op_mask,
            self.fea_m,
            self.mch_mask,
            self.dynamic_pair_mask,
            self.comp_idx,
            self.candidate,
            self.fea_pairs,
        )

        self.old_op_mask = np.copy(self.op_mask)
        self.old_mch_mask = np.copy(self.mch_mask)
        self.old_op_ct_lb = np.copy(self.op_ct_lb)
        self.old_op_match_job_left_op_nums = np.copy(self.op_match_job_left_op_nums)
        self.old_op_match_job_remain_work = np.copy(self.op_match_job_remain_work)
        self.old_init_quality = np.copy(self.init_quality)
        self.old_candidate_pt = np.copy(self.candidate_pt)
        self.old_candidate_process_relation = np.copy(self.candidate_process_relation)
        self.old_mch_current_available_op_nums = np.copy(
            self.mch_current_available_op_nums
        )
        self.old_mch_current_available_jc_nums = np.copy(
            self.mch_current_available_jc_nums
        )

        self.state = copy.deepcopy(self.old_state)
        return self.state

    def reset(self):
        self.initial_vars()
        self.op_mask = np.copy(self.old_op_mask)
        self.mch_mask = np.copy(self.old_mch_mask)
        self.op_ct_lb = np.copy(self.old_op_ct_lb)
        self.op_match_job_left_op_nums = np.copy(self.old_op_match_job_left_op_nums)
        self.op_match_job_remain_work = np.copy(self.old_op_match_job_remain_work)
        self.init_quality = np.copy(self.old_init_quality)
        self.max_endTime = self.init_quality
        self.candidate_pt = np.copy(self.old_candidate_pt)
        self.candidate_process_relation = np.copy(self.old_candidate_process_relation)
        self.mch_current_available_op_nums = np.copy(
            self.old_mch_current_available_op_nums
        )
        self.mch_current_available_jc_nums = np.copy(
            self.old_mch_current_available_jc_nums
        )
        self.state = copy.deepcopy(self.old_state)
        return self.state

    def initial_vars(self):
        self.step_count = 0
        self.current_makespan = np.full(self.number_of_envs, float("-inf"))
        self.op_ct = np.zeros((self.number_of_envs, self.number_of_ops))
        self.mch_free_time = np.zeros((self.number_of_envs, self.number_of_machines))
        self.mch_remain_work = np.zeros((self.number_of_envs, self.number_of_machines))
        self.mch_waiting_time = np.zeros((self.number_of_envs, self.number_of_machines))
        self.mch_working_flag = np.zeros((self.number_of_envs, self.number_of_machines))
        self.next_schedule_time = np.zeros(self.number_of_envs)
        self.candidate_free_time = np.zeros((self.number_of_envs, self.number_of_jobs))
        self.true_op_ct = np.zeros((self.number_of_envs, self.number_of_ops))
        self.true_candidate_free_time = np.zeros(
            (self.number_of_envs, self.number_of_jobs)
        )
        self.true_mch_free_time = np.zeros(
            (self.number_of_envs, self.number_of_machines)
        )
        self.candidate = np.copy(self.job_first_op_id)
        self.mask = np.full(
            shape=(self.number_of_envs, self.number_of_jobs), fill_value=0, dtype=bool
        )
        self.op_scheduled_flag = np.zeros((self.number_of_envs, self.number_of_ops))
        self.op_waiting_time = np.zeros((self.number_of_envs, self.number_of_ops))
        self.op_remain_work = np.zeros((self.number_of_envs, self.number_of_ops))
        self.op_available_mch_nums = (
            np.copy(self.compatible_op) / self.number_of_machines
        )
        self.pair_free_time = np.zeros(
            (self.number_of_envs, self.number_of_jobs, self.number_of_machines)
        )
        self.remain_process_relation = np.copy(self.process_relation)
        self.delete_mask_fea_j = np.full(
            shape=(self.number_of_envs, self.number_of_ops, self.op_fea_dim),
            fill_value=0,
            dtype=bool,
        )
        self.deleted_op_nodes = np.full(
            shape=(self.number_of_envs, self.number_of_ops), fill_value=0, dtype=bool
        )

    def logic_operator(self, x, flagT=True):
        if flagT:
            x = x.transpose(0, 2, 1)
        d1 = np.expand_dims(x, 2)
        d2 = np.expand_dims(x, 1)
        return np.logical_and(d1, d2).astype(np.float32)

    def init_op_mask(self):
        self.op_mask = np.full(
            shape=(self.number_of_envs, self.number_of_ops, 3),
            fill_value=0,
            dtype=np.float32,
        )
        self.op_mask[self.env_job_idx, self.job_first_op_id, 0] = 1
        self.op_mask[self.env_job_idx, self.job_last_op_id, 2] = 1

    def init_mch_mask(self):
        self.mch_mask = (
            self.logic_operator(self.remain_process_relation).sum(axis=-1).astype(bool)
        )
        self.delete_mask_fea_m = np.tile(
            ~(np.sum(self.mch_mask, keepdims=True, axis=-1).astype(bool)),
            (1, 1, self.mch_fea_dim),
        )
        self.mch_mask[self.multi_env_mch_diag] = 1
        
    def construct_op_features(self):
        self.fea_j = np.stack(
            (
                self.op_scheduled_flag,
                self.op_ct_lb,
                self.op_min_pt,
                self.pt_span,
                self.op_mean_pt,
                self.op_waiting_time,
                self.op_remain_work,
                self.op_match_job_left_op_nums,
                self.op_match_job_remain_work,
                self.op_available_mch_nums,
            ),
            axis=2,
        )
        if self.step_count != self.number_of_ops:
            self.norm_op_features()

    def norm_op_features(self):
        self.fea_j[self.delete_mask_fea_j] = 0
        num_delete_nodes = np.count_nonzero(self.deleted_op_nodes, axis=1)
        num_delete_nodes = num_delete_nodes[:, np.newaxis]
        num_left_nodes = self.number_of_ops - num_delete_nodes
        mean_fea_j = np.sum(self.fea_j, axis=1) / num_left_nodes
        temp = np.where(
            self.delete_mask_fea_j, mean_fea_j[:, np.newaxis, :], self.fea_j
        )
        var_fea_j = np.var(temp, axis=1)
        std_fea_j = np.sqrt(var_fea_j * self.number_of_ops / num_left_nodes)
        self.fea_j = (temp - mean_fea_j[:, np.newaxis, :]) / (
            std_fea_j[:, np.newaxis, :] + 1e-8
        )

    def construct_mch_features(self):
        self.fea_m = np.stack(
            (
                self.mch_current_available_jc_nums,
                self.mch_current_available_op_nums,
                self.mch_min_pt,
                self.mch_mean_pt,
                self.mch_waiting_time,
                self.mch_remain_work,
                self.mch_free_time,
                self.mch_working_flag,
            ),
            axis=2,
        )
        if self.step_count != self.number_of_ops:
            self.norm_machine_features()

    def norm_machine_features(self):
        self.fea_m[self.delete_mask_fea_m] = 0
        num_delete_mchs = np.count_nonzero(self.delete_mask_fea_m[:, :, 0], axis=1)
        num_delete_mchs = num_delete_mchs[:, np.newaxis]
        num_left_mchs = self.number_of_machines - num_delete_mchs
        mean_fea_m = np.sum(self.fea_m, axis=1) / num_left_mchs
        temp = np.where(
            self.delete_mask_fea_m, mean_fea_m[:, np.newaxis, :], self.fea_m
        )
        var_fea_m = np.var(temp, axis=1)
        std_fea_m = np.sqrt(var_fea_m * self.number_of_machines / num_left_mchs)
        self.fea_m = (temp - mean_fea_m[:, np.newaxis, :]) / (
            std_fea_m[:, np.newaxis, :] + 1e-8
        )

    def construct_pair_features(self):
        remain_op_pt = ma.array(self.op_pt, mask=~self.remain_process_relation)
        chosen_op_max_pt = np.expand_dims(
            self.op_max_pt[self.env_job_idx, self.candidate], axis=-1
        )
        max_remain_op_pt = np.max(
            np.max(remain_op_pt, axis=1, keepdims=True), axis=2, keepdims=True
        ).filled(0 + 1e-8)
        mch_max_remain_op_pt = np.max(remain_op_pt, axis=1, keepdims=True).filled(
            0 + 1e-8
        )
        pair_max_pt = (
            np.max(
                np.max(self.candidate_pt, axis=1, keepdims=True), axis=2, keepdims=True
            )
            + 1e-8
        )
        mch_max_candidate_pt = np.max(self.candidate_pt, axis=1, keepdims=True) + 1e-8
        pair_wait_time = (
            self.op_waiting_time[self.env_job_idx, self.candidate][:, :, np.newaxis]
            + self.mch_waiting_time[:, np.newaxis, :]
        )
        chosen_job_remain_work = (
            np.expand_dims(
                self.op_match_job_remain_work[self.env_job_idx, self.candidate], axis=-1
            )
            + 1e-8
        )
        self.fea_pairs = np.stack(
            (
                self.candidate_pt,
                self.candidate_pt / chosen_op_max_pt,
                self.candidate_pt / mch_max_candidate_pt,
                self.candidate_pt / max_remain_op_pt,
                self.candidate_pt / mch_max_remain_op_pt,
                self.candidate_pt / pair_max_pt,
                self.candidate_pt / chosen_job_remain_work,
                pair_wait_time,
            ),
            axis=-1,
        )

    def update_op_mask(self):
        object_mask = np.zeros_like(self.op_mask)
        object_mask[:, :, 2] = self.deleted_op_nodes
        object_mask[:, 1:, 0] = self.deleted_op_nodes[:, :-1]
        self.op_mask = np.logical_or(object_mask, self.op_mask).astype(np.float32)

    def update_mch_mask(self):
        self.mch_mask = (
            self.logic_operator(self.remain_process_relation).sum(axis=-1).astype(bool)
        )
        self.delete_mask_fea_m = np.tile(
            ~(np.sum(self.mch_mask, keepdims=True, axis=-1).astype(bool)),
            (1, 1, self.mch_fea_dim),
        )
        self.mch_mask[self.multi_env_mch_diag] = 1

    def done(self):
        return np.ones(self.number_of_envs) * (self.step_count >= self.number_of_ops)

    def _step_core(self, chosen_job, chosen_mch, active_envs):
        # Requires self.env_idxs to be set to active_envs
        # This is a modified version of the original step logic
        
        chosen_op = self.candidate[active_envs, chosen_job]
        
        # Validation checks can be skipped if we trust the loop calling this
        
        self.step_count += 1 # Note: This might be inaccurate if we step multiple times per macro-step

        candidate_add_flag = chosen_op != self.job_last_op_id[active_envs, chosen_job]
        self.candidate[active_envs, chosen_job] += candidate_add_flag
        self.mask[active_envs, chosen_job] = 1 - candidate_add_flag

        chosen_op_st = np.maximum(
            self.candidate_free_time[active_envs, chosen_job],
            self.mch_free_time[active_envs, chosen_mch],
        )

        self.op_ct[active_envs, chosen_op] = (
            chosen_op_st + self.op_pt[active_envs, chosen_op, chosen_mch]
        )
        self.candidate_free_time[active_envs, chosen_job] = self.op_ct[
            active_envs, chosen_op
        ]
        self.mch_free_time[active_envs, chosen_mch] = self.op_ct[
            active_envs, chosen_op
        ]

        true_chosen_op_st = np.maximum(
            self.true_candidate_free_time[active_envs, chosen_job],
            self.true_mch_free_time[active_envs, chosen_mch],
        )
        self.true_op_ct[active_envs, chosen_op] = (
            true_chosen_op_st + self.true_op_pt[active_envs, chosen_op, chosen_mch]
        )
        self.true_candidate_free_time[active_envs, chosen_job] = self.true_op_ct[
            active_envs, chosen_op
        ]
        self.true_mch_free_time[active_envs, chosen_mch] = self.true_op_ct[
            active_envs, chosen_op
        ]

        self.current_makespan[active_envs] = np.maximum(
            self.current_makespan[active_envs], self.true_op_ct[active_envs, chosen_op]
        )

        # Partial update of candidate message
        mask_temp = candidate_add_flag
        # Note: indexing with mask_temp inside active_envs
        # active_envs[mask_temp] indices where flag is true
        
        # self.candidate_pt[active_envs, chosen_job] is problematic if we update inplace for all envs
        # But here we only update active_envs rows.
        
        # Indices in full arrays:
        # active_envs is shape [K]
        # chosen_job is shape [K]
        # mask_temp is shape [K]
        
        # Rows to update: active_envs[mask_temp]
        # Jobs to update: chosen_job[mask_temp]
        
        idxs = active_envs[mask_temp]
        jobs = chosen_job[mask_temp]
        ops = chosen_op[mask_temp] + 1
        
        self.candidate_pt[idxs, jobs] = self.unmasked_op_pt[idxs, ops]
        self.candidate_process_relation[idxs, jobs] = self.reverse_process_relation[idxs, ops]
        
        idxs_finished = active_envs[~mask_temp]
        jobs_finished = chosen_job[~mask_temp]
        self.candidate_process_relation[idxs_finished, jobs_finished] = 1

        # Calculate next schedule time (Global update or Partial?)
        # We need next schedule time for ALL envs, but we only changed some.
        # It's safer to recompute entirely or carefully update.
        # Recomputing is safer.
        
        # [E, J, M]
        candidateFT_for_compare = np.expand_dims(self.candidate_free_time, axis=2)
        mchFT_for_compare = np.expand_dims(self.mch_free_time, axis=1)
        self.pair_free_time = np.maximum(candidateFT_for_compare, mchFT_for_compare)

        schedule_matrix = ma.array(
            self.pair_free_time, mask=self.candidate_process_relation
        )

        self.next_schedule_time = np.min(
            schedule_matrix.reshape(self.number_of_envs, -1), axis=1
        ).data

        self.remain_process_relation[active_envs, chosen_op] = 0
        self.op_scheduled_flag[active_envs, chosen_op] = 1

        # deleted_op_nodes update
        self.deleted_op_nodes = np.logical_and(
            (self.op_ct <= self.next_schedule_time[:, np.newaxis]),
            self.op_scheduled_flag,
        )
        self.delete_mask_fea_j = np.tile(
            self.deleted_op_nodes[:, :, np.newaxis], (1, 1, self.op_fea_dim)
        )
        
        # Trigger Full Feature Updates (simplest way to ensure consistency)
        self.update_op_mask()
        
        # Update op_ct_lb (complex update)
        diff = (
            self.op_ct[active_envs, chosen_op]
            - self.op_ct_lb[active_envs, chosen_op]
        )
        # Broadcasting diff to relevant ops is tricky with partial updates.
        # Let's just loop for simplicity in this port
        for k, d in zip(active_envs, diff):
             j = chosen_job[np.where(active_envs == k)[0][0]]
             op = chosen_op[np.where(active_envs == k)[0][0]]
             
             # op_ct_lb update
             mask1 = (self.op_idx[0] >= op) & (self.op_idx[0] < self.job_last_op_id[k, j] + 1)
             self.op_ct_lb[k, mask1] += d
             
             # op_match_job... update
             mask2 = (self.op_idx[0] >= self.job_first_op_id[k, j]) & (self.op_idx[0] < self.job_last_op_id[k, j] + 1)
             self.op_match_job_left_op_nums[k, mask2] -= 1
             self.op_match_job_remain_work[k, mask2] -= self.op_mean_pt[k, op]

        # Waiting time
        self.op_waiting_time = np.zeros((self.number_of_envs, self.number_of_ops))
        # vectorized full update
        self.op_waiting_time[self.env_job_idx, self.candidate] = (
            1 - self.mask
        ) * np.maximum(
            np.expand_dims(self.next_schedule_time, axis=1) - self.candidate_free_time,
            0,
        ) + self.mask * self.op_waiting_time[
            self.env_job_idx, self.candidate
        ]
        
        self.op_remain_work = np.maximum(
             self.op_ct - np.expand_dims(self.next_schedule_time, axis=1), 0
        )
        
        self.construct_op_features()
        
        self.dynamic_pair_mask = np.copy(self.candidate_process_relation)
        self.unavailable_pairs = (
            self.pair_free_time > self.next_schedule_time[:, np.newaxis, np.newaxis]
        )
        self.dynamic_pair_mask = np.logical_or(
            self.dynamic_pair_mask, self.unavailable_pairs
        )
        self.comp_idx = self.logic_operator(x=~self.dynamic_pair_mask)
        self.update_mch_mask()
        
        self.mch_current_available_jc_nums = np.sum(~self.dynamic_pair_mask, axis=1)
        
        # Partial update for current op removal?
        # self.process_relation[active_envs, chosen_op] is for specific ops
        # We need to subtract 1 from mch_current_available_op_nums[active_envs] for the valid machines of chosen_op
        # This is getting complicated to be exact. 
        # But we can just RECOMPUTE from scratch if needed.
        # Or trust the update:
        # self.mch_current_available_op_nums -= self.process_relation[active_envs, chosen_op] # shape mismatch
        
        # Simple full recompute of mch_current..
        # compatible_mch is static. 
        # mch_current_available_op_nums is num of ops available for mch.
        # It should decrease as ops are scheduled.
        # Actually doing full recompute is safer.
        # But implementation uses incremental update:
        # self.mch_current_available_op_nums -= self.process_relation[self.env_idxs, chosen_op]
        # Here we have subset keys.
        # We need to broadcast.
        
        # Loop mainly for correct logic
        for k, op in zip(active_envs, chosen_op):
             self.mch_current_available_op_nums[k] -= self.process_relation[k, op]

        mch_free_duration = (
            np.expand_dims(self.next_schedule_time, axis=1) - self.mch_free_time
        )
        mch_free_flag = mch_free_duration < 0
        self.mch_working_flag = mch_free_flag + 0
        self.mch_waiting_time = (1 - mch_free_flag) * mch_free_duration
        self.mch_remain_work = np.maximum(-mch_free_duration, 0)
        
        self.construct_mch_features()
        self.construct_pair_features()

    def step_madrl(self, actions_matrix):
        # actions_matrix: [batch_size, n_m] tensor/array.
        # Contains Job ID to schedule, or -1 for no-op.
        
        # We iterate over machines to apply updates sequentially WITHIN the time step.
        # Order is shuffled to ensure fairness/stochasticity.
        
        m_order = np.arange(self.number_of_machines)
        np.random.shuffle(m_order)
        
        # Track which envs have been updated to avoid double counting or sync issues?
        # No, sequential updates are fine.
        
        rewards = np.zeros(self.number_of_envs)
        
        # old_makespan
        old_quality = self.max_endTime.copy()
        
        for m in m_order:
            actions = actions_matrix[:, m]
            
            # Identify valid actions
            # condition: action != -1
            # AND job is not already taken by another machine in this loop?
            # We check if 'candidate[e, action]' is still valid for machine m.
            # But process_relation might have changed if another machine took the job? 
            # (No, if another machine took the job, 'candidate' index incremented)
            # So checking self.process_relation[e, self.candidate[e, action], m] verifies if op is still available for m.
            
            active_candidate_mask = []
            active_envs = []
            
            for e in range(self.number_of_envs):
                j = actions[e]
                if j == -1: continue
                # Check if job done
                if self.mask[e, j]: continue # Job done
                
                op = self.candidate[e, j]
                # Check compatibility
                if not self.process_relation[e, op, m]: continue
                
                # Check valid (not scheduled yet)
                if self.op_scheduled_flag[e, op]: continue
                
                active_envs.append(e)
                active_candidate_mask.append(True)
            
            if not active_envs: continue
            
            active_envs = np.array(active_envs)
            chosen_job = actions[active_envs]
            
            # Execute core step logic
            # We temporarily set self.env_idxs to specific envs if needed by helpers,
            # but helpers use broadcasting. We need to be careful.
            # actually most helpers use [self.env_idxs, ...], so just setting self.env_idxs works.
            
            original_env_idxs = self.env_idxs
            self.env_idxs = active_envs
            
            self._step_core(chosen_job, m, active_envs)
            
            self.env_idxs = original_env_idxs
            
        # Reward calculation: Change in max_endTime (Lower Bound of makespan)
        reward = old_quality - np.max(self.op_ct_lb, axis=1)
        self.max_endTime = np.max(self.op_ct_lb, axis=1)

        self.state.update(
            self.fea_j,
            self.op_mask,
            self.fea_m,
            self.mch_mask,
            self.dynamic_pair_mask,
            self.comp_idx,
            self.candidate,
            self.fea_pairs,
        )
        
        return self.state, reward, self.done()
