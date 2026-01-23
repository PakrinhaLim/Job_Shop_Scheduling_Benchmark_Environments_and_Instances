
from solution_methods.MADRL.network.madrl_model import MADRL_Network
import torch.nn as nn
import torch
from copy import deepcopy
import numpy as np

def eval_actions_madrl(pis, actions):
    # pis: [sz_b, J, M]
    # actions: [sz_b, M]
    
    # We need log_prob of chosen actions.
    # Actions are indices in dim 1 (J).
    
    # Handle -1 in actions (no-op)
    # Clamp -1 to 0 to avoid index error, then mask result
    mask = (actions != -1)
    safe_actions = actions.clone()
    safe_actions[~mask] = 0
    
    # pis for chosen actions
    # gather dim=1. index shape must match pis
    # We want to select, for each (b, m), the probability at index safe_actions[b, m]
    
    # pis is [B, J, M]
    # Permute to [B, M, J] to align with gather
    pis_perm = pis.permute(0, 2, 1) # [B, M, J]
    
    # safe_actions is [B, M]. Unsqueeze to [B, M, 1]
    actions_idx = safe_actions.unsqueeze(-1)
    
    # gather
    selected_probs = pis_perm.gather(2, actions_idx).squeeze(-1) # [B, M]
    
    dist_entropy = -(pis * torch.log(pis + 1e-9)).sum(dim=1) # [B, M]
    
    log_probs = torch.log(selected_probs + 1e-9)
    
    # apply mask
    log_probs = log_probs * mask.float()
    dist_entropy = dist_entropy * mask.float()
    
    return log_probs, dist_entropy, mask

class Memory:
    def __init__(self, gamma, gae_lambda):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        # input variables
        self.fea_j_seq = []
        self.op_mask_seq = []
        self.fea_m_seq = []
        self.mch_mask_seq = []
        self.dynamic_pair_mask_seq = []
        self.comp_idx_seq = []
        self.candidate_seq = []
        self.fea_pairs_seq = []

        self.action_seq = []  # [sz_b, M]
        self.reward_seq = []  # [sz_b]
        self.val_seq = []     # [sz_b]
        self.done_seq = []    # [sz_b]
        self.log_probs = []   # [sz_b, M]

    def clear_memory(self):
        del self.fea_j_seq[:]
        del self.op_mask_seq[:]
        del self.fea_m_seq[:]
        del self.mch_mask_seq[:]
        del self.dynamic_pair_mask_seq[:]
        del self.comp_idx_seq[:]
        del self.candidate_seq[:]
        del self.fea_pairs_seq[:]
        del self.action_seq[:]
        del self.reward_seq[:]
        del self.val_seq[:]
        del self.done_seq[:]
        del self.log_probs[:]

    def push(self, state):
        self.fea_j_seq.append(state.fea_j_tensor)
        self.op_mask_seq.append(state.op_mask_tensor)
        self.fea_m_seq.append(state.fea_m_tensor)
        self.mch_mask_seq.append(state.mch_mask_tensor)
        self.dynamic_pair_mask_seq.append(state.dynamic_pair_mask_tensor)
        self.comp_idx_seq.append(state.comp_idx_tensor)
        self.candidate_seq.append(state.candidate_tensor)
        self.fea_pairs_seq.append(state.fea_pairs_tensor)

    def transpose_data(self):
        t_Fea_j_seq = torch.stack(self.fea_j_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_op_mask_seq = torch.stack(self.op_mask_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_Fea_m_seq = torch.stack(self.fea_m_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_mch_mask_seq = torch.stack(self.mch_mask_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_dynamicMask_seq = torch.stack(self.dynamic_pair_mask_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_Compete_m_seq = torch.stack(self.comp_idx_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_candidate_seq = torch.stack(self.candidate_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_pairMessage_seq = torch.stack(self.fea_pairs_seq, dim=0).transpose(0, 1).flatten(0, 1)
        
        # [N, sz_b, M] -> [sz_b, N, M] -> [sz_b*N, M]
        t_action_seq = torch.stack(self.action_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_logprobs_seq = torch.stack(self.log_probs, dim=0).transpose(0, 1).flatten(0, 1)
        
        # [N, sz_b] -> [sz_b, N] -> [sz_b*N]
        t_reward_seq = torch.stack(self.reward_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_done_seq = torch.stack(self.done_seq, dim=0).transpose(0, 1).flatten(0, 1)
        
        self.t_old_val_seq = torch.stack(self.val_seq, dim=0).transpose(0, 1)
        t_val_seq = self.t_old_val_seq.flatten(0, 1)

        return t_Fea_j_seq, t_op_mask_seq, t_Fea_m_seq, t_mch_mask_seq, t_dynamicMask_seq, \
               t_Compete_m_seq, t_candidate_seq, t_pairMessage_seq, \
               t_action_seq, t_reward_seq, t_val_seq, t_done_seq, t_logprobs_seq

    def get_gae_advantages(self):
        reward_arr = torch.stack(self.reward_seq, dim=0)
        values = self.t_old_val_seq.transpose(0, 1)
        len_trajectory, len_envs = reward_arr.shape

        advantage = torch.zeros(len_envs, device=values.device)
        advantage_seq = []
        for i in reversed(range(len_trajectory)):
            if i == len_trajectory - 1:
                delta_t = reward_arr[i] - values[i]
            else:
                delta_t = reward_arr[i] + self.gamma * values[i + 1] - values[i]
            advantage = delta_t + self.gamma * self.gae_lambda * advantage
            advantage_seq.insert(0, advantage)

        t_advantage_seq = torch.stack(advantage_seq, dim=0).transpose(0, 1).to(torch.float32)
        v_target_seq = (t_advantage_seq + self.t_old_val_seq).flatten(0, 1)
        t_advantage_seq = (t_advantage_seq - t_advantage_seq.mean(dim=1, keepdim=True)) \
                          / (t_advantage_seq.std(dim=1, keepdim=True) + 1e-8)

        return t_advantage_seq.flatten(0, 1), v_target_seq


class PPO_MADRL:
    def __init__(self, config):
        self.lr = config["PPO_Algorithm"]["lr"]
        self.gamma = config["PPO_Algorithm"]["gamma"]
        self.gae_lambda = config["PPO_Algorithm"]["gae_lambda"]
        self.eps_clip = config["PPO_Algorithm"]["eps_clip"]
        self.k_epochs = config["PPO_Algorithm"]["k_epochs"]
        self.tau = config["PPO_Algorithm"]["tau"]
        self.ploss_coef = config["PPO_Algorithm"]["ploss_coef"]
        self.vloss_coef = config["PPO_Algorithm"]["vloss_coef"]
        self.entloss_coef = config["PPO_Algorithm"]["entloss_coef"]
        self.minibatch_size = config["training"]["minibatch_size"]

        self.policy = MADRL_Network(config)
        self.policy_old = deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.V_loss_2 = nn.MSELoss()
        self.device = torch.device(config["device"]["name"])

    def update(self, memory):
        t_data = memory.transpose_data()
        t_advantage_seq, v_target_seq = memory.get_gae_advantages()

        full_batch_size = len(t_data[-1])
        num_batch = np.ceil(full_batch_size / self.minibatch_size)

        loss_epochs = 0
        v_loss_epochs = 0
        
        # t_advantage_seq is [Batch]. We need to broadcast to [Batch, M] for actor update
        # But we can just repeat_interleave logic or broadcast in calc
        
        for _ in range(self.k_epochs):
            for i in range(int(num_batch)):
                start_idx = i * self.minibatch_size
                end_idx = min((i + 1) * self.minibatch_size, full_batch_size) 
                if start_idx >= end_idx: break

                pis, vals = self.policy(fea_j=t_data[0][start_idx:end_idx],
                                        op_mask=t_data[1][start_idx:end_idx],
                                        candidate=t_data[6][start_idx:end_idx],
                                        fea_m=t_data[2][start_idx:end_idx],
                                        mch_mask=t_data[3][start_idx:end_idx],
                                        comp_idx=t_data[5][start_idx:end_idx],
                                        dynamic_pair_mask=t_data[4][start_idx:end_idx],
                                        fea_pairs=t_data[7][start_idx:end_idx])
                
                # pis: [MiniBatch, J, M]
                # actions: [MiniBatch, M]
                action_batch = t_data[8][start_idx: end_idx]
                logprobs, ent_loss, mask = eval_actions_madrl(pis, action_batch)
                
                # logprobs: [MiniBatch, M]
                # old_logprobs: [MiniBatch, M]
                old_logprobs = t_data[12][start_idx: end_idx].detach()
                
                ratios = torch.exp(logprobs - old_logprobs)
                
                # Advantage: [MiniBatch] -> [MiniBatch, M]
                advantages = t_advantage_seq[start_idx: end_idx].unsqueeze(1).expand_as(ratios)
                
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                
                # Mask out invalid actions
                surr1 = surr1 * mask.float()
                surr2 = surr2 * mask.float()
                
                p_loss = - torch.min(surr1, surr2).sum() / (mask.sum() + 1e-8)
                ent_loss = - ent_loss.sum() / (mask.sum() + 1e-8)
                
                v_loss = self.V_loss_2(vals.squeeze(1), v_target_seq[start_idx: end_idx])
                
                loss = self.vloss_coef * v_loss + self.ploss_coef * p_loss + self.entloss_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                loss_epochs += loss.item()
                v_loss_epochs += v_loss.item()
                
        # soft update
        for policy_old_params, policy_params in zip(self.policy_old.parameters(), self.policy.parameters()):
            policy_old_params.data.copy_(self.tau * policy_old_params.data + (1 - self.tau) * policy_params.data)
            
        return loss_epochs / self.k_epochs, v_loss_epochs / self.k_epochs

def PPO_initialize(config):
    return PPO_MADRL(config)
