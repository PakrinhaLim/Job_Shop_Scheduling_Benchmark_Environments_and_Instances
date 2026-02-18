
import torch
import torch.nn as nn
import torch.nn.functional as F
from solution_methods.DANIEL.network.main_model import DualAttentionNetwork
from solution_methods.DANIEL.network.sub_layers import Actor, Critic

class MADRL_Network(nn.Module):
    def __init__(self, config):
        """
            MADRL Network re-using DANIEL's Dual Attention Network
        """
        super(MADRL_Network, self).__init__()
        device = torch.device(config["device"]["name"])

        # pair features input dim with fixed value
        self.pair_input_dim = 8

        self.embedding_output_dim = config["network"]["layer_fea_output_dim"][-1]

        self.feature_exact = DualAttentionNetwork(config).to(device)
        self.actor = Actor(config["network"]["num_mlp_layers_actor"], 4 * self.embedding_output_dim + self.pair_input_dim,
                           config["network"]["hidden_dim_actor"], 1).to(device)
        self.critic = Critic(config["network"]["num_mlp_layers_critic"], 2 * self.embedding_output_dim, config["network"]["hidden_dim_critic"],
                             1).to(device)

    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, dynamic_pair_mask, fea_pairs):
        """
        Returns:
            pi: [sz_b, J, M] - Probability of each job for each machine
            v: [sz_b, 1] - Value of state
        """

        fea_j, fea_m, fea_j_global, fea_m_global = self.feature_exact(fea_j, op_mask, candidate, fea_m, mch_mask,
                                                                      comp_idx)
        sz_b, M, _, J = comp_idx.size()
        d = fea_j.size(-1)

        # collect the input of decision-making network
        candidate_idx = candidate.unsqueeze(-1).repeat(1, 1, d)
        candidate_idx = candidate_idx.type(torch.int64)

        Fea_j_JC = torch.gather(fea_j, 1, candidate_idx)

        # [sz_b, J, M, d] -> [sz_b, J*M, d]
        Fea_j_JC_serialized = Fea_j_JC.unsqueeze(2).repeat(1, 1, M, 1).reshape(sz_b, M * J, d)
        Fea_m_serialized = fea_m.unsqueeze(1).repeat(1, J, 1, 1).reshape(sz_b, M * J, d)

        Fea_Gj_input = fea_j_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
        Fea_Gm_input = fea_m_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)

        fea_pairs_reshaped = fea_pairs.reshape(sz_b, -1, self.pair_input_dim)
        
        candidate_feature = torch.cat((Fea_j_JC_serialized, Fea_m_serialized, Fea_Gj_input,
                                       Fea_Gm_input, fea_pairs_reshaped), dim=-1)

        candidate_scores = self.actor(candidate_feature) # [sz_b, J*M, 1]
        candidate_scores = candidate_scores.squeeze(-1).reshape(sz_b, J, M) # [sz_b, J, M]

        # masking incompatible op-mch pairs
        # dynamic_pair_mask is [sz_b, J, M]
        candidate_scores[dynamic_pair_mask] = float('-inf')
        
        # Softmax over J for each M
        pi = F.softmax(candidate_scores, dim=1) # [sz_b, J, M]

        # Handle NaNs if a machine has no compatible jobs (all -inf)
        if torch.isnan(pi).any():
            pi = torch.where(torch.isnan(pi), torch.full_like(pi, 1.0/J), pi)

        global_feature = torch.cat((fea_j_global, fea_m_global), dim=-1)
        v = self.critic(global_feature)
        
        return pi, v
