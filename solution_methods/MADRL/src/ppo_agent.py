import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=64):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor head: Outputs scores for each item in the queue
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1) # Score for one item
        )
        
        # Critic head: Value of the state
        # We need to aggregate features from all items in queue for the state value
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self):
        raise NotImplementedError

    def get_action_prob(self, state, queue_mask):
        """
        state: [Batch, MaxQueue, ObsDim]
        queue_mask: [Batch, MaxQueue] (1 if item exists, 0 if padding)
        """
        batch_size, max_queue, obs_dim = state.shape
        
        # Flatten to process each item independently
        flat_state = state.view(-1, obs_dim)
        features = self.feature_extractor(flat_state) #[Batch*MaxQueue, Hidden]
        
        # Actor
        scores = self.actor(features).view(batch_size, max_queue)
        
        # Masking
        scores = scores.masked_fill(queue_mask == 0, -1e9)
        probs = F.softmax(scores, dim=1)
        
        return probs

    def get_value(self, state, queue_mask):
        batch_size, max_queue, obs_dim = state.shape
        flat_state = state.view(-1, obs_dim)
        features = self.feature_extractor(flat_state).view(batch_size, max_queue, -1)
        
        # Global pooling (e.g., mean of valid items) represents the machine state
        # Simplified: Sum masked features / Sum mask
        mask_expanded = queue_mask.unsqueeze(-1)
        sum_features = (features * mask_expanded).sum(dim=1)
        count = mask_expanded.sum(dim=1) + 1e-5
        pooled_features = sum_features / count
        
        value = self.critic(pooled_features)
        return value

class PPOAgent:
    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=4):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(obs_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(obs_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        # State: [MaxQueue, ObsDim]
        # We need to add batch dim
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Create mask (check which rows are not all zeros)
        # obs_dim is last dim
        queue_mask = (torch.sum(torch.abs(state_tensor), dim=2) > 0).float()
        
        with torch.no_grad():
            probs = self.policy_old.get_action_prob(state_tensor, queue_mask)
        
        dist = Categorical(probs)
        action = dist.sample()
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        memory.masks.append(queue_mask)
        
        return action.item()

    def update(self, memory):
        # Convert list to tensor
        old_states = torch.FloatTensor(np.array(memory.states))
        old_actions = torch.stack(memory.actions, dim=0).detach().squeeze()
        old_logprobs = torch.stack(memory.logprobs, dim=0).detach().squeeze()
        old_masks = torch.stack(memory.masks, dim=0).detach().squeeze()
        
        # Monte Carlo estimate of rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32)
        # Normalize
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluate old actions and values
            logprobs = []
            state_values = []
            dist_entropy = []
            
            # Re-evaluate logic (simplified, usually done in batch)
            curr_probs = self.policy.get_action_prob(old_states, old_masks)
            dist = Categorical(curr_probs)
            
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            state_values = self.policy.get_value(old_states, old_masks).squeeze()
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

import numpy as np

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.masks = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.masks[:]
