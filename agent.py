import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataclasses import dataclass
from torch.distributions.normal import Normal


@dataclass
class LearningBatch:
    states: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    def __post_init__(self):
        length = self.states.shape[0]
        assert length > 0
        assert self.actions.shape[0]  == length
        assert self.logprobs.shape[0] == length
        assert self.advantages.shape[0]   == length
        assert self.returns.shape[0]  == length

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, key):
        return LearningBatch(self.states[key], self.actions[key],
                           self.logprobs[key], self.advantages[key],
                           self.returns[key])


class Agent(nn.Module):
    def __init__(self, state_size, action_size, lr=2.5e-4, weight_mul=1e-3,
                 preload_file: str = None):
        super().__init__()

        def layer_init(layer, std=np.sqrt(2)):
            """TODO:"""
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, 0.0)
            layer.weight.data.mul_(weight_mul) # CRITICAL!
            return layer

        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_size, 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, 1), std=1.0),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_size, 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, action_size), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, action_size))
        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-5)

        if preload_file is not None:
            print(f'Loading pre-trained model: {preload_file}')
            self.load_state_dict(torch.load(preload_file))

    def get_value(self, states):
        return self.critic(states)

    def sample_action(self, states):
        probs = self.get_action_probs(states)
        action = probs.sample()
        return action, probs.log_prob(action).sum(1)

    def eval_action(self, states, action):
        probs = self.get_action_probs(states)
        return probs.log_prob(action).sum(1), probs.entropy().sum(1)

    def get_action_probs(self, states):
        action_mean = self.actor_mean(states)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        return Normal(action_mean, action_std)

    def learn(self, batch: LearningBatch, entropy_coeff=0.0, clip_coeff=0.1,
              max_grad_norm=0.75, normalize_advantages=False):
        newlogprobs, entropy = self.eval_action(batch.states, batch.actions)

        logratio = newlogprobs - batch.logprobs
        ratio = logratio.exp()
        clipped_ratio = torch.clamp(ratio, 1 - clip_coeff, 1 + clip_coeff)
        advantages = batch.advantages

        # Normalizing the advantages (at the mini-batch level) doesn't seem to improve
        # the learning process. Just leaving the code for more experiments :)
        if normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss.
        L_entropy = entropy_coeff * entropy.mean()
        L_clipped = -torch.min(advantages * ratio, advantages * clipped_ratio).mean()
        L_actor = L_clipped - L_entropy

        self.optimizer.zero_grad()
        L_actor.backward()
        nn.utils.clip_grad_norm_(self.actor_mean.parameters(), max_grad_norm)
        self.optimizer.step()

        # Value loss
        newvalues = self.get_value(batch.states).view(-1)
        L_critic = F.mse_loss(newvalues, batch.returns)
        
        self.optimizer.zero_grad()
        L_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
        self.optimizer.step()

        # PPO loss
        # vf_coeff = 0.5 # This works better if 1.0 in this case
        # L_ppo = L_actor + L_critic * VF_COEFF
        # self.optimizer.zero_grad()
        # L_ppo.backward()
        # nn.utils.clip_grad_norm_(self.parameters(), MAX_GRAD_NORM)
        # self.optimizer.step()

    def save(self):
        """Save the agent weights in the 'weights.pth' file."""
        torch.save(self.state_dict(), 'weights.pth')
