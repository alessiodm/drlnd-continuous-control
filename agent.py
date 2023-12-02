import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple
from torch.distributions.normal import Normal

CLIP_COEFF=0.1
# ENT_COEFF=0.01
# VF_COEFF=0.5
ENT_COEFF=0.0
VF_COEFF=1.0
MAX_GRAD_NORM=0.75

MiniBatch = namedtuple("MiniBatch", field_names=["states", "actions", "logprobs",
                                                 "advantages", "returns"])

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """TODO:"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    layer.weight.data.mul_(1e-3) # CRITICAL!
    return layer

class Agent(nn.Module):
    def __init__(self, state_size, action_size, lr=2.5e-4):
        super().__init__()
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

    def learn(self, batch: MiniBatch):
        newlogprobs, entropy = self.eval_action(batch.states, batch.actions)

        logratio = newlogprobs - batch.logprobs
        ratio = logratio.exp()
        clip = torch.clamp(ratio, 1 - CLIP_COEFF, 1 + CLIP_COEFF)
        advantages = batch.advantages

        # Policy loss.
        pg_loss = torch.max(-advantages * ratio, -advantages * clip).mean()

        newvalues = self.get_value(batch.states).view(-1)

        v_loss = F.mse_loss(newvalues, batch.returns)
        entropy_loss = entropy.mean()

        ppo_loss = pg_loss + VF_COEFF * v_loss - ENT_COEFF * entropy_loss

        self.optimizer.zero_grad()
        ppo_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()

        return (ppo_loss, pg_loss, v_loss, logratio)
