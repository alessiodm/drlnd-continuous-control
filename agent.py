import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataclasses import dataclass
from torch.distributions.normal import Normal


@dataclass
class LearningBatch:
    """Data used for learning by the Agent.

    The learning batch contains (shuffled and flattened) collected experiences for N environment
    bots, and their corresponding advantages and returns. Assuming the size of the batch is B, then
    for example the `states` dimension is (B, <state_dim>).

    Effectively, the learning batch is a bag of B tuples of:

        (state, action, logprob, advantage, return)

    Each tuple represents a single environment bot collected experience, hence the experiences in
    the learning batch can be randomly of different bots and in different `quantities` (in the
    case of mini-batches).
    """
    states: torch.Tensor        # (B, S_dim)
    actions: torch.Tensor       # (B, A_dim)
    logprobs: torch.Tensor      # (B)
    advantages: torch.Tensor    # (B)
    returns: torch.Tensor       # (B)

    def __post_init__(self):
        """Basic sanity dimensional checks."""
        length = self.states.shape[0]
        assert length > 0
        assert self.actions.shape[0]  == length
        assert self.logprobs.shape[0] == length
        assert self.advantages.shape[0]   == length
        assert self.returns.shape[0]  == length

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, key):
        """Ability to slice a learning batch in mini-batches."""
        return LearningBatch(self.states[key], self.actions[key],
                           self.logprobs[key], self.advantages[key],
                           self.returns[key])


class Agent(nn.Module):
    """The PPO agent implementing actor-critic learning."""
    def __init__(self, state_size, action_size, lr=2.5e-4, weight_mul=1e-3,
                 preload_file: str = None):
        super().__init__()

        def layer_init(layer, std=np.sqrt(2)):
            """Layer initialization for the neural-netowork linear layers.

            Scaling the weights might affect learning speed.
            """
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, 0.0)
            layer.weight.data.mul_(weight_mul)
            return layer

        # Critic network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_size, 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, 1), std=1.0),
        )

        # Actor network for the mean (and std deviation parameter)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_size, 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, action_size), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_size))
        # Adam optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-5)

        if preload_file is not None:
            print(f'Loading pre-trained model: {preload_file}')
            self.load_state_dict(torch.load(preload_file))

    def get_value(self, states):
        """Returns the estimated value of a state given by the critic."""
        return self.critic(states)

    def sample_action(self, states):
        """Samples an action using the current policy for the `states` passed as input.

        It returns the action itself, and its log-probability over the action space.
        """
        probs = self.get_action_probs(states)
        action = probs.sample()
        return action, probs.log_prob(action).sum(1)

    def eval_action(self, states, action):
        """Evaluates an action using the *current* (possibly updated) policy.

        It returns the log-probability of the action, along with the entropy (for entropy loss). 
        """
        probs = self.get_action_probs(states)
        return probs.log_prob(action).sum(1), probs.entropy().sum(1)

    def get_action_probs(self, states):
        """Returns the probability distribution over the action space.

        Generally, policy-gradient methods assume the continuous actions are sampled from a normal
        distribution (hence, our neural network outputs mean and std of the gaussian).
        """
        action_mean = self.actor_mean(states)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        return Normal(action_mean, action_std)

    def learn(self, batch: LearningBatch, entropy_coeff=0.0, clip_coeff=0.1,
              max_grad_norm=0.75, normalize_advantages=False):
        """PPO learning step on a mini-batch.

        Instead of using the PPO single loss function (which is required if actor and critic
        share layers), actor and critic losses are trained separately - just to keep more along
        the lines of the Udacity lectures on actor-critic methods.

        Paper: https://arxiv.org/abs/1707.06347
        """
        newlogprobs, entropy = self.eval_action(batch.states, batch.actions)

        logratio = newlogprobs - batch.logprobs
        ratio = logratio.exp()
        clipped_ratio = torch.clamp(ratio, 1 - clip_coeff, 1 + clip_coeff)
        advantages = batch.advantages

        # Normalizing the advantages (at the mini-batch level) doesn't seem to improve
        # the learning process. Just leaving the code for more experiments :)
        if normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss. The actor effectively maximizes the advantages scaled by the
        # probability ratio (~re-weighting factor in importance sampling to be able
        # to share previous experiences in a new version of the policy) clipped to
        # effectively keep the policy updates in the vicinity of the previous version.
        L_entropy = entropy_coeff * entropy.mean()
        L_clipped = -torch.min(advantages * ratio, advantages * clipped_ratio).mean()
        L_actor = L_clipped - L_entropy

        self.optimizer.zero_grad()
        L_actor.backward()
        # Gradients are also clipped to avoid too large of an update.
        nn.utils.clip_grad_norm_(self.actor_mean.parameters(), max_grad_norm)
        self.optimizer.step()

        # Value loss. Mean squared error of the predicted values vs. the actual returns.
        newvalues = self.get_value(batch.states).view(-1)
        L_critic = F.mse_loss(newvalues, batch.returns)
        
        self.optimizer.zero_grad()
        L_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
        self.optimizer.step()

        # PPO loss (in case we want to actually implement PPO from the paper).
        # vf_coeff = 0.5 # This works better if 1.0 in this case
        # L_ppo = L_actor + L_critic * VF_COEFF
        # self.optimizer.zero_grad()
        # L_ppo.backward()
        # nn.utils.clip_grad_norm_(self.parameters(), MAX_GRAD_NORM)
        # self.optimizer.step()

    def save(self):
        """Save the agent weights in the 'weights.pth' file."""
        torch.save(self.state_dict(), 'weights.pth')
