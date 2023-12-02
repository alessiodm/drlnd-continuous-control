import numpy as np
import torch

from typing import Tuple
from unityagents import UnityEnvironment

from agent import Agent, MiniBatch

# TODO: Make these constants configurable
NUM_EPISODES = 300
NUM_UPDATE_EPOCHS = 10
NUM_MINI_BATCHES = 100
# GAMMA=0.995
GAMMA=0.99
GAE_LAMBDA=0.95

ROLLOUT_SIZE=250

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPO:
    """PPO implementation."""
    def __init__(self, env: UnityEnvironment, agent: Agent, solved_score = 30.):
        self.env = env
        self.agent = agent.to(device)
        self.solved_score = solved_score
        self.brain_name: str = env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.action_size = self.brain.vector_action_space_size
        env_info = self.env.reset(train_mode=False)[self.brain_name]
        self.num_agents = len(env_info.agents) # TODO: We should rename this to "experiences" or something.
        states = env_info.vector_observations
        self.state_size = states.shape[1]

    def train(self):
        def flatten(t: Tuple[torch.Tensor, ...]) -> Tuple:
            return tuple(x.flatten(0, 1) for x in t)

        # Using episodes and max-steps-per-episode have many downsides, see:
        #   https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
        # We keep this strategy b/c the Udacity project rubric specifically requires
        # averaging all agents per episode for the success criteria.
        n_episode = 1
        scores = torch.zeros((0, self.num_agents))
        next_states = torch.Tensor(self.env_reset()).to(device)   # get the initial state (for each agent)
        while True:
            # TODO: anneal learning rate?

            # Policy rollout phase.
            # Get more trajectories from many agents to reduce noise.
            # TODO: Improve explanation, pointing to the lecture slides about noise.            
            states, values, actions, logprobs, rewards, dones = self.collect_trajectories(next_states)
            next_states = states[-1]
            scores = torch.cat((scores, rewards), 0) # TODO: Weird the first iteration goes up to 1250...

            batch_size = self.num_agents * ROLLOUT_SIZE
            mini_batch_size = int(batch_size // NUM_MINI_BATCHES)

            # Bootstrapping value and compute advantages and returns.
            advantages, returns = self.advantages_and_returns(values, rewards, dones)

            states, values, actions, logprobs, rewards, dones, advantages, returns = flatten(
                (states, values, actions, logprobs, rewards, dones, advantages, returns))

            # Policy learning phase
            ppo_losses, pg_losses, v_losses, clipfracs = [], [], [], []
            indices = np.arange(batch_size)
            for epoch in range(NUM_UPDATE_EPOCHS):
                np.random.shuffle(indices)
                for start in range(0, batch_size, mini_batch_size):
                    end = start + mini_batch_size
                    inds = indices[start:end]
                    mini_batch = MiniBatch(states[inds], actions[inds], logprobs[inds],
                                          advantages[inds], returns[inds])
                    ppo_loss, pg_loss, v_loss, logratio = self.agent.learn(mini_batch)

                    # Just debug information...
                    ppo_losses.append(ppo_loss.item())
                    pg_losses.append(pg_loss.item())
                    v_losses.append(v_loss.item())
                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((logratio.exp() - 1) - logratio).mean()
                        clipfracs += [((logratio.exp() - 1.0).abs() > 0.1).float().mean().item()]

            if np.any(dones.numpy()):
                print(f'Episode n.{n_episode} completed. Avg score: {scores.sum(0).mean()}')
                print(f'\tppo_loss: {np.mean(ppo_losses)}, pg_loss: {np.mean(pg_losses)}, v_loss: {np.mean(v_losses)}')
                print(f'\told_approx_kl: {old_approx_kl:.2f}, approx_kl: {approx_kl:.2f}, clipfracs: {np.mean(clipfracs):.2f}')
                print()
                n_episode += 1
                scores = torch.zeros((0, self.num_agents))
                if n_episode > NUM_EPISODES:
                    break
 
    def collect_trajectories(self, init_states):
        #
        # https://knowledge.udacity.com/questions/558456
        #
        batch_dim = (ROLLOUT_SIZE, self.num_agents)

        states_list = torch.zeros(batch_dim + (self.state_size,)).to(device)
        actions_list = torch.zeros(batch_dim + (self.action_size,)).to(device)
        logprobs_list = torch.zeros(batch_dim).to(device) # TODO: logprobs reduced to single value!!!
        values_list = torch.zeros(batch_dim).to(device)
        rewards_list = torch.zeros(batch_dim).to(device)
        dones_list = torch.zeros(batch_dim).to(device)

        states = init_states
        for step in range(ROLLOUT_SIZE):
            # Do not track gradients on policy rollout.
            with torch.no_grad():
                actions, logprobs = self.agent.sample_action(states)
                clipped_actions = torch.clamp(actions, -1, 1)   # all actions between -1 and 1
                values = self.agent.get_value(states)  # (20, 1)

            # (20, 33),   (20),    (20)
            next_states, rewards, dones = self.env_step(clipped_actions.cpu().numpy())

            states_list[step] = states
            actions_list[step] = actions
            logprobs_list[step] = logprobs
            values_list[step] = values.flatten()
            rewards_list[step] = torch.Tensor(rewards).to(device)
            dones_list[step] = torch.Tensor(dones).to(device)

            # If done, the next_states is gonna be the new state from which to start a new episode.
            states = torch.Tensor(next_states).to(device)      # roll over states to next time step
            if np.any(dones):                                  # exit loop if (any) episode finished
                assert np.all(dones), 'Unexpected environment behavior!'
                # WE DO NOT BREAK: WE KEEP COLLECTING DATA POINTS.
                # break

        return states_list, values_list, actions_list, logprobs_list, rewards_list, dones_list

    # def compute_returns(self, rewards, dones, values):
    #     last = len(dones) - 1 # last returns index
    #     print(f'\tlast: {last}')
    #     print(f'\trewards: {rewards.shape}')
    #     print(f'\tdones: {dones.shape}')
    #     returns = torch.zeros_like(rewards).to(device).detach()
    #     with torch.no_grad():
    #         # Terminal v(s): depending on whether the episode completed or not.
    #         returns[last] = values[last] * (1 - dones[last])
    #         for t in reversed(range(last)):
    #             # v(s) = r + discount * v(s+1)
    #             returns[t] = rewards[t] + GAMMA * (1. - dones[last]) * returns[t+1]
    #     return returns

    # def advantages_and_returns(self, values, rewards, dones):
    #     """Computes returns and advantages for an episode.

    #     TODO: Consider implementing GAE here instead.
    #     """
    #     last = len(values) - 1 # last returns index
    #     returns = torch.zeros_like(rewards).to(device).detach()
    #     with torch.no_grad():
    #         # Terminal v(s): depending on whether the episode completed or not.
    #         returns[last] = values[last] * (1 - dones[last])
    #         # TODO: Maybe instead? :/
    #         # returns[last_ri] = rewards[last_ri] + GAMMA * values[last_ri] * (1 - dones[last_ri])
    #         for t in reversed(range(last)):
    #             # v(s) = r + discount * v(s+1)
    #             returns[t] = rewards[t] + GAMMA * returns[t+1]
    #         # A(s,a) = Q(s,a) - V(s)
    #         # https://www.youtube.com/watch?v=vQ_ifavFBkI
    #         advantages = returns - values
    #     # TODO: Check whether normalizing advantages makes a difference, and whether it is more
    #     #  performant to do so at the mini-batch level instead.
    #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    #     return advantages, returns

    def advantages_and_returns(self, values, rewards, dones):
        with torch.no_grad():
            # next_value = agent.get_value(next_obs).reshape(1, -1)
            next_value = values[-1].reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            num_steps = len(values)
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - dones[-1]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            returns = advantages + values
        return advantages, returns

    def env_reset(self):
        """Reset the environment for a new training episode."""
        env_info = self.env.reset(train_mode=True)[self.brain_name]   # reset the environment
        states = env_info.vector_observations                         # get the current state (for each agent)
        return states

    def env_step(self, actions):
        """Shortcut method to take an action / step in the Unity environment."""
        env_info = self.env.step(actions)[self.brain_name]    # send all actions to the environment
        next_states = env_info.vector_observations            # get the next state (for each agent)
        rewards = env_info.rewards                            # get the reward (for each agent)
        dones = env_info.local_done                           # see if episode has finished
        return next_states, rewards, dones
