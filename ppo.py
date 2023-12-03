import numpy as np
import torch

from typing import List
from unityagents import UnityEnvironment

from agent import Agent
from trajectories import Batcher, TrajectorySegment


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPO:
    """PPO implementation."""
    def __init__(self, env: UnityEnvironment, agent: Agent, solved_score = 30.0,
                 rollout_sizes = [250, 250, 250, 251], max_episodes = 314):
        self.env             = env
        self.agent           = agent.to(device)
        self.solved_score    = solved_score
        self.max_episodes    = max_episodes
        self.brain_name: str = env.brain_names[0]
        self.brain           = self.env.brains[self.brain_name]
        self.action_size     = self.brain.vector_action_space_size
        env_info             = self.env.reset(train_mode=False)[self.brain_name]
        self.num_bots        = len(env_info.agents)
        states               = env_info.vector_observations
        self.state_size      = states.shape[1]
        self.ep_mean_scores  = []
        # TODO: Explain.
        self.rollout_sizes   = np.array(rollout_sizes)
        self.episode_len     = self.rollout_sizes.sum()
        assert self.episode_len == 1001

    def train(self, n_update_epochs: int = 10, n_mini_batches: int = 100) -> List[float]:
        # Using episodes and max-steps-per-episode have many downsides, see:
        #   https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
        self.n_episode = 1
        self.ep_agent_scores = torch.zeros((0, self.num_bots)).detach()
        start_state = torch.Tensor(self.env_reset()).to(device)   # get the initial state (for each agent)

        while True:
            # Policy rollout phase.
            # Get more trajectories from many agents to reduce noise.
            # TODO: Improve explanation, pointing to the lecture slides about noise.            
            segment = self.collect_trajectory_segment(start_state)

            # Bootstrapping value and compute advantages and returns.
            advantages, returns = self.compute_gae_advantages_and_returns(segment)

            # Policy learning phase
            batcher = Batcher(segment, advantages, returns, n_mini_batches)
            for _ in range(n_update_epochs):
                for mini_batch in batcher.shuffle():
                    self.agent.learn(mini_batch)

            start_state = segment.next_start_state  # prepare for next rollout

            # Checking scores and overall episode.
            self.ep_agent_scores = torch.cat((self.ep_agent_scores, segment.rewards), 0)
            if self.training_checkpoint(segment):
                break
        
        return self.ep_mean_scores
 
    def collect_trajectory_segment(self, start_state):
        #
        # https://knowledge.udacity.com/questions/558456
        #
        rollout_size = self.next_rollout_size()
        batch_dim = (rollout_size, self.num_bots)

        s_states   = torch.zeros(batch_dim + (self.state_size,)).to(device)
        s_actions  = torch.zeros(batch_dim + (self.action_size,)).to(device)
        s_logprobs = torch.zeros(batch_dim).to(device) # TODO: logprobs reduced to single value!!!
        s_values   = torch.zeros(batch_dim).to(device)
        s_rewards  = torch.zeros(batch_dim).to(device)
        s_dones    = torch.zeros(batch_dim).to(device)

        state = start_state
        for step in range(rollout_size):
            # Do not track gradients on policy rollout.
            with torch.no_grad():
                action, logprob = self.agent.sample_action(state)
                clipped_action = torch.clamp(action, -1, 1)   # all actions between -1 and 1
                value = self.agent.get_value(state)  # (20, 1)

            # (20, 33),   (20),    (20)
            next_state, reward, done = self.env_step(clipped_action.cpu().numpy())

            s_states[step]   = state
            s_actions[step]  = action
            s_logprobs[step] = logprob
            s_values[step]   = value.flatten()
            s_rewards[step]  = torch.Tensor(reward).to(device)
            s_dones[step]    = torch.Tensor(done).to(device)

            # If done, the next_states is gonna be the new state from which to start a new episode.
            state = torch.Tensor(next_state).to(device)      # roll over states to next time step

            if np.any(done):
                assert np.all(done), 'Unexpected environment behavior!'
                assert step == rollout_size - 1, 'Rollouts are not even!'
                # Do not break, continue collecting new trajectory segments instead.

        return TrajectorySegment(s_states, s_actions, s_logprobs, s_values,
                                 s_rewards, s_dones, next_start_state=state)

    def compute_gae_advantages_and_returns(self, segment, gamma=0.99, gae_lambda=0.95):
        with torch.no_grad():
            # next_value = agent.get_value(next_obs).reshape(1, -1)
            # next_value = values[-1].reshape(1, -1)
            next_value = segment.values[-1]

            # print(values[-1].shape)
            # print(values[-1].reshape(1, -1).shape)
            # print(next_value.shape)

            advantages = torch.zeros_like(segment.rewards).to(device)
            lastgaelam = 0
            num_steps = len(segment.values)
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - segment.dones[-1]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - segment.dones[t + 1]
                    nextvalues = segment.values[t + 1]
                delta = segment.rewards[t] + gamma * nextvalues * nextnonterminal - segment.values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + segment.values
        return advantages, returns

    def next_rollout_size(self) -> int:
        rs = self.rollout_sizes[0]
        self.rollout_sizes = np.roll(self.rollout_sizes, -1)
        return rs

    def training_checkpoint(self, segment: TrajectorySegment) -> bool:
        # TODO: explain how we can simplify episode_end detection.
        is_episode_end = np.any(segment.dones.flatten(0, 1).numpy())
        if is_episode_end:
            ep_mean_score = self.ep_agent_scores.sum(0).mean().item()
            self.ep_mean_scores.append(ep_mean_score)
            print(f'Episode n.{self.n_episode} completed. Avg score: {ep_mean_score}')
            assert len(self.ep_agent_scores) == self.episode_len, \
                f'Episode length of: {len(self.ep_agent_scores)}?!'
            self.n_episode += 1
            self.ep_agent_scores = torch.zeros((0, self.num_bots))
            # Check for environment solved
            episodes_100_mean = np.mean(self.ep_mean_scores[-100:])
            if episodes_100_mean > self.solved_score:
                print(f'Reacher environment solved! 100 episodes score: {episodes_100_mean}')
                return True
            if self.n_episode > self.max_episodes:
                print('Reached the maximum number of episodes, terminating...')
                return True
        return False

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
