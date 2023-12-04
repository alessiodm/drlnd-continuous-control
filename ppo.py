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
        # Given that the Reacher environment episodes always last 1001 steps, we
        # fine-tune the policy rollout sizes accordingly to have an even training
        # loop and be able to compute statistics more nicely (and according to the
        # rubric of the Udacity project). This is not necessary for PPO in general.
        # See: https://knowledge.udacity.com/questions/558456
        self.ep_mean_scores  = []   # scores per agent for the *current* episode.
        self.rollout_sizes   = np.array(rollout_sizes)
        self.episode_len     = self.rollout_sizes.sum()
        assert self.episode_len == 1001

    def train(self, n_update_epochs=10, n_mini_batches=100,
              gae_enabled=True, persist=False) -> List[float]:
        """PPO training loop.
        
        The loop is effectively:
            * Collect a trajectory segment
            * Compute advantages and returns
            * Learn in mini-batches
            * Repeat until max_episodes or environment solved.

        In PPO learning we don't loop over episodes (and max-steps-per episode), which has
        various downsides, especially for solving long-lasting environments. Please, see
        https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ (point 1)
        for a detailed explanation.

        Nonetheless, the Reacher environment has deterministic episodes after a fixed number
        of steps (i.e., 1001). So in this training loop we take advantage of that to report
        statistics of single episodes and reason about "solved environment" (details in the
        `training_checkpoint` method).
        """
        self.n_episode = 1
        self.ep_agent_scores = torch.zeros((0, self.num_bots)).detach()
        start_state = torch.Tensor(self.env_reset()).to(device) # initial state (for each bot)

        while True:
            # Policy rollout. Given the "vectorized" environment that collects experiences from
            # a different number of bots (20 in the Reacher environment), we reduce noise.      
            segment = self.collect_trajectory_segment(start_state)

            # Advantages and returns computation for the learning phase (GAE or standard).
            advantages, returns = self.gae_advantages_and_returns(segment) \
                                    if gae_enabled else self.basic_advantages_and_returns(segment)

            # Policy learning. The agent learns on mini-batches provided by the Batcher.
            batcher = Batcher(segment, advantages, returns, n_mini_batches)
            for _ in range(n_update_epochs):
                for mini_batch in batcher.shuffle():
                    self.agent.learn(mini_batch)

            start_state = segment.next_start_state  # prepare for next rollout

            # Checking scores and overall episode.
            self.ep_agent_scores = torch.cat((self.ep_agent_scores, segment.rewards), 0)
            if self.training_checkpoint(segment):
                break

        # Save the model on disk if instructed to do so.
        if persist:
            self.agent.save()
            np.savetxt("scores.csv", np.asarray(self.ep_mean_scores), delimiter=",")
        
        return self.ep_mean_scores
 
    def collect_trajectory_segment(self, start_state):
        """Collect a trajectory segment for a round of PPO policy rollout."""
        rollout_size = self.next_rollout_size()
        batch_dim = (rollout_size, self.num_bots)

        s_states   = torch.zeros(batch_dim + (self.state_size,)).to(device)
        s_actions  = torch.zeros(batch_dim + (self.action_size,)).to(device)
        s_logprobs = torch.zeros(batch_dim).to(device)
        s_values   = torch.zeros(batch_dim).to(device)
        s_rewards  = torch.zeros(batch_dim).to(device)
        s_dones    = torch.zeros(batch_dim).to(device)

        state = start_state
        for step in range(rollout_size):
            with torch.no_grad(): # Do not track gradients on policy rollout.
                action, logprob = self.agent.sample_action(state)
                clipped_action = torch.clamp(action, -1, 1)   # all actions between -1 and 1
                value = self.agent.get_value(state)

            # Dimensions: next_state.shape=(20, 33), reward.shape=(20), done.shape=(20)
            # NOTE: if done, the next_state is the new state from which to start a new episode.
            next_state, reward, done = self.env_step(clipped_action.cpu().numpy())

            s_states[step]   = state
            s_actions[step]  = action
            s_logprobs[step] = logprob
            s_values[step]   = value.flatten()
            s_rewards[step]  = torch.Tensor(reward).to(device)
            # dones[t] corresponds to whether the state[t+1] was done. But if that's the case, we
            # do not store such state b/c what the environment returns as `next_state` is the reset
            # start state for the new episode. This is relevant for advantage / return computation.
            s_dones[step]    = torch.Tensor(done).to(device)

            state = torch.Tensor(next_state).to(device)      # roll over states to next time step

            if np.any(done):
                # These assertions are Reacher environment-specific.
                assert np.all(done), 'Unexpected environment behavior!'
                assert step == rollout_size - 1, 'Rollouts are not even!'
                # DO NOT break, continue collecting new trajectory segments instead.

        return TrajectorySegment(s_states, s_actions, s_logprobs, s_values,
                                 s_rewards, s_dones, next_start_state=state)

    @torch.no_grad()
    def gae_advantages_and_returns(self, segment: TrajectorySegment,
                                   gamma=0.99, gae_lambda=0.95):
        """Compute advantages via generalized advantage estimation (GAE).

        Paper: https://arxiv.org/abs/1506.02438"""
        last_gae_lambda = 0
        advantages = torch.zeros_like(segment.rewards).to(device)
        next_value = self.agent.get_value(segment.next_start_state).flatten()

        for t in reversed(range(len(segment))):
            next_non_terminal = 1.0 - segment.dones[t]
            td_error = segment.rewards[t] + (
                gamma * next_value * next_non_terminal) - segment.values[t]
            advantages[t] = td_error + gamma * gae_lambda * next_non_terminal * last_gae_lambda
            next_value = segment.values[t]
            # Reset the last_gae_lambda if an episode terminates half-way (per bot).
            last_gae_lambda = advantages[t] * next_non_terminal

        returns = advantages + segment.values
        return advantages, returns

    @torch.no_grad()
    def basic_advantages_and_returns(self, segment: TrajectorySegment, gamma=0.99):
        """Computes returns and advantages for an segment in the "standard" way."""
        returns = torch.zeros_like(segment.rewards).to(device).detach()
        next_return = self.agent.get_value(segment.next_start_state).flatten()

        for t in reversed(range(len(segment))):
            next_non_terminal = 1.0 - segment.dones[t]
            # v(s) = r + discount * v(s+1)
            returns[t] = segment.rewards[t] + gamma * next_non_terminal * next_return
            # Reset the next_return if an episode terminates half-way (per bot).
            next_return = returns[t] * next_non_terminal

        # A(s,a) = Q(s,a) - V(s) - https://www.youtube.com/watch?v=vQ_ifavFBkI
        advantages = returns - segment.values
        return advantages, returns

    def next_rollout_size(self) -> int:
        """Returns the size of the next trajectory segment to collect.

        Usually, they should all be of the same length. But because of the Reacher environment
        uneven episode steps, we just adjust the rollout_size to have evenly split loops. This
        is a detail for this specific implementation for Reacher (not general PPO).
        """
        rs = self.rollout_sizes[0]
        self.rollout_sizes = np.roll(self.rollout_sizes, -1)
        return rs

    def training_checkpoint(self, segment: TrajectorySegment) -> bool:
        """Check whether an episode is completed and determines whether to terminate training.

        As mentioned above, the Reacher environment has 20 bots whose episodes always last 1001
        timesteps. We take advantage of that in order to determine if an episode is ended (and
        we perform assertions to make sure assumptions are not violated), and to compute the
        average over 100 episodes to determine whether the environment is solved.
        """
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

    def env_reset(self):
        """Reset the environment for a new training episode."""
        env_info = self.env.reset(train_mode=True)[self.brain_name]   # reset the environment
        states = env_info.vector_observations                         # get the current state (for each bot))
        return states

    def env_step(self, actions):
        """Shortcut method to take an action / step in the Unity environment."""
        env_info = self.env.step(actions)[self.brain_name]    # send all actions to the environment
        next_states = env_info.vector_observations            # get the next state (for each bot)
        rewards = env_info.rewards                            # get the reward (for each bot)
        dones = env_info.local_done                           # see if episode has finished
        return next_states, rewards, dones
