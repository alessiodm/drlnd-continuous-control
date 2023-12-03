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

    def train(self, n_update_epochs=10, n_mini_batches=100,
              gae_enabled=True, persist=False) -> List[float]:
        # Using episodes and max-steps-per-episode have many downsides, see:
        #   https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
        self.n_episode = 1
        self.ep_agent_scores = torch.zeros((0, self.num_bots)).detach()
        # get the initial state (for each agent / bot)
        start_state = torch.Tensor(self.env_reset()).to(device)

        while True:
            # Policy rollout.
            # Get more trajectories from many agents to reduce noise.
            # TODO: Improve explanation, pointing to the lecture slides about noise.            
            segment = self.collect_trajectory_segment(start_state)

            # Advantages and returns computation for the learning phase.
            advantages, returns = self.gae_advantages_and_returns(segment) \
                                    if gae_enabled else self.basic_advantages_and_returns(segment)

            # Policy learning.
            batcher = Batcher(segment, advantages, returns, n_mini_batches)
            for _ in range(n_update_epochs):
                for mini_batch in batcher.shuffle():
                    self.agent.learn(mini_batch)

            start_state = segment.next_start_state  # prepare for next rollout

            # Checking scores and overall episode.
            self.ep_agent_scores = torch.cat((self.ep_agent_scores, segment.rewards), 0)
            if self.training_checkpoint(segment):
                break

        if persist:
            self.agent.save()
            np.savetxt("scores.csv", np.asarray(self.ep_mean_scores), delimiter=",")
        
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

    @torch.no_grad()
    def gae_advantages_and_returns(self, segment: TrajectorySegment,
                                           gamma=0.99, gae_lambda=0.95):
        lastgaelam = 0
        advantages = torch.zeros_like(segment.rewards).to(device)
        next_value = self.agent.get_value(segment.next_start_state).flatten()

        for t in reversed(range(len(segment))):
            next_non_terminal = 1.0 - segment.dones[t]
            td_error = segment.rewards[t] + (
                gamma * next_value * next_non_terminal) - segment.values[t]
            advantages[t] = td_error + gamma * gae_lambda * next_non_terminal * lastgaelam
            next_value = segment.values[t]
            lastgaelam = advantages[t]

        returns = advantages + segment.values
        return advantages, returns

    @torch.no_grad()
    def basic_advantages_and_returns(self, segment: TrajectorySegment, gamma=0.99):
        """Computes returns and advantages for an episode."""
        returns = torch.zeros_like(segment.rewards).to(device).detach()
        next_return = self.agent.get_value(segment.next_start_state).flatten()

        for t in reversed(range(len(segment))):
            next_non_terminal = 1.0 - segment.dones[t]
            # v(s) = r + discount * v(s+1)
            returns[t] = segment.rewards[t] + gamma * next_non_terminal * next_return
            next_return = returns[t]

        # A(s,a) = Q(s,a) - V(s) - https://www.youtube.com/watch?v=vQ_ifavFBkI
        advantages = returns - segment.values
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
