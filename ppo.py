import numpy as np
import torch

from typing import Tuple
from unityagents import UnityEnvironment

from agent import Agent, MiniBatch

# TODO: Make these constants configurable
NUM_EPISODES = 100
MAX_STEPS_PER_EPISODE = 1000
NUM_UPDATE_EPOCHS = 4
NUM_MINI_BATCHES = 4
GAMMA=0.995

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
        # Using episodes and max-steps-per-episode have many downsides, see:
        #   https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
        # We keep this strategy b/c the Udacity project rubric specifically requires
        # averaging all agents per episode for the success criteria.
        for n_episode in range(1, NUM_EPISODES + 1):
            # TODO: anneal learning rate?

            # Policy rollout phase.
            # Get more trajectories from many agents to reduce noise.
            # TODO: Improve explanation, pointing to the lecture slides about noise.
            states, values, actions, logprobs, rewards, dones = self.collect_trajectories()

            batch_size = len(states)
            mini_batch_size = int(batch_size // NUM_MINI_BATCHES)

            # Compute advantages and returns
            with torch.no_grad():
              advantages, returns = self.advantages_and_returns(values, rewards, dones)

            # Policy learning phase
            indices = np.arange(batch_size)
            for epoch in range(NUM_UPDATE_EPOCHS):
                # TODO: Shuffle the indices...
                np.random.shuffle(indices)
                for start in range(0, batch_size, mini_batch_size):
                  inds = indices[start:(start + NUM_MINI_BATCHES)]
                  mini_batch = MiniBatch(states[inds], actions[inds], logprobs[inds],
                                         advantages[inds], returns[inds])
                  self.agent.learn(mini_batch)

            print(f'Episode n.{n_episode} completed. Avg score: {rewards.mean()}')

    def collect_trajectories(self):
        """TODO: """
        states_list = torch.zeros(
            (MAX_STEPS_PER_EPISODE, self.num_agents) + (self.state_size,)
          ).to(device)
        actions_list = torch.zeros(
            (MAX_STEPS_PER_EPISODE, self.num_agents) + (self.action_size,)
          ).to(device)
        # TODO: Note the logprobs that are reduced to a single value from a continuous set!!!
        logprobs_list = torch.zeros((MAX_STEPS_PER_EPISODE, self.num_agents)).to(device)
        values_list = torch.zeros((MAX_STEPS_PER_EPISODE, self.num_agents)).to(device)
        rewards_list = torch.zeros((MAX_STEPS_PER_EPISODE, self.num_agents)).to(device)
        dones_list = torch.zeros((MAX_STEPS_PER_EPISODE, self.num_agents)).to(device)

        states = torch.Tensor(self.env_reset()).to(device) # get the initial state (for each agent)
        scores = np.zeros(self.num_agents)                 # initialize the score (for each agent)
        total_steps = MAX_STEPS_PER_EPISODE                # track the actual number of steps executed
        for step in range(MAX_STEPS_PER_EPISODE):
            with torch.no_grad():  # Do not track gradients on policy rollout.
              actions, logprobs = self.agent.sample_action(states)
              actions = torch.clamp(actions, -1, 1)              # all actions between -1 and 1
              values = self.agent.get_value(states)
            next_states, rewards, dones = self.env_step(actions.cpu().numpy())

            states_list[step] = states
            actions_list[step] = actions
            logprobs_list[step] = logprobs
            values_list[step] = values.reshape(-1)
            rewards_list[step] = torch.Tensor(rewards).to(device)
            dones_list[step] = torch.Tensor(dones).to(device)

            states = torch.Tensor(next_states).to(device)      # roll over states to next time step
            scores += rewards                                  # update the score (for each agent)
            if np.any(dones):                                  # exit loop if (any) episode finished
                total_steps = step + 1
                break

        def flatten(t: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
            return tuple(x[:total_steps].flatten(0, 1) for x in t)

        return flatten((states_list, values_list, actions_list,
                       logprobs_list, rewards_list, dones_list))

    def advantages_and_returns(self, values, rewards, dones):
        """Computes returns and advantages for an episode.

        TODO: Consider implementing GAE here instead.
        """
        last_ri = len(values) - 1 # last returns index
        returns = torch.zeros_like(rewards).to(device)
        returns[last_ri] = rewards[last_ri] + GAMMA * values[last_ri] * (1 - dones[last_ri])
        for t in reversed(range(last_ri)):
            returns[t] = rewards[t] + GAMMA * returns[t+1]
        advantages = returns - values # TODO: reconcile advantages computation here with lectures!
        # TODO: Check whether normalizing advantages makes a difference, and whether it is more
        #  performant to do so at the mini-batch level instead.
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
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
