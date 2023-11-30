import numpy as np
import torch

from typing import List
from unityagents import UnityEnvironment

from agent import Agent

# TODO: Make these constants configurable
NUM_EPISODES = 100
MAX_STEPS_PER_EPISODE = 100
NUM_UPDATE_EPOCHS = 4
NUM_MINI_BATCHES = 4

# TODO: We should rename this to "experiences" or something.
NUM_AGENTS = 20 # TODO: get this from the environment
ACTION_SIZE = -1 # TODO: get this appropriately
GAMMA=0.995

class PPO:
    """PPO implementation."""
    def __init__(self, env: UnityEnvironment, agent: Agent, solved_score = 30.):
        self.env = env
        self.agent = agent
        self.solved_score = solved_score
        self.brain_name: str = env.brain_names[0]

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

            batch_size = -1 # TODO: Should be the size of `states`say * num_agents (=20)
            mini_batch_size = int(batch_size // NUM_MINI_BATCHES)

            # Compute advantages and returns
            advantages, returns = self.advantages_and_returns(values)

            # Policy learning phase
            for epoch in range(NUM_UPDATE_EPOCHS):
                # TODO: Shuffle the indices...
                for start in range(0, batch_size, mini_batch_size):
                  mini_batch = [] # TODO: build mini-batch.
                  self.agent.learn(mini_batch)

    def collect_trajectories(self):
        """TODO: """
        states = []
        values = []
        actions = []
        logprobs = []
        rewards = []
        dones = []

        states = self.env_reset()                  # get the initial state (for each agent)
        scores = np.zeros(NUM_AGENTS)                # initialize the score (for each agent)
        for step in range(MAX_STEPS_PER_EPISODE):
            # TODO: select action not tracking gradients.
            with torch.no_grad():
              actions = np.random.randn(NUM_AGENTS, ACTION_SIZE) # select an action (for each agent)
              actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1

            # TODO: need to move everything on device.
            next_states, rewards, dones = self.env_step(actions)
            scores += rewards                                  # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            if np.any(dones):                                  # exit loop if (any) episode finished
                break

        # TODO: Keep track of the scores for each agent?
        return states, values, actions, logprobs, rewards, dones

    def advantages_and_returns(self, values, rewards):
        """Computes returns and advantages for an episode."""
        # TODO: Consider implementing GAE here instead.
        discounts = GAMMA ** np.arange(len(rewards))
        returns = np.asarray(rewards) * discounts[:, np.newaxis]
        advantages = returns - values
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
        env_info = self.env.step(actions)[self.brain_name]   # send all actions to the environment
        next_states = env_info.vector_observations            # get the next state (for each agent)
        rewards = env_info.rewards                            # get the reward (for each agent)
        dones = env_info.local_done                           # see if episode has finished
        return next_states, rewards, dones
