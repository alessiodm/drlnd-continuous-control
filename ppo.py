import numpy as np
import torch

from typing import List
from unityagents import UnityEnvironment

from agent import Agent

# TODO: Make these constants configurable
NUM_EPISODES = 100
MAX_STEPS_PER_EPISODE = 100

# TODO: We should rename this to "experiences" or something.
NUM_AGENTS = 20 # TODO: get this from the environment
ACTION_SIZE = -1 # TODO: get this appropriately

class PPO:
    """PPO implementation."""
    def __init__(self, env: UnityEnvironment, agent: Agent, solved_score = 30.):
        self.env = env
        self.agent = agent
        self.solved_score = solved_score
        self.brain_name: str = env.brain_names[0]

    def train(self):
        # Get more trajectories to reduce noise.
        print('Training started.')
        for n_episode in range(1, NUM_EPISODES + 1):
            # TODO: anneal learning rate?

            # ---------------------
            # Policy rollout phase
            # ---------------------  
            states = self.__env_reset()                  # get the initial state (for each agent)
            scores = np.zeros(NUM_AGENTS)                # initialize the score (for each agent)
            for step in range(MAX_STEPS_PER_EPISODE):
                # TODO: select action not tracking gradients.
                with torch.no_grad():
                  actions = np.random.randn(NUM_AGENTS, ACTION_SIZE) # select an action (for each agent)
                  actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1

                # TODO: need to move everything on device.
                next_states, rewards, dones = self.__env_step(actions)
                scores += rewards                                  # update the score (for each agent)
                states = next_states                               # roll over states to next time step

                # TODO: Instead of breaking if *any* episodes end, let's keep collecting
                #   the ones that are not ended instead? How can I do that?
                if np.any(dones):                                  # exit loop if episode finished
                    break

            # ---------------------
            # Policy learning phase
            # ---------------------
            # TODO:
        
        print('Training complete!')
    
    def __env_reset(self):
        """Reset the environment for a new training episode."""
        env_info = self.env.reset(train_mode=True)[self.brain_name]   # reset the environment
        states = env_info.vector_observations                         # get the current state (for each agent)
        return states

    def __env_step(self, actions):
        """Shortcut method to take an action / step in the Unity environment."""
        env_info = self.env.step(actions)[self.brain_name]   # send all actions to the environment
        next_states = env_info.vector_observations            # get the next state (for each agent)
        rewards = env_info.rewards                            # get the reward (for each agent)
        dones = env_info.local_done                           # see if episode has finished
        return next_states, rewards, dones
