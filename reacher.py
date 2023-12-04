import argparse
from distutils.util import strtobool
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import sys
import torch
from unityagents import UnityEnvironment

from agent import Agent
from ppo import PPO


class ReacherWorld:
    """A world full of reachers!
    
    This class encapsulates the Unity Reacher environment provided by Udacity for the deep-learning
    nanodegree continuous control project. It instantiates the Unity environment, allowing to train
    and simulate an agent via PPO.
    """

    def __init__(self, visual=False, seed=0, persist=False):
        """Initialize the reacher world.

        Params:
        =======
            visual (bool): whether to run in visual mode (default False for headless)
            seed (int): seed to initialize various randoms.
            checkpoint (bool): whether to save PyTorch model and scores checkpoint files.
        """
        suffix = "" if visual else "_NoVis"
        env_file = f"unity_env/Reacher_Linux_Multi{suffix}/Reacher.x86_64"
        self.env = UnityEnvironment(file_name=env_file)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.action_size = self.brain.vector_action_space_size
        env_info = self.env.reset(train_mode=False)[self.brain_name]
        self.num_agents = len(env_info.agents)
        states = env_info.vector_observations
        self.state_size = states.shape[1]
        self.persist = persist
        # Setup deterministic seed across the board.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        # Just a nicer print newline considering Unity outputs.
        print()

    def simulate(self, agent: Agent):
        """Simulate the environment for a given agent"""
        env = self.env
        brain_name = self.brain_name
        env_info = env.reset(train_mode=False)[brain_name]    # reset the environment    
        states = torch.Tensor(env_info.vector_observations)   # get current state (for each bot)
        scores = np.zeros(self.num_agents)                    # initialize score (for each bot)
        while True:
            actions, _ = agent.sample_action(states)          # select an action (for each bot)
            actions = torch.clamp(actions, -1, 1).numpy()     # all actions between -1 and 1
            env_info = env.step(actions)[brain_name]          # send all actions to tne environment
            next_states = env_info.vector_observations        # get next state (for each bot)
            dones = env_info.local_done                       # see if episode finished
            scores += env_info.rewards                        # update the score (for each bot)
            states = torch.Tensor(next_states)                # roll over states to next time step
            if np.any(dones):                                 # exit loop if episode finished
                break
        print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

    def train(self, agent: Agent, max_episodes=314, n_update_epochs=10,
              n_mini_batches=100, gae_enabled=True):
        """Train a PPO agent in the Reacher world."""
        ppo = PPO(self.env, agent, max_episodes=max_episodes)
        scores = ppo.train(n_update_epochs, n_mini_batches, gae_enabled, self.persist)
        return scores

    def new_agent(self, weight_mul=1e-3, preload_file=None):
        """Create a new raw agent for the Reacher world.

        Shortcut and convenient method to create agents tailored to the Reacher world environment.
        """
        return Agent(self.state_size, self.action_size,
                     weight_mul=weight_mul, preload_file=preload_file)

    def close(self):
        self.env.close()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, run the world in training mode",
    )
    parser.add_argument(
        "--simulation",
        type=str,
        default="",
        nargs="?",
        const="pretrained",
        help="if toggled, run a world simulation (use the suffix to select it)",
    )
    args = parser.parse_args()
    return args

# Run this file from the command line to see the default simulated agent.
if __name__ == "__main__":
    args = parse_args()

    if args.train and args.simulation:
        print("Only one of --train or --simulation[=<suffix>] can be specified.")
        sys.exit(1)

    if args.simulation:
        print("Showing the plot of the scores achieved during learning.")
        print("Close the plot window to watch the simulation of the agent.")

        # Show the plot of the scores during learning
        scores = np.loadtxt(f'scores_{args.simulation}.csv', delimiter=',', dtype=np.float)
        avgs = pd.Series(scores).rolling(100).mean()
        x = np.arange(len(scores))
        plt.figure('Episode scores')
        plt.plot(x, scores, label='Scores')
        plt.plot(x, avgs, 'r', label='Running average')
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()

        # Simulate the pre-trained agent.
        world = ReacherWorld(visual=True)
        agent = world.new_agent(preload_file=f'weights_{args.simulation}.pth')
        world.simulate(agent)

    elif args.train:
        world = ReacherWorld(persist=True)
        agent = world.new_agent()
        scores = world.train(agent)
