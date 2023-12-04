import numpy as np
import torch

from dataclasses import dataclass
from typing import List,Tuple

from agent import LearningBatch


@dataclass
class TrajectorySegment:
    """A trajectory segment collected for PPO training.

    In particular, the segment contains M timesteps for N bots performing actions in the
    environment. So for example `states` has dimension: (M, N, <single_state_dim>).
    """

    states: torch.Tensor    # (M, N, S_dim)
    """The states for the N bots collected for M timesteps."""
    actions: torch.Tensor   # (M, N, A_dim)
    """The actions taken by the N bots in the M timesteps."""
    logprobs: torch.Tensor  # (M, N)
    """The log-probabilty of the action for the N bots in the M timesteps."""
    values: torch.Tensor    # (M, N)
    """The estimated state value for the N bots for the M timesteps."""
    rewards: torch.Tensor   # (M, N)
    """The rewards for the N bots at the M timesteps for the action taken."""
    dones: torch.Tensor     # (M, N)
    """Whether the action taken at the current state transitioned to a terminal state.

    Note that dones[t] refers to whether the state[t+1] is a terminal state. Also, we
    never really store the terminal state in `states` because if the episode completes
    the environment resets and returns the next starting state for a new episode (at
    least for this specific Unity Environment).
    """
    next_start_state: torch.Tensor  # (N, S_dim)
    """The next state from which to start the next trajectory segment collection."""

    def __post_init__(self):
        """Basic sanity dimensional checks."""
        length = self.states.shape[0]
        assert length > 0
        assert self.actions.shape[0]  == length
        assert self.logprobs.shape[0] == length
        assert self.values.shape[0]   == length
        assert self.rewards.shape[0]  == length
        assert self.dones.shape[0]    == length

    def __len__(self):
        return self.states.shape[0]


class Batcher:
    """Helper class to perform mini-batch learning after a trajectory segment is collected.

    In particular, the Batcher:
        * stores the data relevant for learning (segment, advantages, returns)
        * flattens learning data (thanks to the Markovian property)
        * shuffles the (flattened) segment and iterates through it in mini-batches.
    """
    def __init__(self, seg: TrajectorySegment,
                 advantages: torch.Tensor, returns: torch.Tensor,
                 n_mini_batches: int):
        self.batch_size = seg.states.shape[0] * seg.states.shape[1]  # rollout_len * num_bots
        self.mini_batch_size = int(self.batch_size // n_mini_batches)
        self.indices = np.arange(self.batch_size)
        self.experiences = LearningBatch(*Batcher.flatten(
            (seg.states, seg.actions, seg.logprobs, advantages, returns)))

    def shuffle(self):
        """Shuffles the learning data and returns a new mini-batch iterator."""
        indices = np.arange(self.batch_size)
        np.random.shuffle(indices)
        return Batcher._BatchIterator(self.experiences, indices, self.mini_batch_size)

    @staticmethod
    def flatten(t: Tuple[torch.Tensor, ...]) -> Tuple:
        """Utility function to flatten a multi-agents / bots trajectory segment.

        In particular input tensors have shape (segment_length, num_bots, ...), and they
        are flattened to (segment_length * num_bots, ...).

        That is useful to probe independent experiences (Markovian ) from a trajectory.
        """
        return tuple(x.flatten(0, 1) for x in t)

    class _BatchIterator:
        """Iterator for a learning batch that loops over mini-batches."""
        def __init__(self, experiences: LearningBatch, indices: List[int], mini_batch_size: int):
            self.experiences = experiences
            self.indices = indices
            self.mini_batch_size = mini_batch_size
            self.start = 0
        
        def __iter__(self):
            return self

        def __next__(self):
            if self.start >= len(self.experiences):
                raise StopIteration()
            start = self.start
            end = start + self.mini_batch_size
            inds = self.indices[start:end]
            self.start = end
            return self.experiences[inds]
