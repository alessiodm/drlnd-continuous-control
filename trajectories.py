import numpy as np
import torch

from dataclasses import dataclass
from typing import List,Tuple

from agent import LearningBatch


@dataclass
class TrajectorySegment:
    states: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    next_start_state: torch.Tensor

    def __post_init__(self):
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
    def __init__(self, seg: TrajectorySegment,
                 advantages: torch.Tensor, returns: torch.Tensor,
                 n_mini_batches: int):
        self.batch_size = seg.states.shape[0] * seg.states.shape[1]  # rollout_len * num_bots
        self.mini_batch_size = int(self.batch_size // n_mini_batches)
        self.indices = np.arange(self.batch_size)
        self.experiences = LearningBatch(*Batcher.__flatten(
            (seg.states, seg.actions, seg.logprobs, advantages, returns)))

    def shuffle(self):
        indices = np.arange(self.batch_size)
        np.random.shuffle(indices)
        return Batcher._BatchIterator(self.experiences, indices, self.mini_batch_size)

    @staticmethod
    def __flatten(t: Tuple[torch.Tensor, ...]) -> Tuple:
        """Utility function to flatten a multi-agents / bots trajectory segment.

        In particular input tensors have shape (segment_length, num_bots, ...), and they
        are flattened to (segment_length * num_bots, ...).

        That is useful to probe independent experiences (Markovian ) from a trajectory.
        """
        return tuple(x.flatten(0, 1) for x in t)

    class _BatchIterator:
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
