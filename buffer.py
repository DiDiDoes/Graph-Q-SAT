from collections import deque, namedtuple
from typing import Deque, List, Optional
import random

from torch_geometric.data import Data

# ----------------------------
# Replay buffer
# ----------------------------

Transition = namedtuple(
    "Transition",
    ["state", "action", "reward", "next_state", "done"],
)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: Data,
        action: int,
        reward: float,
        next_state: Optional[Data],
        done: bool,
    ) -> None:
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)
