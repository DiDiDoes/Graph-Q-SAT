import argparse
from dataclasses import dataclass
import random
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from minisat_wrapper import MiniSAT

from buffer import ReplayBuffer
from cnf import CNFLoader, build_vcg_from_solver
from optim_config import OptimConfig, build_optim_config

# ----------------------------
# Hyperparameters from the paper
# ----------------------------

@dataclass(frozen=True)
class DQNTrainConfig(OptimConfig):
    batch_size: int = 64
    replay_size: int = 20_000
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 30_000
    initial_exploration_steps: int = 5_000
    gamma: float = 0.99
    update_frequency: int = 4
    target_update_frequency: int = 10


def build_dqn_config(args: argparse.Namespace) -> DQNTrainConfig:
    optim_cfg = build_optim_config(args)
    return DQNTrainConfig(
        batch_updates=optim_cfg.batch_updates,
        lr=optim_cfg.lr,
        max_decisions_train=optim_cfg.max_decisions_train,
        step_penalty=optim_cfg.step_penalty,
        truncate_penalty=optim_cfg.truncate_penalty,
        grad_clip_norm=optim_cfg.grad_clip_norm,
        eval_frequency=optim_cfg.eval_frequency,
    )

# ----------------------------
# Action selection
# ----------------------------

def epsilon_by_env_steps(env_steps: int, cfg: DQNTrainConfig) -> float:
    if env_steps < cfg.initial_exploration_steps:
        return cfg.eps_start

    t = min(env_steps - cfg.initial_exploration_steps, cfg.eps_decay_steps)
    frac = t / float(cfg.eps_decay_steps)
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)

# ----------------------------
# DQN loss / update
# ----------------------------

def dqn_update(
    model: nn.Module,
    target_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    device: torch.device,
    cfg: DQNTrainConfig,
) -> float:
    batch = replay.sample(cfg.batch_size)

    states = Batch.from_data_list([t.state for t in batch]).to(device)
    next_states = Batch.from_data_list([t.next_state for t in batch if t.next_state is not None]).to(device)
    actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=device)
    rewards = torch.tensor([t.reward for t in batch], dtype=torch.float, device=device)
    dones = torch.tensor([t.done for t in batch], dtype=torch.float, device=device)

    q_pred_all, _ = model(states)
    q_pred_all = q_pred_all.flatten()
    q_pred_ptr = states.ptr[:-1] * 2
    q_pred = q_pred_all[q_pred_ptr + actions]

    with torch.no_grad():
        q_next_all, var_mask = target_model(next_states)
        q_next_batch = next_states.batch[var_mask].repeat_interleave(2, dim=0)
        q_next_var = q_next_all[var_mask].flatten()
        q_next = torch.zeros_like(rewards).scatter_reduce_(
            0, q_next_batch, q_next_var, reduce="amax", include_self=False
        )

    target = rewards + (1.0 - dones) * cfg.gamma * q_next
    loss = F.mse_loss(q_pred, target.detach())

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
    optimizer.step()

    return float(loss.item())

# ----------------------------
# One environment episode
# ----------------------------

def run_training_episode(
    cnf_file,
    model: nn.Module,
    replay: ReplayBuffer,
    device: torch.device,
    total_env_steps: int,
    cfg: DQNTrainConfig,
) -> Tuple[int, int]:
    """
    Runs one SAT episode and pushes transitions into replay.
    Returns:
        env_steps_added, terminal_state
    terminal_state:
        10 = SAT, 20 = UNSAT, else solver-specific
    """
    loader = CNFLoader(cnf_file)
    loader.load_cnf()

    solver = MiniSAT(cnf=loader.clauses)
    solver.step()
    next_state = state = build_vcg_from_solver(solver, device)

    episode_steps = 0

    while solver.state == 0 and episode_steps < cfg.max_decisions_train:
        eps = epsilon_by_env_steps(total_env_steps, cfg)
        if random.random() < eps:
            action = random.randrange(len(solver.candidates))
        else:
            action = model.select_action(state)

        solver.step(solver.candidates[action])

        done = False
        if episode_steps + 1 >= cfg.max_decisions_train:
            # Truncate long episodes with a penalty
            done = True
            reward = cfg.truncate_penalty
        else:
            reward = cfg.step_penalty
            if solver.state == 0:
                next_state = build_vcg_from_solver(solver, device)
            else:
                done = True

        replay.push(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )
        state = next_state

        total_env_steps += 1
        episode_steps += 1

        if done:
            break

    return episode_steps, solver.state
