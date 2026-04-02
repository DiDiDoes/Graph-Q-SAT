import argparse
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn
from torch.distributions import Categorical
from torch_geometric.data import Data

from minisat_wrapper import MiniSAT

from cnf import CNFLoader, build_vcg_from_solver
from optim_config import OptimConfig, build_optim_config


@dataclass(frozen=True)
class GRPOTrainConfig(OptimConfig):
    batch_size: int = 1
    group_size: int = 4
    clip_eps: float = 0.2
    ppo_epochs: int = 4
    advantage_eps: float = 1e-6
    entropy_coef: float = 0.0


def build_grpo_config(args: argparse.Namespace) -> GRPOTrainConfig:
    if args.grpo_batch_size < 1:
        raise ValueError("--grpo-batch-size must be at least 1.")
    if args.grpo_group_size < 2:
        raise ValueError("--grpo-group-size must be at least 2.")
    if args.grpo_clip_eps < 0:
        raise ValueError("--grpo-clip-eps must be non-negative.")
    if args.grpo_epochs < 1:
        raise ValueError("--grpo-epochs must be at least 1.")

    optim_cfg = build_optim_config(args)
    return GRPOTrainConfig(
        batch_updates=optim_cfg.batch_updates,
        lr=optim_cfg.lr,
        max_decisions_train=optim_cfg.max_decisions_train,
        step_penalty=optim_cfg.step_penalty,
        truncate_penalty=optim_cfg.truncate_penalty,
        grad_clip_norm=optim_cfg.grad_clip_norm,
        eval_frequency=optim_cfg.eval_frequency,
        batch_size=args.grpo_batch_size,
        group_size=args.grpo_group_size,
        clip_eps=args.grpo_clip_eps,
        ppo_epochs=args.grpo_epochs,
    )


@dataclass
class GRPOStep:
    state: Data
    action: int
    old_log_prob: float


@dataclass
class GRPOEpisode:
    steps: List[GRPOStep]
    reward: float
    terminal_state: int


def candidate_logits(model: nn.Module, state) -> torch.Tensor:
    qs, var_mask = model(state)
    return qs[var_mask].flatten()


@torch.no_grad()
def run_training_episode(
    cnf_file,
    model: nn.Module,
    device: torch.device,
    cfg: GRPOTrainConfig,
) -> GRPOEpisode:
    loader = CNFLoader(cnf_file)
    loader.load_cnf()

    solver = MiniSAT(cnf=loader.clauses)
    solver.step()

    steps: List[GRPOStep] = []
    episode_steps = 0

    while solver.state == 0 and episode_steps < cfg.max_decisions_train:
        state = build_vcg_from_solver(solver, device)
        logits = candidate_logits(model, state)
        action_dist = Categorical(logits=logits)
        action_tensor = action_dist.sample()
        action = int(action_tensor.item())

        steps.append(
            GRPOStep(
                state=state,
                action=action,
                old_log_prob=float(action_dist.log_prob(action_tensor).item()),
            )
        )

        solver.step(solver.candidates[action])
        episode_steps += 1

    if solver.state in [10, 20]:
        reward = -float(solver.decisions)
    elif solver.state == 0 and episode_steps >= cfg.max_decisions_train:
        reward = cfg.truncate_penalty
    else:
        raise ValueError(f"Unexpected solver state: {solver.state}")

    return GRPOEpisode(
        steps=steps,
        reward=reward,
        terminal_state=solver.state,
    )


@torch.no_grad()
def collect_training_group(
    cnf_file,
    model: nn.Module,
    device: torch.device,
    cfg: GRPOTrainConfig,
) -> List[GRPOEpisode]:
    return [
        run_training_episode(
            cnf_file=cnf_file,
            model=model,
            device=device,
            cfg=cfg,
        )
        for _ in range(cfg.group_size)
    ]


def grpo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    episode_groups: List[List[GRPOEpisode]],
    device: torch.device,
    cfg: GRPOTrainConfig,
) -> Optional[float]:
    step_records = []
    for episodes in episode_groups:
        rewards = torch.tensor([episode.reward for episode in episodes], dtype=torch.float, device=device)
        if rewards.numel() == 0 or torch.allclose(rewards, rewards[0]):
            continue

        advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + cfg.advantage_eps)
        for episode, advantage in zip(episodes, advantages):
            for step in episode.steps:
                step_records.append((step, advantage))

    if not step_records:
        return None

    last_loss = None
    for _ in range(cfg.ppo_epochs):
        policy_losses = []
        entropies = []

        for step, advantage in step_records:
            logits = candidate_logits(model, step.state)
            action = torch.tensor(step.action, dtype=torch.long, device=device)
            old_log_prob = torch.tensor(step.old_log_prob, dtype=torch.float, device=device)

            action_dist = Categorical(logits=logits)
            new_log_prob = action_dist.log_prob(action)
            ratio = torch.exp(new_log_prob - old_log_prob)

            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantage

            policy_losses.append(-torch.minimum(unclipped, clipped))
            entropies.append(action_dist.entropy())

        policy_loss = torch.stack(policy_losses).mean()
        entropy = torch.stack(entropies).mean()
        loss = policy_loss - cfg.entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        optimizer.step()

        last_loss = float(loss.item())

    return last_loss
