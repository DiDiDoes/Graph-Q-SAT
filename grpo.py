import argparse
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.distributions import Categorical
from torch_geometric.data import Batch, Data

from minisat_wrapper import MiniSAT

from cnf import CNFLoader, build_vcg_from_solver
from model import GraphQSat
from optim_config import OptimConfig, build_optim_config


@dataclass(frozen=True)
class GRPOTrainConfig(OptimConfig):
    batch_size: int = 1
    buffer_size: int = 1
    group_size: int = 4
    clip_eps: float = 0.2
    ppo_epochs: int = 4
    advantage_eps: float = 1e-6
    entropy_coef: float = 0.0
    inference_batch_graphs: int = 0


def build_grpo_config(args: argparse.Namespace) -> GRPOTrainConfig:
    if args.grpo_batch_size < 1:
        raise ValueError("--grpo-batch-size must be at least 1.")
    buffer_size = args.grpo_buffer_size if args.grpo_buffer_size is not None else args.grpo_batch_size
    if buffer_size < 1:
        raise ValueError("--grpo-buffer-size must be at least 1.")
    if buffer_size < args.grpo_batch_size:
        raise ValueError("--grpo-buffer-size must be greater than or equal to --grpo-batch-size.")
    if buffer_size % args.grpo_batch_size != 0:
        raise ValueError("--grpo-buffer-size must be divisible by --grpo-batch-size.")
    if args.grpo_group_size < 2:
        raise ValueError("--grpo-group-size must be at least 2.")
    if args.grpo_clip_eps < 0:
        raise ValueError("--grpo-clip-eps must be non-negative.")
    if args.grpo_epochs < 1:
        raise ValueError("--grpo-epochs must be at least 1.")
    if args.grpo_inference_batch_graphs < 0:
        raise ValueError("--grpo-inference-batch-graphs must be non-negative.")

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
        buffer_size=buffer_size,
        group_size=args.grpo_group_size,
        clip_eps=args.grpo_clip_eps,
        ppo_epochs=args.grpo_epochs,
        inference_batch_graphs=args.grpo_inference_batch_graphs,
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


@dataclass
class RolloutState:
    solver: MiniSAT
    steps: List[GRPOStep]
    state: Optional[Data] = None
    episode_steps: int = 0
    reward: Optional[float] = None
    terminal_state: int = 0


def batched_candidate_logits(
    model: GraphQSat,
    states: List[Data],
    device: torch.device,
) -> List[torch.Tensor]:
    if not states:
        return []

    batch = Batch.from_data_list(states).to(device)
    qs, var_mask = model(batch)
    return model.split_candidate_logits(batch, qs, var_mask)


def chunked_candidate_logits(
    model: GraphQSat,
    states: List[Data],
    device: torch.device,
    max_graphs: int,
) -> List[torch.Tensor]:
    if not states:
        return []

    chunk_size = max_graphs if max_graphs > 0 else len(states)
    logits = []
    for start in range(0, len(states), chunk_size):
        logits.extend(
            batched_candidate_logits(
                model=model,
                states=states[start:start + chunk_size],
                device=device,
            )
        )
    return logits


@torch.no_grad()
def collect_training_groups(
    cnf_files: List[str],
    model: GraphQSat,
    device: torch.device,
    cfg: GRPOTrainConfig,
) -> List[List[GRPOEpisode]]:
    episode_groups: List[List[RolloutState]] = []
    active_rollouts: List[RolloutState] = []

    for cnf_file in cnf_files:
        loader = CNFLoader(cnf_file)
        loader.load_cnf()

        group_rollouts = []
        for _ in range(cfg.group_size):
            solver = MiniSAT(cnf=loader.clauses)
            solver.step()

            rollout = RolloutState(
                solver=solver,
                steps=[],
                terminal_state=solver.state,
            )
            if solver.state == 0:
                rollout.state = build_vcg_from_solver(solver, device)
                active_rollouts.append(rollout)
            elif solver.state in [10, 20]:
                rollout.reward = -float(solver.decisions)
            else:
                raise ValueError(f"Unexpected solver state: {solver.state}")

            group_rollouts.append(rollout)
        episode_groups.append(group_rollouts)

    while active_rollouts:
        active_states = [rollout.state for rollout in active_rollouts if rollout.state is not None]
        active_logits = chunked_candidate_logits(
            model=model,
            states=active_states,
            device=device,
            max_graphs=cfg.inference_batch_graphs,
        )

        next_active_rollouts = []
        for rollout, logits in zip(active_rollouts, active_logits):
            if rollout.state is None:
                continue

            action_dist = Categorical(logits=logits)
            action_tensor = action_dist.sample()
            action = int(action_tensor.item())
            rollout.steps.append(
                GRPOStep(
                    state=rollout.state,
                    action=action,
                    old_log_prob=float(action_dist.log_prob(action_tensor).item()),
                )
            )

            rollout.solver.step(rollout.solver.candidates[action])
            rollout.episode_steps += 1
            rollout.terminal_state = rollout.solver.state

            if rollout.solver.state in [10, 20]:
                rollout.reward = -float(rollout.solver.decisions)
                rollout.state = None
            elif rollout.episode_steps >= cfg.max_decisions_train:
                rollout.reward = cfg.truncate_penalty
                rollout.state = None
            elif rollout.solver.state == 0:
                rollout.state = build_vcg_from_solver(rollout.solver, device)
                next_active_rollouts.append(rollout)
            else:
                raise ValueError(f"Unexpected solver state: {rollout.solver.state}")

        active_rollouts = next_active_rollouts

    return [
        [
            GRPOEpisode(
                steps=rollout.steps,
                reward=rollout.reward if rollout.reward is not None else cfg.truncate_penalty,
                terminal_state=rollout.terminal_state,
            )
            for rollout in group_rollouts
        ]
        for group_rollouts in episode_groups
    ]


def grpo_update(
    model: GraphQSat,
    optimizer: torch.optim.Optimizer,
    episode_groups: List[List[GRPOEpisode]],
    device: torch.device,
    cfg: GRPOTrainConfig,
) -> Optional[float]:
    states = []
    actions = []
    old_log_probs = []
    advantages = []

    for episodes in episode_groups:
        rewards = torch.tensor([episode.reward for episode in episodes], dtype=torch.float, device=device)
        if rewards.numel() == 0 or torch.allclose(rewards, rewards[0]):
            continue

        group_advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + cfg.advantage_eps)
        for episode, advantage in zip(episodes, group_advantages):
            for step in episode.steps:
                states.append(step.state)
                actions.append(step.action)
                old_log_probs.append(step.old_log_prob)
                advantages.append(float(advantage.item()))

    if not states:
        return None

    actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
    old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float, device=device)
    advantages_tensor = torch.tensor(advantages, dtype=torch.float, device=device)

    last_loss = None
    for _ in range(cfg.ppo_epochs):
        new_log_probs = []
        entropies = []
        chunk_size = cfg.inference_batch_graphs if cfg.inference_batch_graphs > 0 else len(states)

        for start in range(0, len(states), chunk_size):
            end = start + chunk_size
            chunk_logits = batched_candidate_logits(
                model=model,
                states=states[start:end],
                device=device,
            )
            for offset, logits in enumerate(chunk_logits):
                action_dist = Categorical(logits=logits)
                action = actions_tensor[start + offset]
                new_log_probs.append(action_dist.log_prob(action))
                entropies.append(action_dist.entropy())

        new_log_probs_tensor = torch.stack(new_log_probs)
        entropy = torch.stack(entropies).mean()
        ratio = torch.exp(new_log_probs_tensor - old_log_probs_tensor)
        unclipped = ratio * advantages_tensor
        clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages_tensor
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        loss = policy_loss - cfg.entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        optimizer.step()

        last_loss = float(loss.item())

    return last_loss
