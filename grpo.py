import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch.distributions import Categorical
from torch_geometric.data import Batch, Data
from torch_geometric.utils import scatter

from minisat_wrapper import MiniSAT

from cnf import CNFLoader, build_vcg_from_solver
from model import GraphQSat
from optim_config import OptimConfig, build_optim_config


@dataclass(frozen=True)
class GRPOTrainConfig(OptimConfig):
    batch_size: int = 1
    step_batch_size: int = 0
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
    if args.grpo_step_batch_size < 0:
        raise ValueError("--grpo-step-batch-size must be non-negative.")
    buffer_size = args.grpo_buffer_size if args.grpo_buffer_size is not None else args.grpo_batch_size
    if buffer_size < 1:
        raise ValueError("--grpo-buffer-size must be at least 1.")
    if args.grpo_step_batch_size == 0 and buffer_size < args.grpo_batch_size:
        raise ValueError("--grpo-buffer-size must be greater than or equal to --grpo-batch-size.")
    if args.grpo_step_batch_size == 0 and buffer_size % args.grpo_batch_size != 0:
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
        step_batch_size=args.grpo_step_batch_size,
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
class GRPOTrainingStep:
    state: Data
    action: int
    old_log_prob: float
    advantage: float
    episode_id: int
    episode_reward: float


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


def batched_packed_candidate_logits(
    model: GraphQSat,
    states: List[Data],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not states:
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty_float = torch.empty(0, dtype=torch.float, device=device)
        return empty_float, empty_long

    batch = Batch.from_data_list(states).to(device)
    qs, var_mask = model(batch)
    return model.pack_candidate_logits(batch, qs, var_mask)


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
                rollout.state = build_vcg_from_solver(solver)
                active_rollouts.append(rollout)
            elif solver.state in [10, 20]:
                rollout.reward = -float(solver.decisions)
            else:
                raise ValueError(f"Unexpected solver state: {solver.state}")

            group_rollouts.append(rollout)
        episode_groups.append(group_rollouts)

    while active_rollouts:
        next_active_rollouts = []
        chunk_size = cfg.inference_batch_graphs if cfg.inference_batch_graphs > 0 else len(active_rollouts)
        for start in range(0, len(active_rollouts), chunk_size):
            chunk_rollouts = active_rollouts[start:start + chunk_size]
            chunk_states = [rollout.state for rollout in chunk_rollouts if rollout.state is not None]
            candidate_logits, candidate_ptr = batched_packed_candidate_logits(
                model=model,
                states=chunk_states,
                device=device,
            )

            candidate_counts = candidate_ptr[1:] - candidate_ptr[:-1]
            num_graphs = int(candidate_counts.numel())
            if torch.any(candidate_counts <= 0):
                raise ValueError("Encountered an active GRPO rollout with no candidate actions.")

            candidate_batch = torch.repeat_interleave(
                torch.arange(num_graphs, device=device),
                candidate_counts,
            )
            max_logits = scatter(candidate_logits, candidate_batch, dim=0, dim_size=num_graphs, reduce="max")
            exp_logits = torch.exp(candidate_logits - max_logits[candidate_batch])
            sum_exp_logits = scatter(exp_logits, candidate_batch, dim=0, dim_size=num_graphs, reduce="sum")
            probs = exp_logits / sum_exp_logits[candidate_batch]
            cumulative_probs = torch.cumsum(probs, dim=0)
            cumulative_offsets = torch.zeros(num_graphs, dtype=probs.dtype, device=device)
            last_candidate_indices = candidate_ptr[1:] - 1
            if num_graphs > 1:
                cumulative_offsets[1:] = cumulative_probs[last_candidate_indices[:-1]]
            relative_cumulative_probs = cumulative_probs - cumulative_offsets[candidate_batch]
            relative_cumulative_probs[last_candidate_indices] = 1.0
            thresholds = torch.rand(num_graphs, device=device)
            candidate_indices = torch.arange(candidate_logits.numel(), device=device)
            sentinel = candidate_logits.numel()
            chosen_packed_indices = scatter(
                torch.where(
                    relative_cumulative_probs >= thresholds[candidate_batch],
                    candidate_indices,
                    candidate_indices.new_full(candidate_indices.shape, sentinel),
                ),
                candidate_batch,
                dim=0,
                dim_size=num_graphs,
                reduce="min",
            )
            chosen_packed_indices = torch.where(
                chosen_packed_indices == sentinel,
                last_candidate_indices,
                chosen_packed_indices,
            )
            action_indices = chosen_packed_indices - candidate_ptr[:-1]
            log_normalizers = torch.log(sum_exp_logits) + max_logits
            old_log_probs = candidate_logits[chosen_packed_indices] - log_normalizers
            action_indices_cpu = action_indices.cpu().tolist()
            old_log_probs_cpu = old_log_probs.cpu().tolist()

            for rollout, action, old_log_prob in zip(chunk_rollouts, action_indices_cpu, old_log_probs_cpu):
                if rollout.state is None:
                    continue

                rollout.steps.append(
                    GRPOStep(
                        state=rollout.state,
                        action=action,
                        old_log_prob=old_log_prob,
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
                    rollout.state = build_vcg_from_solver(rollout.solver)
                    next_active_rollouts.append(rollout)
                else:
                    raise ValueError(f"Unexpected solver state: {rollout.solver.state}")

        active_rollouts = next_active_rollouts

    episode_groups_result = [
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
    return episode_groups_result


def prepare_grpo_training_steps(
    episode_groups: List[List[GRPOEpisode]],
    device: torch.device,
    cfg: GRPOTrainConfig,
) -> List[GRPOTrainingStep]:
    training_steps = []
    episode_id = 0

    for episodes in episode_groups:
        rewards = torch.tensor([episode.reward for episode in episodes], dtype=torch.float, device=device)
        if rewards.numel() == 0 or torch.allclose(rewards, rewards[0]):
            episode_id += len(episodes)
            continue

        group_advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + cfg.advantage_eps)
        for episode, advantage in zip(episodes, group_advantages):
            for step in episode.steps:
                training_steps.append(
                    GRPOTrainingStep(
                        state=step.state,
                        action=step.action,
                        old_log_prob=step.old_log_prob,
                        advantage=float(advantage.item()),
                        episode_id=episode_id,
                        episode_reward=episode.reward,
                    )
                )
            episode_id += 1

    return training_steps


def grpo_update_steps(
    model: GraphQSat,
    optimizer: torch.optim.Optimizer,
    training_steps: List[GRPOTrainingStep],
    device: torch.device,
    cfg: GRPOTrainConfig,
) -> Optional[float]:
    if not training_steps:
        return None

    states = [step.state for step in training_steps]
    actions_tensor = torch.tensor([step.action for step in training_steps], dtype=torch.long, device=device)
    old_log_probs_tensor = torch.tensor(
        [step.old_log_prob for step in training_steps],
        dtype=torch.float,
        device=device,
    )
    advantages_tensor = torch.tensor([step.advantage for step in training_steps], dtype=torch.float, device=device)

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


def grpo_update(
    model: GraphQSat,
    optimizer: torch.optim.Optimizer,
    episode_groups: List[List[GRPOEpisode]],
    device: torch.device,
    cfg: GRPOTrainConfig,
) -> Optional[float]:
    training_steps = prepare_grpo_training_steps(
        episode_groups=episode_groups,
        device=device,
        cfg=cfg,
    )
    return grpo_update_steps(
        model=model,
        optimizer=optimizer,
        training_steps=training_steps,
        device=device,
        cfg=cfg,
    )
