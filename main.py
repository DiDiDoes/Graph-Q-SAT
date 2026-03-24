import copy
import random
from tqdm import tqdm
from typing import Tuple

import torch
from torch import nn
from torch_geometric.data import Data

from minisat_wrapper import MiniSAT

from buffer import ReplayBuffer
from cnf import CNFLoader, build_vcg_from_solver
from dataset import CNFDataset
from dqn import DQNTrainConfig, epsilon_by_env_steps, run_training_episode, dqn_update
from model import GraphQSat

def compute_median(values):
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n == 0:
        return None
    if n % 2 == 1:
        return sorted_values[n // 2]
    else:
        return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2

def compute_median_reduction(values, baselines):
    reductions = [b / v for v, b in zip(values, baselines)]
    return compute_median(reductions)

def eval_solver(dataset: CNFDataset) -> Tuple[list, list, float, float]:
    decisions = []
    propagations = []
    for cnf_file in tqdm(dataset, desc="Evaluating MiniSAT"):
        loader = CNFLoader(cnf_file)
        loader.load_cnf()
        solver = MiniSAT(cnf=loader.clauses)
        solver.step()
        while solver.state == 0:
            solver.step(solver.pick_default_branch_literal())
        if solver.state in [10, 20]:
            decisions.append(solver.decisions)
            propagations.append(solver.propagations)
        else:
            raise ValueError(f"Unexpected solver state: {solver.state}")
    median_decisions = compute_median(decisions)
    median_propagations = compute_median(propagations)
    tqdm.write(f"MiniSAT median decisions: {median_decisions}, median propagations: {median_propagations}")
    return decisions, propagations, median_decisions, median_propagations

@torch.inference_mode()
def eval_model(dataset: CNFDataset, model: nn.Module, device: torch.device) -> Tuple[list, list, float, float]:
    decisions = []
    propagations = []
    for cnf_file in tqdm(dataset, desc="Evaluating model"):
        loader = CNFLoader(cnf_file)
        loader.load_cnf()

        solver = MiniSAT(cnf=loader.clauses)
        solver.step()

        while solver.state == 0:
            graph = build_vcg_from_solver(solver, device)
            action = model.select_action(graph)
            solver.step(solver.candidates[action])

        if solver.state in [10, 20]:
            decisions.append(solver.decisions)
            propagations.append(solver.propagations)
        else:
            raise ValueError(f"Unexpected solver state: {solver.state}")
    median_decisions = compute_median(decisions)
    median_propagations = compute_median(propagations)
    tqdm.write(f"Model median decisions: {median_decisions}, median propagations: {median_propagations}")
    return decisions, propagations, median_decisions, median_propagations

def train_model(
    train_dataset: CNFDataset,
    valid_dataset: CNFDataset,
    model: nn.Module,
    device: torch.device,
):
    cfg = DQNTrainConfig()

    model = model.to(device)
    target_model = copy.deepcopy(model).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    replay = ReplayBuffer(cfg.replay_size)

    total_env_steps = 0
    pending_env_steps = 0
    updates = 0
    best_valid_score = float("inf")
    best_state_dict = copy.deepcopy(model.state_dict())

    progress = tqdm(total=cfg.batch_updates, desc="DQN updates")
    indices = list(range(len(train_dataset)))
    random.shuffle(indices)
    next_train_idx = 0

    while updates < cfg.batch_updates:
        if next_train_idx >= len(indices):
            random.shuffle(indices)
            next_train_idx = 0

        cnf_file = train_dataset[indices[next_train_idx]]
        next_train_idx += 1

        env_steps_added, _ = run_training_episode(
            cnf_file=cnf_file,
            model=model,
            replay=replay,
            device=device,
            total_env_steps=total_env_steps,
            cfg=cfg,
        )
        total_env_steps += env_steps_added
        if (
            total_env_steps >= cfg.initial_exploration_steps
            and len(replay) >= cfg.batch_size
        ):
            pending_env_steps += env_steps_added

        while (
            pending_env_steps >= cfg.update_frequency
            and updates < cfg.batch_updates
        ):
            model.train()
            loss = dqn_update(
                model=model,
                target_model=target_model,
                optimizer=optimizer,
                replay=replay,
                device=device,
                cfg=cfg,
            )
            updates += 1
            progress.update(1)
            pending_env_steps -= cfg.update_frequency
            progress.set_postfix(
                loss=f"{loss:.4f}",
                eps=f"{epsilon_by_env_steps(total_env_steps, cfg):.3f}",
                replay=len(replay),
            )

            if updates % cfg.target_update_frequency == 0:
                target_model.load_state_dict(model.state_dict())

            if updates % cfg.eval_frequency == 0:
                model.eval()
                valid_decisions, valid_propagations, valid_median_decisions, valid_median_propagations = eval_model(valid_dataset, model, device)

                if valid_median_decisions < best_valid_score:
                    best_valid_score = valid_median_decisions
                    best_state_dict = copy.deepcopy(model.state_dict())

    progress.close()
    model.load_state_dict(best_state_dict)
    target_model.load_state_dict(best_state_dict)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphQSat()
    model.eval()
    model.to(device)

    test_dataset = CNFDataset(num_variables=50, sat=True, split="test")
    solver_decisions, solver_propagations, solver_median_decisions, solver_median_propagations = eval_solver(test_dataset)

    train_dataset = CNFDataset(num_variables=50, sat=True, split="train")
    valid_dataset = CNFDataset(num_variables=50, sat=True, split="valid")
    train_results = train_model(train_dataset, valid_dataset, model, device)

    model_decisions, model_propagations, model_median_decisions, model_median_propagations = eval_model(test_dataset, model, device)
    print(f"MiniSAT median decisions: {solver_median_decisions}, median propagations: {solver_median_propagations}")
    print(f"Model median decisions: {model_median_decisions}, median propagations: {model_median_propagations}")
    print(f"Median decision reduction: {compute_median_reduction(model_decisions, solver_decisions):.2f}x")
    print(f"Median propagation reduction: {compute_median_reduction(model_propagations, solver_propagations):.2f}x")

    torch.save(model.state_dict(), "graphqsat_model.pth")
