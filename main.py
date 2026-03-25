import argparse
import copy
import random
from pathlib import Path
from typing import Tuple

from tqdm import tqdm

import torch
from torch import nn

from minisat_wrapper import MiniSAT

from buffer import ReplayBuffer
from cnf import CNFLoader, build_vcg_from_solver
from dataset import CNFDataset, build_dataset, infer_dataset_source, parse_legacy_dataset_spec
from dqn import DQNTrainConfig, epsilon_by_env_steps, run_training_episode, dqn_update
from model import GraphQSat


def compute_median(values):
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n == 0:
        return None
    if n % 2 == 1:
        return sorted_values[n // 2]
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
                _, _, valid_median_decisions, _ = eval_model(valid_dataset, model, device)

                if valid_median_decisions < best_valid_score:
                    best_valid_score = valid_median_decisions
                    best_state_dict = copy.deepcopy(model.state_dict())

    progress.close()
    model.load_state_dict(best_state_dict)
    target_model.load_state_dict(best_state_dict)


def parse_bool_arg(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("Expected 'true' or 'false'.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sat_label(sat: bool) -> str:
    return "SAT" if sat else "UNSAT"


def describe_dataset_spec(dataset_spec: str) -> str:
    source = infer_dataset_source(dataset_spec)
    if source == "legacy":
        return f"legacy dataset {int(dataset_spec)}"
    return f"mas_sat dataset {dataset_spec}"


def validate_train_dataset_spec(dataset_spec: str) -> None:
    num_variables = parse_legacy_dataset_spec(dataset_spec)
    if num_variables is not None and num_variables != 50:
        raise ValueError("Legacy training only supports '--dataset 50'. Use a mas_sat dataset id for other training datasets.")


def validate_test_dataset_spec(dataset_spec: str) -> None:
    num_variables = parse_legacy_dataset_spec(dataset_spec)
    if num_variables is not None and num_variables not in {50, 100, 250}:
        raise ValueError("Legacy testing only supports '--dataset 50', '--dataset 100', or '--dataset 250'.")


def evaluate_and_report(dataset: CNFDataset, model: nn.Module, device: torch.device) -> None:
    solver_decisions, solver_propagations, solver_median_decisions, solver_median_propagations = eval_solver(dataset)
    model_decisions, model_propagations, model_median_decisions, model_median_propagations = eval_model(dataset, model, device)
    print(f"MiniSAT median decisions: {solver_median_decisions}, median propagations: {solver_median_propagations}")
    print(f"Model median decisions: {model_median_decisions}, median propagations: {model_median_propagations}")
    print(f"Median decision reduction: {compute_median_reduction(model_decisions, solver_decisions):.2f}x")
    print(f"Median propagation reduction: {compute_median_reduction(model_propagations, solver_propagations):.2f}x")


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)


def run_train(args: argparse.Namespace) -> None:
    validate_train_dataset_spec(args.dataset)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphQSat()
    model.eval()
    model.to(device)

    dataset_description = describe_dataset_spec(args.dataset)
    print(f"Training on {sat_label(args.sat)} {dataset_description} with seed {args.seed}")
    train_dataset = build_dataset(args.dataset, sat=args.sat, split="train")
    valid_dataset = build_dataset(args.dataset, sat=args.sat, split="valid")
    test_dataset = build_dataset(args.dataset, sat=args.sat, split="test")

    train_model(train_dataset, valid_dataset, model, device)
    evaluate_and_report(test_dataset, model, device)

    torch.save(model.state_dict(), args.checkpoint_path)
    print(f"Saved checkpoint to {args.checkpoint_path}")


def run_test(args: argparse.Namespace) -> None:
    validate_test_dataset_spec(args.dataset)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = GraphQSat()
    model.to(device)
    load_checkpoint(model, str(checkpoint_path), device)
    model.eval()

    dataset_description = describe_dataset_spec(args.dataset)
    print(
        f"Testing checkpoint {checkpoint_path} on "
        f"{sat_label(args.sat)} {dataset_description} "
        f"with seed {args.seed}"
    )
    test_dataset = build_dataset(args.dataset, sat=args.sat, split="test")
    evaluate_and_report(test_dataset, model, device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or evaluate Graph-Q-SAT.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset_help = (
        "Dataset specification: use an integer like '50' for legacy datasets or "
        "'<family>/<name>' like '3-sat/easy' for mas_sat datasets."
    )

    train_parser = subparsers.add_parser("train", help="Train a model on a legacy or mas_sat dataset.")
    train_parser.add_argument(
        "--dataset",
        required=True,
        help=dataset_help,
    )
    train_parser.add_argument(
        "--sat",
        type=parse_bool_arg,
        required=True,
        metavar="{true,false}",
        help="Use satisfiable data with 'true' or unsatisfiable data with 'false'.",
    )
    train_parser.add_argument(
        "--checkpoint-path",
        default="graphqsat_model.pth",
        help="Path where the trained checkpoint will be saved.",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible training and evaluation.",
    )
    train_parser.set_defaults(func=run_train)

    test_parser = subparsers.add_parser("test", help="Evaluate a checkpoint on a legacy or mas_sat test split.")
    test_parser.add_argument(
        "--dataset",
        required=True,
        help=dataset_help,
    )
    test_parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to a saved model checkpoint.",
    )
    test_parser.add_argument(
        "--sat",
        type=parse_bool_arg,
        required=True,
        metavar="{true,false}",
        help="Use satisfiable data with 'true' or unsatisfiable data with 'false'.",
    )
    test_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible evaluation.",
    )
    test_parser.set_defaults(func=run_test)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
