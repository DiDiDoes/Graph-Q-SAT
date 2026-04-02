import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class OptimConfig:
    batch_updates: int = 50_000
    lr: float = 2e-5
    max_decisions_train: int = 500
    step_penalty: float = -0.1
    truncate_penalty: float = -10.0
    grad_clip_norm: float = 1.0
    eval_frequency: int = 1_000


def build_optim_config(args: argparse.Namespace) -> OptimConfig:
    if args.batch_updates < 1:
        raise ValueError("--batch-updates must be at least 1.")
    if args.lr <= 0:
        raise ValueError("--lr must be positive.")
    if args.max_decisions_train < 1:
        raise ValueError("--max-decisions-train must be at least 1.")
    if args.grad_clip_norm <= 0:
        raise ValueError("--grad-clip-norm must be positive.")
    if args.eval_frequency < 1:
        raise ValueError("--eval-frequency must be at least 1.")

    return OptimConfig(
        batch_updates=args.batch_updates,
        lr=args.lr,
        max_decisions_train=args.max_decisions_train,
        step_penalty=args.step_penalty,
        truncate_penalty=args.truncate_penalty,
        grad_clip_norm=args.grad_clip_norm,
        eval_frequency=args.eval_frequency,
    )
