"""All settings in one place, filled from the command line."""

import argparse
from dataclasses import dataclass


@dataclass
class Config:
    epochs: int = 3
    lr: float = 1e-3
    batch_size: int = 64
    hidden: int = 256
    n_train: int = 10_000
    seed: int = 0
    out: str = "runs/fashion"
    data: str = "data"


def parse_args(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(description="Train a FashionMNIST MLP.")
    for name, default in vars(Config()).items():
        parser.add_argument(f"--{name.replace('_', '-')}",
                            type=type(default), default=default)
    return Config(**vars(parser.parse_args(argv)))
