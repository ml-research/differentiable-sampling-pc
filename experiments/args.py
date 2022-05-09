import argparse
import os
import pathlib

from experiments.data import Dist


def get_general_experiment_parser() -> argparse.ArgumentParser:
    """
    Returns a parser for general experiment arguments.

    Returns:
        parser: argparse.ArgumentParser

    """

    home = os.getenv("HOME")
    data_dir = os.getenv("DATA_DIR", os.path.join(home, "data"))
    results_dir = os.getenv("RESULTS_DIR", os.path.join(home, "results"))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        help="Dataset to use for training.",
    )
    parser.add_argument("--data-dir", default=data_dir, help="path to dataset")
    parser.add_argument("--results-dir", default=results_dir, help="path to results")
    parser.add_argument(
        "--num-workers", type=int, help="number of data loading workers", default=4
    )
    parser.add_argument("--batch-size", type=int, default=128, help="input batch size")
    parser.add_argument(
        "--epochs", type=int, default=100, help="number of epochs to train for"
    )
    parser.add_argument("--gpu", type=int, default=0, help="using gpu device id")
    # SPN-specific arguments
    parser.add_argument("--spn-K", type=int, default=5)
    parser.add_argument("--spn-R", type=int, default=3)
    parser.add_argument("--spn-D", type=int, default=3)
    parser.add_argument("--spn-min-sigma", type=float, default=1e-3)
    parser.add_argument("--spn-max-sigma", type=float, default=1.0)
    parser.add_argument(
        "--spn-tau",
        type=float,
        default=1.0,
        help="tau value for differentiable " "sampling",
    )
    parser.add_argument(
        "--spn-hard", type=int, help="use hard differentiable sampling (0: off, 1: on)"
    )

    parser.add_argument(
        "--epochs-pretrain",
        type=int,
        default=0,
        help="number of epochs of pretraining",
    )
    parser.add_argument(
        "--lr-pretrain",
        type=float,
        default=1e-1,
        help="learning rate for pretraining",
    )

    parser.add_argument("--tag", type=str, help="tag for experiment")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--dist",
        type=Dist,
        choices=list(Dist),
        default=Dist.BINOMIAL,
        help="data distribution",
    )
    parser.add_argument(
        "--load-and-eval",
        default=None,
        type=pathlib.Path,
        help="path to a result directory with a "
        "model and stored args. if set, "
        "training is skipped and model is "
        "evaluated",
    )
    parser.add_argument(
        "--spn-tau-method",
        default="constant",
        choices=["constant", "annealed", "learned"],
        help="how the tau parameter is decided",
    )
    parser.add_argument(
        "--fid",
        action="store_true",
        help="compute FID score for the model",
    )

    return parser