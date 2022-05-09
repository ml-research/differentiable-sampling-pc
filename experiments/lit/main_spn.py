#!/usr/bin/env python
from argparse import Namespace

import torch
import torch.nn.parallel
import torch.utils.data

from experiments.args import get_general_experiment_parser
from experiments.data import get_data_shape, Dist
from experiments.lit import train_test
from experiments.lit.model import AbstractLitGenerator, make_einet
from experiments.utils import (
    setup_experiment_lit,
    load_from_checkpoint,
)


def get_args():
    """
    Get the arguments for the SPN experiment.

    Returns:
        args: The arguments.
    """
    parser = get_general_experiment_parser()
    parser.add_argument(
        "--lr", type=float, default=0.05, help="learning rate, default=0.05"
    )
    return parser.parse_args()


class SPN(AbstractLitGenerator):
    def __init__(self, args: Namespace):
        super().__init__(args=args, name="spn")
        self.spn = make_einet(args)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(0.7 * args.epochs), int(0.9 * args.epochs)],
            gamma=0.1,
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, train_batch, batch_idx):
        data, labels = train_batch
        nll = self.compute_loss(data)
        self.log("train_loss", nll)
        return nll

    def validation_step(self, val_batch, batch_idx):
        data, labels = val_batch
        nll = self.compute_loss(data)
        self.log("val_loss", nll)
        return nll

    def compute_loss(self, data):
        if self.args.dist == Dist.BINOMIAL:
            data *= 255
        nll = -1 * self.spn(data).mean()
        return nll

    def generate_samples(self, num_samples: int):
        samples = self.spn.sample(num_samples=num_samples, mpe_at_leaves=True).view(
            -1, *self.image_shape
        )
        samples = samples / 255.0
        return samples

    def get_spn(self):
        return self.spn


if __name__ == "__main__":
    args = get_args()
    results_dir, args = setup_experiment_lit(
        name="spn", args=args, remove_if_exists=True
    )

    # Load or create model
    if args.load_and_eval:
        model = load_from_checkpoint(
            results_dir, load_fn=SPN.load_from_checkpoint, args=args
        )
    else:
        model = SPN(args)

    normalize = args.dist == Dist.NORMAL
    hparams = {
        "lr": args.lr,
    }
    train_test.main(
        args=args,
        model=model,
        results_dir=results_dir,
        normalize=normalize,
        hparams=hparams,
    )