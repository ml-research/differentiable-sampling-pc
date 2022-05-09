#!/usr/bin/env python

import timeit

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from experiments.args import get_general_experiment_parser
from experiments.mmd_gan import base_module
from experiments.mmd_gan.mmd import mix_rbf_mmd2
from experiments.synth import main_lit
from experiments.synth.utils_2d import (
    generate_data,
    SynthDataset,
)
from experiments.synth.model_2d import FlatSpn2D
from experiments.synth.plot_synth import plot_distribution, final_plot
from experiments.utils import (
    setup_experiment,
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
    parser.add_argument(
        "--data-synth",
        required=True
    )
    return parser.parse_args()


class Spn(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.spn = FlatSpn2D(args)
        self.args = args
        self.current_step = 0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(0.7 * args.epochs), int(0.9 * args.epochs)], gamma=0.1
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, train_batch, batch_idx):
        nll = -1 * self.spn(train_batch).mean()
        self.log("train_loss", nll)
        return nll

    def _save_img(self):
        with torch.no_grad():
            x = generate_data(self.args.data_synth, self.device, n_samples=1000)
            plot_distribution(
                self.spn,
                x,
                writer=self.logger.experiment,
                step=self.current_epoch,
            )

    def on_epoch_end(self) -> None:
        self._save_img()
        data = generate_data(self.args.data_synth, self.device, n_samples=5000)
        final_plot(self.spn, data=data, save_dir=results_dir, step=self.current_epoch)
        rtpt.step()


if __name__ == "__main__":
    args = get_args()
    (
        args,
        results_dir,
        writer,
        device,
        dataloader_train,
        dataloader_test,
        image_shape,
        rtpt,
    ) = setup_experiment(name="spn" + args.data_synth, args=args, with_tensorboard=False)

    # model
    model = Spn(args)
    main_lit.main(args=args, model=model, results_dir=results_dir)