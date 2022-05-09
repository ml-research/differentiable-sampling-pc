import os
import warnings
from abc import abstractmethod, ABC, ABCMeta
from argparse import Namespace

import torch
import torchvision
from rtpt import RTPT
from torch import nn
from torch.nn import functional as F

from experiments.evaluation import construct_marginalization_indices
from simple_einet.einet import EinetConfig, Einet
from experiments.data import get_distribution, get_data_shape, Dist
from experiments.synth.utils_2d import get_num_components
import pytorch_lightning as pl


class AbstractLitGenerator(pl.LightningModule, ABC):
    def __init__(self, args: Namespace, name: str):
        """
        Abstract Generator class.

        Args:
            args: Arguments parsed from argparse.
            name: Name of the model used for RTPT.
        """
        super().__init__()
        self.args = args
        self.image_shape = get_data_shape(args.dataset)
        self.rtpt = RTPT(
            name_initials="SL",
            experiment_name=name + "_" + str(args.tag),
            max_iterations=args.epochs + 1,
        )

        # Prepare masks for test evaluation
        self.test_marginalization_masks = construct_marginalization_indices(
            self.image_shape
        )

        self.mask_fixed = None

    def on_train_start(self) -> None:
        self.rtpt.start()

    @abstractmethod
    def generate_samples(self, num_samples: int):
        """Sample pixels should be in range [0, 1].

        Args:
            num_samples: Number of samples to generate.

        Returns:
            Samples.
        """
        pass

    def samples_generator(self, batch_size: int):
        """
        Generator that yields batches of samples.
        Args:
            batch_size: Number of samples per batch.

        Returns:
            Generator that yields batches of samples.
        """
        while True:
            samples = self.generate_samples(batch_size)
            yield samples

    @abstractmethod
    def get_spn(self):
        """Return the SPN of the model if available."""
        pass

    def on_train_epoch_end(self) -> None:
        with torch.no_grad():
            samples = self.generate_samples(num_samples=25)
            grid = torchvision.utils.make_grid(
                samples.data[:25], nrow=5, pad_value=0.0, normalize=True
            )
            self.logger.experiment.add_image("samples", grid, self.current_epoch)
            self.rtpt.step()

    def save_samples(self, samples_dir, num_samples, nrow):
        """
        Save samples to a directory.

        Args:
            samples_dir: Directory to save samples to.
            num_samples: Number of samples to save.
            nrow: Number of samples per row.

        """
        for i in range(5):
            samples = self.generate_samples(num_samples)
            grid = torchvision.utils.make_grid(
                samples.data[:25], nrow=nrow, pad_value=0.0, normalize=True
            )
            torchvision.utils.save_image(grid, os.path.join(samples_dir, f"{i}.png"))


def make_einet(args):
    """
    Make an EinsumNetworks model based off the given arguments.

    Args:
        args: Arguments parsed from argparse.

    Returns:
        EinsumNetworks model.
    """
    image_shape = get_data_shape(args.dataset)
    leaf_kwargs, leaf_type = get_distribution(
        args.dist, args.spn_min_sigma, args.spn_max_sigma
    )
    config = EinetConfig(
        num_features=image_shape.num_pixels,
        num_channels=image_shape.channels,
        depth=args.spn_D,
        num_sums=args.spn_K,
        num_leaves=args.spn_K,
        num_repetitions=args.spn_R,
        num_classes=1,
        leaf_kwargs=leaf_kwargs,
        leaf_type=leaf_type,
        dropout=0.0,
    )
    return Einet(config)