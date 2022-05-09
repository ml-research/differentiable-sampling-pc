#!/usr/bin/env python
# encoding: utf-8
# Modified from: https://github.com/OctoberChang/MMD-GAN

import timeit
from argparse import Namespace
from typing import Tuple

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
from experiments.data import Dist
from experiments.lit.model import make_einet, AbstractLitGenerator
from experiments.mmd_gan import base_module
from experiments.mmd_gan.mmd import mix_rbf_mmd2
from experiments.lit import train_test
from experiments.utils import (
    setup_experiment,
    setup_experiment_lit,
    anneal_tau, load_from_checkpoint,
)


class GanGenerator(nn.Module):
    def __init__(self, image_shape, latent_dim):
        super(GanGenerator, self).__init__()

        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.image_shape = image_shape
        self.model = nn.Sequential(
            *block(latent_dim, 128),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(image_shape.num_pixels * image_shape.channels)),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.image_shape)
        return img


class SpnGenerator(nn.Module):
    def __init__(
        self,
        num_channels: int,
        height: int,
        width: int,
        D: int,
        K: int,
        R: int,
        min_sigma: float,
        max_sigma: float,
        tau: float,
        hard: bool,
        dist: Dist,
        args,
    ):
        """
        The generative SPN.

        Args:
            num_channels: The number of channels in the input image.
            height: The height of the input image.
            width: The width of the input image.
            D: The depth of the SPN.
            K: The number of leaves/sums in the SPN.
            R: The number of repetitions in the SPN.
            tau: The temperature for the differentiable sampling.
            hard: Whether to use hard or soft sampling.
            dist: The distribution to use for the leaves.
        """
        super(SpnGenerator, self).__init__()
        self.D = D
        self.K = K
        self.R = R
        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.hard = hard
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.dist = dist
        self.args = args

        self.spn = make_einet(args)

        self.epoch = 0
        if self.args.spn_tau_method == "constant":
            self._tau = args.spn_tau
        elif self.args.spn_tau_method == "learned":
            self._tau = nn.Parameter(torch.tensor(0.0))

    @property
    def tau(self):
        if self.args.spn_tau_method == "constant":
            return self.args.spn_tau
        elif self.args.spn_tau_method == "learned":
            # Cast into valid range
            return torch.sigmoid(self._tau)
        elif self.args.spn_tau_method == "annealed":
            return anneal_tau(self.epoch, max_epochs=self.args.epochs)
        else:
            raise ValueError()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate a new batch of samples.

        Args:
            z: Ignored -- just for code compatibility with the GanGenerator.

        Returns:
            Batch of samples.
        """
        num_samples = z.shape[0]
        samples = self.spn.sample_differentiable(
            num_samples=num_samples, hard=self.hard, tau=self.tau
        )
        samples = samples.view(num_samples, self.num_channels, self.width, self.height)
        return samples


class SpnDiscriminator(nn.Module):
    def __init__(
        self,
        num_channels: int,
        height: int,
        width: int,
        D: int,
        K: int,
        R: int,
        min_sigma: float,
        max_sigma: float,
        dist: Dist,
    ):
        """
        The generative SPN.

        Args:
            num_channels: The number of channels in the input image.
            height: The height of the input image.
            width: The width of the input image.
            D: The depth of the SPN.
            K: The number of leaves/sums in the SPN.
            R: The number of repetitions in the SPN.
            min_sigma: The minimum sigma for the leaves.
            max_sigma: The maximum sigma for the leaves.
            dist: The distribution to use for the leaves.
        """
        super(SpnDiscriminator, self).__init__()
        self.D = D
        self.K = K
        self.R = R
        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.dist = dist

        self.model = self._make_spn()

    def _make_spn(self):
        leaf_kwargs, leaf_type = self.get_distribution(
            self.dist, self.min_sigma, self.max_sigma
        )
        config = EinetConfig(
            num_features=self.height * self.width,
            num_channels=self.num_channels,
            depth=self.D,
            num_sums=self.K,
            num_leaves=self.K,
            num_repetitions=self.R,
            num_classes=2,
            leaf_kwargs=leaf_kwargs,
            leaf_type=leaf_type,
            dropout=0.0,
        )
        return Einet(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify a batch of samples.

        Args:
            x: The batch of samples to classify.

        Returns:
            The logits of the classifier.

        """
        # Class conditional data distribution
        p_x_g_c = self.model(x)  # p(x | c)

        # Bayes: p(c | x ) = p(c) * p(x | c) / p(x)
        p_c = np.log(1 / 2)
        p_c_g_x = p_c + p_x_g_c - torch.logsumexp(p_x_g_c + p_c, dim=1, keepdim=True)

        # Only return likelihood for first class (equals 1-second)
        return p_c_g_x[:, [0]].exp()


class GanDiscriminator(nn.Module):
    def __init__(self, image_shape):
        super(GanDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(image_shape.num_pixels * image_shape.channels, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


def get_args():
    parser = get_general_experiment_parser()
    parser.add_argument(
        "--lr-d", type=float, default=0.0002, help="adam: learning rate"
    )
    parser.add_argument(
        "--lr-g", type=float, default=0.0002, help="adam: learning rate"
    )
    parser.add_argument(
        "--b1",
        type=float,
        default=0.5,
        help="adam: decay of first order momentum of gradient",
    )
    parser.add_argument(
        "--b2",
        type=float,
        default=0.999,
        help="adam: decay of first order momentum of gradient",
    )
    parser.add_argument(
        "--latent-dim", type=int, default=100, help="dimensionality of the latent space"
    )
    parser.add_argument(
        "-G",
        "--model",
        # "--model-generator",
        type=str,
        choices=["spn", "gan"],
        help="model type (gan " "or spn)",
    )
    parser.add_argument(
        "-D",
        "--model-discriminator",
        type=str,
        default="gan",
        choices=["spn", "gan"],
        help="model type (" "gan or spn)",
    )
    parser.add_argument(
        "--disc-step", type=int, default=10, help="discriminator step interval"
    )
    return parser.parse_args()


class ADV(AbstractLitGenerator):
    def __init__(self, args: Namespace):
        super().__init__(args, name="gan")
        self.args = args
        self.automatic_optimization = False

        # Count current iteration to decide if a discriminator step is necessary
        self.current_iteration = 0

        # Loss function
        self.adversarial_loss = torch.nn.BCELoss()

        # Initialize generator and discriminator
        if args.model == "gan":
            self.generator = GanGenerator(self.image_shape, args.latent_dim)
        elif args.model == "spn":
            self.generator = SpnGenerator(
                num_channels=self.image_shape.channels,
                width=self.image_shape.width,
                height=self.image_shape.height,
                D=args.spn_D,
                R=args.spn_R,
                K=args.spn_K,
                tau=args.spn_tau,
                hard=args.spn_hard,
                min_sigma=args.spn_min_sigma,
                max_sigma=args.spn_max_sigma,
                dist=args.dist,
                args=args,
            )
        else:
            raise Exception()

        if args.model_discriminator == "gan":
            self.discriminator = GanDiscriminator(self.image_shape)
        elif args.model_discriminator == "spn":
            self.discriminator = SpnDiscriminator(
                num_channels=self.image_shape.channels,
                width=self.image_shape.width,
                height=self.image_shape.height,
                D=args.spn_D,
                R=args.spn_R,
                K=args.spn_K,
                min_sigma=args.spn_min_sigma,
                max_sigma=args.spn_max_sigma,
                dist=args.dist,
            )
        else:
            raise NotImplementedError()

        self.fixed_z = nn.Parameter(
            torch.randn(25, args.latent_dim), requires_grad=False
        )

    def configure_optimizers(self):
        # Optimizers
        optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=args.lr_g,
            betas=(self.args.b1, self.args.b2),
        )
        optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=args.lr_d,
            betas=(self.args.b1, self.args.b2),
        )

        # # Scheduler
        # lr_scheduler_G = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer_G,
        #     milestones=[int(0.7 * args.epochs), int(0.9 * args.epochs)],
        #     gamma=0.1,
        # )
        # lr_scheduler_D = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer_D,
        #     milestones=[int(0.7 * args.epochs), int(0.9 * args.epochs)],
        #     gamma=0.1,
        # )

        return [optimizer_G, optimizer_D]  # , [lr_scheduler_G, lr_scheduler_D]

    def get_spn(self):
        if self.args.model == "spn":
            return self.generator.spn
        else:
            return None

    def training_step(self, train_batch, batch_idx):
        imgs, _ = train_batch
        # Adversarial ground truths
        valid = torch.ones(imgs.size(0), 1, device=self.device)

        # Configure input
        real_imgs = imgs

        #  Generator step
        gen_imgs = self.step_generator(imgs, valid)

        #  Discriminator step
        if self.current_iteration % args.disc_step == 0:
            self.step_discriminator(gen_imgs, real_imgs, valid)

    def step_generator(self, imgs, valid):
        optimizer_G, _ = self.optimizers()
        g_loss, gen_imgs = self.loss_generator(imgs, valid)
        optimizer_G.zero_grad()
        self.manual_backward(g_loss)
        optimizer_G.step()

        self.log("Loss/g_loss", g_loss, prog_bar=True)

        return gen_imgs

    def loss_generator(self, imgs, valid) -> Tuple[torch.Tensor, torch.Tensor]:
        # Sample noise as generator input
        z = torch.randn(
            imgs.size(0), args.latent_dim, device=self.device, requires_grad=True
        )
        # Generate a batch of images
        gen_imgs = self.generator(z)
        if args.model == "spn" and args.dist == Dist.BINOMIAL:
            gen_imgs /= 255
        # Loss measures generator's ability to fool the discriminator
        g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)
        return g_loss, gen_imgs

    def step_discriminator(self, gen_imgs, real_imgs, valid):
        _, optimizer_D = self.optimizers()
        d_loss = self.loss_discriminator(gen_imgs, real_imgs, valid)
        optimizer_D.zero_grad()
        self.manual_backward(d_loss)
        optimizer_D.step()

        self.log("Loss/d_loss", d_loss, prog_bar=True)

    def loss_discriminator(self, gen_imgs, real_imgs, valid):
        # Create fake labels
        fake = torch.zeros(real_imgs.size(0), 1, device=self.device)
        # Measure discriminator's ability to classify real from generated samples
        real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
        fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        return d_loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        # Adversarial ground truths
        valid = torch.ones(x.size(0), 1, device=self.device)
        g_loss, gen_imgs = self.loss_generator(imgs=x, valid=valid)
        self.log("Loss/val_g_loss", g_loss)

        d_loss = self.loss_discriminator(gen_imgs=gen_imgs, real_imgs=x, valid=valid)
        self.log("Loss/val_d_loss", d_loss)

    def generate_samples(self, num_samples: int) -> torch.Tensor:
        """Generate samples from the generator."""
        if args.model == "spn":
            samples = self.generator.spn.sample(
                num_samples=num_samples, mpe_at_leaves=True
            ).view(num_samples, *self.image_shape)
            samples = samples / 255
        elif args.model == "gan":
            samples = self.generator(self.fixed_z).view(num_samples, *self.image_shape)
        else:
            raise ValueError(f"Unknown model generator: {args.model}")

        return samples


if __name__ == "__main__":
    args = get_args()
    results_dir, args = setup_experiment_lit(
        name="adv",
        args=args,
        remove_if_exists=True
    )

    # model
    # Load or create model
    if args.load_and_eval:
        model = load_from_checkpoint(results_dir, load_fn=ADV.load_from_checkpoint, args=args)
    else:
        model = ADV(args)
    normalize = args.model == "gan" or (
        args.model == "spn" and args.dist == Dist.NORMAL
    )

    # hparams
    hparams = {
        "lr-g": args.lr_g,
        "lr-d": args.lr_d,
        "latent-dim": args.latent_dim,
        "model": args.model,
    }

    train_test.main(
        args=args,
        model=model,
        results_dir=results_dir,
        normalize=normalize,
        hparams=hparams,
    )