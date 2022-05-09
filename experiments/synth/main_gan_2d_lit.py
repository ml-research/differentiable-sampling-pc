#!/usr/bin/env python
# encoding: utf-8
# Modified from: https://github.com/OctoberChang/MMD-GAN

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
        "--model-generator",
        type=str,
        choices=["spn", "gan"],
        required=True,
        help="model type (gan " "or spn)",
    )
    parser.add_argument(
        "-D",
        "--model-discriminator",
        type=str,
        required=True,
        choices=["spn", "gan"],
        help="model type (" "gan or spn)",
    )
    parser.add_argument(
        "--disc-step", type=int, default=10, help="discriminator step interval"
    )
    parser.add_argument("--data-synth", required=True)
    return parser.parse_args()


class GanDiscriminator(nn.Module):
    def __init__(self):
        super(GanDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(2, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


class GanSpn(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.spn = FlatSpn2D(args)
        self.args = args

        # Loss function
        self.adversarial_loss = torch.nn.BCELoss()

        if args.model_discriminator == "gan":
            self.discriminator = GanDiscriminator()
        elif args.model_discriminator == "spn":
            self.discriminator = SpnDiscriminator(
                num_channels=1,
                width=2,
                height=1,
                D=1,
                R=args.spn_R,
                K=args.spn_K,
                min_sigma=args.spn_min_sigma,
                max_sigma=args.spn_max_sigma,
                dist=Dist.NORMAL,
            )
        else:
            raise NotImplementedError()

    def configure_optimizers(self):
        # Optimizers
        optimizer_G = torch.optim.Adam(
            self.spn.parameters(), lr=args.lr_d  # , betas=(opt.b1, opt.b2)
        )
        optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), lr=args.lr_g  # , betas=(opt.b1, opt.b2)
        )

        # Scheduler
        lr_scheduler_G = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_G,
            milestones=[int(0.7 * args.epochs), int(0.9 * args.epochs)],
            gamma=0.1,
        )
        lr_scheduler_D = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_D,
            milestones=[int(0.7 * args.epochs), int(0.9 * args.epochs)],
            gamma=0.1,
        )

        return [optimizer_G, optimizer_D], [lr_scheduler_G, lr_scheduler_D]

    def training_step(self, train_batch, batch_idx, optimizer_idx):
        imgs = train_batch
        real_imgs = imgs
        # Adversarial ground truths
        valid = torch.Tensor(imgs.size(0), 1).fill_(1.0).to(device)
        fake = torch.Tensor(imgs.size(0), 1).fill_(0.0).to(device)
        if optimizer_idx == 0:
            # Generate a batch of images
            gen_imgs = self.spn.sample_diff(n_samples=real_imgs.shape[0])

            # Loss measures generator's ability to fool the discriminator
            pred_gen = self.discriminator(gen_imgs)
            g_loss = self.adversarial_loss(pred_gen, valid)
            self.log("train_loss", g_loss)
            return g_loss

        if optimizer_idx == 1:
            # Generate a batch of images
            gen_imgs = self.spn.sample_diff(n_samples=real_imgs.shape[0])

            # Measure discriminator's ability to classify real from generated samples
            pred_real = self.discriminator(real_imgs)
            pred_gen = self.discriminator(gen_imgs.detach())
            real_loss = self.adversarial_loss(pred_real, valid)
            fake_loss = self.adversarial_loss(pred_gen, fake)
            d_loss = (real_loss + fake_loss) / 2

            self.log("d_loss", d_loss)

            num_correct = ((pred_real > 0.5) == valid).sum() + (
                (pred_gen < 0.5) == fake
            ).sum()
            accuracy = num_correct / (valid.shape[0] + fake.shape[0])
            self.log("accuracy", accuracy)
            return d_loss

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
        data = generate_data(self.args.data_synth, self.device, n_samples=10000)
        final_plot(self.spn, data=data, save_dir=results_dir, step=self.current_epoch)
        rtpt.step()


if __name__ == "__main__":
    args = get_args()
    (args, results_dir, writer, device, image_shape, rtpt,) = setup_experiment(
        name="gan-synth_" + args.data_synth, args=args, with_tensorboard=False
    )

    # model
    model = GanSpn(args)
    main_lit.main(args=args, model=model, results_dir=results_dir)