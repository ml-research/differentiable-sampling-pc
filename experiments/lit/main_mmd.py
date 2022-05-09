#!/usr/bin/env python
# encoding: utf-8
# Modified from: https://github.com/OctoberChang/MMD-GAN
import os.path
import timeit
from collections import defaultdict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torchvision
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from torch.autograd import Variable
from torch.utils.data import DataLoader

from experiments.args import get_general_experiment_parser
from experiments.data import Dist, get_distribution
from experiments.lit import train_test
from experiments.lit.model import AbstractLitGenerator, make_einet
from experiments.mmd_gan import base_module, util
from experiments.mmd_gan.mmd import mix_rbf_mmd2
from experiments.utils import (
    setup_experiment,
    anneal_tau,
    setup_experiment_lit, load_from_checkpoint,
)
from experiments.mmd_gan.mmd import mix_rbf_mmd2


def get_args():
    """
    Get the arguments for the MMD experiment.

    Returns:
        args: The arguments.
    """
    parser = get_general_experiment_parser()
    parser.add_argument(
        "--nz", type=int, default=100, help="size of the latent z vector"
    )
    parser.add_argument(
        "--lr-g", type=float, default=0.05, help="learning rate, default=0.05"
    )
    parser.add_argument(
        "--lr-d", type=float, default=0.00005, help="learning rate, default=0.00005"
    )
    parser.add_argument(
        "--Diters", type=int, default=5, help="number of D iters per each G iter"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["spn", "gan"],
        help="choose model ('spn' or " "'gan'",
    )
    return parser.parse_args()


# NetG is a decoder
# input: batch_size * nz * 1 * 1
# output: batch_size * nc * image_size * image_size
class NetG(nn.Module):
    def __init__(self, decoder):
        super(NetG, self).__init__()
        self.decoder = decoder

    def forward(self, input):
        output = self.decoder(input)
        return output


# NetD is an encoder + decoder
# input: batch_size * nc * image_size * image_size
# f_enc_X: batch_size * k * 1 * 1
# f_dec_X: batch_size * nc * image_size * image_size
class NetD(nn.Module):
    def __init__(self, encoder, decoder):
        super(NetD, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):
        f_enc_X = self.encoder(input)
        f_dec_X = self.decoder(f_enc_X)

        f_enc_X = f_enc_X.view(input.size(0), -1)
        f_dec_X = f_dec_X.view(input.size(0), -1)
        return f_enc_X, f_dec_X


class ONE_SIDED(nn.Module):
    def __init__(self):
        super(ONE_SIDED, self).__init__()

        main = nn.ReLU()
        self.main = main

    def forward(self, input):
        output = self.main(-input)
        output = -output.mean()
        return output


class SpnDecoder(nn.Module):
    def __init__(
        self,
        num_channels: int,
        height: int,
        width: int,
        D: int,
        K: int,
        R: int,
        hard: bool,
        min_sigma: float,
        max_sigma: float,
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
            min_sigma: The minimum sigma for the Gaussian distributions.
            max_sigma: The maximum sigma for the Gaussian distributions.
            dist: The distribution to use.
        """
        super().__init__()
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

        self.spn = make_einet(args)

        self.args = args
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

    def sample(self, num_samples: int):
        """
        Samples from the generative SPN.

        Args:
            num_samples: The number of samples to draw.

        Returns:
            A batch of samples.
        """
        with torch.no_grad():
            return self.spn.sample(num_samples, mpe_at_leaves=True).view(
                num_samples, self.num_channels, self.height, self.width
            )

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
        samples = samples.view(num_samples, self.num_channels, self.height, self.width)
        return samples


class MMD(AbstractLitGenerator):
    def __init__(self, args):
        super().__init__(args=args, name="mmd")

        # sigma for MMD
        base = 1.0
        sigma_list = [1, 2, 4, 8]
        self.sigma_list = [sigma / base for sigma in sigma_list]

        self.automatic_optimization = False
        self.gen_iterations = 0

        self.lambda_MMD = 1.0
        self.lambda_AE_X = 8.0
        self.lambda_AE_Y = 8.0
        self.lambda_rg = 16.0

        self.fixed_noise = nn.Parameter(
            torch.Tensor(25, args.nz, 1, 1).normal_(0, 1).float(), requires_grad=False
        )

        # construct encoder/decoder modules
        hidden_dim = args.nz
        image_shape = self.image_shape
        if args.model == "spn":
            G_decoder = SpnDecoder(
                num_channels=image_shape.channels,
                width=image_shape.width,
                height=image_shape.height,
                D=args.spn_D,
                R=args.spn_R,
                K=args.spn_K,
                hard=args.spn_hard,
                min_sigma=args.spn_min_sigma,
                max_sigma=args.spn_max_sigma,
                dist=args.dist,
                args=args,
            )
        elif args.model == "gan":
            G_decoder = base_module.Decoder(
                image_shape.height, image_shape.channels, k=args.nz, ngf=64
            )
        else:
            raise ValueError(f"Unknown model: {args.model}")

        # Keep decoder as in original MMD paper
        D_encoder = base_module.Encoder(
            image_shape.width, image_shape.channels, k=hidden_dim, ndf=64
        )
        D_decoder = base_module.Decoder(
            image_shape.width, image_shape.channels, k=hidden_dim, ngf=64
        )

        self.generator = NetG(G_decoder)
        self.discriminator = NetD(D_encoder, D_decoder)
        self.one_sided = ONE_SIDED()

        self.iteration_g = 0
        self.iteration_d = 0

        self.optimize_generator = False

    def configure_optimizers(self):
        # Generator optimizer
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr_g)
        lr_scheduler_G = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_G,
            milestones=[int(0.7 * args.epochs), int(0.9 * args.epochs)],
            gamma=0.1,
        )

        # Discriminator optimizer
        optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.args.lr_d
        )
        lr_scheduler_D = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_D,
            milestones=[int(0.7 * args.epochs), int(0.9 * args.epochs)],
            gamma=0.1,
        )
        return [optimizer_G, optimizer_D], [lr_scheduler_G, lr_scheduler_D]

    def training_step(self, train_batch, batch_idx):
        Diters, Giters = self.get_D_G_iterations()

        print(f"Diters: {Diters}, Giters: {Giters}, iteration_g: {self.iteration_g}, iteration_d: {self.iteration_d}")

        x, _ = train_batch
        if self.optimize_generator:
            self.step_generator(Giters, x)
        else:
            self.step_discriminator(Diters, x)

    def get_D_G_iterations(self):
        if self.gen_iterations < 25 or self.gen_iterations % 500 == 0:
            Diters = 100
            Giters = 1
        else:
            Diters = 5
            Giters = 1
        return Diters, Giters

    def step_generator(self, Giters, x):
        optimizer_G, _ = self.optimizers()
        for p in self.discriminator.parameters():
            p.requires_grad = False

        errG, f_enc_X, f_enc_Y, mmd2_G, one_side_errG = self.loss_generator(x)

        self.log("Loss/train_err_G", errG, prog_bar=True)

        optimizer_G.zero_grad()
        self.manual_backward(loss=errG)
        optimizer_G.step()

        self.gen_iterations += 1
        self.iteration_g += 1
        if self.iteration_g % Giters == 0:
            self.optimize_generator = False
            self.iteration_g = 0


    def loss_generator(self, x):
        f_enc_X, f_dec_X = self.discriminator(x)
        noise = torch.FloatTensor(x.size(0), args.nz, 1, 1).normal_(0, 1)
        noise = Variable(noise).to(self.device)
        y = self.generator(noise)
        if args.dist == Dist.BINOMIAL and args.model == "spn":
            y = y / 255

        f_enc_Y, f_dec_Y = self.discriminator(y)
        self.last_f_dec_X = f_dec_X.detach()

        # compute biased MMD2 and use ReLU to prevent negative value
        mmd2_G = mix_rbf_mmd2(f_enc_X, f_enc_Y, self.sigma_list)
        mmd2_G = F.relu(mmd2_G)

        # compute rank hinge loss
        one_side_errG = self.one_sided(f_enc_X.mean(0) - f_enc_Y.mean(0))
        errG = torch.sqrt(mmd2_G) + self.lambda_rg * one_side_errG
        return errG, f_enc_X, f_enc_Y, mmd2_G, one_side_errG

    def step_discriminator(self, Diters, x):
        # ---------------------------
        #        Optimize over NetD
        # ---------------------------
        for p in self.discriminator.parameters():
            p.requires_grad = True

        _, optimizer_D = self.optimizers()
        # clamp parameters of NetD encoder to a cube
        # do not clamp paramters of NetD decoder!!!
        for p in self.discriminator.encoder.parameters():
            p.data.clamp_(-0.01, 0.01)
        errD = self.loss_discriminator(x)
        optimizer_D.zero_grad()
        self.manual_backward(loss=errD)
        optimizer_D.step()
        self.log("Loss/train_err_D", errD, prog_bar=True)
        self.iteration_d += 1
        if self.iteration_d % Diters == 0:
            self.optimize_generator = True
            self.iteration_d = 0

    def loss_discriminator(self, x):
        f_enc_X_D, f_dec_X_D = self.discriminator(x)
        batch_size = x.size(0)
        noise = (
            torch.FloatTensor(batch_size, args.nz, 1, 1).normal_(0, 1).to(self.device)
        )
        with torch.no_grad():
            y = Variable(self.generator(noise).data)

            if args.dist == Dist.BINOMIAL and args.model == "spn":
                y = y / 255
        f_enc_Y_D, f_dec_Y_D = self.discriminator(y)
        # compute biased MMD2 and use ReLU to prevent negative value
        mmd2_D = mix_rbf_mmd2(f_enc_X_D, f_enc_Y_D, self.sigma_list)
        mmd2_D = F.relu(mmd2_D)
        # self.log("mmd2_D", mmd2_D)
        # compute rank hinge loss
        # print('f_enc_X_D:', f_enc_X_D.size())
        # print('f_enc_Y_D:', f_enc_Y_D.size())
        one_side_errD = self.one_sided(f_enc_X_D.mean(0) - f_enc_Y_D.mean(0))
        # self.log("one_side_err_D", one_side_errD)
        # compute L2-loss of AE
        L2_AE_X_D = util.match(x.view(batch_size, -1), f_dec_X_D, "L2")
        L2_AE_Y_D = util.match(y.view(batch_size, -1), f_dec_Y_D, "L2")
        # self.log("L2_AE_X_D", L2_AE_X_D)
        # self.log("L2_AE_Y_D", L2_AE_Y_D)
        errD = (
            torch.sqrt(mmd2_D)
            + self.lambda_rg * one_side_errD
            - self.lambda_AE_X * L2_AE_X_D
            - self.lambda_AE_Y * L2_AE_Y_D
        )
        errD = -1 * errD
        return errD

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        g_loss = self.loss_generator(x=x)
        err_G = g_loss[0]
        self.log("Loss/val_err_G", err_G)

        err_D = self.loss_discriminator(x=x)
        self.log("Loss/val_err_D", err_D)

    def get_spn(self):
        if self.args.model == "spn":
            return self.generator.decoder.spn
        else:
            return None

    def generate_samples(self, num_samples: int):
        """Generate samples from the generator."""
        if args.model == "spn":
            samples = self.generator.decoder.sample(num_samples)
            samples = samples / 255
        else:
            samples = self.generator(self.fixed_noise)
            samples = samples.mul(0.5).add(0.5)

        return samples.view(num_samples, *self.image_shape)

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        with torch.no_grad():
            last_f_dec_X = self.last_f_dec_X.data.mul(0.5).add(0.5)
            last_f_dec_X = last_f_dec_X.view(last_f_dec_X.size(0), *self.image_shape)
            grid_f_dec_X_D = torchvision.utils.make_grid(
                last_f_dec_X[:25], nrow=5, pad_value=0.0, normalize=False
            )
            self.logger.experiment.add_image(
                "reconstructions", grid_f_dec_X_D, self.current_epoch
            )

if __name__ == "__main__":
    args = get_args()
    results_dir, args = setup_experiment_lit(
        name="mmd",
        args=args,
        remove_if_exists=True
    )

    # model
    normalize = args.model == "gan" or (
        args.model == "spn" and args.dist == Dist.NORMAL
    )

    # Load or create model
    if args.load_and_eval:
        model = load_from_checkpoint(results_dir, load_fn=MMD.load_from_checkpoint, args=args)
    else:
        model = MMD(args)

    # hparams
    hparams = {
        "model": args.model,
        "lr-g": args.lr_g,
        "lr-d": args.lr_d,
        "nz": args.nz,
    }

    train_test.main(
        args=args,
        model=model,
        results_dir=results_dir,
        normalize=normalize,
        hparams=hparams,
    )