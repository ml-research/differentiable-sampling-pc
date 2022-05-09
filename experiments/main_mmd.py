#!/usr/bin/env python
# encoding: utf-8
# Modified from: https://github.com/OctoberChang/MMD-GAN
import pprint
from collections import defaultdict

from experiments import utils

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.utils
from torch.autograd import Variable
import torch.nn.functional as F
import os
import timeit

from tqdm import tqdm

from simple_einet.distributions.binomial import Binomial
from experiments.args import get_general_experiment_parser
from experiments.evaluation import (
    save_samples,
    evaluate_reconstruction_error,
    eval_fid_kid_celeba,
)
from experiments.pretrain_spn import pretrain_spn
from simple_einet.distributions import RatNormal
from simple_einet.einet import EinetConfig, Einet
from experiments.mmd_gan import util
import numpy as np

from experiments.data import get_distribution, Dist, build_dataloader

from experiments.mmd_gan import base_module
from experiments.utils import (
    count_params,
    setup_experiment, anneal_tau,
)
from mmd_gan.mmd import mix_rbf_mmd2


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
        "--netG", default="", help="path to netG (to continue training)"
    )
    parser.add_argument(
        "--netD", default="", help="path to netD (to continue training)"
    )
    parser.add_argument(
        "--Diters", type=int, default=5, help="number of D iters per each G iter"
    )
    parser.add_argument(
        "--model", type=str, choices=["spn", "gan"], help="choose model ('spn' or 'gan'"
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
        tau: float,
        hard: bool,
        min_sigma: float,
        max_sigma: float,
        dist: Dist,
            args
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

        self.model = self._make_spn()


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

    def _make_spn(self):
        leaf_kwargs, leaf_type = get_distribution(
            self.dist, self.min_sigma, self.max_sigma
        )
        config = EinetConfig(
            num_features=self.height * self.width,
            num_channels=self.num_channels,
            depth=self.D,
            num_sums=self.K,
            num_leaves=self.K,
            num_repetitions=self.R,
            num_classes=1,
            dropout=0.0,
            leaf_kwargs=leaf_kwargs,
            leaf_type=leaf_type,
        )
        return Einet(config)

    def sample(self, num_samples: int):
        """
        Samples from the generative SPN.

        Args:
            num_samples: The number of samples to draw.

        Returns:
            A batch of samples.
        """
        with torch.no_grad():
            return self.model.sample(num_samples, mpe_at_leaves=True).view(
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
        samples = self.model.sample_differentiable(
            num_samples=num_samples, hard=self.hard, tau=self.tau
        )
        samples = samples.view(num_samples, self.num_channels, self.height, self.width)
        return samples


def main_train():
    global args
    # setup optimizer
    # optimizer_G = torch.optim.RMSprop(net_G.parameters(), lr=args.lr_g)
    # optimizer_D = torch.optim.RMSprop(net_D.parameters(), lr=args.lr_d)
    optimizer_G = torch.optim.Adam(net_G.parameters(), lr=args.lr_g)
    optimizer_D = torch.optim.Adam(net_D.parameters(), lr=args.lr_d)
    lambda_MMD = 1.0
    lambda_AE_X = 8.0
    lambda_AE_Y = 8.0
    lambda_rg = 16.0
    if args.debug:
        print("Debug mode")
        Diters = 1
        Giters = 1
        args.epochs = 1
    # ----------
    #  Pre-Training
    # ----------
    if args.model == "spn":
        pretrain_spn(
            spn=net_G.decoder.model,
            dataloader=dataloader_train,
            device=device,
            args=args,
            learning_rate=args.lr_g,
        )
        gen_imgs = net_G.decoder.sample(25).view(-1, *image_shape)
        grid = torchvision.utils.make_grid(
            gen_imgs[:25], nrow=5, pad_value=0.0, normalize=True
        )
        writer.add_image("Image/pretrained", grid, 0)
    time = timeit.default_timer()
    gen_iterations = 0

    for t in range(args.epochs):
        net_G.decoder.epoch = t
        data_iter = iter(dataloader_train)
        i = 0
        while i < len(dataloader_train):
            # ---------------------------
            #        Optimize over NetD
            # ---------------------------
            for p in net_D.parameters():
                p.requires_grad = True

            if gen_iterations < 25 or gen_iterations % 500 == 0:
                Diters = 100
                Giters = 1
            else:
                Diters = 5
                Giters = 1

            metrics = defaultdict(list)

            for j in range(Diters):
                if i == len(dataloader_train):
                    break

                # clamp parameters of NetD encoder to a cube
                # do not clamp paramters of NetD decoder!!!
                for p in net_D.encoder.parameters():
                    p.data.clamp_(-0.01, 0.01)

                data = data_iter.next()
                i += 1
                optimizer_D.zero_grad()

                x, _ = data

                x = x.to(device)
                batch_size = x.size(0)

                f_enc_X_D, f_dec_X_D = net_D(x)

                noise = (
                    torch.FloatTensor(batch_size, args.nz, 1, 1)
                    .normal_(0, 1)
                    .to(device)
                )
                with torch.no_grad():
                    y = Variable(net_G(noise).data)

                    if args.dist == Dist.BINOMIAL and args.model == "spn":
                        y = y / 255

                f_enc_Y_D, f_dec_Y_D = net_D(y)

                # compute biased MMD2 and use ReLU to prevent negative value
                mmd2_D = mix_rbf_mmd2(f_enc_X_D, f_enc_Y_D, sigma_list)
                mmd2_D = F.relu(mmd2_D)

                metrics["mmd2_D"].append(mmd2_D.item())

                # compute rank hinge loss
                # print('f_enc_X_D:', f_enc_X_D.size())
                # print('f_enc_Y_D:', f_enc_Y_D.size())
                one_side_errD = one_sided(f_enc_X_D.mean(0) - f_enc_Y_D.mean(0))
                metrics["one_side_errD"].append(one_side_errD.item())

                # compute L2-loss of AE
                L2_AE_X_D = util.match(x.view(batch_size, -1), f_dec_X_D, "L2")
                L2_AE_Y_D = util.match(y.view(batch_size, -1), f_dec_Y_D, "L2")
                metrics["L2_AE_X_D"].append(L2_AE_X_D.item())
                metrics["L2_AE_Y_D"].append(L2_AE_Y_D.item())

                errD = (
                    torch.sqrt(mmd2_D)
                    + lambda_rg * one_side_errD
                    - lambda_AE_X * L2_AE_X_D
                    - lambda_AE_Y * L2_AE_Y_D
                )
                # errD.backward(mone)  # old pytorch code
                errD = -1 * errD
                errD.backward()
                optimizer_D.step()

                metrics["errD"].append(errD.item())

                if args.debug:
                    break

            # ---------------------------
            #        Optimize over NetG
            # ---------------------------
            for p in net_D.parameters():
                p.requires_grad = False

            for j in range(Giters):
                if i == len(dataloader_train):
                    break

                data = data_iter.next()
                i += 1
                optimizer_G.zero_grad()

                x, _ = data
                # HACK: fix this
                # if args.dist == Dist.BINOMIAL and args.model == "spn":
                #     x = x * 255
                # ENDHACK
                x = x.to(device)
                batch_size = x.size(0)

                f_enc_X, f_dec_X = net_D(x)

                noise = torch.FloatTensor(batch_size, args.nz, 1, 1).normal_(0, 1)
                noise = Variable(noise).to(device)
                y = net_G(noise)

                if args.dist == Dist.BINOMIAL and args.model == "spn":
                    y = y / 255

                f_enc_Y, f_dec_Y = net_D(y)

                # compute biased MMD2 and use ReLU to prevent negative value
                mmd2_G = mix_rbf_mmd2(f_enc_X, f_enc_Y, sigma_list)
                mmd2_G = F.relu(mmd2_G)

                metrics["mmd2_G"].append(mmd2_G.item())

                # compute rank hinge loss
                one_side_errG = one_sided(f_enc_X.mean(0) - f_enc_Y.mean(0))
                metrics["one_side_errG"].append(one_side_errG.item())

                errG = torch.sqrt(mmd2_G) + lambda_rg * one_side_errG
                errG.backward()
                optimizer_G.step()

                metrics["errG"].append(errG.item())

                metrics["f_enc_X"].append(f_enc_X.mean().item())
                metrics["f_enc_Y"].append(f_enc_Y.mean().item())

                gen_iterations += 1

                if args.debug:
                    break

            # Reduce collected values to their mean
            metrics = {k: np.mean(v) for k, v in metrics.items()}

            # Print stuff
            run_time = (timeit.default_timer() - time) / 60.0
            if i % 100 == 0:
                print(
                    "[%3d/%3d][%3d/%3d] [%5d] (%.2f m) %s |gD| %.4f |gG| %.4f"
                    % (
                        t,
                        args.epochs,
                        i,
                        len(dataloader_train),
                        gen_iterations,
                        run_time,
                        " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]),
                        base_module.grad_norm(net_D),
                        base_module.grad_norm(net_G),
                    )
                )

            # Write to tensorboard
            for key, value in metrics.items():
                writer.add_scalar(f"Loss/{key}", value, gen_iterations)

            if gen_iterations % 100 == 0:
                with torch.no_grad():
                    if args.model == "spn":
                        samples = net_G.decoder.sample(25)
                        if args.dist == Dist.BINOMIAL:
                            samples = samples / 255
                    else:
                        samples = net_G(fixed_noise)
                        samples = samples.mul(0.5).add(0.5)
                    f_dec_X_D = f_dec_X_D.view(
                        f_dec_X_D.size(0),
                        image_shape.channels,
                        image_shape.height,
                        image_shape.width,
                    )
                    f_dec_X_D.data = f_dec_X_D.data.mul(0.5).add(0.5)
                    grid_samples = torchvision.utils.make_grid(
                        samples[:25], nrow=5, pad_value=0.0, normalize=True
                    )
                    grid_f_dec_X_D = torchvision.utils.make_grid(
                        f_dec_X_D[:25], nrow=5, pad_value=0.0, normalize=True
                    )
                    writer.add_image("Image/samples", grid_samples, gen_iterations)
                    writer.add_image(
                        "Image/decoded_samples", grid_f_dec_X_D, gen_iterations
                    )

            if args.debug:
                break

        rtpt.step()

        if args.debug:
            break

    # Save models
    torch.save(net_G.state_dict(), model_gen_path)
    torch.save(net_D.state_dict(), model_disc_path)


if __name__ == "__main__":
    args = get_args()
    (args, results_dir, writer, device, image_shape, rtpt) = setup_experiment(
        name="mmd", args=args
    )

    # Check if we skip training and only load and evaluate in this run
    load_and_eval = args.load_and_eval is not None
    do_train = not load_and_eval

    # Model paths
    model_gen_path = os.path.join(results_dir, "netG.pth")
    model_disc_path = os.path.join(results_dir, "netD.pth")

    # Get data
    normalize = args.model == "gan" or (
            args.model == "spn" and args.dist == Dist.NORMAL
    )
    dataloader_train, dataloader_val, dataloader_test = build_dataloader(
        args, loop=False, normalize=normalize
    )

    # construct encoder/decoder modules
    hidden_dim = args.nz
    if args.model == "spn":
        G_decoder = SpnDecoder(
            num_channels=image_shape.channels,
            width=image_shape.width,
            height=image_shape.height,
            D=args.spn_D,
            R=args.spn_R,
            K=args.spn_K,
            tau=args.spn_tau,
            hard=args.spn_hard,
            min_sigma=args.spn_min_sigma,
            max_sigma=args.spn_max_sigma,
            dist=args.dist,
            args=args
        )
    else:
        G_decoder = base_module.Decoder(
            image_shape.height, image_shape.channels, k=args.nz, ngf=64
        )

    # Keep decoder as in original MMD paper
    D_encoder = base_module.Encoder(
        image_shape.width, image_shape.channels, k=hidden_dim, ndf=64
    )
    D_decoder = base_module.Decoder(
        image_shape.width, image_shape.channels, k=hidden_dim, ngf=64
    )

    net_G = NetG(G_decoder)
    net_D = NetD(D_encoder, D_decoder)
    one_sided = ONE_SIDED()
    print("netG:", net_G)
    print("netD:", net_D)
    print("oneSide:", one_sided)

    print(f"Number of parameters netG        :{count_params(net_G) / 1e6:.3f} M")
    print(f"Number of parameters netD        :{count_params(net_D) / 1e6:.3f} M")
    print(
        f"Number of parameters netD.encoder:{count_params(net_D.encoder) / 1e6:.3f} M"
    )
    print(
        f"Number of parameters netD.decoder:{count_params(net_D.decoder) / 1e6:.3f} M"
    )

    if args.model == "gan":
        net_G.apply(base_module.weights_init)

    net_D.apply(base_module.weights_init)
    one_sided.apply(base_module.weights_init)

    # sigma for MMD
    base = 1.0
    sigma_list = [1, 2, 4, 8, 16]
    sigma_list = [sigma / base for sigma in sigma_list]

    # put variable into cuda device
    fixed_noise = torch.Tensor(64, args.nz, 1, 1).normal_(0, 1).float().to(device)
    net_G.to(device)
    net_D.to(device)
    one_sided.to(device)

    if do_train:
        main_train()
    else:
        print("Loading generator model from:", model_gen_path)
        print("Loading discriminator model from:", model_disc_path)
        state_dict_gen = torch.load(model_gen_path, map_location=torch.device("cpu"))
        state_dict_disc = torch.load(model_disc_path, map_location=torch.device("cpu"))
        net_G.load_state_dict(state_dict_gen)
        net_D.load_state_dict(state_dict_disc)

    net_G.eval()

    def generate_samples(batch_size: int = 100):
        """
        Construct a generator from the trained model.
        Returns:
            Generator that yields samples.

        """
        # while True:
        z = torch.Tensor(batch_size, args.nz, 1, 1).normal_(0, 1).float().to(device)
        samples = net_G.decoder(z)
        return samples
            # yield samples

    # if "celeba" in args.dataset.lower():
    #     metrics = eval_fid_kid_celeba(
    #         args=args,
    #         results_dir=results_dir,
    #         generate_samples=generate_samples,
    #         device=device,
    #     )
    # else:
    #     metrics = {}
    metrics = {}

    hparams = {
        "lr_g": args.lr_g,
        "lr_d": args.lr_d,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "dataset": args.dataset,
    }

    # if args.spn == "spn":
    #     spn = net_G.decoder.model
    #     rec_error_test = evaluate_reconstruction_error(
    #         args,
    #         spn,
    #         dataloader_test,
    #         device,
    #     )
    #
    #     rec_error_val = evaluate_reconstruction_error(
    #         args,
    #         spn,
    #         dataloader_val,
    #         device,
    #     )
    #     metrics.update(
    #         {"rec_error_test": rec_error_val, "rec_error_val": rec_error_test}
    #     )
    #
    #     # Collect hyperparameters
    #     hparams.update(
    #         {
    #             "spn_D": args.spn_D,
    #             "spn_K": args.spn_K,
    #             "spn_R": args.spn_R,
    #             "spn_tau": args.spn_tau,
    #             "spn_hard": args.spn_hard,
    #         }
    #     )

    writer.add_hparams(hparams, metrics)
    writer.close()


    # Save some samples
    samples_dir = os.path.join(results_dir, "samples_9")
    os.makedirs(exist_ok=True, name=samples_dir)
    utils.save_samples(samples_dir=samples_dir, num_samples=9, nrow=3, generate_samples=generate_samples)

    samples_dir = os.path.join(results_dir, "samples_25")
    os.makedirs(exist_ok=True, name=samples_dir)
    utils.save_samples(samples_dir=samples_dir, num_samples=25, nrow=5,
                       generate_samples=generate_samples)

    print("#" * 80)
    print("FINAL METRICS")
    pprint.pprint(metrics)