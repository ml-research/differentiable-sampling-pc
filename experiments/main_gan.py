import os
import pprint

import numpy as np
import torch
import torch.nn as nn
import torchvision.utils

from experiments.args import get_general_experiment_parser
from experiments.data import Dist, get_distribution, build_dataloader
from experiments.evaluation import (
    save_samples,
    evaluate_reconstruction_error,
    evaluate_fid_kid_scores,
    eval_fid_kid_celeba,
)
from experiments.utils import (
    setup_experiment, anneal_tau,
)
from experiments.pretrain_spn import pretrain_spn
from simple_einet.distributions.binomial import Binomial
from simple_einet.distributions.normal import RatNormal
from simple_einet.einet import EinetConfig, Einet


class GanGenerator(nn.Module):
    def __init__(self):
        super(GanGenerator, self).__init__()

        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(args.latent_dim, 128),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(image_shape.num_pixels * image_shape.channels)),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *image_shape)
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

        self.model = self._make_spn()


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
            leaf_kwargs=leaf_kwargs,
            leaf_type=leaf_type,
            dropout=0.0,
        )
        return Einet(config)

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
    def __init__(self):
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
        "--model-generator",
        type=str,
        choices=["spn", "gan"],
        help="model type (gan " "or spn)",
    )
    parser.add_argument(
        "-D",
        "--model-discriminator",
        type=str,
        choices=["spn", "gan"],
        help="model type (" "gan or spn)",
    )
    parser.add_argument(
        "--disc-step", type=int, default=10, help="discriminator step interval"
    )
    return parser.parse_args()


def main_train():
    # Optimizers
    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=args.lr_d#, betas=(args.b1, args.b2)
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=args.lr_g#, betas=(args.b1, args.b2)
    )
    # scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=50, gamma=0.1)
    # scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=50, gamma=0.1)
    # ----------
    #  Pre-Training
    # ----------
    if args.model_generator == "spn":
        pretrain_spn(
            spn=generator.model,
            dataloader=dataloader_train,
            args=args,
            device=device,
            learning_rate=args.lr_g,
        )
        gen_imgs = generator.model.sample(25, mpe_at_leaves=True).view(-1, *image_shape)
        grid = torchvision.utils.make_grid(
            gen_imgs[:25], nrow=5, pad_value=0.0, normalize=True
        )
        writer.add_image("Image/pretrained", grid, 0)
    # ----------
    #  Training
    # ----------
    for epoch in range(args.epochs):
        generator.epoch = epoch
        if args.debug:
            break
        for i, (imgs, _) in enumerate(dataloader_train):
            # if args.model_generator == "spn" and args.dist == Dist.BINOMIAL:
            #     imgs *= 255

            # Adversarial ground truths
            valid = torch.ones(imgs.size(0), 1, device=device)
            fake = torch.zeros(imgs.size(0), 1, device=device)

            imgs = imgs.to(device)

            # Configure input
            real_imgs = imgs

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = torch.randn(
                imgs.size(0), args.latent_dim, device=device, requires_grad=True
            )
            # # Sample noise as generator input
            # z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)

            if args.model_generator == "spn" and args.dist == Dist.BINOMIAL:
                gen_imgs /= 255

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            batches_done = epoch * len(dataloader_train) + i
            writer.add_scalar("g_loss", g_loss.item(), global_step=batches_done)

            # Compute LL
            # if args.model_generator == "spn":
            #     lls = generator.model(imgs)
            #     g_loss += (-1) * lls.mean() * 0.001

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            if i % args.disc_step == 0:
                # Only perform discriminator step every args.disc_step steps

                optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(discriminator(real_imgs), valid)
                fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()

                writer.add_scalar("d_loss", d_loss.item(), global_step=batches_done)

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (
                    epoch,
                    args.epochs,
                    i,
                    len(dataloader_train),
                    d_loss.item(),
                    g_loss.item(),
                )
            )

            # Save samples
            batches_done = epoch * len(dataloader_train) + i
            if batches_done % args.sample_interval == 0:
                if args.model_generator == "spn":
                    samples = generator.model.sample(
                        num_samples=25, mpe_at_leaves=True
                    ).view(25, *image_shape)
                else:
                    samples = gen_imgs
                grid = torchvision.utils.make_grid(
                    samples.data[:25], nrow=5, pad_value=0.0, normalize=True
                )
                writer.add_image("Image/samples", grid, batches_done)

            if args.debug:
                break

        # scheduler_G.step()
        # scheduler_D.step()
        rtpt.step()

        if args.debug:
            break

    # Save models
    torch.save(generator.state_dict(), model_generator_path)
    torch.save(discriminator.state_dict(), model_discriminator_path)


if __name__ == "__main__":
    args = get_args()
    (
        args,
        results_dir,
        writer,
        device,
        image_shape,
        rtpt,
    ) = setup_experiment(name="gan", args=args)

    # Check if we skip training and only load and evaluate in this run
    load_and_eval = args.load_and_eval is not None
    do_train = not load_and_eval

    # Paths for gen/disc saving
    model_generator_path = os.path.join(results_dir, "generator.pth")
    model_discriminator_path = os.path.join(results_dir, "discriminator.pth")

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    if args.model_generator == "gan":
        generator = GanGenerator()
    elif args.model_generator == "spn":
        generator = SpnGenerator(
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
        raise Exception()

    if args.model_discriminator == "gan":
        discriminator = GanDiscriminator()
    elif args.model_discriminator == "spn":
        discriminator = SpnDiscriminator(
            num_channels=image_shape.channels,
            width=image_shape.width,
            height=image_shape.height,
            D=args.spn_D,
            R=args.spn_R,
            K=args.spn_K,
            min_sigma=args.spn_min_sigma,
            max_sigma=args.spn_max_sigma,
            dist=args.dist,
        )
    else:
        raise NotImplementedError()

    # Move to correct device
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    adversarial_loss = adversarial_loss.to(device)

    # Get data
    normalize = args.model_generator == "gan" or (
        args.model_generator == "spn" and args.dist == Dist.NORMAL
    )
    dataloader_train, dataloader_val, dataloader_test = build_dataloader(
        args, loop=False, normalize=normalize
    )

    if do_train:
        main_train()
    else:
        print("Loading generator model from:", model_generator_path)
        print("Loading discriminator model from:", model_discriminator_path)
        state_dict_gen = torch.load(
            model_generator_path, map_location=torch.device("cpu")
        )
        state_dict_disc = torch.load(
            model_discriminator_path, map_location=torch.device("cpu")
        )
        generator.load_state_dict(state_dict_gen)
        discriminator.load_state_dict(state_dict_disc)

    generator.eval()

    def generate_samples(batch_size: int = 100):
        """
        Construct a generator from the trained model.
        Returns:
            Generator that yields samples.

        """
        while True:
            # Sample noise as generator input
            z = torch.randn(batch_size, args.latent_dim).to(device)

            # Generate a batch of images
            samples = generator(z).view(-1, *image_shape)
            yield samples

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

    if args.model_generator == "spn":
        spn = generator.model
        rec_error_test = evaluate_reconstruction_error(
            args,
            spn,
            dataloader_test,
            device,
        )

        rec_error_val = evaluate_reconstruction_error(
            args,
            spn,
            dataloader_val,
            device,
        )
        metrics.update(
            {
                "rec_error_test": rec_error_val,
                "rec_error_val": rec_error_test,
            }
        )

        # Collect hyperparameters
        hparams.update(
            {
                "spn_D": args.spn_D,
                "spn_K": args.spn_K,
                "spn_R": args.spn_R,
                "spn_tau": args.spn_tau,
                "spn_hard": args.spn_hard,
            }
        )

    writer.add_hparams(hparams, metrics)

    print("#" * 80)
    print("FINAL METRICS")
    pprint.pprint(metrics)

    writer.close()