"""Source: https://github.com/pytorch/examples/blob/main/vae/main.py"""
from __future__ import print_function
import os
import pprint
from argparse import Namespace

import torch
import torch.utils.data
import torchvision
from torch import nn, optim
from torch.nn import functional as F

from simple_einet.einet import EinetConfig, Einet
from experiments.args import get_general_experiment_parser
from experiments.data import Shape, Dist, get_distribution, build_dataloader
from experiments.evaluation import (
    save_samples,
    evaluate_reconstruction_error,
    evaluate_fid_kid_scores,
    eval_fid_kid_celeba,
    evaluate_auto_encoding_reconstruction_error,
)
from experiments.utils import setup_experiment, count_params, anneal_tau


def get_args():
    parser = get_general_experiment_parser()
    parser.add_argument(
        "--latent-dim", type=int, default=20, help="size of latent space"
    )
    parser.add_argument(
        "--model", type=str, choices=["spn", "vae"], default="vae", help="model to use"
    )
    parser.add_argument(
        "--leaf-temperature", type=float, default=1.0, help="temperature for leaf nodes"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    return parser.parse_args()


class VAE(nn.Module):
    def __init__(self, image_shape: Shape, latent_dim: int):
        super(VAE, self).__init__()
        self.image_shape = image_shape
        self.fc1 = nn.Linear(image_shape.num_pixels * image_shape.channels, 500)
        self.fc21 = nn.Linear(500, latent_dim)
        self.fc22 = nn.Linear(500, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 500)
        self.fc4 = nn.Linear(500, image_shape.num_pixels * image_shape.channels)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3)).view(-1, image_shape.channels, image_shape.num_pixels)

    def forward(self, x):
        mu, logvar = self.encode(
            x.view(-1, image_shape.num_pixels * image_shape.channels)
        )
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, x, return_recon=False):
        x_recon, mu, logvar = self(x)
        loss = vae_loss_function(x_recon, x, mu, logvar)
        if return_recon:
            return loss, x_recon
        else:
            return loss


# Reconstruction + KL divergence losses summed over all elements and batch
def vae_loss_function(recon_x, x, mu, logvar, reduction="sum"):
    BCE = F.binary_cross_entropy(
        recon_x.view(-1, image_shape.num_pixels * image_shape.channels),
        x.view(-1, image_shape.num_pixels * image_shape.channels),
        reduction=reduction,
    )

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    if reduction == "sum":
        _red = torch.sum
    elif reduction == "mean":
        _red = torch.mean
    else:
        raise ValueError("reduction must be sum or mean")

    # KL divergence
    KLD = -0.5 * _red(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


class SPNAE(nn.Module):
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
        latent_dim: int,
        dist: Dist,
        args: Namespace,
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
            min_sigma: The minimum sigma for the leaves.
            max_sigma: The maximum sigma for the leaves.
            latent_dim: The dimension of the latent space.
            dist: The distribution to use for the data.
        """
        super().__init__()
        self.args = args
        self.D = D
        self.K = K
        self.R = R
        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.hard = hard
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.latent_dim = latent_dim
        self.dist = dist

        self.epoch = 0

        self.model = self._make_spn()

        self.x_scopes = torch.arange(0, self.width * self.height)
        self.z_scopes = torch.arange(
            self.width * self.height, self.width * self.height + self.latent_dim
        )

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

    def _make_spn(self) -> Einet:
        leaf_kwargs, leaf_type = get_distribution(
            self.dist, self.min_sigma, self.max_sigma
        )

        config = EinetConfig(
            num_features=self.height * self.width + self.latent_dim,
            num_channels=self.num_channels,
            depth=self.D,
            num_sums=self.K,
            num_leaves=self.K,
            num_repetitions=self.R,
            num_classes=1,
            leaf_type=leaf_type,
            leaf_kwargs=leaf_kwargs,
            dropout=0.0,
        )
        return Einet(config)

    def encode(self, x: torch.Tensor, mpe=False) -> torch.Tensor:

        # Sample z ~ p( z | x )
        # Construct evidence
        evidence = torch.cat(
            [
                x,
                self.z_scopes.new_zeros(
                    x.shape[0], x.shape[1], self.latent_dim, device=x.device
                ),
            ],
            dim=2,
        ).to(x.device)

        if self.dist == Dist.BINOMIAL:
            # Round the to the next integer
            evidence = torch.floor(evidence)
            evidence = torch.clip(evidence, min=0, max=255)

        x_and_z = self.model.sample_differentiable(
            evidence=evidence,
            marginalized_scopes=self.z_scopes,
            temperature_leaves=args.leaf_temperature,  # low temp to be closer to MPE
            hard=self.hard or mpe,
            tau=1.0 if mpe else self.tau
        )
        z = x_and_z[:, :, self.z_scopes]
        return z

    def decode(self, z: torch.Tensor, mpe=False) -> torch.Tensor:
        # Sample x' ~ p( x | z )
        # Construct evidence
        evidence = torch.cat(
            [
                self.x_scopes.new_zeros(
                    z.shape[0], z.shape[1], self.height * self.width, device=z.device
                ),
                z,
            ],
            dim=2,
        ).to(z.device)

        if self.dist == Dist.BINOMIAL:
            # Round the to the next integer
            evidence = torch.floor(evidence)
            evidence = torch.clip(evidence, min=0, max=255)

        x_and_z = self.model.sample_differentiable(
            evidence=evidence,
            marginalized_scopes=self.x_scopes,
            temperature_leaves=args.leaf_temperature,  # low temp to be closer to MPE
            hard=self.hard or mpe,
            tau=1.0 if mpe else self.tau
        )
        x = x_and_z[:, :, self.x_scopes]
        return x

    def sample_x(self, num_samples):
        x_and_z = self.model.sample(
            num_samples=num_samples,
            marginalized_scopes=self.z_scopes,
            mpe_at_leaves=True,
        )

        x = x_and_z[:, :, self.x_scopes]

        if self.dist == Dist.BINOMIAL:
            # Round the to the next integer
            x = torch.floor(x)
            x = torch.clip(x, min=0, max=255)

        return x

    def sample_z(self, num_samples):
        x_and_z = self.model.sample(
            num_samples=num_samples,
            marginalized_scopes=self.x_scopes,
            mpe_at_leaves=True,
        )

        z = x_and_z[:, :, self.z_scopes]

        if self.dist == Dist.BINOMIAL:
            # Round the to the next integer
            z = torch.floor(z)
            z = torch.clip(z, min=0, max=255)

        return z

    def forward(self, x: torch.Tensor, mpe: bool = False) -> torch.Tensor:
        """
        Forward pass through the generative SPN.
        Args:
            x:

        Returns:

        """
        x = x.view(-1, image_shape.channels, image_shape.num_pixels)
        z = self.encode(x, mpe=mpe)
        x_rec = self.decode(z, mpe=mpe)

        return x_rec

    def loss(self, x: torch.Tensor, reduction="sum", return_recon=False):
        """
        Compute the loss for the generative SPN.
        Args:
            x:

        Returns:

        """
        x = x.view(-1, image_shape.channels, image_shape.num_pixels)

        # Reconstruction
        x_rec = self(x)

        # Compute data likelihood
        # z_dummy = self.z_scopes.new_zeros(
        #     x.shape[0], x.shape[1], args.latent_dim, device=x.device
        # )
        # x_and_z = torch.cat([x, z_dummy], dim=2)
        # ...

        # Reconstruction loss
        if reduction == "mean":
            loss_rec = torch.mean((x_rec - x) ** 2)
        elif reduction == "sum":
            loss_rec = torch.sum((x_rec - x) ** 2)
        else:
            raise ValueError(f"Reduction {reduction} not supported.")

        loss = loss_rec

        # If return_recon is True, return the reconstructed image as well
        if return_recon:
            return loss, x_rec
        else:
            return loss


def train(epoch):
    model.train()
    model.epoch = epoch - 1  # epochs are shifted by 1 here (starts at 1)
    train_loss = 0
    for batch_idx, (data, _) in enumerate(dataloader_train):

        # Stop early in debug mode
        if args.debug and batch_idx > 10:
            break

        if args.dist == Dist.BINOMIAL and args.model == "spn":
            data = data * 255.0

        data = data.to(device)
        optimizer.zero_grad()
        loss = model.loss(data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        writer.add_scalar(
            "Loss/train", loss.item(), epoch * len(dataloader_train) + batch_idx
        )
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(dataloader_train.dataset),
                    100.0 * batch_idx / len(dataloader_train),
                    loss.item() / len(data),
                )
            )

    print(
        "====> Epoch: {} Average loss: {:.4f}".format(
            epoch, train_loss / len(dataloader_train.dataset)
        )
    )


def test(epoch, dataloader, split):
    model.eval()
    test_loss = 0.0
    test_mse = 0.0
    mse_loss = nn.MSELoss(reduction="sum")
    with torch.no_grad():
        for i, (data, _) in enumerate(dataloader):
            data = data.to(device)
            loss, recon_batch = model.loss(data, return_recon=True)
            test_loss += loss.item()
            test_mse += mse_loss(recon_batch.view(-1, *image_shape), data).item()

            if args.debug:
                break

    test_loss /= len(dataloader.dataset)
    test_mse /= len(dataloader.dataset)
    print("====> {} set loss: {:.4f}".format(split, test_loss))
    print("====> {} set mse : {:.4f}".format(split, test_mse))

    writer.add_scalar(f"{split}_loss", test_loss, epoch)
    writer.add_scalar(f"{split}_mse", test_mse, epoch)


def pretrain_spn(
    spn: Einet,
    dataloader,
    args,
    device,
    learning_rate,
):
    """
    Pretrains the SPN using MLE by optimizing the negative data log likelihood.

    Args:
        spn_generator: The SPN generator.
        dataloader: The data on which we want to pretrain.
        args: The optimizer.
        device: The device on which to train the SPN.
    """
    # Optimizer
    optimizer = torch.optim.Adam(spn.parameters(), lr=learning_rate)

    # ==================
    # Pretrain generator
    # ==================
    print("Pretraining SPN...")

    marg_scope = model.z_scopes

    for epoch in range(args.epochs_pretrain):
        for i, (x, _) in enumerate(dataloader):
            # Move data to device
            x = x.to(device)

            if args.dist == Dist.BINOMIAL:
                x = x * 255.0

            x = x.view(-1, image_shape.channels, image_shape.num_pixels)

            x_and_z = torch.cat(
                [
                    x,
                    marg_scope.new_zeros(
                        x.shape[0], x.shape[1], args.latent_dim, device=x.device
                    ),
                ],
                dim=2,
            ).to(x.device)

            # Forward pass through spn_generator
            lls = spn(x_and_z, marginalization_mask=marg_scope)

            loss = -1 * torch.mean(lls)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % args.log_interval == 0:
                print(
                    f"Epoch: {epoch}/{args.epochs_pretrain} | Batch: {i:>3d}/{len(dataloader)} | "
                    f"Loss: {loss.item():.4f}"
                )

            if args.debug:
                break
        if args.debug:
            break
    print("Pretraining SPN finished!")
    print()


def main_train():
    global optimizer
    # ----------
    #  Pre-Training
    # ----------
    if args.model == "spn":
        pretrain_spn(
            spn=model.model,
            dataloader=dataloader_train,
            device=device,
            args=args,
            learning_rate=args.lr_pretrain,
        )
        gen_imgs = model.sample_x(25).view(
            -1, image_shape.channels, image_shape.height, image_shape.width
        )
        grid = torchvision.utils.make_grid(
            gen_imgs[:25], nrow=5, pad_value=0.0, normalize=True
        )
        writer.add_image("Image/pretrained", grid, 0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(0.7 * args.epochs), int(0.9 * args.epochs)],
        gamma=0.1,
        verbose=True,
    )
    for epoch in range(1, args.epochs + 1):

        # Stop early in debug mode
        if args.debug and epoch > 1:
            break

        train(epoch)
        # test(epoch, dataloader_val, "val")
        # test(epoch, dataloader_test, "test")
        with torch.no_grad():
            if args.model == "vae":
                z = torch.randn(64, args.latent_dim).to(device)
                samples = model.decode(z).cpu()
            elif args.model == "spn":
                z = model.sample_z(64)
                samples = model.decode(z).cpu()
            else:
                raise ValueError("Unknown model type")
            samples = samples.view(
                64, image_shape.channels, image_shape.height, image_shape.width
            )
            grid_samples = torchvision.utils.make_grid(
                samples[:25], nrow=5, pad_value=0.0, normalize=True
            )
            writer.add_image("Image/sample", grid_samples, epoch)

        lr_scheduler.step()
        rtpt.step()
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    args = get_args()
    (
        args,
        results_dir,
        writer,
        device,
        image_shape,
        rtpt,
    ) = setup_experiment(name="autoencoder", args=args)

    # Check if we skip training and only load and evaluate in this run
    load_and_eval = args.load_and_eval is not None
    do_train = not load_and_eval

    model_path = os.path.join(results_dir, "model.pth")

    # Get data
    normalize = False
    dataloader_train, dataloader_val, dataloader_test = build_dataloader(
        args, loop=False, normalize=normalize
    )

    if args.spn == "vae":
        model = VAE(image_shape=image_shape, latent_dim=args.latent_dim).to(device)
    elif args.spn == "spn":
        model = SPNAE(
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
            latent_dim=args.latent_dim,
            dist=args.dist,
            args=args
        ).to(device)
    else:
        raise ValueError("Unknown model type")

    print(f"Number of parameters: {count_params(model) / 1e6:.2f}M")

    if do_train:
        main_train()
    else:
        print("Loading model from:", model_path)
        state_dict_gen = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict_gen)

    writer.close()

    model.eval()

    def generate_samples(batch_size: int = 100):
        """
        Construct a generator from the trained model.
        Returns:
            Generator that yields samples.

        """
        while True:
            if args.model == "vae":
                z = torch.randn(batch_size, image_shape.channels, args.latent_dim).to(
                    device
                )
                samples = model.decode(z).cpu()
            elif args.model == "spn":
                samples = model.sample_x(num_samples=batch_size).cpu()
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
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "dataset": args.dataset,
    }

    # autoencoding_error = evaluate_auto_encoding_reconstruction_error(
    #     args, model, dataloader_test, device
    # )
    #
    # metrics.update({"ae_error": autoencoding_error})

    if args.spn == "spn":
        spn = model.model

        rec_error_test = evaluate_reconstruction_error(
            args,
            spn,
            dataloader_test,
            device,
            mask_fixed=model.z_scopes.tolist(),
        )

        rec_error_val = evaluate_reconstruction_error(
            args,
            spn,
            dataloader_val,
            device,
            mask_fixed=model.z_scopes.tolist(),
        )
        metrics.update(
            {
                "rec_error_test": rec_error_val,
                "rec_error_val": rec_error_test,
            }
        )

        hparams.update(
            {
                "spn_D": args.spn_D,
                "spn_K": args.spn_K,
                "spn_R": args.spn_R,
                "spn_tau": args.spn_tau,
                "spn_hard": args.spn_hard,
            }
        )

        # Collect hyperparameters

    writer.add_hparams(hparams, metrics)

    writer.close()

    print("#" * 80)
    print("FINAL METRICS")
    pprint.pprint(metrics)