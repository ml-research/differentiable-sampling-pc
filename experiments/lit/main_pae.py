"""Source: https://github.com/pytorch/examples/blob/main/vae/main.py"""
#!/usr/bin/env python

from argparse import Namespace

import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data
from torch import nn
from torch.nn import functional as F

from experiments.lit import train_test
from experiments.args import get_general_experiment_parser
from experiments.data import Shape, Dist, get_distribution, get_data_shape
from experiments.lit import train_test
from experiments.lit.model import AbstractLitGenerator, make_einet
from experiments.utils import anneal_tau, load_from_checkpoint
from experiments.utils import (
    setup_experiment_lit,
)
from simple_einet.einet import EinetConfig, Einet


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
        return torch.sigmoid(self.fc4(h3)).view(
            -1, image_shape.channels, image_shape.num_pixels
        )

    def reconstruct(self, x):
        x_recon, _, _ = self(x)
        return x_recon

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

        self.spn = self._make_spn()

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

    def round_and_clip(self, x):
        return torch.clamp(torch.floor(x), 0, 255)

    def encode(
        self, x: torch.Tensor, mpe_at_leaves=False, mpe=False, differentiable=True
    ) -> torch.Tensor:

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
        )

        evidence = self.round_and_clip(evidence)
        if differentiable:
            x_and_z = self.spn.sample_differentiable(
                evidence=evidence,
                marginalized_scopes=self.z_scopes,
                temperature_leaves=args.leaf_temperature,  # low temp to be closer to MPE
                hard=self.hard or mpe,
                tau=1.0 if mpe else self.tau,
            )
        else:
            x_and_z = self.spn.sample(
                evidence=evidence,
                marginalized_scopes=self.z_scopes,
                mpe_at_leaves=mpe_at_leaves,
                is_mpe=mpe,
            )

        z = x_and_z[:, :, self.z_scopes]

        z = z.clip(0, 255)
        return z

    def decode(
        self, z: torch.Tensor, mpe=False, mpe_at_leaves=False, differentiable=True
    ) -> torch.Tensor:
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

        evidence = self.round_and_clip(evidence)
        if differentiable:
            x_and_z = self.spn.sample_differentiable(
                evidence=evidence,
                marginalized_scopes=self.x_scopes,
                temperature_leaves=args.leaf_temperature,  # low temp to be closer to MPE
                hard=self.hard or mpe,
                tau=1.0 if mpe else self.tau,
                mpe_at_leaves=mpe_at_leaves,
            )
        else:
            x_and_z = self.spn.sample(
                evidence=evidence,
                marginalized_scopes=self.x_scopes,
                temperature_leaves=args.leaf_temperature,  # low temp to be closer to MPE
                mpe_at_leaves=mpe_at_leaves,
                is_mpe=mpe,
            )

        x = x_and_z[:, :, self.x_scopes]

        x = x.clip(0, 255)
        return x

    def sample_x(self, num_samples):
        x_and_z = self.spn.sample(
            num_samples=num_samples,
            marginalized_scopes=self.z_scopes,
            mpe_at_leaves=True,
        )

        x = x_and_z[:, :, self.x_scopes]
        return x

    def sample_z(self, num_samples):
        x_and_z = self.spn.sample(
            num_samples=num_samples,
            marginalized_scopes=self.x_scopes,
            mpe_at_leaves=True,
        )

        z = x_and_z[:, :, self.z_scopes]
        return z

    def sample(self, num_samples):
        z = self.sample_z(num_samples)
        x = self.decode(z, mpe=False, mpe_at_leaves=True, differentiable=True)
        return x

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

    def reconstruct(self, x):
        x = x.view(-1, image_shape.channels, image_shape.num_pixels)
        z = self.encode(x, mpe=True, differentiable=False)
        x_rec = self.decode(z, mpe=True, differentiable=False)
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

        if self.args.dist == Dist.BINOMIAL:
            x_rec = x_rec / 255
            x = x / 255

        loss_rec = F.binary_cross_entropy(
            input=x_rec.view(-1, image_shape.num_pixels * image_shape.channels),
            target=x.view(-1, image_shape.num_pixels * image_shape.channels),
            reduction=reduction,
        )

        loss = loss_rec

        # If return_recon is True, return the reconstructed image as well
        if return_recon:
            return loss, x_rec
        else:
            return loss


class PAE(AbstractLitGenerator):
    def __init__(self, args: Namespace):
        super().__init__(args=args, name="pae")

        if args.model == "vae":
            self.model = VAE(image_shape=image_shape, latent_dim=args.latent_dim)
        elif args.model == "spn":
            self.model = SPNAE(
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
                args=args,
            )
            self.mask_fixed = self.model.z_scopes.tolist()
        else:
            raise ValueError("Unknown model type")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(0.7 * args.epochs), int(0.9 * args.epochs)],
            gamma=0.1,
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, train_batch, batch_idx):
        """
        Training step.

        Args:
            train_batch: Training batch.
            batch_idx: Batch index.

        Returns:
            Loss for the batch.
        """
        data, _ = train_batch

        loss = self.compute_loss(data)
        self.log("Loss/train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        """
        Compute the loss for a validation batch.

        Args:
            val_batch: Validation batch.
            batch_idx: Index of the batch.

        Returns:
            loss: Loss for the batch.
        """
        data, _ = val_batch

        loss = self.compute_loss(data)
        self.log("Loss/val_loss", loss, prog_bar=True)

    def compute_loss(self, data):
        """
        Compute the loss for the given data.

        Args:
            data: The data to compute the loss for.

        Returns:
            The loss.
        """
        if args.dist == Dist.BINOMIAL and args.model == "spn":
            data = data * 255.0
        loss = self.model.loss(data)
        loss = loss / data.shape[0]
        return loss

    def get_spn(self):
        if self.args.model == "spn":
            return self.model.spn
        else:
            return None

    def generate_samples(self, num_samples: int):
        if self.args.model == "spn":
            samples = self.model.sample(num_samples=num_samples).view(
                num_samples, *self.image_shape
            )
            samples = samples / 255.0
            return samples
        elif self.args.model == "vae":
            z = torch.randn(num_samples, args.latent_dim).to(self.device)
            return self.model.decode(z).view(num_samples, *self.image_shape)
        else:
            raise ValueError("Unknown model type")

    def test_step(self, batch, batch_idx):
        """
        Test step, evaluates reconstruction error in BCE.

        Args:
            batch: batch of data.
            batch_idx: batch index.
        """
        data, _ = batch

        # Perform auto-encoding
        # Map to [0, 255] in spn case
        if self.args.model == "spn":
            data = data * 255.0
        data_rec = self.model.reconstruct(data)

        # Assert that the shapes are correct
        data = data.view(data.shape[0], -1)
        data_rec = data_rec.view(data_rec.shape[0], -1)

        # Map back to [0, 1] in spn case
        if self.args.model == "spn":
            data_rec = data_rec / 255.0
            data = data / 255.0

        # Compute reconstruction loss
        rec_error = F.binary_cross_entropy(input=data_rec, target=data, reduction="sum")
        rec_error = rec_error / data.shape[0]

        # Log
        self.log("Test/autoencoding_rec_error", rec_error)


if __name__ == "__main__":
    args = get_args()
    results_dir, args = setup_experiment_lit(
        name="pae", args=args, remove_if_exists=True
    )

    image_shape = get_data_shape(args.dataset)

    # model
    # Load or create model
    if args.load_and_eval:
        model = load_from_checkpoint(
            results_dir, load_fn=PAE.load_from_checkpoint, args=args
        )
    else:
        model = PAE(args)
    normalize = False
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