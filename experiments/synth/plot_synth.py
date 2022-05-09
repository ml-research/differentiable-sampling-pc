import argparse
import io
import json
import os
import pathlib
import shutil
from typing import Tuple


import PIL
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor

from matplotlib.cm import tab10
from matplotlib import cm

from experiments.synth.utils_2d import generate_data
from experiments.synth.model_2d import FlatSpn2D

bins = 100


def plot_data_distribution(
    data, save_dir: str, step: int, name, xmin, xmax, ymin, ymax
):
    """Plot the distribution of the data."""
    plt.figure(figsize=get_figsize(1.0))

    data = data.cpu().numpy()
    plt.scatter(
        *data[:5000].T,
        ec="black",
        lw=0.5,
        s=10,
        alpha=0.5,
    )

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.tick_params(labelcolor="white", bottom=False, left=False)
    plt.grid("off")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}-dist_{step:0>4d}.pdf"))
    plt.close()


def plot_model_samples(model, data, save_dir: str, step: int, xmin, xmax, ymin, ymax):
    n_samples = data.shape[0]
    samples = model.sample(n_samples)

    plot_data_distribution(
        data=samples,
        save_dir=save_dir,
        step=step,
        name="model-samples",
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
    )


def plot_model_distribution(model, data, save_dir: str, step: int, xmin, xmax, ymin, ymax):
    plt.figure(figsize=get_figsize(1.0))

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    x = np.linspace(xmin, xmax, bins)
    y = np.linspace(ymin, ymax, bins)
    X, Y = np.meshgrid(x, y)

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    Z = torch.exp(
        model(torch.from_numpy(np.c_[X.flatten(), Y.flatten()]).to(data.device))
    )
    Z = Z.view(X.shape).cpu().numpy()
    plt.contour(X, Y, Z, 20, cmap=plt.cm.viridis)

    plt.tick_params(labelcolor="white", bottom=False, left=False)
    plt.grid("off")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"model-dist_{step:0>4d}.pdf"))
    plt.close()


@torch.no_grad()
def final_plot(model, data, save_dir, step):
    """Plot the model distribution."""
    data = data.detach()
    model = model

    xmin, xmax = data[:, 0].min().item(), data[:, 0].max().item()
    ymin, ymax = data[:, 1].min().item(), data[:, 1].max().item()

    plot_model_distribution(
        model=model,
        data=data,
        save_dir=save_dir,
        step=step,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
    )
    plot_model_samples(
        model=model,
        data=data,
        save_dir=save_dir,
        step=step,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
    )
    plot_data_distribution(
        data=data,
        save_dir=save_dir,
        step=step,
        name="data",
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
    )


TEXTWIDTH = 5.78853
LINEWIDTH = 0.75
ARROW_HEADWIDTH = 5
colors = tab10.colors


def get_figsize(scale: float, aspect_ratio=0.8) -> Tuple[float, float]:
    """
    Scale the default figure size to: (scale * TEXTWIDTH, scale * aspect_ratio * TEXTWIDTH).

    Args:
      scale(float): Figsize scale. Should be lower than 1.0.
      aspect_ratio(float): Aspect ratio (as scale), height to width. (Default value = 0.8)

    Returns:
      Tuple: Tuple containing (width, height) of the figure.

    """
    height = aspect_ratio * TEXTWIDTH
    widht = TEXTWIDTH
    return (scale * widht, scale * height)


def set_style():
    matplotlib.use("pgf")
    plt.style.use(["science", "grid"])  # Need SciencePlots pip package
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
        }
    )


def plot_distribution(
    model, data, targets=None, tag=None, writer: SummaryWriter = None, step=0
):
    with torch.no_grad():
        # samples = torch.stack([model.sample() for _ in range(300)], dim=0)

        fig = plt.figure(figsize=get_figsize(1.0))
        data_cpu = data.cpu()
        delta = 0.05
        xmin, xmax = data_cpu[:, 0].min(), data_cpu[:, 0].max()
        ymin, ymax = data_cpu[:, 1].min(), data_cpu[:, 1].max()
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        # xmin, xmax, ymin, ymax = xmin * 3, xmax * 3, ymin * 3, ymax * 3
        x = np.arange(xmin * 1.05, xmax * 1.05, delta)
        y = np.arange(ymin * 1.05, ymax * 1.05, delta)
        X, Y = np.meshgrid(x, y)

        Z = torch.exp(
            model(
                torch.from_numpy(np.c_[X.flatten(), Y.flatten()])
                .to(data.device)
                .float()
            ).float()
        ).cpu()
        Z = Z.view(X.shape)
        CS = plt.contourf(X, Y, Z, 100, cmap=plt.cm.viridis)
        plt.colorbar(CS)

        if targets is None:
            plt.scatter(
                *data_cpu[:500].T,
                label="Data",
                ec="black",
                lw=0.5,
                s=10,
                alpha=0.5,
                color=colors[1],
            )
        else:
            for i, label in enumerate(set(targets.numpy())):
                mask = label == targets
                plt.scatter(
                    *data[mask][:500].T,
                    label="Data",
                    ec="black",
                    lw=0.5,
                    s=10,
                    alpha=0.5,
                    color=colors[i],
                )

        # Plot cluster center
        for ch in model.loc.T:
            plt.scatter(ch[0], ch[1], lw=3, s=75, alpha=1.0, marker="x")



        plt.xlabel("$X_0$")
        plt.ylabel("$X_1$")
        plt.title(f"Learned PDF represented by the SPN ({tag})")

        buf = io.BytesIO()
        plt.savefig(buf, format="jpeg")
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image)

        # Add figure in numpy "image" to TensorBoard writer
        writer.add_image("Distribution", image, step)
        plt.close(fig)

        #
        # if path is not None:
        #     plt.savefig(path, dpi=300)
        # else:
        #     plt.savefig(f"./pdf-{tag}.png", dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=pathlib.Path,
        default=None,
        help="Path to model",
        required=True,
    )
    parser.add_argument(
        "--output-path",
        type=pathlib.Path,
        default=None,
        help="Path to output",
        required=True,
    )
    args = parser.parse_args()
    args.model_path = args.model_path.expanduser()
    args.output_path = args.output_path.expanduser()
    args_file = os.path.join(os.path.dirname(args.model_path), "args.json")
    orig_args = argparse.Namespace(**json.load(open(args_file)))

    output_path = os.path.join(args.output_path, orig_args.data_synth)

    os.makedirs(output_path, exist_ok=True)

    # Copy model to the figs dir
    shutil.copy(args.model_path, os.path.join(output_path, "model.pth"))

    # Copy args
    shutil.copy(args_file, os.path.join(output_path, "args.json"))

    # Load model
    model = FlatSpn2D(orig_args)
    state_dict = torch.load(args.model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)

    model.eval()
    with torch.no_grad():
        data = generate_data(orig_args.data_synth, device="cpu", n_samples=100000)
        final_plot(model, data, output_path, 0)