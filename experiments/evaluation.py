import os
import pathlib
import warnings
from argparse import Namespace
from typing import Callable, Tuple
from cleanfid import fid

import torch
import torchvision
import tqdm
from torch.utils.data import DataLoader

from typing import List
from simple_einet.einet import Einet
from experiments.data import Shape, get_data_shape, Dist
from experiments.utils import ensure_dir

from torch.nn import MSELoss

# Sub directory name to store samples in
SAMPLES_SUB_DIR = "samples"


@torch.no_grad()
def save_samples(args, results_dir, generator: Callable):
    """
    Save samples from a callable generator.

    Args:
        results_dir: Results directory.
        args: Arguments.
        generator: Callable generator to iterate over (generates a batch of samples in each
        iteration).

    Returns:

    """
    samples_dir = os.path.join(results_dir, SAMPLES_SUB_DIR)
    ensure_dir(samples_dir + "/")
    counter = 0
    batch_size = 500
    max_samples = 10_000
    for samples in tqdm.tqdm(
        generator(batch_size), desc="Saving samples", total=max_samples // batch_size
    ):
        for sample in samples:
            counter += 1
            filename = samples_dir + "/sample_%d.png" % counter
            torchvision.utils.save_image(sample, filename)

            if counter >= max_samples:
                print("Finished generating samples")
                return

        if args.debug:
            return

    raise Exception("Could not generate enough samples")


@torch.no_grad()
def eval_fid_kid_celeba(args, results_dir, generate_samples, device):
    save_samples(args, results_dir=results_dir, generator=generate_samples)

    # Compute KID/FID scores
    fid_score_test, kid_score_test = evaluate_fid_kid_scores(
        args, results_dir, device, split="test"
    )
    # fid_score_val, kid_score_val = evaluate_fid_kid_scores(
    #     args, results_dir, device, split="val"
    # )
    metrics = {
        "fid_test": fid_score_test,
        # "fid_val": fid_score_val,
        "kid_test": kid_score_test,
        # "kid_val": kid_score_val,
    }
    return metrics


@torch.no_grad()
def evaluate_fid_kid_scores(args, results_dir, device, split) -> Tuple[float, float]:
    if args.debug:
        return 0., 0.
    assert split in ["test", "val"]
    # Create paths to images
    if split == "test":
        celeba_path = pathlib.Path(args.data_dir, "celeba", "img_test")
    elif split == "val":
        celeba_path = pathlib.Path(args.data_dir, "celeba", "img_val")
    else:
        raise ValueError()

    samples_path = pathlib.Path(results_dir, SAMPLES_SUB_DIR)
    assert celeba_path.is_dir()
    assert samples_path.is_dir()
    celeba_path = str(celeba_path)
    samples_path = str(samples_path)

    celeba_stats_name = "celeba-small-" + split

    # If stats do not exist, create them
    if not fid.test_stats_exists(name=celeba_stats_name, mode="clean", metric="FID"):
        fid.make_custom_stats(
            name=celeba_stats_name,
            fdir=celeba_path,
            num=50_000,
            num_workers=8,
            batch_size=args.batch_size,
            device=device,
            mode="clean",
        )

    fid.test_stats_exists(name=celeba_stats_name, mode="clean", metric="KID")

    # run FID library
    fid_score = fid.compute_fid(
        fdir1=samples_path,
        dataset_name=celeba_stats_name,
        dataset_split="custom",
        batch_size=args.batch_size,
        device=device,
    )

    kid_score = fid.compute_kid(
        fdir1=samples_path,
        dataset_name=celeba_stats_name,
        dataset_split="custom",
        batch_size=args.batch_size,
        device=device,
    )

    print("FID score:", fid_score)
    print("KID score:", kid_score)
    return fid_score, kid_score


def construct_marginalization_indices(image_shape: Shape) -> List[torch.Tensor]:
    mask_set = []

    # horizontal stripes
    h, w = image_shape.height, image_shape.width
    index = torch.arange(h * w).view(1, h, w)
    for i in range(4):
        mask = index[:, int(h * i / 4) : int(h * (i + 1) / 4), :]
        mask = mask.reshape(-1).tolist()
        mask_set.append(mask)

        mask = index[:, :, int(h * i / 4) : int(h * (i + 1) / 4)]
        mask = mask.reshape(-1).tolist()
        mask_set.append(mask)

    return mask_set


@torch.no_grad()
def evaluate_reconstruction_error(
    args: Namespace,
    model: Einet,
    dataloader_test: DataLoader,
    device,
    mask_fixed=None,
):
    if args.debug:
        return 0.0

    image_shape = get_data_shape(args.dataset)
    masks = construct_marginalization_indices(image_shape)

    mse = MSELoss(reduction="mean")

    model.eval()
    error_total = 0.0
    count = 0

    # Iterate over dataloader
    for data, _ in dataloader_test:
        data = data.to(device)
        data = data.view(-1, image_shape.channels, image_shape.num_pixels)

        # Scale if binomial model distribution
        if args.dist == Dist.BINOMIAL:
            data *= 255.0

        # If this is the autoencoding setting with latent variables, append some dummy
        if mask_fixed is not None:
            dummy_z = torch.zeros(
                data.shape[0],
                data.shape[1],
                len(mask_fixed),
                device=data.device,
            )
            data = torch.cat([data, dummy_z], dim=2)

        # Scale over all masks (4x horizontal + 4x vertical stripes)
        for mask in masks:

            # Autoencoding setting
            if mask_fixed is not None:
                mask = mask + mask_fixed

            # Reconstruct
            data_rec = model.sample(
                evidence=data, is_mpe=True, marginalized_scopes=mask
            )

            # Compute error
            if args.dist == Dist.BINOMIAL:
                error = mse(data, data_rec)
            else:
                error = mse(data, data_rec)
            error_total += error
            count += 1

    # Average over number of samples
    final_error = error_total / count
    print("Reconstruction error:", final_error)
    return final_error


@torch.no_grad()
def evaluate_auto_encoding_reconstruction_error(
        args: Namespace,
        model,
        dataloader_test: DataLoader,
        device,
):

    image_shape = get_data_shape(args.dataset)

    mse = MSELoss(reduction="mean")

    model.eval()
    error_total = 0.0
    count = 0

    # Iterate over dataloader
    for data, _ in dataloader_test:
        data = data.to(device)
        data = data.view(-1, image_shape.channels, image_shape.num_pixels)



        # Reconstruct
        if args.model == "spn":
            # Scale if binomial model distribution
            if args.dist == Dist.BINOMIAL:
                data *= 255.0

            data_rec = model(data, mpe=True)
        elif args.model == "vae":
            data_rec, _, _ = model(data)

            # Scale to [0,255]
            data_rec = data_rec * 255
            data = data * 255
        else:
            raise ValueError()

        # Compute error
        error = mse(data, data_rec)
        error_total += error
        count += 1

    # Average over number of samples
    final_error = error_total / count
    print("Autoencoding reconstruction error:", final_error)
    return final_error

if __name__ == "__main__":
    construct_marginalization_indices(Shape(1, 4, 4))