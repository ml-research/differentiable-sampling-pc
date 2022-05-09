#!/usr/bin/env python3

import os
import pprint

import torch
import torchvision
import tqdm
from torch.utils.tensorboard import SummaryWriter

from experiments.args import get_general_experiment_parser
from experiments.data import Dist, get_distribution, build_dataloader
from experiments.evaluation import (
    save_samples,
    evaluate_reconstruction_error,
    evaluate_fid_kid_scores, eval_fid_kid_celeba,
)
from experiments.utils import (
    setup_experiment,
)
from simple_einet.einet import Einet, EinetConfig


def log_likelihoods(outputs, targets=None):
    """Compute the likelihood of an Einet."""
    if targets is None:
        num_roots = outputs.shape[-1]
        if num_roots == 1:
            lls = outputs
        else:
            num_roots = torch.tensor(float(num_roots), device=outputs.device)
            lls = torch.logsumexp(outputs - torch.log(num_roots), -1)
    else:
        lls = outputs.gather(-1, targets.unsqueeze(-1))
    return lls


def train(
    args, model: Einet, device, train_loader, optimizer, epoch, writer: SummaryWriter
):

    model.train()

    pbar = tqdm.tqdm(train_loader)

    for batch_idx, (data, target) in enumerate(pbar):

        # Stop after a few batches in debug mode
        if args.debug and batch_idx > 2:
            break

        # Prepare data
        data, target = data.to(device, memory_format=torch.channels_last), target.to(
            device
        )

        if args.dist == Dist.BINOMIAL:
            data *= 255.0

        optimizer.zero_grad()

        # Generate outputs
        outputs = model(data)
        # # Compute loss
        loss = -1 * log_likelihoods(outputs).mean()

        # Compute gradients
        loss.backward()

        # Update weights
        optimizer.step()

        # Logging
        writer.add_scalar(
            "Loss/lls", loss.item(), epoch * len(train_loader) + batch_idx
        )

        # Logging
        pbar.set_description(
            "Train Epoch: {} [{}/{}] Loss: {:.4f}".format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                loss.item() / data.shape[0],
            )
        )

        batches_done = epoch * len(dataloader_train) + batch_idx
        if batches_done % args.sample_interval == 0:
            gen_imgs = model.sample(num_samples=25, mpe_at_leaves=True).view(
                25, *image_shape
            )
            grid = torchvision.utils.make_grid(
                gen_imgs.data[:25], nrow=5, pad_value=0.0, normalize=True
            )
            writer.add_image("Image/samples", grid, batches_done)

        if args.debug:
            break

        rtpt.step()


def test(model, device, loader, tag) -> float:
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)

            data *= 255.0

            outputs = model(data)

            lls = log_likelihoods(outputs)
            test_loss += -1 * lls.sum()

    test_loss /= len(loader.dataset)

    print()
    print("{} set: Average loss: {:.4f}".format(tag, test_loss))

    print()
    return test_loss


def get_args():
    parser = get_general_experiment_parser()
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--train", action="store_true", help="Train the model")

    return parser.parse_args()


def make_einet(args, image_shape):
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
    return Einet(config).to(device)


def main_train():
    # Optimize Einet parameters (weights and leaf params)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(0.7 * args.epochs), int(0.9 * args.epochs)],
        gamma=0.1,
    )
    print(model)
    try:
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, dataloader_train, optimizer, epoch, writer)
            # test(model, device, test_loader, "Test")
            lr_scheduler.step()

    except KeyboardInterrupt:
        print("Interrupt caught. Stopping training.")
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    args = get_args()
    (args, results_dir, writer, device, image_shape, rtpt) = setup_experiment(
        name="spn", args=args
    )
    # We store and load the model in this path
    model_path = os.path.join(results_dir, "spn.pth")

    # Check if we skip training and only load and evaluate in this run
    load_and_eval = args.load_and_eval is not None
    do_train = not load_and_eval

    normalize = args.dist == Dist.NORMAL
    dataloader_train, dataloader_val, dataloader_test = build_dataloader(
        args, loop=False, normalize=normalize
    )

    model = make_einet(args, image_shape)
    print(
        "Number of parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    if do_train:
        main_train()
    else:
        print("Loading model from:", model_path)
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)

    model.eval()

    # train_lls = test(model, device, dataloader_test, "Train")
    # test_lls = test(model, device, dataloader_test, "Test")
    # val_lls = test(model, device, dataloader_val, "Val")

    def generate_samples(batch_size: int = 100):
        """
        Construct a generator from the trained model.
        Returns:
            Generator that yields samples.

        """
        while True:
            samples = (
                model.sample(num_samples=batch_size, mpe_at_leaves=True).view(
                    batch_size, *image_shape
                )
                / 255.0
            )
            yield samples

    # if "celeba" in args.dataset.lower():
    #     metrics = eval_fid_kid_celeba(args=args, results_dir=results_dir,
    #                         generate_samples=generate_samples, device=device)
    # else:
    #     metrics = {}
    metrics = {}

    spn = model
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
    hparams = {
        "spn_D": args.spn_D,
        "spn_K": args.spn_K,
        "spn_R": args.spn_R,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "dataset": args.dataset,
    }
    writer.add_hparams(hparams, metrics)
    writer.close()

    print("#" * 80)
    print("FINAL METRICS")
    pprint.pprint(metrics)