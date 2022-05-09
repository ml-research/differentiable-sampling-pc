import torch

from experiments.data import Dist
from simple_einet.einet import Einet


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
    optimizer = torch.optim.Adam(spn.parameters(), lr=args.lr_pretrain)

    # ==================
    # Pretrain generator
    # ==================
    print("Pretraining SPN...")
    for epoch in range(args.epochs_pretrain):
        for i, (data, _) in enumerate(dataloader):
            # Move data to device
            data = data.to(device)

            if args.dist == Dist.BINOMIAL:
                data = data * 255.0

            # Forward pass through spn_generator
            lls = spn(data)

            loss = -1 * torch.sum(lls)

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