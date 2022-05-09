import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from experiments.synth.utils_2d import SynthDataset


def main(args, model, results_dir):

    print(model)

    seed_everything(args.seed, workers=True)

    num_samples_train = args.batch_size * 1000
    train_dataset = SynthDataset(args.data_synth, num_samples=num_samples_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8)

    logger = TensorBoardLogger(results_dir, name="tb")
    gpus = 1 if torch.cuda.is_available() else 0
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = [args.gpu] if gpus else None
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        accelerator=accelerator,
        devices=devices
    )
    trainer.fit(model, train_loader)

    print("Training finished.")
    print(f"All results saved in {results_dir}")