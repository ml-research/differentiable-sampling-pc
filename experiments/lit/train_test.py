import os
from typing import Any, Dict

import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from experiments.data import build_dataloader
from experiments.evaluation import eval_fid_kid_celeba
from experiments.lit.model import AbstractLitGenerator


def main(args, model: AbstractLitGenerator, results_dir, normalize: bool, hparams: Dict[str, Any]):
    print(model)
    seed_everything(args.seed, workers=True)

    print("Training model...")
    # Create dataloader
    train_loader, val_loader, test_loader = build_dataloader(
        args=args, loop=False, normalize=normalize
    )

    # Create callbacks
    logger = TensorBoardLogger(results_dir, name="tb")

    # Create trainer
    gpus = 1 if torch.cuda.is_available() else 0
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = [args.gpu] if gpus else None
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        accelerator=accelerator,
        devices=devices,
    )


    if not args.load_and_eval:
        # Fit model
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("Evaluating model...")


    metrics = {}

    # Evaluate spn reconstruction error
    # rec_error_val = trainer.test(model=model, dataloaders=val_loader, verbose=True)[0]
    # rec_error_test = trainer.test(model=model, dataloaders=test_loader, verbose=True)[0]
    #
    #
    # for key, value in rec_error_val.items():
    #     metrics[f"{key}_val"] = value
    #
    # for key, value in rec_error_test.items():
    #     metrics[f"{key}_test"] = value
    #
    #     # # Save results
    #     # metrics_rec = {
    #     #     "rec_error_val": rec_error_val["test_rec_error"],
    #     #     "rec_error_test": rec_error_test["test_rec_error"],
    #     # }
    #     # metrics.update(metrics_rec)
    #
    #
    # # Evaluate FID score
    # if args.fid and "celeba" in args.dataset:
    #     metrics_fid = eval_fid_kid_celeba(
    #         args=args,
    #         results_dir=results_dir,
    #         generate_samples=model.samples_generator,
    #         device=torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device(
    #             "cpu"),
    #     )
    #     metrics.update(metrics_fid)
    #
    # hparams_general = {
    #     "epochs": args.epochs,
    #     "batch-size": args.batch_size,
    #     "seed": args.seed,
    #     "spn-K": args.spn_K,
    #     "spn-D": args.spn_D,
    #     "spn-R": args.spn_R,
    #     "spn-tau": args.spn_tau,
    #     "spn-tau-method": args.spn_tau_method,
    #     "dataset": args.dataset,
    # }
    # hparams.update(hparams_general)
    # logger.experiment.add_hparams(metric_dict=metrics, hparam_dict=hparams)

    # Save some samples
    samples_dir = os.path.join(results_dir, "samples")
    os.makedirs(exist_ok=True, name=samples_dir)
    model.save_samples(samples_dir, num_samples=9, nrow=3)