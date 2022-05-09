# PyTorch Lightning Implementation

This module contains the code for the pytorch lightning implementation. All training/testing/validation procedures are covered by PyTorch Lightning.

Note: For now the MMD code `experiments/lit/main_mmd.py` seems to have some issues for now, results compared to the original MMD code in `experiments/main_mmd.py` could not be reproduced. We suspect that there is something weird happening in PyTorch Lightning when the kernel encoder parameters are set to be frozen during the `train_step` implementation. This needs further investigation. 