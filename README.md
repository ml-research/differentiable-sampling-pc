# Elevating Perceptual Sample Quality in Probabilistic Circuits through Differentiable Sampling

Code for our paper [Elevating Perceptual Sample Quality in Probabilistic Circuits through Differentiable Sampling](https://proceedings.mlr.press/v181/lang22a/lang22a.pdf). 

## Code Structure

- `experiments/lit`: Main experiments code, implemented with PyTorch Lightning.
- `experiments`: Experiments code, implemented with pure PyTorch.
- `scripts`: Bash scripts for hyper-parameter grid-search and other stuff.
- `tools`: Experiments unrelated scripts (Docker, Data)
- `simple_einet`: Fork of [simple-einet](https://github.com/steven-lang/simple-einet) with a differentiable sampling implementation added.
- `environment.yml`: Conda environment file for a simple setup with Conda.
- `requirements.txt`: PIP requirements file for a simple setup with pip.
- `Dockerfile`: Dockerfile for a simple setup with Docker (Note: Use `tools/run.py -g -- python ...` to run experiments via Docker).


## Other Sources

We base our experiments for the adversarial, maximum mean discrepancy, and probabilistic auto-encoding experiment on the following implementations:

- GAN: [Vanilla GAN, PyTorch-GAN Collection](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py)
- MMD: [Original MMD-GAN implementation](https://github.com/OctoberChang/MMD-GAN)
- VAE: [PyTorch Vanilla VAE Tutorial](https://github.com/pytorch/examples/blob/main/vae/main.py)


## Citing This Work


```bib
@inproceedings{lang2022diff-sampling-spns,
  title = {Elevating Perceptual Sample Quality in Probabilistic Circuits through Differentiable Sampling},
  author = {Lang, Steven and Mundt, Martin and Ventola, Fabrizio and Peharz, Robert and Kersting, Kristian},
  booktitle = {NeurIPS 2021 Workshop on Pre-registration in Machine Learning},
  keywords = {Probabilistic Circuits, Sum-Product Networks, Differentiable Sampling, Deep Learning},
  year = {2022},
  volume = {181},
  series = {Proceedings of Machine Learning Research},
  pages = {1--25},
  publisher = {PMLR},
}
```


