#!/usr/bin/env bash
set -x
gpu=$1
seed=$2
datasets=( "mnist-28" "svhn" "cifar" "celeba-small" )

ARGS="--epochs 50 --batch-size 128 --spn-R 10 --spn-D 4 --spn-K 20 --epochs-pretrain 0 --seed $seed"

for dataset in ${datasets[@]};
do
  # VAE
  PYTHONPATH=./ python experiments/lit/main_pae.py $ARGS --dataset $dataset --model vae --gpu $gpu --tag "model=vae_seed=$seed"
  # SPN
  PYTHONPATH=./ python experiments/lit/main_pae.py $ARGS --dataset $dataset --model spn --lr 0.1 --spn-tau-method annealed --gpu $gpu --tag "model=spn_seed=$seed"
done