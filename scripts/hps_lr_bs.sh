#!/usr/bin/env bash
set -x
gpu=$1
lr=$2
bs=$3
datasets=( "mnist" "svhn" "cifar" "celeba-small" )

ARGS="--batch-size $bs --spn-R 10 --spn-D 4 --spn-K 20 --epochs-pretrain 0 --seed 0 --spn-tau 1.0 --spn-tau-method annealed"


tag="lr=${lr}_bs=${bs}"

for dataset in ${datasets[@]};
do
  # PAE
  PYTHONPATH=./ python experiments/lit/main_pae.py $ARGS --epochs 30 --dataset $dataset --model spn --lr $lr --gpu $gpu --tag $tag

  # LL
  PYTHONPATH=./ python experiments/lit/main_spn.py $ARGS --epochs 30 --dataset $dataset --lr $lr --gpu $gpu --tag $tag

  # GAN
  PYTHONPATH=./ python experiments/lit/main_adv.py $ARGS --epochs 30 --dataset $dataset --model spn --lr-g $lr --gpu $gpu --tag $tag

  # MMD
  PYTHONPATH=./ python experiments/lit/main_mmd.py $ARGS --epochs 30 --dataset $dataset --model spn --lr-g $lr --gpu $gpu --tag $tag
done