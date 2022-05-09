#!/usr/bin/env bash
set -x
gpu=$1
lr_g=$2
batch_sizes=( 64 128 256 )
lrs_d=( 0.1 0.05 0.01 0.005 0.001 0.0001 0.00001)
datasets=( "mnist" "svhn" "cifar" "celeba-small")

for dataset in ${datasets[@]};
do
  for bs in ${batch_sizes[@]};
  do
    for lr_d in ${lrs_d[@]};
    do
      tag="lr-g=${lr_g}_lr-d=${lr_d}_bs=${bs}_mmd-old"
      ARGS="--batch-size ${bs} --spn-R 10 --spn-D 4 --spn-K 20 --epochs-pretrain 0 --seed 0 --spn-tau 1.0 --spn-tau-method annealed"
      # MMD
      PYTHONPATH=./ python experiments/main_mmd.py $ARGS --epochs 30 --dataset $dataset --model spn --lr-g $lr_g --lr-d $lr_d --gpu $gpu --tag $tag
    done
  done
done