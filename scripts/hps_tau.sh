#!/usr/bin/env bash
set -x
gpu=$1
tau=$2
method=$3
dataset="celeba"


tag="method=${method}_tau=${tau}"

ARGS="--dataset ${dataset} --model spn --batch-size 128 --spn-R 10 --spn-D 4 --spn-K 20 --epochs-pretrain 0 --seed 0 --spn-tau ${tau} --spn-tau-method ${method} --tag ${tag} --gpu ${gpu}"

# PAE
PYTHONPATH=./ python experiments/lit/main_pae.py $ARGS --epochs 10 --lr 0.1

# GAN
PYTHONPATH=./ python experiments/lit/main_adv.py $ARGS --epochs 30 --lr-g 0.001

# MMD
PYTHONPATH=./ python experiments/main_mmd.py $ARGS --epochs 30 --lr-g 0.010 --lr-d 0.001