#!/usr/bin/env bash
set -x
gpu=$1
seed=$2

ARGS="--dataset celeba-small --epochs 30 --batch-size 64 --spn-R 10 --spn-D 4 --spn-K 20 --epochs-pretrain 0 --seed $seed"


## GAN
PYTHONPATH=./ python experiments/lit/main_adv.py $ARGS --model spn --lr-g 0.0001 --spn-tau 0.5 --gpu $gpu --tag "model=spn_seed=$seed"
PYTHONPATH=./ python experiments/lit/main_adv.py $ARGS --model gan --lr-g 0.0002 --spn-tau 0.05 --gpu $gpu --tag "model=gan_seed=$seed"

## PAE
PYTHONPATH=./ python experiments/lit/main_pae.py $ARGS --model spn --lr 0.0001 --spn-tau 0.5 --gpu $gpu --tag "model=spn_seed=$seed"
PYTHONPATH=./ python experiments/lit/main_pae.py $ARGS --model vae --lr 0.0002 --spn-tau 0.05 --gpu $gpu --tag "model=vae_seed=$seed"

# MMD
PYTHONPATH=./ python experiments/lit/main_mmd.py $ARGS --model spn --lr-g 0.01 --spn-tau 0.05 --gpu $gpu --tag "model=spn_seed=$seed"
PYTHONPATH=./ python experiments/lit/main_mmd.py $ARGS --model gan --lr-g 0.05 --spn-tau 0.05 --gpu $gpu --tag "model=gan_seed=$seed"

# LL
PYTHONPATH=./ python experiments/lit/main_spn.py $ARGS --lr 0.2 --gpu $gpu --tag "seed=$seed"