#!/usr/bin/env bash
set -x
gpu=$1
results_dir=$2
model=$3


for dataset_dir in $results_dir/${model}/*;
do
  for exp_dir in $dataset_dir/*;
  do
        # Run main_mmd
        echo "Running main_${model} in $exp_dir"
        python ./experiments/lit/main_${model}.py --load-and-eval $exp_dir --gpu $gpu --dataset mnist

  done
done


for dataset_dir in $results_dir/spn/*;
do
  for exp_dir in $dataset_dir/*;
  do
        # Run main_spn
        echo "Running main_spn in $exp_dir"
        python ./experiments/lit/main_spn.py --load-and-eval $exp_dir --gpu $gpu --dataset mnist
  done
done

for dataset_dir in $results_dir/adv/*;
do
  for exp_dir in $dataset_dir/*;
  do
        # Run main_gan
        echo "Running main_gan in $exp_dir"
        python ./experiments/lit/main_adv.py --load-and-eval $exp_dir --gpu $gpu --dataset mnist
  done
done

for dataset_dir in $results_dir/pae/*;
do
  for exp_dir in $dataset_dir/*;
  do
        # Run main_ae
        echo "Running main_pae in $exp_dir"
        python ./experiments/lit/main_pae.py --load-and-eval $exp_dir --gpu $gpu --dataset mnist
  done
done

for dataset_dir in $results_dir/mmd/*;
do
  for exp_dir in $dataset_dir/*;
  do
        # Run main_mmd
        echo "Running main_mmd in $exp_dir"
        python ./experiments/lit/main_mmd.py --load-and-eval $exp_dir --gpu $gpu --dataset mnist

  done
done