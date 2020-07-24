#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=semi_supervised_schedule_2

# set number of GPUs
#SBATCH --gres=gpu:1

# set number of CPUs
#SBATCH --cpus-per-task=5

# set partition
#SBATCH --partition=small

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=artur.begyan@kellogg.ox.ac.uk

# Standard output and error log
#SBATCH --output=semi_supervised_schedule_2_%j.out

srun python main.py --model_type mlp --semi_supervised_schedule True --num_iterations 300000
