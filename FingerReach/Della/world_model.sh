#!/usr/bin/env bash
#SBATCH -J 'wnet'
#SBATCH -o slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00


module purge
module load anaconda3/2022.5

conda activate object_env
python -u world_model.py


