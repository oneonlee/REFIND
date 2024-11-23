#!/bin/sh
#SBATCH -J build_index
#SBATCH -p RTX4090
#SBATCH -o /home/donggeonlee/repo/REFIND/logs/%j.out
#SBATCH -e /home/donggeonlee/repo/REFIND/logs/%j.err
#SBATCH -t 3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBTACH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8

echo "Activate conda"
source /home/donggeonlee/miniconda3/bin/activate REFIND

cd /home/donggeonlee/repo/REFIND
pwd

python retriever/build_index.py

echo "Done"
