#!/bin/bash --login
#SBATCH -N 1
#SBATCH --partition=batch ###SBATCH --partition=batch
#SBATCH -J Monodepth3
##SBATCH --cores-per-socket=24
#SBATCH -o slurm/%J
#SBATCH -e slurm/%J
#SBATCH --time=03:58:00  ###SBATCH --time=96:00:00
#SBATCH --mem=48G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:a100:2   ##SBATCH --gres=gpu:v100:4
#SBATCH --reservation=A100

##module purge
##module add anaconda3
#source ~/.bashrc

conda activate ffcv
echo "Activated environment"
python train.py -m toymodel --tags=testrun --epochs=5
#jupyter lab --ip=0.0.0.0 --port 8990
#sleep infinity
