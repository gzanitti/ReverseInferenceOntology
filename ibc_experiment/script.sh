#!/bin/bash

#SBATCH --job-name=RIIBC
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --oversubscribe
#SBATCH --ntasks-per-node=1
#SBATCH --partition=parietal,normal

id_region=`echo $1`
python Intra-IBC.py --id_region $id_region # --lower_prob 0.0