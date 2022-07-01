#!/bin/bash

#SBATCH --job-name=RIJulich
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --oversubscribe
#SBATCH --ntasks-per-node=1
#SBATCH --partition=parietal,normal

id_region=`echo $1`
python python_script_terms_new_atlas.py --id_region $id_region # --lower_prob 0.0
