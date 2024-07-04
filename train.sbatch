#!/bin/bash
#SBATCH --job-name open-calm3b
#SBATCH --output logs/%x-%j.log
#SBATCH --error logs/%x-%j.err
#SBATCH --nodes 1
#SBATCH --gpus tesla:1

date
singularity exec --nv experiment2.sif python3.10  /home/student/e21/e215736/experiment2/files_extract_g-sam.py
date