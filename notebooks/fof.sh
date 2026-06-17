#!/bin/bash
#$ -N FoF
#$ -cwd
#$ -j y
#$ -m ea
#$ -M B.Grashey@campus.lmu.de
#$ -pe smp 4
#$ -l h_vmem=20G
#$ -l h_rt=12:00:00





#export PYTHONPATH=$PYTHONPATH:/data/hetdex/u/bgrashey/notebooks/

# --- Skript ausführen ---
/data/backup/hetdex/u/bgrashey/micromamba run -p /data/backup/hetdex/u/bgrashey/envs/cnn python fof_execution.py
