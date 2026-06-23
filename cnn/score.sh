#!/bin/bash
#$ -N Scoring_A_Catalog
#$ -cwd
#$ -j y
#$ -m ea
#$ -M B.Grashey@campus.lmu.de
#$ -pe smp 1
#$ -l h_vmem=50G
#$ -l h_rt=02:00:00





#export PYTHONPATH=$PYTHONPATH:/data/hetdex/u/bgrashey/notebooks/

# --- Skript ausführen ---
/data/backup/hetdex/u/bgrashey/micromamba run -p /data/backup/hetdex/u/bgrashey/envs/cnn python scoring_.py
