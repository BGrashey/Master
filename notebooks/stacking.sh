#!/bin/bash
#$ -N Stacking
#$ -cwd
#$ -j y
#$ -m ea
#$ -M B.Grashey@campus.lmu.de
#$ -pe smp 4
#$ -l h_vmem=50G
#$ -l h_rt=48:00:00





#export PYTHONPATH=$PYTHONPATH:/data/hetdex/u/bgrashey/notebooks/

# --- Skript ausführen ---
/data/backup/hetdex/u/bgrashey/micromamba run -p /data/backup/hetdex/u/bgrashey/envs/cube_aktuell python stacking.py
