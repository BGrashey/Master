#!/bin/bash
#$ -N Extracting_a_Cube
#$ -cwd
#$ -j y
#$ -m ea
#$ -M B.Grashey@campus.lmu.de
#$ -pe smp 2
#$ -l h_vmem=50G
#$ -l h_rt=24:00:00





#export PYTHONPATH=$PYTHONPATH:/data/hetdex/u/bgrashey/notebooks/

# --- Skript ausführen ---
/data/backup/hetdex/u/bgrashey/micromamba run -p /data/backup/hetdex/u/bgrashey/envs/cnn python cube.py
