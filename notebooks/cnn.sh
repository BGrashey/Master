#!/bin/bash
#$ -N Training
#$ -wd /data/u/bgrashey
#$ -j y
#$ -m ea
#$ -M B.Grashey@campus.lmu.de
#$ -pe smp 4
#$ -l h_vmem=40G
#$ -l h_rt=02:00:00





#export PYTHONPATH=$PYTHONPATH:/data/hetdex/u/bgrashey/notebooks/

# --- Skript ausführen ---
./micromamba run -p /data/backup/hetdex/u/bgrashey/envs/cnn python training.py
