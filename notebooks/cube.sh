#!/bin/bash
#$ -N Cube
#$ -wd /data/u/bgrashey
#$ -j y
#$ -m ea
#$ -M B.Grashey@campus.lmu.de
#$ -pe smp 8
#$ -l h_vmem=30G
#$ -l h_rt=12:00:00





#export PYTHONPATH=$PYTHONPATH:/data/hetdex/u/bgrashey/notebooks/

# --- Skript ausführen ---
./micromamba run -p /data/backup/hetdex/u/bgrashey/envs/cnn python cube_cnn.py
