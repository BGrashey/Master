#!/bin/bash
#$ -N Grashey
#$ -wd /data/hetdex/u/bgrashey/notebooks/
#$ -j y
#$ -m ea
#$ -M B.Grashey@campus.lmu.de
#$ -pe smp 1
#$ -l h_vmem=120G
#$ -l h_rt=08:00:00





export PYTHONPATH=$PYTHONPATH:/data/hetdex/u/bgrashey/notebooks/

# --- Skript ausführen ---
/data/u/bgrashey/micromamba run -p /data/backup/hetdex/u/bgrashey/envs/cube_aktuell python /data/u/bgrashey/ew.py --catalog /data/hetdex/u/bgrashey/data_/fof_run_3_5_cnnscored.fits --ra ra --dec dec --z z 
