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
/data/u/bgrashey/micromamba run -p /data/backup/hetdex/u/bgrashey/envs/cube_aktuell python /data/u/bgrashey/ew.py --catalog /data/hetdex/u/bgrashey/data_/combined_manual_vdfi_rf_cnn_scored.fits --ra ra_vdfi --dec dec_vdfi --z z_vdfi 
