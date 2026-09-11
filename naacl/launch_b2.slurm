#!/bin/bash
#SBATCH --job-name=gl_opt_b2
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=360G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/opt_b2_%j.out
#SBATCH --error=logs/opt_b2_%j.err

set -euo pipefail
exec bash "${SLURM_SUBMIT_DIR:?}/naacl/launch_stage.sh" b2
