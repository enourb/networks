#!/bin/bash
#SBATCH -A e32800
#SBATCH -p short
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH --job-name=concentration_job
#SBATCH --output=logs/concentration_%j.out
#SBATCH --error=logs/concentration_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ethan.nourbash@northwestern.edu

# Load modules
module load python-miniconda3/4.12.0

# Activate environment
source activate networks_env

# Navigate to project directory
cd ~/networks

# Convert notebook to Python script
echo "Converting notebook to Python script..."
jupyter nbconvert --to script code/concentration.ipynb

# Run the script
echo "Running concentration.py..."
python code/concentration.py

echo "Job completed at $(date)"

