#!/bin/bash
#SBATCH -A e32800
#SBATCH --qos=normal
#SBATCH -p short
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH --job-name=concentration_job
#SBATCH --output=../logs/concentration_%j.out
#SBATCH --error=../logs/concentration_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ethan.nourbash@northwestern.edu

# Load modules
module purge
module load python-miniconda3/4.12.0

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Activate environment
conda activate networks_env

# Navigate to CODE directory (not project root)
cd ~/networks/code

# Convert notebook to Python script
echo "Converting notebook to Python script..."
jupyter nbconvert --to script concentration.ipynb

# Run the script FROM the code directory
echo "Running concentration.py..."
python concentration.py

echo "Job completed at $(date)"
