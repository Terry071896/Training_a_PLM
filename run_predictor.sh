#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1 
#SBATCH --time=8:00:00
#SBATCH --mem=40GB
#SBATCH --mail-user=terry.cox@utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o assignment_1-%j
#SBATCH --export=ALL

source /uufs/chpc.utah.edu/common/home/u1427155/miniconda3/etc/profile.d/conda.sh
conda activate train_plm

mkdir -p /scratch/general/vast/u1427155/huggingface_cache
export TRANSFORMER_CACHE="/scratch/general/vast/u1427155/huggingface_cache"
export HF_DATASETS_CACHE="/scratch/general/vast/u1427155/huggingface_cache"

python src/run_model.py 