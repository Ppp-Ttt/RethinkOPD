#!/bin/bash
#SBATCH --job-name=install_flash_attn
#SBATCH --output=logs/install_flash_attn_%j.log
#SBATCH --error=logs/install_flash_attn_%j.log
#SBATCH --account=test
#SBATCH --partition=TEST1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --nodes=1

set -x

nvcc -V
echo "Python: $(/mmu_cd_ssd/pengtiantian/envs/ptt_verl_opd/bin/python --version)"

MAX_JOBS=8 /mmu_cd_ssd/pengtiantian/envs/ptt_verl_opd/bin/pip install flash-attn==2.8.1 --no-build-isolation

echo "=== Verifying installation ==="
/mmu_cd_ssd/pengtiantian/envs/ptt_verl_opd/bin/python -c "import flash_attn; print('flash_attn version:', flash_attn.__version__)"
