#!/bin/bash -l
#SBATCH -o tjob.out.%A.%a.out
#SBATCH -e tjob.err.%A.%a.err
#SBATCH --job-name=process_modelling
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-143%200
#SBATCH --distribution=pack
##SBATCH --mail-type=ALL
#SBATCH --mail-user=surabhi.nath@tuebingen.mpg.de
##SBATCH --time=165:00:00
##SBATCH -p compute

conda activate process_modelling
python runner.py --nosimulate;

exit 0