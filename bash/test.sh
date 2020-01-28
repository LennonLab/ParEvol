#!/bin/sh
#PBS -k o
#PBS -l nodes=1:ppn=8,vmem=20gb,walltime=24:00:00
#PBS -M wrshoema@iu.edu
#PBS -m e
#PBS -m n
# the -j flag isn't acceptd on carbonate any more I think
#    PBS -j oe

# the location of the conda environment
# ~/.conda/envs/ParEvol/bin/python

module unload python/2.7.16
module load anaconda/python3.6/4.3.1
source activate ParEvol

# create asa159 Pyhon file to import
# f2py -c -m asa159 asa159.f90

# call that code from this file
python /N/dc2/projects/Lennon_Sequences/ParEvol/Python/rndm_sample_tenaillon_carbonate.py 25
