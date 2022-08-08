#!/bin/bash
# FILE: wrapper
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q all.q
#$ -N PPS_Sate1
#$ -M mv19977@essex.ac.uk
#$ -m be
#$ -pe smp 10
#$ -o $HOME/PPS/Satellite/out_Satellite1.txt


python36 ./1-Main.py