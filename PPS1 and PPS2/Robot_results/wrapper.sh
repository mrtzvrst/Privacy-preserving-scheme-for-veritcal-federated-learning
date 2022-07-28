#!/bin/bash
# FILE: wrapper
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q all.q
#$ -N Robot_PPS2
#$ -M mv19977@essex.ac.uk
#$ -m be
#$ -pe smp 10
#$ -o $HOME/PPS/PPS_Robot/output_Robot_PPS2.txt


python36 ./1-Main.py