#!/bin/bash
# FILE: wrapper
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q all.q
#$ -N PPS_Rob2
#$ -M mv19977@essex.ac.uk
#$ -m be
#$ -pe smp 10
#$ -o $HOME/PPS/Robot/out_Robot2.txt


python36 ./1-Main.py