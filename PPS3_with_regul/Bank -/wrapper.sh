#!/bin/bash
# FILE: wrapper
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q all.q
#$ -N PPS_Bank4
#$ -M mv19977@essex.ac.uk
#$ -m be
#$ -pe smp 10
#$ -o $HOME/PPS/Bank/out_Bank4.txt


python36 ./1-Main.py