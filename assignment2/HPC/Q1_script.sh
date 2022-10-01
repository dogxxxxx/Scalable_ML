#!/bin/bash
#$ -l h_rt=6:00:00  #time needed
#$ -pe smp 4 #number of cores
#$ -l rmem=8G #number of memery
#$ -o /data/acp21kc/ScalableML/Assignment2/Output/Q1_output.txt  #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M acp21kc@sheffield.ac.uk #Notify you by email, remove this line if you don't like
#$ -m ea #Email you when it finished or aborted
#$ -cwd # Run job from current directory

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

spark-submit --driver-memory 8g --executor-memory 8g /data/acp21kc/ScalableML/Assignment2/Code/Q1_code.py
