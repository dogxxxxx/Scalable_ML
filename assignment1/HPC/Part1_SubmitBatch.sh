#!/bin/bash
#$ -l h_rt=6:00:00  #time needed
#$ -pe smp 10 #number of cores
#$ -l rmem=8G #number of memery
#$ -o /data/acp21kc/ScalableML/Assignment1/Output/output.txt  #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M acp21kc@sheffield.ac.uk #Notify you by email, remove this line if you don't like
#$ -m ea #Email you when it finished or aborted
#$ -cwd # Run job from current directory

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

spark-submit --driver-memory 8g --executor-memory 8g --master local[5] /data/acp21kc/ScalableML/Assignment1/Code/Assignment.py
spark-submit --driver-memory 8g --executor-memory 8g --master local[10] /data/acp21kc/ScalableML/Assignment1/Code/Assignment.py