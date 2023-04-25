#!/bin/bash -x

COMPUTE_NODES=$1
RANKS_PER_NODE=$2
sMult=$3
dataSizeValue=$4
DIR="./"

sbatch -p el8-rpi -N $COMPUTE_NODES --gres=gpu:4 --mail-type=ALL --mail-user=fespadearocks@gmail.com -t 5 -D $DIR -o $DIR/proj$1-$2-$3-$4.stdout -e $DIR/proj$1-$2-$3-$4.stderr $DIR/run_job.sh $RANKS_PER_NODE $sMult $dataSizeValue