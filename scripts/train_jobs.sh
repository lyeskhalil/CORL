#!/bin/bash

U_SIZE=$1
V_SIZE=$2
GRAPH_FAMILY="movielense"
DIST="movielense"
MODEL="ff"
MEAN=-1
VAR=-1
LR=0.004
LR_DECAY=0.97
EXP_BETA=0.7
ENT_RATE=0.06



for i in "-1" 
do
    sbatch --account=def-khalile2 train.sh $U_SIZE $V_SIZE $i $GRAPH_FAMILY $DIST $MEAN $VAR $MODEL $LR $LR_DECAY $EXP_BETA $ENT_RATE 
done
