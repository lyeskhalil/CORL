#!/bin/bash

U_SIZE=$1
V_SIZE=$2
GRAPH_FAMILY="gmission"
DIST="gmission"
MODEL="inv-ff"
MEAN=-1
VAR=-1
LR=0.02
LR_DECAY=0.96
EXP_BETA=0.75



for i in "-1" 
do
    sbatch --account=def-khalile2 train.sh $U_SIZE $V_SIZE $i $GRAPH_FAMILY $DIST $MEAN $VAR $MODEL $LR $LR_DECAY $EXP_BETA 
done
