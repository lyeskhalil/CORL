#!/bin/bash

U_SIZE=$1
V_SIZE=$2
GRAPH_FAMILY="gmission"
DIST="gmission"
MODEL="inv-ff-hist"
MEAN=-1
VAR=-1
LR=0.02
LR_DECAY=0.98
EXP_BETA=0.65
ENT_RATE=0.02

for i in "-1" 
do
    sbatch --account=def-khalile2 generate_dataset.sh $U_SIZE $V_SIZE $i 
    #sbatch --account=def-khalile2 train.sh $U_SIZE $V_SIZE $i $GRAPH_FAMILY $DIST $MEAN $VAR $MODEL $LR $LR_DECAY $EXP_BETA $ENT_RATE 
done
