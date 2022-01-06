#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=192:00:00
#SBATCH --output=%N-%j.out

U_SIZE=$1
V_SIZE=$2
GRAPH_FAMILY=$4
PROBLEM=${13}
FAMILY_PARAMETER=$3
TRAIN_SIZE=20000
VAL_SIZE=1000
EMBEDDING_SIZE=30
MAX_WEIGHT=100
WEIGHT_DIST=$5
MEAN=$6
VARIANCE=$7
a=3
DATASET="$PROBLEM"_"$GRAPH_FAMILY"_"$U_SIZE"by"$V_SIZE"_"p=$FAMILY_PARAMETER"_"$WEIGHT_DIST"_"m=$MEAN"_"v=$VARIANCE"_"a=$a"
MODEL=$8


module load python/3.7
module load scipy-stack
module load gurobi


virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
cd $EBROOTGUROBI
python setup.py build --build-base /tmp/${USER} install

cd ~/scratch/corl/
pip install --no-index --upgrade pip

pip install --no-index -r requirements.txt


# Prepare data
tar xf ~/projects/def-khalile2/alomrani/$DATASET.tar -C $SLURM_TMPDIR/
mkdir $SLURM_TMPDIR/logs_$DATASET

python run.py --problem $PROBLEM --encoder mpnn --batch_size ${15} --eval_batch_size ${15} --embedding_dim $EMBEDDING_SIZE --n_heads 1 --u_size $U_SIZE --v_size $V_SIZE --n_epochs 300 --train_dataset $SLURM_TMPDIR/$DATASET/train --val_dataset $SLURM_TMPDIR/$DATASET/val --dataset_size $TRAIN_SIZE --val_size $VAL_SIZE --checkpoint_epochs 10 --baseline exponential --exp_beta ${11} --lr_model $9 --lr_decay ${10} --ent_rate ${12} --output_dir $SLURM_TMPDIR/output_$DATASET --log_dir $SLURM_TMPDIR/logs_$DATASET --max_grad_norm 1.0 --n_encode_layers ${14} --model $MODEL

cp -r $SLURM_TMPDIR/output_$DATASET ~/projects/def-khalile2/alomrani/
cp -r $SLURM_TMPDIR/logs_$DATASET ~/projects/def-khalile2/alomrani/
