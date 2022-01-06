#!/bin/bash
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=10:00:00
#SBATCH --output=%N-%j.out

U_SIZE=$1
V_SIZE=$2
GRAPH_FAMILY=$3
FAMILY_PARAMETER=$5
PROBLEM=$4
TRAIN_SIZE=20000
VAL_SIZE=1000
EVAL_SIZE=2000
MAX_WEIGHT=100
WEIGHT_DIST=$6
MEAN=$7
VARIANCE=$8
a=3
DATASET="$PROBLEM"_"$GRAPH_FAMILY"_"$U_SIZE"by"$V_SIZE"_"p=$FAMILY_PARAMETER"_"$WEIGHT_DIST"_"m=$MEAN"_"v=$VARIANCE"_"a=$a"

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


# python data/generate_data.py --dataset_size $TRAIN_SIZE --seed 50000 --u_size $U_SIZE --v_size $V_SIZE --graph_family_parameter $FAMILY_PARAMETER  --dataset_folder $SLURM_TMPDIR/$DATASET/train
python data/generate_data.py --weight_distribution_param $MEAN $VARIANCE --weight_distribution $WEIGHT_DIST --graph_family $GRAPH_FAMILY --problem $PROBLEM --dataset_size $TRAIN_SIZE --u_size $U_SIZE --v_size $V_SIZE --graph_family_parameter $FAMILY_PARAMETER --dataset_folder $SLURM_TMPDIR/$DATASET/train

python data/generate_data.py --problem $PROBLEM --weight_distribution_param $MEAN $VARIANCE --weight_distribution $WEIGHT_DIST --graph_family $GRAPH_FAMILY --dataset_size $VAL_SIZE --u_size $U_SIZE --v_size $V_SIZE --graph_family_parameter $FAMILY_PARAMETER --seed 20000 --dataset_folder $SLURM_TMPDIR/$DATASET/val

python data/generate_data.py --problem $PROBLEM --weight_distribution_param $MEAN $VARIANCE --weight_distribution $WEIGHT_DIST --graph_family $GRAPH_FAMILY --dataset_size $EVAL_SIZE --u_size $U_SIZE --v_size $V_SIZE --graph_family_parameter $FAMILY_PARAMETER --seed 40000 --dataset_folder $SLURM_TMPDIR/$DATASET/eval

cd $SLURM_TMPDIR

tar cf ~/projects/def-khalile2/alomrani/$DATASET.tar $DATASET
