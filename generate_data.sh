U_SIZE=10
V_SIZE=10
GRAPH_FAMILY="er"
FAMILY_PARAMETER=1.0
PROBLEM="e-obm"
DATASET="$PROBLEM"_"$GRAPH_FAMILY"_"$U_SIZE"by"$V_SIZE"_"$FAMILY_PARAMETER"
TRAIN_SIZE=1000
VAL_SIZE=10
MAX_WEIGHT=100
module load python/3.6
module load scipy-stack

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r requirements.txt

# python data/generate_data.py --dataset_size $TRAIN_SIZE --seed 50000 --u_size $U_SIZE --v_size $V_SIZE --graph_family_parameter $FAMILY_PARAMETER  --dataset_folder $SLURM_TMPDIR/$DATASET/train
python data/generate_data.py --max_weight $MAX_WEIGHT --problem $PROBLEM --dataset_size $TRAIN_SIZE --u_size $U_SIZE --v_size $V_SIZE --graph_family_parameter $FAMILY_PARAMETER --dataset_folder $SLURM_TMPDIR/$DATASET/train

python data/generate_data.py --problem $PROBLEM --max_weight $MAX_WEIGHT --dataset_size $VAL_SIZE --u_size $U_SIZE --v_size $V_SIZE --graph_family_parameter $FAMILY_PARAMETER --seed 20000 --dataset_folder $SLURM_TMPDIR/$DATASET/val
