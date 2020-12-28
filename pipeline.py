# !/usr/bin/env python
import os

# Refer to opts.py for details about the flags
# graph/dataset flags
problem = 'e-obm'
graph_family = 'er'
weight_distribution = 'uniform'
weight_distribution_param = [5, 4000]
graph_family_parameters = [0.05, 0.1, 0.2, 0.5]
u_size = 10
v_size = 10
dataset_size = 200
val_size = 100
eval_size = 100
num_edges = 50
train_dataset = 'dataset/train' + "/{}_{}_{}_{}_{}by{}_results".format(
    problem, graph_family,
    weight_distribution, weight_distribution_param,
    u_size,
    v_size,
).replace(" ", "")

val_dataset = 'dataset/val' + "/{}_{}_{}_{}_{}by{}_results".format(
    problem, graph_family,
    weight_distribution, weight_distribution_param,
    u_size, v_size
).replace(" ", "")

eval_dataset = 'dataset/eval' + "/{}_{}_{}_{}_{}by{}_results".format(
    problem, graph_family,
    weight_distribution,
    weight_distribution_param,
    u_size,
    v_size,
).replace(" ", "")


# model flags
batch_size = 10
embedding_dim = 16
n_heads = 1
n_epochs = 10
checkpoint_epochs = 5
baselines = ['greedy']   # ******
lr_model = 0.001
lr_decay = 0.9
n_encode_layers = 4

# directory io flags
output_dir = 'figures'
log_dir = 'logs_dataset'

# model evaluation flags
eval_models = 'attention'
# this is the checkpoint. Example: outputs_dataset/e-obm_20/run_20201226T171156/epoch-4.pt
load_path = 'outputs_dataset/e-obm_20/run_20201226T171156/epoch-4.pt'
eval_batch_size = 10
eval_set = graph_family_parameters
eval_baselines = baselines


def make_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists('data'):
        os.makedirs('data')

    if not os.path.exists('data/train'):
        os.makedirs('data/train')

    if not os.path.exists('data/val'):
        os.makedirs('data/val')

    if not os.path.exists('data/eval'):
        os.makedirs('data/eval')


def generate_data():
    for n in graph_family_parameters:
        generate_train = """python data/generate_data.py --problem {} --dataset_size {} --dataset_folder {}
                            --u_size {} --v_size {} --graph_family {} --num_edges {} --weight_distribution {}
                            --weight_distribution_param {} --graph_family_parameter {}""".format(
            problem, dataset_size, train_dataset, u_size, v_size,
            graph_family, num_edges, weight_distribution,
            weight_distribution_param, n)

        generate_val = """python data/generate_data.py --problem {} --dataset_size {} --dataset_folder {}
                            --u_size {} --v_size {} --graph_family {} --num_edges {} --weight_distribution {}
                            --weight_distribution_param {} --graph_family_parameter {}""".format(
            problem, val_size, val_dataset, u_size, v_size,
            graph_family, num_edges, weight_distribution,
            weight_distribution_param, n)

        generate_eval = """python data/generate_data.py --problem {} --dataset_size {} --dataset_folder {}
                            --u_size {} --v_size {} --graph_family {} --num_edges {} --weight_distribution {}
                            --weight_distribution_param {} --graph_family_parameter {}""".format(
            problem, eval_size, eval_dataset, u_size, v_size,
            graph_family, num_edges, weight_distribution,
            weight_distribution_param, n)

        print(generate_train)
        print(generate_val)
        print(generate_eval)
        # os.system(generate_train)
        # os.system(generate_val)
        # os.system(generate_eval)


def train_model():
    train = """python run.py --problem {} --batch_size {} --embedding_dim {} --n_heads {} --u_size {}  --v_size {} --n_epochs {}
        --train_dataset {} --val_dataset {} --dataset_size {} --val_size {} --checkpoint_epochs {} --baseline {} --lr_model {}
        --lr_decay {} --output_dir {} --log_dir {} --n_encode_layers {}""".format(problem, batch_size, embedding_dim, n_heads, u_size, v_size, n_epochs, train_dataset, val_dataset, dataset_size, val_size, checkpoint_epochs, baselines, lr_model, lr_decay, output_dir, log_dir, n_encode_layers)

    print(train)
    # os.system(train)


def evaluate_model():
    evaluate = """python eval.py --problem {} --load_path {} --eval_baselines {} --eval_models  {} --eval_dataset {}
        --u_size {} --v_size {} --eval_set {} --eval_size {} --eval_batch_size {} --n_encode_layers {} --n_heads {}""".format(problem, load_path, eval_baselines, eval_models, eval_dataset, u_size, v_size, eval_set, eval_size, eval_batch_size, n_encode_layers, n_heads)

    print(evaluate)
    # os.system(evaluate)


if __name__ == "__main__":
    # make the directories if they do not exist
    make_dir(output_dir)
    generate_data()
    train_model()
    evaluate_model()
