# !/usr/bin/env python
import os

# Refer to opts.py for details about the flags
# graph/dataset flags
problem = "e-obm"
graph_family = "er"
weight_distribution = "uniform"
weight_distribution_param = [5, 100]
graph_family_parameters = [0.01, 0.05, 0.15, 0.1, 0.2]
u_size = 10
v_size = 30
dataset_size = 1000
val_size = 100
eval_size = 1000
train_dataset = "dataset/train" + "/{}_{}_{}_{}_{}by{}_results".format(
    problem,
    graph_family,
    weight_distribution,
    weight_distribution_param,
    u_size,
    v_size,
).replace(" ", "")

val_dataset = "dataset/val" + "/{}_{}_{}_{}_{}by{}_results".format(
    problem,
    graph_family,
    weight_distribution,
    weight_distribution_param,
    u_size,
    v_size,
).replace(" ", "")

eval_dataset = "dataset/eval" + "/{}_{}_{}_{}_{}by{}_results".format(
    problem,
    graph_family,
    weight_distribution,
    weight_distribution_param,
    u_size,
    v_size,
).replace(" ", "")


# model flags
batch_size = 50
embedding_dim = 60
n_heads = 3
n_epochs = 10
checkpoint_epochs = 5
baselines = ["greedy"]  # ******
lr_model = 0.001
lr_decay = 0.9
n_encode_layers = 3

# directory io flags
output_dir = "figures"
log_dir = "logs_dataset"

# model evaluation flags
eval_models = "attention attention attention attention attention"
# this is the checkpoint. Example: outputs_dataset/e-obm_20/run_20201226T171156/epoch-4.pt
load_path = "../output_e-obm_er_10by30_p=0.15_uniform_m=5_v=100_a=3/e-obm_20/run_20201223T063254/epoch-79.pt ../output_e-obm_er_10by30_p=0.01_uniform_m=5_v=100_a=3/e-obm_20/run_20201223T060349/epoch-79.pt ../output_e-obm_er_10by30_p=0.05_uniform_m=5_v=100_a=3/e-obm_20/run_20201223T060338/epoch-79.pt ../output_e-obm_er_10by30_p=0.1_uniform_m=5_v=100_a=3/e-obm_20/run_20201223T062920/epoch-79.pt ../output_e-obm_er_10by30_p=0.2_uniform_m=5_v=100_a=3/e-obm_20/run_20201223T063830/epoch-79.pt"
eval_batch_size = 50
eval_set = graph_family_parameters
eval_baselines = baselines


def make_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists("data"):
        os.makedirs("data")

    if not os.path.exists("data/train"):
        os.makedirs("data/train")

    if not os.path.exists("data/val"):
        os.makedirs("data/val")

    if not os.path.exists("data/eval"):
        os.makedirs("data/eval")


def generate_data():
    for n in graph_family_parameters:
        generate_train = """python data/generate_data.py --problem {} --dataset_size {} --dataset_folder {}
                            --u_size {} --v_size {} --graph_family {} --weight_distribution {}
                            --weight_distribution_param {} --graph_family_parameter {}""".format(
            problem,
            dataset_size,
            train_dataset,
            u_size,
            v_size,
            graph_family,
            weight_distribution,
            weight_distribution_param,
            n,
        )

        generate_val = """python data/generate_data.py --problem {} --dataset_size {} --dataset_folder {}
                            --u_size {} --v_size {} --graph_family {} --weight_distribution {}
                            --weight_distribution_param {} --graph_family_parameter {} --seed 20000""".format(
            problem,
            val_size,
            val_dataset,
            u_size,
            v_size,
            graph_family,
            weight_distribution,
            weight_distribution_param,
            n,
        )

        generate_eval = """python data/generate_data.py --problem {} --dataset_size {} --dataset_folder {} --u_size {} --v_size {} --graph_family {} --weight_distribution {} --graph_family_parameter {} --weight_distribution_param {} {} --seed 40000""".format(
            problem,
            eval_size,
            eval_dataset,
            u_size,
            v_size,
            graph_family,
            weight_distribution,
            n,
            weight_distribution_param[0],
            weight_distribution_param[1],
        )

        print(generate_train)
        print(generate_val)
        print(generate_eval)
        # os.system(generate_train)
        # os.system(generate_val)
        os.system(generate_eval)


def train_model():
    train = """python run.py --problem {} --batch_size {} --embedding_dim {} --n_heads {} --u_size {}  --v_size {} --n_epochs {} --train_dataset {} --val_dataset {} --dataset_size {} --val_size {} --checkpoint_epochs {} --baseline {} --lr_model {} --lr_decay {} --output_dir {} --log_dir {} --n_encode_layers {}""".format(
        problem,
        batch_size,
        embedding_dim,
        n_heads,
        u_size,
        v_size,
        n_epochs,
        train_dataset,
        val_dataset,
        dataset_size,
        val_size,
        checkpoint_epochs,
        baselines,
        lr_model,
        lr_decay,
        output_dir,
        log_dir,
        n_encode_layers,
    )

    print(train)

    # os.system(train)


def evaluate_model():
    evaluate = """python eval.py --problem {} --embedding_dim {} --load_path {} --eval_baselines {} --eval_models  {} --eval_dataset {} --u_size {} --v_size {} --eval_set {} --eval_size {} --eval_batch_size {} --n_encode_layers {} --n_heads {}""".format(
        problem,
        embedding_dim,
        load_path,
        eval_baselines[0],
        eval_models,
        eval_dataset,
        u_size,
        v_size,
        eval_set,
        eval_size,
        eval_batch_size,
        n_encode_layers,
        n_heads,
    )

    # print(evaluate)
    os.system(evaluate)


if __name__ == "__main__":
    # make the directories if they do not exist
    make_dir(output_dir)
    generate_data()
    # train_model()
    evaluate_model()
