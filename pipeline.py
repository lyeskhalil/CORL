import numpy as np
import os
import subprocess

# Refer to opts.py for details about the flags
# graph/dataset flags
model_type = "ff-supervised"
problem = "e-obm"
graph_family = "gmission"
weight_distribution = "gmission"
weight_distribution_param = "-1 -1"  # seperate by a space
graph_family_parameters = "-1"

u_size = 10
v_size = 30
dataset_size = 1000
val_size = 100
eval_size = 2000
extention = "/{}_{}_{}_{}_{}by{}".format(
    problem,
    graph_family,
    weight_distribution,
    weight_distribution_param,
    u_size,
    v_size,
).replace(" ", "")

train_dataset = "dataset/train" + extention

val_dataset = "dataset/val" + extention

eval_dataset = "dataset/eval" + extention

# model flags
batch_size = 1
eval_batch_size = 100
embedding_dim = 30  # 60
n_heads = 1  # 3
n_epochs = 10
checkpoint_epochs = 0
eval_baselines = "greedy"  # ******
lr_model = 0.001
lr_decay = 0.99
beta_decay = 0.7
ent_rate = 0
n_encode_layers = 1
baseline = "exponential"
# directory io flags
output_dir = "saved_models"
log_dir = "logs_dataset"

# model evaluation flags
eval_models = "inv-ff ff ff-hist ff-supervised inv-ff-hist gnn-hist"
# TODO: ADD MODELS TO ABOVE
eval_output = "figures"
# this is a single checkpoint. Example: outputs_dataset/e-obm_20/run_20201226T171156/epoch-4.pt
load_path = None
test_transfer = True


def get_latest_model(
    m_type,
    u_size,
    v_size,
    problem,
    graph_family,
    weight_dist,
    w_dist_param,
    g_fam_param,
    eval_models,
):
    if m_type not in eval_models:
        return "None"
    m, v = w_dist_param.split(" ")
    dir = f"outputs/output_{problem}_{graph_family}_{u_size}by{v_size}_p={g_fam_param}_{weight_dist}_m={m}_v={v}_a=3"

    list_of_files = sorted(
        os.listdir(dir + f"/{m_type}"), key=lambda s: int(s[8:12] + s[13:])
    )

    return dir + f"/{m_type}/{list_of_files[-1]}/best-model.pt"


arg = [
    u_size,
    v_size,
    problem,
    graph_family,
    weight_distribution,
    weight_distribution_param,
    graph_family_parameters,
    eval_models.split(" "),
]
attention_models = get_latest_model("attention", *arg)

ff_supervised_models = get_latest_model("ff-supervised", *arg)

gnn_hist_models = get_latest_model("gnn-hist", *arg)

gnn_models = get_latest_model("gnn", *arg)

gnn_simp_hist_models = get_latest_model("gnn-simp-hist", *arg)

inv_ff_models = get_latest_model("inv-ff", *arg)

inv_ff_hist_models = get_latest_model("inv-ff-hist", *arg)

ff_models = get_latest_model("ff", *arg)

ff_hist_models = get_latest_model("ff-hist", *arg)

eval_set = graph_family_parameters


def make_dir():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(eval_output):
        os.makedirs(eval_output)

    if not os.path.exists("data"):
        os.makedirs("data")

    if not os.path.exists("data/train"):
        os.makedirs("data/train")

    if not os.path.exists("data/val"):
        os.makedirs("data/val")

    if not os.path.exists("data/eval"):
        os.makedirs("data/eval")


def generate_data():
    for n in graph_family_parameters.split(" "):
        # the naming convention here should not be changed!
        train_dir = train_dataset + "/parameter_{}".format(n)
        val_dir = val_dataset + "/parameter_{}".format(n)
        eval_dir = eval_dataset + "/parameter_{}".format(n)

        generate_train = """python data/generate_data.py --problem {} --dataset_size {} --dataset_folder {} \
                            --u_size {} --v_size {} --graph_family {} --weight_distribution {} \
                            --weight_distribution_param {} --graph_family_parameter {}""".format(
            problem,
            dataset_size,
            train_dir,
            u_size,
            v_size,
            graph_family,
            weight_distribution,
            weight_distribution_param,
            n,
        )

        generate_val = """python data/generate_data.py --problem {} --dataset_size {} --dataset_folder {}  \
                            --u_size {} --v_size {} --graph_family {} --weight_distribution {} \
                            --weight_distribution_param {} --graph_family_parameter {} --seed 20000""".format(
            problem,
            val_size,
            val_dir,
            u_size,
            v_size,
            graph_family,
            weight_distribution,
            weight_distribution_param,
            n,
        )

        generate_eval = """python data/generate_data.py --problem {} --dataset_size {} --dataset_folder {} \
                            --u_size {} --v_size {} --graph_family {} --weight_distribution {} \
                            --weight_distribution_param {} --graph_family_parameter {} --seed 40000""".format(
            problem,
            eval_size,
            eval_dir,
            u_size,
            v_size,
            graph_family,
            weight_distribution,
            weight_distribution_param,
            n,
        )

        # print(generate_train)
        # os.system(generate_train)
        subprocess.run(generate_train, shell=True)

        # print(generate_val)
        # os.system(generate_val)
        subprocess.run(generate_val, shell=True)

        # print(generate_eval)
        # os.system(generate_eval)
        subprocess.run(generate_eval, shell=True)


def train_model():
    for n in graph_family_parameters.split(" "):
        # the naming convention here should not be changed!
        train_dir = train_dataset + "/parameter_{}".format(n)
        val_dir = val_dataset + "/parameter_{}".format(n)
        save_dir = output_dir + extention + "/parameter_{}".format(n)
        train = """python run.py --encoder mpnn --model {} --problem {} --batch_size {} --embedding_dim {} --n_heads {} --u_size {}  --v_size {} --n_epochs {} \
                    --train_dataset {} --val_dataset {} --dataset_size {} --val_size {} --checkpoint_epochs {} --baseline {} \
                    --lr_model {} --lr_decay {} --output_dir {} --log_dir {} --n_encode_layers {} --save_dir {} --graph_family_parameter {} --exp_beta {} --ent_rate {}""".format(
            model_type,
            problem,
            batch_size,
            embedding_dim,
            n_heads,
            u_size,
            v_size,
            n_epochs,
            train_dir,
            val_dir,
            dataset_size,
            val_size,
            checkpoint_epochs,
            baseline,
            lr_model,
            lr_decay,
            output_dir,
            log_dir,
            n_encode_layers,
            save_dir,
            n,
            beta_decay,
            ent_rate,
        )

        # print(train)
        subprocess.run(train, shell=True)


def evaluate_model():
    evaluate = """python eval.py --problem {} --graph_family {} --embedding_dim {} --load_path {} --ff_models {} --attention_models {} --inv_ff_models {} --ff_hist_models {} \
        --inv_ff_hist_models {} --gnn_hist_models {} --gnn_models {} --gnn_simp_hist_models {} --ff_supervised_models {} --eval_baselines {} \
        --baseline {} --eval_models {} --eval_dataset {}  --u_size {} --v_size {} --eval_set {} --eval_size {} --eval_batch_size {} \
        --n_encode_layers {} --n_heads {} --output_dir {} --dataset_size {} --batch_size {} --encoder mpnn --weight_distribution {} --weight_distribution_param {}""".format(
        problem,
        graph_family,
        embedding_dim,
        load_path,
        ff_models,
        attention_models,
        inv_ff_models,
        ff_hist_models,
        inv_ff_hist_models,
        gnn_hist_models,
        gnn_models,
        gnn_simp_hist_models,
        ff_supervised_models,
        eval_baselines,
        baseline,
        eval_models,
        eval_dataset,
        u_size,
        v_size,
        eval_set,
        eval_size,
        eval_batch_size,
        n_encode_layers,
        n_heads,
        output_dir,
        eval_size,
        eval_batch_size,
        weight_distribution,
        weight_distribution_param,
    )
    if test_transfer:
        evaluate += " --test_transfer"
    # print(evaluate)
    subprocess.run(evaluate, shell=True)


if __name__ == "__main__":
    # make the directories if they do not exist
    make_dir()
    # generate_data()
    # train_model()
    evaluate_model()
