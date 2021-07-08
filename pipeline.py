import numpy as np
import os
import subprocess

# Refer to opts.py for details about the flags
# graph/dataset flags
<<<<<<< HEAD
model_type = "gnn-hist"
=======
model_type = "gnn-simp-hist"
>>>>>>> 875534792a266d37890cb8018b3f4b933fde0a0d
problem = "e-obm"
graph_family = "gmission"
weight_distribution = "gmission"
weight_distribution_param = "-1 -1"  # seperate by a space
graph_family_parameters = "-1"
u_size = 10
v_size = 3
dataset_size = 10
val_size = 1
eval_size = 1
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
def get_latest_model(m_type, u_size, v_size , problem, graph_family, weight_dist, w_dist_param, g_fam_param):
    m, v = w_dist_param.split(" ")
    u_size = 10
    v_size = 60
    dire = f"/home/alomrani/projects/def-khalile2/alomrani/output_{problem}_{graph_family}_{u_size}by{v_size}_p={g_fam_param}_{weight_dist}_m={m}_v={v}_a=3"
    list_of_files = sorted(os.listdir(dire + f"/{m_type}"), key=lambda s: int(s[8:12] + s[13:]))

    return dire + f"/{m_type}/{list_of_files[-1]}/best-model.pt"

# load_path = "../output_e-obm_er_5by15_p=0.01_uniform_m=5_v=100_a=3/e-obm_20/run_20201222T163026/epoch-69.pt"
# ../output_e-obm_er_5by15_p=0.05_uniform_m=5_v=100_a=3/e-obm_20/run_20201222T163107/epoch-69.pt \
# ../output_e-obm_er_5by15_p=0.1_uniform_m=5_v=100_a=3/e-obm_20/run_20201222T163157/epoch-69.pt \
# ../output_e-obm_er_5by15_p=0.15_uniform_m=5_v=100_a=3/e-obm_20/run_20201222T163441/epoch-69.pt \
# ../output_e-obm_er_5by15_p=0.2_uniform_m=5_v=100_a=3/outputs_e-obm_er_5by15_p=0.2_uniform_m=5_v=100_a=3/e-obm_20/run_20201222T170215/epoch-79.pt"

# this is a list of attention model checkpoints seperated by space. The number of checkpoints should be the same as the length of eval_set
# Note: checkpoints must be in the same order as eval set (i,e. checkpoint1 must be for graph paramter 0.05, etc.)

# 10by60
# attention_models = "../output_e-obm_er_10by60_p=0.05_fixed-normal_m=5_v=100_a=3/attention/run_20210414T085006/epoch-99.pt \
# ../output_e-obm_er_10by60_p=0.1_fixed-normal_m=5_v=100_a=3/attention/run_20210414T085006/epoch-99.pt \
# ../output_e-obm_er_10by60_p=0.15_fixed-normal_m=5_v=100_a=3/attention/run_20210414T085006/epoch-99.pt \
# ../output_e-obm_er_10by60_p=0.2_fixed-normal_m=5_v=100_a=3/attention/run_20210414T085006/epoch-99.pt"


# # 100by100
# attention_models = "../output_e-obm_er_100by100_p=0.05_uniform_m=5_v=100_a=3/attention/run_20210310T052217/epoch-59.pt \
# ../output_e-obm_er_100by100_p=0.1_uniform_m=5_v=100_a=3/outputs_e-obm_er_100by100_p=0.1_uniform_m=5_v=100_a=3/attention/run_20210310T052324/epoch-59.pt \
# ../output_e-obm_er_100by100_p=0.15_uniform_m=5_v=100_a=3/outputs_e-obm_er_100by100_p=0.15_uniform_m=5_v=100_a=3/attention/run_20210310T052320/epoch-59.pt \
# ../output_e-obm_er_100by100_p=0.2_uniform_m=5_v=100_a=3/outputs_e-obm_er_100by100_p=0.2_uniform_m=5_v=100_a=3/attention/run_20210310T052524/epoch-59.pt \
# "

# gMission
# 10by30
# attention_models = "../output_e-obm_gmission_10by30_p=-1_gmission_m=-1_v=-1_a=3/attention/run_20210415T024155/epoch-119.pt"
# 10by60
# attention_models = "./output_e-obm_gmission_10by60_p=-1_gmission_m=-1_v=-1_a=3/attention/run_20210415T042030/epoch-119.pt"

# 10by30
attention_models = "None"
# attention_models = "../output_e-obm_er_10by30_p=0.05_uniform_m=5_v=100_a=3/attention/run_20210310T022430/epoch-69.pt \
# ../output_e-obm_er_10by30_p=0.1_uniform_m=5_v=100_a=3/attention/run_20210310T022430/epoch-69.pt \
# ../output_e-obm_er_10by30_p=0.15_uniform_m=5_v=100_a=3/attention/run_20210310T022430/epoch-69.pt \
# ../output_e-obm_er_10by30_p=0.2_uniform_m=5_v=100_a=3/attention/run_20210310T022430/epoch-69.pt"
# gnn_hist_models = "None"
# this is a list of feedforward model checkpoints seperated by space. The number of checkpoints should be the same as the length of eval_set
# Note: checkpoints must be in the same order as eval set (i,e. checkpoint1 must be for graph paramter 0.05, etc.)
# 10by30
# ff_models = "../output_e-obm_er_10by30_p=0.05_uniform_m=5_v=100_a=3/outputs_e-obm_er_10by30_p=0.05_uniform_m=5_v=100_a=3/ff/run_20210310T083836/epoch-69.pt \
# ../output_e-obm_er_10by30_p=0.1_uniform_m=5_v=100_a=3/outputs_e-obm_er_10by30_p=0.1_uniform_m=5_v=100_a=3/ff/run_20210310T083922/epoch-69.pt \
# ../output_e-obm_er_10by30_p=0.15_uniform_m=5_v=100_a=3/outputs_e-obm_er_10by30_p=0.15_uniform_m=5_v=100_a=3/ff/run_20210310T083920/epoch-69.pt \
# ../output_e-obm_er_10by30_p=0.2_uniform_m=5_v=100_a=3/outputs_e-obm_er_10by30_p=0.2_uniform_m=5_v=100_a=3/ff/run_20210310T083920/epoch-69.pt"

# inv_ff_models = "../output_e-obm_er_10by30_p=0.05_uniform_m=5_v=100_a=3/inv-ff/run_20210421T062441/epoch-119.pt \
# ../output_e-obm_er_10by30_p=0.1_uniform_m=5_v=100_a=3/inv-ff/run_20210421T062441/epoch-119.pt \
# ../output_e-obm_er_10by30_p=0.15_uniform_m=5_v=100_a=3/inv-ff/run_20210421T063433/epoch-119.pt \
# ../output_e-obm_er_10by30_p=0.2_uniform_m=5_v=100_a=3/inv-ff/run_20210421T071049/epoch-119.pt"

# inv_ff_models = "../output_e-obm_er_10by60_p=0.1_uniform_m=5_v=100_a=3/inv-ff/run_20210421T065258/epoch-119.pt \
# ../output_e-obm_er_10by60_p=0.15_uniform_m=5_v=100_a=3/inv-ff/run_20210421T065311/epoch-119.pt \
# ../output_e-obm_er_10by60_p=0.2_uniform_m=5_v=100_a=3/inv-ff/run_20210421T065256/epoch-119.pt"
ff_supervised_models = "None"
#ff_supervised_models = get_latest_model("ff-supervised", u_size, v_size, problem, graph_family, weight_distribution, weight_distribution_param, graph_family_parameters)
# ff_supervised_models = "outputs/output_e-obm_gmission_10by60_p=-1_gmission_m=-1_v=-1_a=3/ff-supervised/run_20210601T143608/epoch-119.pt"
# ff_supervised_models = "outputs/output_e-obm_gmission_10by30_p=-1_gmission_m=-1_v=-1_a=3/ff-supervised/run_20210601T143608/epoch-119.pt"
# ff_supervised_models = "None"
#gnn_hist_models = "~/projects/def-khalile2/alomrani/output_e-obm_gmission_10by30_p=-1_gmission_m=-1_v=-1_a=3/gnn-hist/run_20210602T075444/epoch-149.pt"
#gnn_hist_models = "~/projects/def-khalile2/alomrani/output_e-obm_gmission_10by60_p=-1_gmission_m=-1_v=-1_a=3/gnn-hist/run_20210602T075439/epoch-149.pt"
gnn_hist_models = get_latest_model("gnn-hist", u_size, v_size, problem, graph_family, weight_distribution, weight_distribution_param, graph_family_parameters)
#inv_ff_models = "~/projects/def-khalile2/alomrani/output_e-obm_gmission_10by30_p=-1_gmission_m=-1_v=-1_a=3/inv-ff/run_20210602T083225/epoch-149.pt"
# inv_ff_models = "outputs/output_osbm_movielense_10by30_p=-1_movielense_m=-1_v=-1_a=3/inv-ff/run_20210602T102500/epoch-149.pt"
# gnn_hist_models = "outputs/output_e-obm_gmission-max_10by30_p=-1_gmission-max_m=-1_v=-1_a=3/gnn-hist/run_20210520T182339/epoch-119.pt"
# gnn_hist_models = "outputs/output_e-obm_gmission-max_10by60_p=-1_gmission-max_m=-1_v=-1_a=3/gnn-hist/run_20210520T182204/epoch-119.pt"
# inv_ff_models = "outputs/output_e-obm_gmission_100by100_p=-1_gmission_m=-1_v=-1_a=3/inv-ff/run_20210502T062345/epoch-119.pt"
#inv_ff_models = "~/projects/def-khalile2/alomrani/output_e-obm_gmission_10by60_p=-1_gmission_m=-1_v=-1_a=3/inv-ff/run_20210602T083223/epoch-149.pt"
inv_ff_models = get_latest_model("inv-ff", u_size, v_size, problem, graph_family, weight_distribution, weight_distribution_param, graph_family_parameters)
# inv_ff_models = "outputs/output_e-obm_gmission-var_10by30_p=-1_gmission-var_m=-1_v=-1_a=3/inv-ff/run_20210513T051323/epoch-119.pt"
# inv_ff_models = "outputs/output_e-obm_gmission-var_10by60_p=-1_gmission-var_m=-1_v=-1_a=3/inv-ff/run_20210513T051318/epoch-119.pt"
# inv_ff_models = "outputs/output_e-obm_gmission-max_10by30_p=-1_gmission-max_m=-1_v=-1_a=3/inv-ff/run_20210520T173501/epoch-119.pt"
# inv_ff_models = "outputs/output_e-obm_gmission-max_10by60_p=-1_gmission-max_m=-1_v=-1_a=3/inv-ff/run_20210520T173503/epoch-119.pt"

#inv_ff_hist_models = "~/projects/def-khalile2/alomrani/output_e-obm_gmission_10by30_p=-1_gmission_m=-1_v=-1_a=3/inv-ff-hist/run_20210602T075550/epoch-149.pt"
# inv_ff_hist_models = "outputs/output_osbm_movielense_10by30_p=-1_movielense_m=-1_v=-1_a=3/inv-ff-hist/run_20210602T102854/epoch-149.pt"
# inv_ff_hist_models = "outputs/output_e-obm_gmission-var_10by30_p=-1_gmission-var_m=-1_v=-1_a=3/inv-ff-hist/run_20210513T032250/epoch-119.pt"
# inv_ff_hist_models = "outputs/output_e-obm_gmission-var_10by60_p=-1_gmission-var_m=-1_v=-1_a=3/inv-ff-hist/run_20210513T032442/epoch-119.pt"
#inv_ff_hist_models = "~/projects/def-khalile2/alomrani/output_e-obm_gmission_10by60_p=-1_gmission_m=-1_v=-1_a=3/inv-ff-hist/run_20210602T075656/epoch-149.pt"
inv_ff_hist_models = get_latest_model("inv-ff-hist", u_size, v_size, problem, graph_family, weight_distribution, weight_distribution_param, graph_family_parameters)
# inv_ff_hist_models = "outputs/output_e-obm_gmission_100by100_p=-1_gmission_m=-1_v=-1_a=3/inv-ff-hist/run_20210502T062345/epoch-119.pt"
# inv_ff_hist_models = "outputs/output_e-obm_gmission-max_10by30_p=-1_gmission-max_m=-1_v=-1_a=3/inv-ff-hist/run_20210520T173506/epoch-119.pt"
# inv_ff_hist_models = "outputs/output_e-obm_gmission-max_10by60_p=-1_gmission-max_m=-1_v=-1_a=3/inv-ff-hist/run_20210520T173504/epoch-119.pt"

## 10by60
# ff_models = "../output_e-obm_er_10by60_p=0.05_fixed-normal_m=5_v=100_a=3/ff/run_20210414T085002/epoch-99.pt \
# ff_models = "../output_e-obm_er_10by60_p=0.1_fixed-normal_m=5_v=100_a=3/ff/run_20210414T085002/epoch-99.pt \
# ../output_e-obm_er_10by60_p=0.15_fixed-normal_m=5_v=100_a=3/ff/run_20210414T085002/epoch-99.pt \
# ../output_e-obm_er_10by60_p=0.2_fixed-normal_m=5_v=100_a=3/ff/run_20210414T085002/epoch-99.pt"
# output_e-obm_er_10by60_p=0.2_uniform_m=5_v=100_a=3/outputs_e-obm_er_10by60_p=0.2_uniform_m=5_v=100_a=3/ff/run_20210310T083907
# ff_models = "../output_e-obm_er_10by60_p=0.1_uniform_m=5_v=100_a=3/outputs_e-obm_er_10by60_p=0.1_uniform_m=5_v=100_a=3/ff/run_20210310T083914/epoch-69.pt \
#     ../output_e-obm_er_10by60_p=0.15_uniform_m=5_v=100_a=3/outputs_e-obm_er_10by60_p=0.15_uniform_m=5_v=100_a=3/ff/run_20210310T083914/epoch-69.pt \
#     ../output_e-obm_er_10by60_p=0.2_uniform_m=5_v=100_a=3/outputs_e-obm_er_10by60_p=0.2_uniform_m=5_v=100_a=3/ff/run_20210310T083907/epoch-69.pt"

# 100by100
# ff_models = "../output_e-obm_er_100by100_p=0.05_uniform_m=5_v=100_a=3/outputs_e-obm_er_100by100_p=0.05_uniform_m=5_v=100_a=3/ff/run_20210310T084054/epoch-69.pt \
# ../output_e-obm_er_100by100_p=0.1_uniform_m=5_v=100_a=3/ff/run_20210310T084054/epoch-69.pt \
# ../output_e-obm_er_100by100_p=0.15_uniform_m=5_v=100_a=3/ff/run_20210310T084103/epoch-69.pt \
# ../output_e-obm_er_100by100_p=0.2_uniform_m=5_v=100_a=3/ff/run_20210310T084210/epoch-69.pt"
ff_models = "None"
#ff_models = get_latest_model("ff", u_size, v_size, problem, graph_family, weight_distribution, weight_distribution_param, graph_family_parameters)
# gMission
# 10by30
# ff_models = "outputs/output_e-obm_gmission_10by30_p=-1_gmission_m=-1_v=-1_a=3/ff/run_20210602T083312/epoch-149.pt"
# ff_models = "outputs/output_e-obm_gmission-var_10by30_p=-1_gmission-var_m=-1_v=-1_a=3/ff/run_20210513T032052/epoch-119.pt"
# ff_models = "outputs/output_osbm_movielense_10by30_p=-1_movielense_m=-1_v=-1_a=3/ff/run_20210602T101325/epoch-149.pt"
# ff_models = "outputs/output_e-obm_gmission-var_10by60_p=-1_gmission-var_m=-1_v=-1_a=3/ff/run_20210513T044317/epoch-119.pt"

# ff_models = "outputs/output_e-obm_gmission-max_10by30_p=-1_gmission-max_m=-1_v=-1_a=3/ff/run_20210520T173459/epoch-119.pt"
# 10by60
# ff_models = "outputs/output_e-obm_gmission_10by60_p=-1_gmission_m=-1_v=-1_a=3/ff/run_20210602T083310/epoch-149.pt"
# ff_models = "outputs/output_e-obm_gmission_100by100_p=-1_gmission_m=-1_v=-1_a=3/ff/run_20210503T040359/epoch-119.pt"
# ff_models = "outputs/output_e-obm_gmission-max_10by60_p=-1_gmission-max_m=-1_v=-1_a=3/ff/run_20210520T173459/epoch-119.pt"
# ff_hist_models = "outputs/output_e-obm_gmission_10by30_p=-1_gmission_m=-1_v=-1_a=3/ff-hist/run_20210602T083223/epoch-149.pt"
# ff_hist_models = "outputs/output_osbm_movielense_10by30_p=-1_movielense_m=-1_v=-1_a=3/ff-hist/run_20210602T102601/epoch-149.pt"
# ff_hist_models = "outputs/output_e-obm_gmission-var_10by30_p=-1_gmission-var_m=-1_v=-1_a=3/ff-hist/run_20210513T032049/epoch-119.pt"
# ff_hist_models = "outputs/output_e-obm_gmission-var_10by60_p=-1_gmission-var_m=-1_v=-1_a=3/ff-hist/run_20210513T032043/epoch-119.pt"
# ff_hist_models = "outputs/output_e-obm_gmission_10by60_p=-1_gmission_m=-1_v=-1_a=3/ff-hist/run_20210602T083223/epoch-149.pt"
# ff_hist_models = "outputs/output_e-obm_gmission_100by100_p=-1_gmission_m=-1_v=-1_a=3/ff-hist/run_20210502T095427/epoch-119.pt"
# ff_hist_models = "outputs/output_e-obm_gmission-max_10by30_p=-1_gmission-max_m=-1_v=-1_a=3/ff-hist/run_20210520T173507/epoch-119.pt"
ff_hist_models = "None"
#ff_hist_models = get_latest_model("ff-hist", u_size, v_size, problem, graph_family, weight_distribution, weight_distribution_param, graph_family_parameters)
# ff_hist_models = "outputs/output_e-obm_gmission-max_10by60_p=-1_gmission-max_m=-1_v=-1_a=3/ff-hist/run_20210520T173507/epoch-119.pt"
eval_batch_size = 200


def get_latest_model(
    m_type,
    u_size,
    v_size,
    problem,
    graph_family,
    weight_dist,
    w_dist_param,
    g_fam_param,
):
    m, v = w_dist_param.split(" ")
    dir = f"outputs/output_{problem}_{graph_family}_{u_size}by{v_size}_p={g_fam_param}_{weight_dist}_m={m}_v={v}_a=3"

    list_of_files = sorted(
        os.listdir(dir + f"/{m_type}"), key=lambda s: int(s[8:12] + s[13:])
    )

    return dir + f"/{m_type}/{list_of_files[-1]}/best-model.pt"


attention_models = "None"

ff_supervised_models = get_latest_model(
    "ff-supervised",
    u_size,
    v_size,
    problem,
    graph_family,
    weight_distribution,
    weight_distribution_param,
    graph_family_parameters,
)

gnn_hist_models = get_latest_model(
    "gnn-hist",
    u_size,
    v_size,
    problem,
    graph_family,
    weight_distribution,
    weight_distribution_param,
    graph_family_parameters,
)

inv_ff_models = get_latest_model(
    "inv-ff",
    u_size,
    v_size,
    problem,
    graph_family,
    weight_distribution,
    weight_distribution_param,
    graph_family_parameters,
)

inv_ff_hist_models = get_latest_model(
    "inv-ff-hist",
    u_size,
    v_size,
    problem,
    graph_family,
    weight_distribution,
    weight_distribution_param,
    graph_family_parameters,
)

ff_models = get_latest_model(
    "ff",
    u_size,
    v_size,
    problem,
    graph_family,
    weight_distribution,
    weight_distribution_param,
    graph_family_parameters,
)

ff_hist_models = get_latest_model(
    "ff-hist",
    u_size,
    v_size,
    problem,
    graph_family,
    weight_distribution,
    weight_distribution_param,
    graph_family_parameters,
)

eval_batch_size = 100
>>>>>>> 875534792a266d37890cb8018b3f4b933fde0a0d
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
    evaluate = """python eval.py --problem {} --embedding_dim {} --load_path {} --ff_models {} --attention_models {} --inv_ff_models {} --ff_hist_models {} \
        --inv_ff_hist_models {} --gnn_hist_models {} --ff_supervised_models {} --eval_baselines {} \
        --baseline {} --eval_models {} --eval_dataset {}  --u_size {} --v_size {} --eval_set {} --eval_size {} --eval_batch_size {} \
        --n_encode_layers {} --n_heads {} --output_dir {} --dataset_size {} --batch_size {} --encoder mpnn --weight_distribution {}""".format(
        problem,
        embedding_dim,
        load_path,
        ff_models,
        attention_models,
        inv_ff_models,
        ff_hist_models,
        inv_ff_hist_models,
        gnn_hist_models,
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
    )

    # print(evaluate)
    subprocess.run(evaluate, shell=True)


if __name__ == "__main__":
    # make the directories if they do not exist
    make_dir()
#    generate_data()
#    train_model()
    evaluate_model()
