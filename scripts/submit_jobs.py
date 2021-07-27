import subprocess
import sys
import time

if len(sys.argv) < 5:
    raise "Must input u_size, v_size, dataset, models to train"

#subprocess.run("source ~/env/bin/activate", shell=True)

u_size = int(sys.argv[3])
v_size = int(sys.argv[4])
dataset = sys.argv[2]
model = sys.argv[5]
n_encoding_layers = sys.argv[-2]
problem = sys.argv[1]
mode = sys.argv[-1]
batch_size = 200
num_per_agent = 2
print(n_encoding_layers, model)
problems = ["e-obm"]
g_sizes = [(10, 30), (10, 60)]
datasets = ["gmission", "gmission-var", "er"]

param = {"e-obm":{}, "osbm":{}}   # Contains best hyperparameter for each model

param["e-obm"]["ff"] = {"lr": 0.008, "lr_decay": 0.96, "exp_beta": 0.75, "ent_rate": 0.05}
param["e-obm"]["inv-ff"] = {"lr": 0.02, "lr_decay": 0.98, "exp_beta": 0.7, "ent_rate": 0.003}
param["e-obm"]["ff-hist"] = {"lr": 0.003, "lr_decay": 0.98, "exp_beta": 0.75, "ent_rate": 0.03}
param["e-obm"]["ff-supervised"] = {"lr": 0.0006, "lr_decay": 0.99, "exp_beta": 0.0, "ent_rate": 0.0}
param["e-obm"]["inv-ff-hist"] = {"lr": 0.006, "lr_decay": 0.97, "exp_beta": 0.8, "ent_rate": 0.0006}
param["e-obm"]["gnn-hist"] = {"lr": 0.002, "lr_decay": 0.99, "exp_beta": 0.95, "ent_rate": 0.05}
param["e-obm"]["gnn"] = {"lr": 0.002, "lr_decay": 0.97, "exp_beta": 0.8, "ent_rate": 0.05}
param["e-obm"]["gnn-simp-hist"] = {"lr": 0.002, "lr_decay": 0.97, "exp_beta": 0.8, "ent_rate": 0.05}

param["osbm"]["ff"] = {"lr": 0.008, "lr_decay": 0.96, "exp_beta": 0.75, "ent_rate": 0.05}
param["osbm"]["inv-ff"] = {"lr": 0.02, "lr_decay": 0.98, "exp_beta": 0.7, "ent_rate": 0.003}
param["osbm"]["ff-hist"] = {"lr": 0.003, "lr_decay": 0.98, "exp_beta": 0.75, "ent_rate": 0.03}
param["osbm"]["ff-supervised"] = {"lr": 0.0006, "lr_decay": 0.99, "exp_beta": 0.0, "ent_rate": 0.0}
param["osbm"]["inv-ff-hist"] = {"lr": 0.006, "lr_decay": 0.97, "exp_beta": 0.8, "ent_rate": 0.0006}
param["osbm"]["gnn-hist"] = {"lr": 0.002, "lr_decay": 0.99, "exp_beta": 0.8, "ent_rate": 0.006}
param["osbm"]["gnn"] = {"lr": 0.002, "lr_decay": 0.97, "exp_beta": 0.8, "ent_rate": 0.05}
param["osbm"]["gnn-simp-hist"] = {"lr": 0.002, "lr_decay": 0.97, "exp_beta": 0.8, "ent_rate": 0.05}

if problem == "osbm":
    batch_size = 100

graph_fam_list = [-1]

if dataset == "er":
    graph_fam_list = [0.05, 0.1, 0.15, 0.2]

models = []

if model == "all":
    models = ["inv-ff", "ff", "ff-hist", "ff-supervised", "inv-ff-hist", "gnn-hist"]
else:
    models = sys.argv[5:-2]  # Train/tune one model only

def submit_job(model, u_size, v_size, dataset, dist, lr, lr_decay, exp_beta, ent_rate, problem, n_encode_layers, mode, batch_size, num_per_agent, g_fam_param, m, v):
    e = 1
    i = 0

    if mode == "tune":
        o = subprocess.run(f"python scripts/run_sweep.py noent {model}_{u_size}by{v_size}_{problem}_{dataset}", shell=True, capture_output=True, text=True)
        sweep_id = o.stdout[22: 22 + 8]
        print(sweep_id)
    while e != 0:  # Make sure job submitted successfully with exit code 0
        if i != 0:
            print(f"Failed to submit job for {model} {dataset} {g_fam_param}, resubmitting...")
            time.sleep(30)
        if mode == "train":
            e = subprocess.run(f"sbatch --account=def-khalile2 train.sh {u_size} {v_size} {g_fam_param} {dataset} {dist} {m} {v} {model} {lr} {lr_decay} {exp_beta} {ent_rate} {problem} {n_encode_layers} {batch_size}", shell=True)
            print(e)
        elif mode == "tune":
            e = subprocess.run(f"sbatch --account=def-khalile2 tune.sh {u_size} {v_size} {dataset} {problem} {g_fam_param} {dataset} {m} {v} {model} {n_encode_layers} {sweep_id} {num_per_agent} {batch_size}", shell=True)
        else:
            e = subprocess.run(f"sbatch --account=def-khalile2 generate_dataset.sh {u_size} {v_size} {dataset} {problem} {g_fam_param} {dist} {m} {v}", shell=True)
        e = e.returncode
        i += 1
    return


if "gmission" in dataset or "movielense" in dataset or "er" in dataset:
    if mode != "generate":
        for dataset in datasets:
            mean = -1 if dataset != "er" else 0
            var = -1 if dataset != "er" else 1
            weight_dist = dataset if dataset != "er" else "uniform"
            graph_fam_list = [-1] if dataset != "er" else [0.01, 0.1, 0.15, 0.2]
            for u_size, v_size in g_sizes:
                for m in models:
                    lr = param[problem][m]["lr"]
                    lr_decay = param[problem][m]["lr_decay"]
                    exp_beta = param[problem][m]["exp_beta"]
                    ent_rate = param[problem][m]["ent_rate"]
                    for p in graph_fam_list:
                        print("Waiting 30s before submitting job...")
                        time.sleep(30)
                        submit_job(m, u_size, v_size, dataset, weight_dist, lr, lr_decay, exp_beta, ent_rate, problem, n_encoding_layers, mode, batch_size, num_per_agent, p, mean, var)
    else:
        for p in graph_fam_list:
            print("Waiting 30s before submitting job...")
            time.sleep(30)
            submit_job("ff", u_size, v_size, dataset, weight_dist, -1, -1, -1, -1, problem, n_encoding_layers, mode, batch_size, num_per_agent, p, mean, var)
