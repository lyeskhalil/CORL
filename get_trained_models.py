import subprocess
import sys


# dataset = sys.argv[2]
# u_size = sys.argv[3]
# v_size = sys.argv[4]
# problem = sys.argv[1]
# mode = sys.argv[5]
mode = "output"
problems = ["e-obm"]
g_sizes = [(100, 100)]
datasets = ["gmission", "gmission-var"]


def get_models(model, u, v, dataset, problem, p, mode):
    if dataset == "er":
        weight_dist = "uniform"
        m, var = 0, 1
    else:
        weight_dist = dataset
        m, var = -1, -1

    if mode == "logs":
        subprocess.run(
            f"scp -r alomrani@cedar.computecanada.ca:~/projects/def-khalile2/alomrani/{mode}_{problem}_{dataset}_{u}by{v}_p={p}_{weight_dist}_m={m}_v={var}_a=3/{model} \
        {mode}/{mode}_{problem}_{dataset}_{u}by{v}_p={p}_{dataset}_m={m}_v={var}_a=3",
            shell=True,
        )
    else:
        subprocess.run(
            f"scp -r alomrani@cedar.computecanada.ca:~/projects/def-khalile2/alomrani/{mode}_{problem}_{dataset}_{u}by{v}_p={p}_{weight_dist}_m={m}_v={var}_a=3/{model} \
        {mode}s/{mode}_{problem}_{dataset}_{u}by{v}_p={p}_{dataset}_m={m}_v={var}_a=3",
            shell=True,
        )


models = ["inv-ff", "ff", "ff-hist", "ff-supervised", "inv-ff-hist"]

for problem in problems:
    for dataset in datasets:
        for u_size, v_size in g_sizes:
            g_params = [-1] if dataset != "er" else [0.05, 0.1, 0.15, 0.2]
            for m in models:
                for p in g_params:
                    get_models(m, u_size, v_size, dataset, problem, p, mode)
