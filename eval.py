import os
import numpy as np

import pprint as pp

import torch
from torch_geometric.data import DataLoader

# from nets.critic_network import CriticNetwork
from options import get_options
from train import get_inner_model, evaluate
from policy.attention_model import AttentionModel
from policy.ff_model import FeedForwardModel
from policy.ff_model_invariant import InvariantFF
from policy.ff_model_hist import FeedForwardModelHist
from policy.inv_ff_history import InvariantFFHist
from policy.gnn_hist import GNNHist
from policy.greedy import Greedy
from policy.greedy_rt import GreedyRt
from policy.greedy_theshold import GreedyThresh
from policy.greedy_matching import GreedyMatching
from policy.simple_greedy import SimpleGreedy
from policy.msvv import MSVV
from policy.balance import Balance
from policy.ff_supervised import SupervisedFFModel
from policy.gnn import GNN
from policy.gnn_simp_hist import GNNSimpHist

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import traceback
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils.functions import torch_load_cpu, load_problem

matplotlib.use("Agg")


def get_model_op_ratios(opts, model, problem):
    """
    given the model, run the model on the evaluation dataset and return the optmiality ratios
    """
    # get the path to the test set dir
    ops = []
    batch_size = opts.eval_batch_size
    # for i in graph family parameters
    for i in range(len(opts.eval_set)):
        dataset = opts.eval_dataset + "/parameter_{}".format(opts.eval_set[i])
        # get the eval dataset as a pytorch dataset object
        eval_dataset = problem.make_dataset(
            dataset, opts.eval_size, opts.eval_size, opts.problem, opts
        )
        if model.model_name == "greedy-m":
            opts.eval_batch_size = 2
            opts.batch_size = 2
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=opts.eval_batch_size, num_workers=0
        )
        opts.graph_family_parameter = opts.eval_set[i]
        avg_cost, cr, avg_cr, op, *_ = evaluate([model, model], eval_dataloader, opts)
        ops.append(op.cpu().numpy())
    opts.eval_batch_size = batch_size
    opts.batch_size = batch_size
    return np.array(ops)


def get_models_op_ratios(opts, models, problem):
    """
    given the model, run the model trained on a parameter on the evaluation dataset for
    that parameter and return the optmiality ratios
    """
    # get the path to the test set dir
    ops = []
    # for i in graph family parameters
    for i in range(len(opts.eval_set)):
        dataset = opts.eval_dataset + "/parameter_{}".format(opts.eval_set[i])
        # get the eval dataset as a pytorch dataset object
        eval_dataset = problem.make_dataset(
            dataset, opts.eval_size, opts.eval_size, opts.problem, opts
        )
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=opts.eval_batch_size, num_workers=0
        )

        avg_cost, cr, avg_cr, op, *_ = evaluate(models[i], eval_dataloader, opts)
        ops.append(op.cpu().numpy())
    return np.array(ops)


def set_box_color(bp, color):
    plt.setp(bp["boxes"], color=color)
    plt.setp(bp["whiskers"], color=color)
    plt.setp(bp["caps"], color=color)
    plt.setp(bp["medians"], color="#4d4d4d")


def plot_box(opts, data):
    """
    plots the box data.
    data is a list of (|graph family param| x |training examples|) arrays
    """
    # plt.figure()
    sns.set_theme()
    sns.set_style(style="darkgrid")
    # sns.set(style="ticks", context="talk")
    # plt.style.use("dark_background")
    # custom_style = {
    #     'axes.labelcolor': 'white',
    #     'xtick.color': 'white',
    #     'ytick.color': 'white',
    #     "axes.labelcolor": 'white',
    #     'text.color': 'white',
    # }
    # sns.set_style("whitegrid", rc=custom_style)
    f = plt.figure()

    colors = sns.color_palette()
    if opts.problem == "osbm":
        colors = [colors[0]] + colors[3:]
    if opts.problem == "adwords":
        colors = colors[0:2] + colors[3:]
    # plt.xlabel("Graph family parameter")
    # plt.ylabel("Optimality ratio")
    # plt.title("Bipartite graphs of size {}×{}".format(opts.u_size, opts.v_size))
    ticks = ["0.05", "0.1", "0.15", "0.2"]
    i = 0
    baselines = ["greedy"]
    if opts.problem == "e-obm":
        baselines += ["greedy-rt", "greedy-t"]
    elif opts.problem == "adwords":
        baselines += ["balance", "msvv"]
    m = baselines + opts.eval_models
    # sns.set_style(style="whitegrid")
    if opts.graph_family != "er":
        sns.set(font_scale=6)
        models = []
        avg_cr = []
        for i, d in enumerate(data):
            avg_cr += d.flatten().tolist()
            models += [m[i]] * len(d.flatten().tolist())
        data_p = pd.DataFrame({"Model": models, "Optimality Ratio": avg_cr})
        b = sns.boxplot(
            data=data_p,
            x="Model",
            y="Optimality Ratio",
            linewidth=6,
            width=0.8,
            palette=colors,
            showfliers=False,
        )
        g_fam_name = opts.graph_family
        problem_name = opts.problem
        if g_fam_name == "movielense":
            g_fam_name = "MovieLens"
        elif g_fam_name == "movielense-var":
            g_fam_name = "MovieLens-var"
        elif g_fam_name == "gmission":
            g_fam_name = "gMission"
        elif g_fam_name == "gmission-var":
            g_fam_name = "gMission-var"
        elif g_fam_name == "ba":
            g_fam_name = "BA"
        elif g_fam_name == "gmission-perm":
            g_fam_name = "gMission-perm"
        if problem_name == "e-obm":
            problem_name = "E-OBM"
        elif problem_name == "osbm":
            problem_name = "OSBM"
        b.set_title(
            f"{problem_name} {g_fam_name} {opts.u_size}x{opts.v_size}", fontsize=100
        )
        # plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
        plt.xticks([])

        # plt.yticks(color="white")
        b.set_xlabel("")
        b.set_ylabel("Optimality Ratio", fontsize=80)
        if not (opts.u_size == 10 and opts.v_size == 30):
            b.legend().remove()
        f.set_size_inches(h=30, w=27)
        # for line in b.get_lines():
        #     line.set_color('white')
        # _, xlabels = plt.xticks()
        # b.set_xticklabels(b.get_xticks(), size=10)
        # print(b.get_yticks().astype('float32'))
        b.set_yticklabels(b.get_yticks().astype("float32"), size=80)
    else:
        sns.set(font_scale=6)
        models = []
        avg_cr = []
        p = []
        for i, d in enumerate(data):
            avg_cr += d.flatten().tolist()
            models += [m[i]] * len(d.flatten().tolist())
            for t in ticks:
                p += [t] * int(len(d.flatten().tolist()) / len(ticks))
        data_p = pd.DataFrame({"Model": models, "Optimality Ratio": avg_cr, "p": p})

        b = sns.boxplot(
            data=data_p,
            x="p",
            y="Optimality Ratio",
            hue="Model",
            linewidth=3,
            palette=colors,
            showfliers=False,
        )
        problem_name = opts.problem
        g_fam_name = opts.graph_family
        if problem_name == "e-obm":
            problem_name = "E-OBM"
        if g_fam_name == "er":
            g_fam_name = "ER"
        # elif problem_name == "osbm":
        #     problem_name == "OSBM"
        b.set_title(
            f"{problem_name} {g_fam_name} {opts.u_size}x{opts.v_size}", fontsize=100
        )
        b.set_xlabel("p")
        b.set_ylabel("Optimality Ratio", fontsize=80)
        # plt.yticks(color="white")
        # plt.xticks(color="white")
        f.set_size_inches(h=40, w=27)
        [b.axvline(x, color="k", linestyle="--") for x in [0.5, 1.5, 2.5]]
        # b.get_legend().get_frame().set_alpha(None)
        # b.get_legend().get_frame().set_facecolor((0, 0, 1, 0))
        # for text in b.get_legend().get_texts():
        #     text.set_color("white")
        b.get_legend().set_title("")
        if not (opts.u_size == 10 and opts.v_size == 30):
            b.legend().remove()
        # for line in b.get_lines():
        #     line.set_color('white')
        # # _, xlabels = plt.xticks()
        # b.set_xticklabels(b.get_yticks(), size=7)
        # # print(b.get_yticks().astype('float32'))
        # b.set_yticklabels(b.get_yticks().astype('float32'), size=10)
    f.tight_layout()
    plt.savefig(
        opts.eval_output
        + "/{}_{}_{}x{}_{}-{}_boxplot.pdf".format(
            opts.problem,
            opts.graph_family,
            opts.u_size,
            opts.v_size,
            opts.weight_distribution_param[0] if opts.problem != "adwords" else -1,
            opts.weight_distribution_param[1] if opts.problem != "adwords" else -1,
        ).replace(" ", ""),
        dpi=300,
        # transparent=True,
    )


def make_legend(opts):
    sns.set(font="monospace")
    # plt.rc('font',family='monospace')
    colors = sns.color_palette()
    plt.axis("off")
    if opts.problem == "osbm":
        colors = [colors[0]] + colors[3:]
    if opts.problem == "adwords":
        colors = colors[0:2] + colors[3:]

    def f(m, c):
        return plt.plot([], [], marker=m, color=c, ls="none", markersize=20)[0]

    if opts.problem == "e-obm":
        labels = [
            "greedy",
            "greedy-rt",
            "greedy-t",
            "ff-supervised",
            "ff",
            "ff-hist",
            "inv-ff",
            "inv-ff-hist",
            "gnn-hist",
        ]
    else:
        labels = [
            "greedy",
            "ff-supervised",
            "ff",
            "ff-hist",
            "inv-ff",
            "inv-ff-hist",
            "gnn-hist",
        ]
    handles = [f("s", colors[i]) for i in range(len(colors))]
    legend = plt.legend(
        handles, labels, loc=10, frameon=False, fontsize=20, framealpha=1
    )

    def export_legend(legend, filename=f"{opts.problem}_legend.pdf"):
        fig = legend.figure
        fig.canvas.draw()
        bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename, dpi=300, bbox_inches=bbox)

    export_legend(legend)
    plt.show()


def plot_agreemant(opts, data, with_opt=False):
    """
    plots the box data.
    data is a list of (|graph family param| x |training examples|) arrays
    """
    fig = plt.figure()
    sns.set_theme()
    sns.set_style("darkgrid")
    # custom_style = {
    #     'axes.labelcolor': 'white',
    #     'xtick.color': 'white',
    #     'ytick.color': 'white',
    #     "axes.labelcolor": 'white',
    #     'text.color': 'white',
    # }
    # sns.set_style("whitegrid", rc=custom_style)
    colors = sns.color_palette()
    if opts.problem == "e-obm" and with_opt:
        colors = [colors[0]] + colors[3:]
    elif opts.problem == "osbm" and with_opt:
        colors = [colors[0]] + colors[3:]
    else:
        colors = colors[3:]
    fig, axs = plt.subplots(
        ncols=1, nrows=len(opts.eval_set), sharex=True, sharey=True, figsize=(8, 10)
    )
    ag = "Optimal" if with_opt else "Greedy"
    g_fam_name = opts.graph_family
    if g_fam_name == "movielense":
        g_fam_name = "MovieLens"
    elif g_fam_name == "gmission":
        g_fam_name = "gMission"
    elif g_fam_name == "gmission-var":
        g_fam_name = "gMission-var"
    elif g_fam_name == "er":
        g_fam_name = "ER"
    fig.suptitle(
        f"Agreement with {ag} for {g_fam_name} {opts.u_size}x{opts.v_size}", fontsize=25
    )
    plots = []
    for j, d in enumerate(data):
        for i, a in enumerate(d):
            if len(opts.eval_set) != 1:
                (a,) = axs[i].plot(
                    np.arange(opts.v_size), np.array(a) * 100.0, color=colors[j]
                )
                axs[i].set_title(opts.eval_set[i])
            else:
                (a,) = axs.plot(
                    np.arange(opts.v_size), np.array(a) * 100.0, color=colors[j]
                )
        plots.append(a)
    if with_opt and opts.graph_family != "er":
        plt.legend(plots, ["greedy"] + opts.eval_models, fontsize=20)
    elif opts.graph_family != "er":
        plt.legend(plots, opts.eval_models, fontsize=20)
    # for line in axs.get_lines():
    #     line.set_color('white')
    # for text in axs.get_legend().get_texts():
    #     text.set_color("white")
    # axs.get_legend().get_frame().set_alpha(None)
    # axs.get_legend().get_frame().set_facecolor((0, 0, 1, 0))

    plt.xlabel("Timestep", fontsize=20)
    fig.text(
        0.06,
        0.5,
        "Agreement per Timestep %",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=20,
    )
    # plt.legend(bps, opts.eval_baselines + opts.eval_models)
    s = ""
    if with_opt:
        s = "_with_opt"
    # fig.tight_layout()
    plt.savefig(
        opts.eval_output
        + "/{}_{}_{}x{}_agreementplot{}.pdf".format(
            opts.problem,
            opts.graph_family,
            opts.u_size,
            opts.v_size,
            s,
        ).replace(" ", ""),
        dpi=300,
    )


def load_model(opts):
    """
    Load models (here we refer to them as data) from load_path
    """
    load_data = {}
    load_datas = []
    path = opts.load_path if opts.load_path is not None else []
    if path is not None:
        print("  [*] Loading data from {}".format(path))
        load_data = torch_load_cpu(path)
        load_datas.append(load_data)
    return load_datas


def load_models(opts, models_paths):
    """
    load models from the attention models dir
    """
    load_data = {}
    load_datas = []
    assert len(models_paths) == len(
        opts.eval_set
    ), "the number of models and the eval_set should be equal"
    for path in models_paths:
        print(" Loading the model from {}".format(path))
        load_data = torch_load_cpu(path)
        load_datas.append(load_data)
    return load_datas


def initialize_models(opts, models, load_datas, Model):
    problem = load_problem(opts.problem)
    for m in range(len(load_datas)):
        model = Model(
            opts.embedding_dim,
            opts.hidden_dim,
            problem=problem,
            n_encode_layers=opts.n_encode_layers,
            mask_inner=True,
            mask_logits=True,
            normalization=opts.normalization,
            tanh_clipping=opts.tanh_clipping,
            checkpoint_encoder=opts.checkpoint_encoder,
            shrink_size=opts.shrink_size,
            num_actions=opts.u_size + 1,
            n_heads=opts.n_heads,
            opts=opts,
        ).to(opts.device)

        if opts.use_cuda and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        # Overwrite model parameters × parameters to load
        model_ = get_inner_model(model)
        model_.load_state_dict(
            {**model_.state_dict(), **load_datas[m].get("model", {})}
        )
        models.append(model)


def compare_actions(opts, models, greedy, problem):
    ops = []
    ps = []
    ps1 = []
    ps2 = []
    counts = []
    counts1 = []
    # for i in graph family parameters
    for i in range(len(opts.eval_set)):
        dataset = opts.eval_dataset + "/parameter_{}".format(opts.eval_set[i])
        # get the eval dataset as a pytorch dataset object
        eval_dataset = problem.make_dataset(
            dataset, opts.eval_size, opts.eval_size, opts.problem, opts
        )
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=opts.eval_batch_size, num_workers=0
        )

        avg_cost, cr, avg_cr, op, p, p1, p2, count1, count2, avg_j, wil = evaluate(
            [models[i], greedy], eval_dataloader, opts
        )
        print(f"Average Jaccard Index: {opts.eval_set[i]}: {avg_j}")
        # print(f"Wilcoxon test p-value: {opts.eval_set[i]}: {wil}")
        ops.append(op.cpu().numpy())
        ps.append(p.cpu().numpy() / float(opts.eval_size))
        ps1.append(p1.cpu().numpy() / float(opts.eval_size))
        ps2.append(p2.cpu().numpy() / float(opts.eval_size))
        counts.append(count1.cpu().numpy() / float(count1.sum()))
        counts1.append(count2.cpu().numpy() / float(count2.sum()))
    return (
        np.array(ops),
        np.array(ps),
        np.array(ps1),
        np.array(ps2),
        np.array(counts),
        np.array(counts1),
    )


def test_transeferability(opts, models, greedy, problem):

    # sns.set_style("darkgrid")
    # plt.figure()
    trained_on = (opts.u_size, opts.v_size)
    g_sizes = [(10, 30), (10, 60), (100, 100), (100, 200)]
    data = {"Model": [], "Graph Size": [], "Average Optimality Ratio": []}
    data_matrix = []
    models = [greedy[0], greedy[2]] + models
    for g in g_sizes:
        g_list = []
        extention = "{}_{}_{}_{}{}_{}by{}".format(
            opts.problem,
            opts.graph_family,
            opts.weight_distribution,
            opts.weight_distribution_param[0],
            opts.weight_distribution_param[1],
            g[0],
            g[1],
        ).replace(" ", "")

        eval_dataset = f"dataset/eval/{extention}/parameter_-1"
        opts.u_size = g[0]
        opts.v_size = g[1]

        for m in models:
            eval_dataset = problem.make_dataset(
                eval_dataset, opts.eval_size, opts.eval_size, opts.problem, opts
            )
            eval_dataloader = DataLoader(
                eval_dataset, batch_size=opts.eval_batch_size, num_workers=0
            )
            if not (
                m.model_name in ["ff", "ff-hist", "ff-supervised"]
                and g[0] != trained_on[0]
            ):
                (
                    avg_cost,
                    cr,
                    avg_cr,
                    op,
                    p,
                    p1,
                    p2,
                    count1,
                    count2,
                    avg_j,
                    wil,
                ) = evaluate([m, greedy[0]], eval_dataloader, opts)
                data["Model"].append(m.model_name)
                data["Graph Size"].append(f"{g[0]}x{g[1]}")
                data["Average Optimality Ratio"].append(avg_cr.item())
                g_list.append(avg_cr.item())
            else:
                g_list.append(0.0)
        data_matrix.append(g_list)
        # else:
        #     data["Model"].append(m.model_name)
        #     data["Graph Size"].append(f"{g[0]}×{g[1]}")
        #     data["Average Optimality Ratio"].append(0.0)
    data = pd.DataFrame(data)
    # b = sns.catplot(
    #     data=data,
    #     hue="Graph Size",
    #     x="Model",
    #     y="Average Optimality Ratio",
    #     legend_out=False,
    #     height=7,
    # )

    # data_matrix = np.array(
    # data = pd.read_pickle(
    #     "./{}_{}_{}x{}_graph_transfer.pkl".format(
    #         opts.problem, opts.graph_family, trained_on[0], trained_on[1]
    #     ).replace(" ", "")
    # )
    data.to_pickle(
        "./{}_{}_{}x{}_graph_transfer.pkl".format(
            opts.problem, opts.graph_family, trained_on[0], trained_on[1]
        ).replace(" ", "")
    )
    # print(data_matrix.shape)
    # data_matrix = data_matrix[:, [0, 4, 2, 3, 1, 5, 6]]
    # b = sns.heatmap(data=data_matrix, annot=True, fmt="d")
    models = [
        "greedy",
        "greedy-t",
        "ff-supervised",
        "ff",
        "ff-hist",
        "inv-ff",
        "inv-ff-hist",
        "gnn-hist",
    ]
    data_matrix = np.array(data_matrix)
    g_sizes = ["10x30", "10x60", "100x100", "100x200"]
    fig, ax = plt.subplots()
    d1 = data_matrix.copy()
    data = 1.0 - data_matrix
    data[d1 == 0.0] = 0.1
    ax.imshow(data)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(g_sizes)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(list(m for m in models))
    ax.set_yticklabels(g_sizes)

    # Rotate the tick labels and set their alignment.
    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=15,
    )
    # Loop over data dimensions and create text annotations.
    for i in range(len(g_sizes)):
        for j in range(len(models)):
            num = str(d1[i, j])[:4] if d1[i, j] != 0.0 else "-"
            ax.text(j, i, num, ha="center", va="center", color="w", fontsize=15)

    # b.set_xticklabels(size=11)
    # b.set_ylabels(size=15)
    # b.set_xlabels(size=15)
    ax.set_title(
        f"Graph Transferability Trained On {trained_on[0]}x{trained_on[1]}",
        fontsize=20,
    )
    fig.tight_layout()
    plt.savefig(
        opts.eval_output
        + "/{}_{}_{}x{}_graph_transfer.pdf".format(
            opts.problem,
            opts.graph_family,
            trained_on[0],
            trained_on[1],
        ).replace(" ", ""),
        dpi=300,
    )
    return


def tflog2pandas(path):
    """convert single tensorflow log file to pandas DataFrame
    Parameters
    ----------
    path : str
        path to tensorflow log file
    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data


def get_log_data(opts, g_param):
    m = -1 if opts.graph_family != "er" else 0
    v = -1 if opts.graph_family != "er" else 1
    log_dir = f"logs/logs_{opts.problem}_{opts.graph_family}_{opts.u_size}by{opts.v_size}_p={g_param}_{opts.graph_family}_m={m}_v={v}_a=3"
    complete_df = pd.DataFrame({"metric": [], "value": [], "step": [], "Model": []})
    for m_type in opts.eval_models:
        list_of_files = sorted(
            os.listdir(log_dir + f"/{m_type}"), key=lambda s: int(s[8:12] + s[13:])
        )
        log_file = log_dir + f"/{m_type}/" + list_of_files[-1]
        df = tflog2pandas(log_file)
        df["Model"] = [m_type] * len(df)
        df = df[df["metric"] == "val_avg_reward"]
        complete_df = pd.concat([complete_df, df])
    return complete_df


def plot_val_reward(opts):
    fig = plt.figure()
    colors = sns.color_palette()[3:-1]
    for g_param in opts.eval_set:
        fig = plt.figure()
        sns.set_theme()
        sns.set_style("darkgrid")
        log_data = get_log_data(opts, g_param)
        sns.set_style("darkgrid")
        ax = sns.lineplot(
            data=log_data,
            x="step",
            y="value",
            hue="Model",
            hue_order=opts.eval_models,
            palette=colors,
        )
        ax.set_ylim((4.0, log_data["value"].max() + 0.1))
        ax.set_title("Average Validation Reward", fontsize=20)
        ax.set_ylabel("Reward", fontsize=15)
        ax.set_xlabel("Step", fontsize=15)
        plt.legend(title=None)
        fig.tight_layout()
        plt.savefig(
            opts.eval_output
            + "/{}_{}_{}_{}x{}_reward_plot.pdf".format(
                opts.problem,
                opts.graph_family,
                g_param,
                opts.u_size,
                opts.v_size,
            ).replace(" ", ""),
            dpi=300,
        )


def run(opts):

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Save arguments so exact configuration can always be found
    #    with open(os.path.join(opts.save_dir, "args.json"), "w") as f:
    #        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Figure out what's the problem
    problem = load_problem(opts.problem)
    if opts.eval_plot:
        t = torch.load(opts.eval_results_file)
        plot_box(opts, np.array(t))
        return

    # load the basline and neural net models and save them in models, attention_models, ff_models, baseline_models

    # load models
    assert (
        opts.load_path is None
        and opts.eval_ff_dir is None
        and opts.eval_attention_dir is None
    ) or opts.resume is None, "either one of load_path, attention_models, ff_models as well as resume should be given"
    # single_model = None if opts.load_path == ["None"] else opts.load_path
    att_models = None if opts.attention_models == ["None"] else opts.attention_models
    ff_models = None if opts.ff_models == ["None"] else opts.ff_models
    inv_ff_models = None if opts.inv_ff_models == ["None"] else opts.inv_ff_models
    ff_hist_models = None if opts.ff_hist_models == ["None"] else opts.ff_hist_models
    gnn_hist_models = None if opts.gnn_hist_models == ["None"] else opts.gnn_hist_models
    gnn_simp_hist_models = (
        None if opts.gnn_simp_hist_models == ["None"] else opts.gnn_simp_hist_models
    )
    gnn_models = None if opts.gnn_models == ["None"] else opts.gnn_models
    ff_supervised_models = (
        None if opts.ff_supervised_models == ["None"] else opts.ff_supervised_models
    )
    inv_ff_hist_models = (
        None if opts.inv_ff_hist_models == ["None"] else opts.inv_ff_hist_models
    )
    model_paths = [
        (att_models, AttentionModel),
        (ff_supervised_models, SupervisedFFModel),
        (ff_models, FeedForwardModel),
        (ff_hist_models, FeedForwardModelHist),
        (inv_ff_models, InvariantFF),
        (inv_ff_hist_models, InvariantFFHist),
        (gnn_hist_models, GNNHist),
        (gnn_models, GNN),
        (gnn_simp_hist_models, GNNSimpHist),
    ]
    models = []

    for m_path, m_class in model_paths:
        if m_path is not None:
            model_param = load_models(opts, m_path)
            initialize_models(opts, models, model_param, m_class)

    # Initialize baseline models
    baseline_models = []
    for i in range(len(opts.eval_baselines)):
        baseline_model_class = {
            "greedy": Greedy,
            "greedy-rt": GreedyRt,
            "simple-greedy": SimpleGreedy,
            "greedy-t": GreedyThresh,
            "greedy-m": GreedyMatching,
            "msvv": MSVV,
            "balance": Balance,
        }.get(opts.eval_baselines[i], None)
        assert baseline_model_class is not None, "Unknown baseline model: {}".format(
            opts.eval_baselines[i]
        )
        model = baseline_model_class(
            opts.embedding_dim,
            opts.hidden_dim,
            problem=problem,
            n_encode_layers=opts.n_encode_layers,
            mask_inner=True,
            mask_logits=True,
            normalization=opts.normalization,
            tanh_clipping=opts.tanh_clipping,
            checkpoint_encoder=opts.checkpoint_encoder,
            shrink_size=opts.shrink_size,
            num_actions=opts.u_size + 1,
            opts=opts,
        ).to(opts.device)
        baseline_models.append(model)
    if opts.test_transfer:
        test_transeferability(opts, models, baseline_models, problem)
        return
    if len(opts.eval_set) > 0:
        if opts.save_eval_data:
            baseline_results = []
            trained_models_results = []
            # plot_data = []
            for m in baseline_models:  # Get the performance of the baselines
                ops = get_model_op_ratios(opts, m, problem)
                baseline_results.append(ops)

            if len(models) > 0:
                # Get the performance of the trained models
                for j in range(0, len(models), len(opts.eval_set)):
                    trained_models_results.append(
                        compare_actions(
                            opts,
                            models[j : j + len(opts.eval_set)],
                            baseline_models[0],
                            problem,
                        )
                    )

            results = [
                np.array(baseline_results[i]) for i in range(len(baseline_results))
            ] + [
                np.array(trained_models_results[i][0])
                for i in range(len(trained_models_results))
            ]

            results2 = [
                np.array(trained_models_results[i][1])
                for i in range(len(trained_models_results))
            ]

            results3 = [np.array(trained_models_results[0][2])] + [
                np.array(trained_models_results[i][3])
                for i in range(len(trained_models_results))
            ]
            np.save(
                f"dataset/eval/{opts.problem}_{opts.graph_family}_{opts.u_size}x{opts.v_size}_eval_output_op_ratio.npy",
                results,
            )
            np.save(
                f"dataset/eval/{opts.problem}_{opts.graph_family}_{opts.u_size}x{opts.v_size}_eval_output_agr.npy",
                results2,
            )
            np.save(
                f"dataset/eval/{opts.problem}_{opts.graph_family}_{opts.u_size}x{opts.v_size}_eval_output_agr_opt.npy",
                results3,
            )
        results = np.load(
            f"dataset/eval/{opts.problem}_{opts.graph_family}_{opts.u_size}x{opts.v_size}_eval_output_op_ratio.npy"
        )
        results2 = np.load(
            f"dataset/eval/{opts.problem}_{opts.graph_family}_{opts.u_size}x{opts.v_size}_eval_output_agr.npy"
        )
        results3 = np.load(
            f"dataset/eval/{opts.problem}_{opts.graph_family}_{opts.u_size}x{opts.v_size}_eval_output_agr_opt.npy"
        )
        # make_legend(opts)
        plot_box(opts, results)
        # plot_val_reward(opts)
        # test_transeferability(opts, models, baseline_models, problem)
        # plot_agreemant(opts, results2)
        # plot_agreemant(opts, results3, with_opt=True)


if __name__ == "__main__":
    run(get_options())
