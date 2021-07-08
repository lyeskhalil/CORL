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
from policy.simple_greedy import SimpleGreedy
from policy.ff_supervised import SupervisedFFModel
from policy.gnn import GNN
from policy.gnn_simp_hist import GNNSimpHist

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils.functions import torch_load_cpu, load_problem

matplotlib.use("Agg")


def get_model_op_ratios(opts, model, problem):
    """
    given the model, run the model on the evaluation dataset and return the optmiality ratios
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

        avg_cost, cr, avg_cr, op, *_ = evaluate([model, model], eval_dataloader, opts)
        ops.append(op.cpu().numpy())
    return np.array(ops)


def get_models_op_ratios(opts, models, problem):
    """
    given the model, run the model trained on a parameter on the evaluation dataset for that parameter and return the optmiality ratios
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
    sns.set_style(style="darkgrid")
    f = plt.figure()
    # plt.xlabel("Graph family parameter")
    # plt.ylabel("Optimality ratio")
    # plt.title("Bipartite graphs of size {}by{}".format(opts.u_size, opts.v_size))
    ticks = ["0.01", "0.05", "0.1", "0.15", "0.2"]
    i = 0
    m = ["greedy"] + opts.eval_models
    # sns.set_style(style="whitegrid")
    if opts.graph_family != "er":
        sns.set(font_scale=3)
        models = []
        avg_cr = []
        for i, d in enumerate(data):
            avg_cr += d.flatten().tolist()
            models += [m[i]] * len(d.flatten().tolist())
        data_p = pd.DataFrame({"Model": models, "Average Optimality Ratio": avg_cr})
        b = sns.boxplot(
            data=data_p, x="Model", y="Average Optimality Ratio", linewidth=3, width=0.5
        )
        b.set_title(f"{opts.problem} {opts.graph_family} {opts.u_size}by{opts.v_size}")
        # b.set_xlabel("Model", fontsize=15)
        # b.set_ylabel("Average Optimality Ratio", fontsize=15)

        f.set_size_inches(h=15, w=25)
        # # _, xlabels = plt.xticks()
        # b.set_xticklabels(b.get_yticks(), size=7)
        # # print(b.get_yticks().astype('float32'))
        # b.set_yticklabels(b.get_yticks().astype('float32'), size=10)
    else:
        sns.set(font_scale=2)
        models = []
        avg_cr = []
        p = []
        for i, d in enumerate(data):
            avg_cr += d.flatten().tolist()
            models += [m[i]] * len(d.flatten().tolist())
            print(len(d.flatten().tolist()), len(ticks))
            for t in ticks:
                p += [t] * (len(d.flatten().tolist()) / len(ticks))
        data_p = pd.DataFrame(
            {"Model": models, "Average Optimality Ratio": avg_cr, "p": p}
        )

        b = sns.boxplot(
            data=data_p, x="p", y="Average Optimality Ratio", hue="Model", linewidth=3
        )

        b.set_title(f"{opts.problem} {opts.graph_family} {opts.u_size}by{opts.v_size}")
        # b.set_xlabel("Model", fontsize=15)
        # b.set_ylabel("Average Optimality Ratio", fontsize=15)

        f.set_size_inches(h=15, w=25)
        # # _, xlabels = plt.xticks()
        # b.set_xticklabels(b.get_yticks(), size=7)
        # # print(b.get_yticks().astype('float32'))
        # b.set_yticklabels(b.get_yticks().astype('float32'), size=10)

    plt.savefig(
        opts.eval_output
        + "/{}_{}_{}by{}_boxplot".format(
            opts.problem, opts.graph_family, opts.u_size, opts.v_size,
        ).replace(" ", "")
    )


def plot_agreemant(opts, data, with_opt=False):
    """
    plots the box data.
    data is a list of (|graph family param| x |training examples|) arrays
    """
    colors = [
        "#d53e4f",
        "#3288bd",
        "#7fbf7b",
        "#fee08b",
        "#fc8d59",
        "#e6f598",
        "#ff69b4",
    ]
    fig, axs = plt.subplots(
        ncols=1, nrows=len(opts.eval_set), sharex=True, sharey=True, figsize=(8, 10)
    )
    fig.suptitle("Agreemant plots for {}by{} graphs".format(opts.u_size, opts.v_size))
    plots = []
    for j, d in enumerate(data):
        c = colors[j]
        for i, a in enumerate(d):
            (a,) = axs.plot(np.arange(opts.v_size), np.array(a) * 100.0, c)
            plots.append(a)
            axs.set_title(opts.eval_set[i])
    print(len(plots))
    plt.legend(plots, opts.eval_models)

    plt.xlabel("Timestep")
    fig.text(
        0.06,
        0.5,
        "Agreemant per timestep %",
        ha="center",
        va="center",
        rotation="vertical",
    )
    # plt.legend(bps, opts.eval_baselines + opts.eval_models)
    s = ""
    if with_opt:
        s = "_with_opt"
    plt.savefig(
        opts.eval_output
        + "/{}_{}_{}_{}_{}by{}_agreemantplot{}".format(
            opts.problem,
            opts.graph_family,
            opts.weight_distribution,
            opts.weight_distribution_param,
            opts.u_size,
            opts.v_size,
            s,
        ).replace(" ", "")
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

        # Overwrite model parameters by parameters to load
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

    sns.set_style("darkgrid")
    plt.figure()
    trained_on = (opts.u_size, opts.v_size)
    g_sizes = [(10, 30), (10, 60), (100, 100), (100, 200)]
    data = {"Model": [], "Graph Size": [], "Average Optimality Ratio": []}
    for g in g_sizes:
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
        models = [greedy] + models
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
                ) = evaluate([m, greedy], eval_dataloader, opts)
                data["Model"].append(m.model_name)
                data["Graph Size"].append(f"{g[0]}by{g[1]}")
                data["Average Optimality Ratio"].append(avg_cr.item())
            else:
                data["Model"].append(m.model_name)
                data["Graph Size"].append(f"{g[0]}by{g[1]}")
                data["Average Optimality Ratio"].append(0.0)
    data = pd.DataFrame(data)
    sns.set_style("darkgrid")
    b = sns.catplot(
        data=data,
        hue="Graph Size",
        x="Model",
        y="Average Optimality Ratio",
        legend_out=False,
        height=7,
    )
    b.set_xticklabels(size=11)
    b.set_ylabels(size=15)
    b.set_xlabels(size=15)
    b.ax.set_title(
        f"Graph Transferability Trained On {trained_on[0]}by{trained_on[1]}",
        fontsize=20,
    )
    plt.savefig(
        opts.eval_output
        + "/{}_{}_{}by{}_graph_transfer".format(
            opts.problem, opts.graph_family, trained_on[0], trained_on[1],
        ).replace(" ", "")
    )
    return


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
        (inv_ff_models, InvariantFF),
        (ff_models, FeedForwardModel),
        (ff_hist_models, FeedForwardModelHist),
        (ff_supervised_models, SupervisedFFModel),
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
        test_transeferability(opts, models, baseline_models[0], problem)
        return
    if len(opts.eval_set) > 0:
        baseline_results = []
        trained_models_results = []
        # plot_data = []
        for m in baseline_models:  # Get the performance of the baselines
            ops = get_model_op_ratios(opts, m, problem)
            baseline_results.append(ops)

        print(len(models))
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

        # results2 = [np.array(trained_models_results[i][1]) for i in range(len(trained_models_results))]

        # results3 = [np.array(trained_models_results[i][3]) for i in range(len(trained_models_results))]

        plot_box(opts, results)
        # test_transeferability(opts, models, baseline_models[0], problem)
        # plot_agreemant(opts, results2)
        # plot_agreemant(opts, results3, with_opt=True)


if __name__ == "__main__":
    run(get_options())
