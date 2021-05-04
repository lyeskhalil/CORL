# !/usr/bin/env python

import os
import json
import pprint as pp
import ast

import torch
import torch.optim as optim
from torch_geometric.data import DataLoader

# from nets.critic_network import CriticNetwork
from options import get_options
from train import train_epoch, validate, get_inner_model, eval_model, evaluate
from policy.attention_model import AttentionModel
from policy.ff_model_v2 import FeedForwardModel
from policy.ff_model_invariant import InvariantFF
from policy.ff_model_hist import FeedForwardModelHist
from policy.inv_ff_history import InvariantFFHist
from policy.greedy import Greedy
from policy.greedy_rt import GreedyRt
from policy.simple_greedy import SimpleGreedy

import numpy as np
import time
from tqdm import tqdm

import math
import matplotlib

import matplotlib.pyplot as plt

from torch.nn import DataParallel
from policy.attention_model_v2 import set_decode_type
from log_utils import log_values
from functions import move_to

# from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from functions import torch_load_cpu, load_problem

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
    num = len(data)
    plt.xlabel("Graph family parameter")
    plt.ylabel("Optimality ratio")
    plt.title("Bipartite graphs of size {}by{}".format(opts.u_size, opts.v_size))
    ticks = opts.eval_set  # ["0.01", "0.05", "0.1", "0.15", "0.2"]
    colors = ["#d53e4f", "#3288bd", "#7fbf7b", "#fee08b", "#fc8d59", "#e6f598"]
    i = 0
    bps = []
    for d in data:
        bp = plt.boxplot(
            d.T,
            positions=np.array(range(len(d))) * num + (0.5 * i),
            sym="",
            widths=0.6,
            whis=(0, 100),
        )
        bps.append(bp["boxes"][0])
        set_box_color(bp, colors[i])
        i += 1

    plt.xlim(-1 * num, len(ticks) * num)
    # plt.ylim(0, 1)
    print(ticks)
    plt.xticks(range(0, len(ticks) * num, num), ticks)
    plt.legend(bps, opts.eval_baselines + opts.eval_models)
    plt.savefig(
        opts.eval_output
        + "/{}_{}_{}_{}_{}by{}_boxplot".format(
            opts.problem,
            opts.graph_family,
            opts.weight_distribution,
            opts.weight_distribution_param,
            opts.u_size,
            opts.v_size,
        ).replace(" ", "")
    )


def plot_agreemant(opts, data):
    """
    plots the box data.
    data is a list of (|graph family param| x |training examples|) arrays
    """
    colors = ["#d53e4f", "#3288bd", "#7fbf7b", "#fee08b", "#fc8d59", "#e6f598"]
    fig, axs = plt.subplots(
        ncols=1, nrows=len(opts.eval_set), sharex=True, sharey=True, figsize=(8, 10)
    )
    fig.suptitle("Agreemant plots for {}by{} graphs".format(opts.u_size, opts.v_size))
    plots = []
    for j, d in enumerate(data):
        c = colors[j]
        for i, a in enumerate(d):
            (a,) = axs[i].plot(np.arange(opts.v_size), np.array(a) * 100.0, c)
            plots.append(a)
            axs[i].set_title(opts.eval_set[i])

    plt.legend([plots[0], plots[-1]], opts.eval_models)

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
    plt.savefig(
        opts.eval_output
        + "/{}_{}_{}_{}_{}by{}_agreemantplot".format(
            opts.problem,
            opts.graph_family,
            opts.weight_distribution,
            opts.weight_distribution_param,
            opts.u_size,
            opts.v_size,
        ).replace(" ", "")
    )


def plot_action_distribution(opts, data):
    """
    plots the box data.
    data is a list of (|graph family param| x |training examples|) arrays
    """

    labels = list(range(opts.u_size + 1))

    x = np.arange(len(labels))  # the label locations
    width = 0.75  # the width of the bars

    # Add some text for labels, title and custom x-axis tick labels, etc.

    colors = ["#d53e4f", "#3288bd", "#7fbf7b", "#fee08b", "#fc8d59", "#e6f598"]
    fig, axs = plt.subplots(
        ncols=1, nrows=len(opts.eval_set), sharex=True, sharey=True, figsize=(8, 15)
    )
    fig.suptitle(
        "Action Distribution for {}by{} graphs".format(opts.u_size, opts.v_size)
    )
    plots = []
    models = ["attention", "ff", "greedy"]
    for j, d in enumerate(data):
        c = colors[j]
        for i, a in enumerate(d):
            # (a,) = axs[i].plot(np.arange(opts.v_size), np.array(a) * 100.0, c)

            rects1 = axs[i].bar(
                x - (width) * (j + 1), a, width, label=models[j], color=c
            )
            plots.append(rects1)
            axs[i].set_title(opts.eval_set[i])

    plt.legend([plots[0], plots[-5], plots[-1]], models)
    plt.set_xticks(x)
    plt.set_xticklabels(labels)
    plt.xlabel("Actions")
    fig.text(
        0.06,
        0.5,
        "Action Distribution %",
        ha="center",
        va="center",
        rotation="vertical",
    )
    # plt.legend(bps, opts.eval_baselines + opts.eval_models)
    plt.savefig(
        opts.eval_output
        + "/{}_{}_{}_{}_{}by{}_actiondistribution".format(
            opts.problem,
            opts.graph_family,
            opts.weight_distribution,
            opts.weight_distribution_param,
            opts.u_size,
            opts.v_size,
        ).replace(" ", "")
    )


def line_graph(opts, models, problem):
    """
    Evaluate the models on a range of graph family parameters.
    Draw the line graph of optimality and competative ratios optimality ratio
    """
    plt.figure(1)
    plt.xlabel("Graph family parameter")
    plt.ylabel("Competitive ratio")

    plt.figure(2)
    plt.xlabel("Graph family parameter")
    plt.ylabel("Average ratio to optimal")

    min_p, max_p = float(opts.eval_range[0]), float(opts.eval_range[1])
    #    for i, j in enumerate(
    #        np.arange(min_p, max_p, (min_p + max_p) / opts.eval_num_range)
    #    ):
    eval_dataset = problem.make_dataset(opts.eval_dataset, opts.eval_size, opts.problem)
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=opts.eval_batch_size, num_workers=1
    )

    for model in models:
        crs = []
        avg_crs = []
        avg_ratio, cr, avg_cr = evaluate(model, eval_dataloader, opts)
        crs.append(cr)
        avg_crs.append(avg_cr)

        plt.figure(1)
        plt.plot(np.arange(min_p, max_p, (min_p + max_p) / opts.eval_num_range), crs)

        plt.figure(2)
        plt.plot(
            np.arange(min_p, max_p, (min_p + max_p) / opts.eval_num_range), avg_crs
        )

    plt.savefig(
        opts.eval_output
        + "/{}_{}_{}_{}_{}by{}_competitive_ratio".format(
            opts.problem,
            opts.graph_family,
            opts.weight_distribution,
            opts.weight_distribution_param,
            opts.u_size,
            opts.v_size,
        ).replace(" ", "")
    )

    plt.savefig(
        opts.eval_output
        + "/{}_{}_{}_{}_{}by{}_avg_opt_ratio".format(
            opts.problem,
            opts.graph_family,
            opts.weight_distribution,
            opts.weight_distribution_param,
            opts.u_size,
            opts.v_size,
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


# def initialize_models(opts, models, load_datas):
#     problem = load_problem(opts.problem)
#     for m in range(len(opts.eval_models)):
#         model_class = {"attention": AttentionModel, "ff": FeedForwardModel}.get(
#             opts.eval_models[m], None
#         )
#         model = model_class(
#             opts.embedding_dim,
#             opts.hidden_dim,
#             problem=problem,
#             opts=opts,
#             n_encode_layers=opts.n_encode_layers,
#             mask_inner=True,
#             mask_logits=True,
#             normalization=opts.normalization,
#             tanh_clipping=opts.tanh_clipping,
#             checkpoint_encoder=opts.checkpoint_encoder,
#             shrink_size=opts.shrink_size,
#             num_actions=opts.u_size + 1,
#             n_heads=opts.n_heads,
#             encoder=opts.encoder,
#         ).to(opts.device)

#         if opts.use_cuda and torch.cuda.device_count() > 1:
#             model = torch.nn.DataParallel(model)

#         # Overwrite model parameters by parameters to load
#         model_ = get_inner_model(model)
#         model_.load_state_dict(
#             {**model_.state_dict(), **load_datas[m].get("model", {})}
#         )
#         models.append(model)


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

        avg_cost, cr, avg_cr, op, p, count1, count2, avg_j, wil = evaluate(
            [models[i], greedy], eval_dataloader, opts
        )
        print(f"Average Jaccard Index: {opts.eval_set[i]}: {avg_j}")
        print(f"Wilcoxon test p-value: {opts.eval_set[i]}: {wil}")
        ops.append(op.cpu().numpy())
        ps.append(p.cpu().numpy() / float(opts.eval_size))
        counts.append(count1.cpu().numpy() / float(count1.sum()))
        counts1.append(count2.cpu().numpy() / float(count2.sum()))
    return np.array(ops), np.array(ps), np.array(counts), np.array(counts1)


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
    inv_ff_hist_models = (
        None if opts.inv_ff_hist_models == ["None"] else opts.inv_ff_hist_models
    )

    models = []
    # Initialize models
    # if single_model is not None:
    #     load_datas = load_model(opts)
    #     initialize_models(opts, models, load_datas)
    if att_models is not None:
        load_attention_datas = load_models(opts, opts.att_models)
        initialize_models(
            opts, models, load_attention_datas, AttentionModel
        )  # attention models from the directory
    if ff_models is not None:
        load_ff_datas = load_models(opts, ff_models)
        initialize_models(
            opts, models, load_ff_datas, FeedForwardModel
        )  # feed forwad models from the directory
    if inv_ff_models is not None:
        load_inv_ff_datas = load_models(opts, inv_ff_models)
        initialize_models(
            opts, models, load_inv_ff_datas, InvariantFF
        )  # feed forwad models from the directory
    if ff_hist_models is not None:
        load_ff_datas = load_models(opts, ff_hist_models)
        initialize_models(
            opts, models, load_ff_datas, FeedForwardModelHist
        )  # feed forwad models from the directory
    if inv_ff_hist_models is not None:
        load_inv_ff_datas = load_models(opts, inv_ff_hist_models)
        initialize_models(
            opts, models, load_inv_ff_datas, InvariantFFHist
        )  # feed forwad models from the directory

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

    if len(opts.eval_set) > 0:
        baseline_results = []
        trained_models_results = []
        # plot_data = []
        for m in baseline_models:  # Get the performance of the baselines
            ops = get_model_op_ratios(opts, m, problem)
            baseline_results.append(ops)
        # if single_model is not None:
        #     trained_models_results.append(get_model_op_ratios(opts, models[0], problem))
        print(len(models))
        if att_models is not None or ff_models is not None:
            # Get the performance of the trained models
            trained_models_results.append(
                compare_actions(
                    opts, models[: len(opts.eval_set)], baseline_models[0], problem
                )
            )
            trained_models_results.append(
                compare_actions(
                    opts,
                    models[len(opts.eval_set) : len(opts.eval_set) * 2],
                    baseline_models[0],
                    problem,
                )
            )
            trained_models_results.append(
                compare_actions(
                    opts,
                    models[2 * len(opts.eval_set) : 3 * len(opts.eval_set)],
                    baseline_models[0],
                    problem,
                )
            )
            trained_models_results.append(
                compare_actions(
                    opts, models[3 * len(opts.eval_set) :], baseline_models[0], problem
                )
            )
            # print('baseline_results[0]: ', baseline_results[0])
            # print('trained_models_results ', trained_models_results)
        # print('baseline_results: ', baseline_results)
        # print('trained_models_results ', trained_models_results)

        results = [
            np.array(baseline_results[0]),
            # np.array(baseline_results[1]),
            np.array(trained_models_results[0][0]),
            np.array(trained_models_results[1][0]),
            np.array(trained_models_results[2][0]),
            np.array(trained_models_results[3][0]),
            # np.array(trained_models_results[2][0])
        ]
        # results2 = [
        #     np.array(trained_models_results[0][1]),
        #     np.array(trained_models_results[1][1]),
        # ]
        # results3 = [
        #     np.array(trained_models_results[0][2]),
        #     np.array(trained_models_results[1][2]),
        #     np.array(trained_models_results[1][3]),
        # ]

        # torch.save(
        #    torch.tensor(results),
        #    opts.eval_output + "/{}_{}_{}_{}_{}by{}_results".format(
        #    opts.problem, opts.graph_family,
        #    opts.weight_distribution, opts.weight_distribution_param,
        #    opts.u_size, opts.v_size
        # ).replace(" ",""),
        # )
        plot_box(opts, results)
        # plot_agreemant(opts, results2)
        # plot_action_distribution(opts, results3)
        # line_graph(opts, models + baseline_models , problem)

    # if opts.eval_plot:
    # plot_box(opts, np.array(torch.load(opts.eval_results_folder)))

    # elif opts.eval_model:
    #     model1 = FeedForwardModel(
    #         (opts.u_size + 1) * 2,
    #         opts.hidden_dim,
    #         problem,
    #         n_encode_layers=opts.n_encode_layers,
    #         mask_inner=True,
    #         mask_logits=True,
    #         normalization=opts.normalization,
    #         tanh_clipping=opts.tanh_clipping,
    #         checkpoint_encoder=opts.checkpoint_encoder,
    #         shrink_size=opts.shrink_size,
    #         num_actions=opts.u_size + 1,
    #     ).to(opts.device)
    #     model1_ = get_inner_model(model1)
    #     model1_.load_state_dict({**model1_.state_dict(), **load_data2.get("model", {})})
    #     eval_model([model, model1], problem, opts)


if __name__ == "__main__":
    run(get_options())
