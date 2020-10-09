#!/usr/bin/env python

import os
import json
import pprint as pp

import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger
from torch.utils.data import DataLoader

# from nets.critic_network import CriticNetwork
from options import get_options
from train import train_epoch, validate, get_inner_model, eval_model
from policy.attention_model_v2 import AttentionModel
from policy.ff_model_v2 import FeedForwardModel
from policy.greedy import Greedy
from policy.greedy_rt import GreedyRt
from policy.simple_greedy import SimpleGreedy

import numpy as np
import time
from tqdm import tqdm

import math
import matplotlib.pyplot as plt

from torch.nn import DataParallel

from policy.attention_model_v2 import set_decode_type
from log_utils import log_values
from functions import move_to

# from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from functions import torch_load_cpu, load_problem


def validate_many(opts, model, problem):

    crs = []
    avg_crs = []
    min_p, max_p = float(opts.eval_range[0]), float(opts.eval_range[1])
    for i, j in enumerate(
        np.arange(min_p, max_p, (min_p + max_p) / opts.eval_num_range)
    ):
        dataset_folder = opts.eval_dataset + "/eval{}".format(i)

        val_dataset = problem.make_dataset(dataset_folder, opts.val_size, opts.problem)
        val_dataloader = DataLoader(
            val_dataset, batch_size=opts.eval_batch_size, num_workers=1
        )

        avg_ratio, cr, avg_cr = validate(model, val_dataloader, opts)
        crs.append(cr)
        avg_crs.append(avg_cr)
    plt.figure(1)
    plt.plot(np.arange(min_p, max_p, (min_p + max_p) / opts.eval_num_range), crs)
    plt.xlabel("Graph family parameter")
    plt.ylabel("Competitive ratio")

    plt.savefig(opts.eval_output + "/competitive_ratio.png")

    plt.figure(2)
    plt.plot(np.arange(min_p, max_p, (min_p + max_p) / opts.eval_num_range), avg_crs)
    plt.xlabel("Graph family parameter")
    plt.ylabel("Average ratio to optimal")

    plt.savefig(opts.eval_output + "/avg_optim_ratio.png")


def run(opts):

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), "w") as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Figure out what's the problem
    problem = load_problem(opts.problem)

    # Load data from load_path
    load_data = {}
    assert (
        opts.load_path is None or opts.resume is None
    ), "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print("  [*] Loading data from {}".format(load_path))
        load_data = torch_load_cpu(load_path)
    if opts.load_path2 is not None:
        print("  [*] Loading data from {}".format(opts.load_path2))
        load_data2 = torch_load_cpu(opts.load_path2)
    # Initialize model
    model_class = {
        "attention": AttentionModel,
        "ff": FeedForwardModel,
        "greedy": Greedy,
        "greedy-rt": GreedyRt,
        "simple-greedy": SimpleGreedy,
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)

    model = model_class(
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
    ).to(opts.device)

    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get("model", {})})

    if opts.eval_family:
        validate_many(opts, model, problem)
    elif opts.eval_model:
        model1 = FeedForwardModel(
            (opts.u_size + 1) * 2,
            opts.hidden_dim,
            problem,
            n_encode_layers=opts.n_encode_layers,
            mask_inner=True,
            mask_logits=True,
            normalization=opts.normalization,
            tanh_clipping=opts.tanh_clipping,
            checkpoint_encoder=opts.checkpoint_encoder,
            shrink_size=opts.shrink_size,
            num_actions=opts.u_size + 1,
        ).to(opts.device)
        model1_ = get_inner_model(model1)
        model1_.load_state_dict({**model1_.state_dict(), **load_data2.get("model", {})})
        eval_model([model, model1], problem, opts)


if __name__ == "__main__":
    run(get_options())
