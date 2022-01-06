import os
import numpy as np
import json
import pprint as pp

import torch
import torch.optim as optim
from itertools import product
import wandb

# from tensorboard_logger import Logger as TbLogger
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader as geoDataloader

# from nets.critic_network import CriticNetwork
from options import get_options
from train import train_epoch, validate, get_inner_model
from utils.reinforce_baselines import (
    NoBaseline,
    ExponentialBaseline,
    RolloutBaseline,
    WarmupBaseline,
    GreedyBaseline,
)

from policy.attention_model import AttentionModel as AttentionModelgeo
from policy.ff_model import FeedForwardModel
from policy.ff_model_invariant import InvariantFF
from policy.ff_model_hist import FeedForwardModelHist
from policy.inv_ff_history import InvariantFFHist
from policy.greedy import Greedy
from policy.greedy_rt import GreedyRt
from policy.greedy_theshold import GreedyThresh
from policy.greedy_matching import GreedyMatching
from policy.simple_greedy import SimpleGreedy
from policy.supervised import SupervisedModel
from policy.ff_supervised import SupervisedFFModel
from policy.gnn_hist import GNNHist
from policy.gnn_simp_hist import GNNSimpHist
from policy.gnn import GNN

# from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils.functions import torch_load_cpu, load_problem


def run(opts):

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)
    # torch.backends.cudnn.benchmark = True
    # torch.autograd.set_detect_anomaly(True)
    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = SummaryWriter(
            os.path.join(
                opts.log_dir,
                opts.model,
                opts.run_name,
            )
        )
    if not opts.eval_only and not os.path.exists(opts.save_dir):
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
    # if opts.load_path2 is not None:
    #     print("  [*] Loading data from {}".format(opts.load_path2))
    #     load_data2 = torch_load_cpu(opts.load_path2)
    # Initialize model
    model_class = {
        "attention": AttentionModelgeo,
        "ff": FeedForwardModel,
        "greedy": Greedy,
        "greedy-rt": GreedyRt,
        "greedy-t": GreedyThresh,
        "greedy-m": GreedyMatching,
        "simple-greedy": SimpleGreedy,
        "inv-ff": InvariantFF,
        "inv-ff-hist": InvariantFFHist,
        "ff-hist": FeedForwardModelHist,
        "supervised": SupervisedModel,
        "ff-supervised": SupervisedFFModel,
        "gnn-hist": GNNHist,
        "gnn-simp-hist": GNNSimpHist,
        "gnn": GNN,
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    # if not opts.tune:
    model, lr_schedulers, optimizers, val_dataloader, baseline = setup_training_env(
        opts, model_class, problem, load_data, tb_logger
    )

    training_dataset = problem.make_dataset(
        opts.train_dataset, opts.dataset_size, opts.problem, seed=None, opts=opts
    )
    # training_dataloader = DataLoader(
    #    baseline.wrap_dataset(training_dataset), batch_size=opts.batch_size, num_workers=1, shuffle=True,
    # )
    # training_dataloader = training_dataset
    if opts.eval_only:
        validate(model, val_dataloader, opts)
    elif opts.tune_wandb:
        # wandb.login(key="e49f6e29371d2198953129649f6352f26d5a6fd5", relogin=True)
        wandb.agent(
            sweep_id=opts.sweep_id,
            function=lambda config=None: train_wandb(
                model_class, problem, tb_logger, opts, config=config
            ),
            count=opts.num_per_agent,
            project="CORL",
        )
    elif opts.tune:
        PARAM_GRID = list(
            product(
                [
                    0.01,
                    0.001,
                    0.0001,
                    0.00001,
                    0.02,
                    0.002,
                    0.0002,
                    0.00002,
                    0.03,
                    0.003,
                    0.0003,
                    0.00003,
                ],  # learning_rate
                #            [(20, 1), (30, 1), (40, 4)],  # embedding size
                [0.75, 0.85, 0.8, 0.9, 0.95],  # baseline exponential decay
                [1.0, 0.99, 0.98, 0.97, 0.96],  # lr decay
            )
        )
        # total number of slurm workers detected
        # defaults to 1 if not running under SLURM
        N_WORKERS = int(os.getenv("SLURM_ARRAY_TASK_COUNT", 1))

        # this worker's array index. Assumes slurm array job is zero-indexed
        # defaults to zero if not running under SLURM
        this_worker = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
        SCOREFILE = os.path.expanduser(
            f"./val_rewards_{opts.model}_{opts.u_size}_{opts.v_size}_{opts.graph_family}_{opts.graph_family_parameter}_2.csv"
        )
        for param_ix in range(this_worker, len(PARAM_GRID), N_WORKERS):
            torch.manual_seed(opts.seed)
            params = PARAM_GRID[param_ix]
            lr = params[0]
            #            embedding_dim = params[1][0]
            #            n_heads = params[1][1]
            exp_decay = params[1]
            lr_decay = params[2]
            opts.lr_model = lr
            opts.lr_decay = lr_decay
            opts.exp_beta = exp_decay
            # opts.embedding_dim = embedding_dim
            # opts.n_heads = n_heads
            if not opts.no_tensorboard:
                tb_logger = SummaryWriter(
                    os.path.join(
                        opts.log_dir,
                        "{}_{}_{}_{}_{}".format(
                            opts.lr_decay,
                            opts.exp_beta,
                            opts.lr_model,
                            opts.embedding_dim,
                            opts.n_heads,
                        ),
                        opts.run_name,
                    )
                )
            load_data = {}
            (
                model,
                lr_schedulers,
                optimizers,
                val_dataloader,
                baseline,
            ) = setup_training_env(opts, model_class, problem, load_data, tb_logger)
            training_dataset = problem.make_dataset(
                opts.train_dataset,
                opts.dataset_size,
                opts.problem,
                seed=None,
                opts=opts,
            )

            # training_dataloader = DataLoader(
            #    baseline.wrap_dataset(training_dataset), batch_size=opts.batch_size, num_workers=1, shuffle=True,
            # )
            best_avg_cr = 0
            for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
                training_dataloader = geoDataloader(
                    baseline.wrap_dataset(training_dataset),
                    batch_size=opts.batch_size,
                    num_workers=0,
                    shuffle=True,
                )
                avg_reward, min_cr, avg_cr, loss = train_epoch(
                    model,
                    optimizers,
                    baseline,
                    lr_schedulers,
                    epoch,
                    val_dataloader,
                    training_dataloader,
                    problem,
                    tb_logger,
                    opts,
                    best_avg_cr,
                )
                best_avg_cr = max(best_avg_cr, avg_cr)
            avg_reward, min_cr, avg_cr = avg_reward.item(), min_cr, avg_cr.item()
            with open(SCOREFILE, "a") as f:
                f.write(f'{",".join(map(str, params + (avg_reward,min_cr,avg_cr)))}\n')
    elif opts.tune_baseline:
        PARAM_GRID = np.round(np.linspace(0, 1, 100).tolist(), decimals=2) # Threshold for greedy-t
        N_WORKERS = int(os.getenv("SLURM_ARRAY_TASK_COUNT", 1))

        # this worker's array index. Assumes slurm array job is zero-indexed
        # defaults to zero if not running under SLURM
        this_worker = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
        SCOREFILE = os.path.expanduser(
            f"./val_rewards_{opts.model}_{opts.u_size}_{opts.v_size}_{opts.graph_family}_{opts.graph_family_parameter}.csv"
        )
        for param_ix in range(this_worker, len(PARAM_GRID), N_WORKERS):
            torch.manual_seed(opts.seed)
            params = PARAM_GRID[param_ix]
            training_dataloader = geoDataloader(
                baseline.wrap_dataset(training_dataset),
                batch_size=opts.batch_size,
                num_workers=0,
                shuffle=True,
            )

            opts.threshold = params
            (
                model,
                lr_schedulers,
                optimizers,
                val_dataloader,
                baseline,
            ) = setup_training_env(opts, model_class, problem, load_data, tb_logger)
            avg_cost, *_ = validate(model, training_dataloader, opts)
            with open(SCOREFILE, "a") as f:
                f.write(f'{",".join(map(str, (params,) + (avg_cost.item(),)))}\n')

    else:
        best_avg_cr = 0.0
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            # with profiler.profile() as prof:
            #    with profiler.record_function("model_inference"):

            training_dataloader = geoDataloader(
                baseline.wrap_dataset(training_dataset),
                batch_size=opts.batch_size,
                num_workers=0,
                shuffle=True,
            )
            avg_reward, min_cr, avg_cr, loss = train_epoch(
                model,
                optimizers,
                baseline,
                lr_schedulers,
                epoch,
                val_dataloader,
                training_dataloader,
                problem,
                tb_logger,
                opts,
                best_avg_cr,
            )
            # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
            best_avg_cr = max(best_avg_cr, avg_cr)


def train_wandb(model_class, problem, tb_logger, opts, config=None):
    with wandb.init(config=config):
        torch.manual_seed(opts.seed)
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        opts.lr_model = config.lr_model
        opts.lr_decay = config.lr_decay
        opts.exp_beta = config.exp_beta
        opts.ent_rate = config.ent_rate
        load_data = {}
        (
            model,
            lr_schedulers,
            optimizers,
            val_dataloader,
            baseline,
        ) = setup_training_env(opts, model_class, problem, load_data, tb_logger)
        training_dataset = problem.make_dataset(
            opts.train_dataset,
            opts.dataset_size,
            opts.problem,
            seed=None,
            opts=opts,
        )

        # training_dataloader = DataLoader(
        #    baseline.wrap_dataset(training_dataset), batch_size=opts.batch_size, num_workers=1, shuffle=True,
        # )
        best_avg_cr = 0.0
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            training_dataloader = geoDataloader(
                baseline.wrap_dataset(training_dataset),
                batch_size=opts.batch_size,
                num_workers=0,
                shuffle=True,
            )
            avg_reward, min_cr, avg_cr, loss = train_epoch(
                model,
                optimizers,
                baseline,
                lr_schedulers,
                epoch,
                val_dataloader,
                training_dataloader,
                problem,
                tb_logger,
                opts,
                best_avg_cr,
            )
            best_avg_cr = max(best_avg_cr, avg_cr)
            if "supervised" in opts.model:

                wandb.log(
                    {
                        "val_reward": abs(avg_reward),
                        "avg_cr": abs(avg_cr),
                        "min_cr": abs(min_cr),
                        "val_loss": loss,
                    }
                )
            else:
                wandb.log(
                    {
                        "val_reward": abs(avg_reward),
                        "avg_cr": abs(avg_cr),
                        "min_cr": abs(min_cr),
                    }
                )


def setup_training_env(opts, model_class, problem, load_data, tb_logger):
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
        n_heads=opts.n_heads,
        encoder=opts.encoder,
        opts=opts,
    ).to(opts.device)

    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get("model", {})})

    # Initialize baseline
    if opts.baseline == "exponential":
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == "greedy":
        baseline_class = {"e-obm": Greedy, "obm": SimpleGreedy}.get(opts.problem, None)

        greedybaseline = baseline_class(
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
            # n_heads=opts.n_heads,
        )
        baseline = GreedyBaseline(greedybaseline, opts)
    elif opts.baseline == "rollout":
        baseline = RolloutBaseline(model, problem, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(
            baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta
        )

    # Load baseline from data, make sure script is called with same type of baseline
    if "baseline" in load_data:
        baseline.load_state_dict(load_data["baseline"])
    # init_node_embedding_weights = (
    #     "project_node_features.weight",
    #     "project_node_features.bias",
    # )
    # parameters = (
    #     p
    #     for name, p in model.named_parameters()
    # )
    # parameters1 = (
    #     p for name, p in model.named_parameters() if name in init_node_embedding_weights
    # )
    # Initialize optimizer
    optimizer = optim.Adam(
        [{"params": model.parameters(), "lr": opts.lr_model}]
        + (
            [{"params": baseline.get_learnable_parameters(), "lr": opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )
    # optimizer1 = optim.Adam([{"params": parameters1, "lr": opts.lr_model}])

    # Load optimizer state
    if "optimizer" in load_data:
        optimizer.load_state_dict(load_data["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: opts.lr_decay ** epoch
    )
    # lr_scheduler1 = optim.lr_scheduler.LambdaLR(
    #     optimizer1, lambda epoch: opts.lr_decay ** epoch
    # )
    # Start the actual training loop
    val_dataset = problem.make_dataset(
        opts.val_dataset, opts.val_size, opts.problem, seed=None, opts=opts
    )
    val_dataloader = geoDataloader(
        val_dataset, batch_size=opts.batch_size, num_workers=1
    )
    if opts.resume:  # TODO: This does not resume both optimizers
        epoch_resume = int(
            os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1]
        )

        torch.set_rng_state(load_data["rng_state"])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data["cuda_rng_state"])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1
    return (
        model,
        [lr_scheduler],
        [optimizer],
        val_dataloader,
        baseline,
    )


if __name__ == "__main__":
    run(get_options())
