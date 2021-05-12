import os
import time
from tqdm import tqdm
import torch
import math
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from policy.attention_model_v2 import set_decode_type
from log_utils import log_values
from functions import move_to

import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def evaluate(models, dataset, opts):
    print("Evaluating...")
    cost, cr, p, count1, count2, avg_j, wil = rollout_eval(models, dataset, opts)
    avg_cost = cost.mean()

    min_cr = min(cr)
    avg_cr = cr.mean()

    min_cr = min(cr)
    avg_cr = cr.mean()
    print(
        "Evaluation overall avg_cost: {} +- {}".format(
            avg_cost, torch.std(cost) / math.sqrt(len(cost))
        )
    )
    print(
        "\nEvaluation overall avg ratio to optimal: {} +- {}".format(
            avg_cr, torch.std(cr) / math.sqrt(len(cr))
        )
    )
    print("\nEvaluation competitive ratio", min_cr.item())

    return avg_cost, min_cr.item(), avg_cr, cr, p, count1, count2, avg_j, wil


def validate(model, dataset, opts):
    # Validate
    print("Validating...")
    cost, cr = rollout(model, dataset, opts)
    avg_cost = cost.mean()

    min_cr = min(cr)
    avg_cr = cr.mean()
    print(
        "Validation overall avg_cost: {} +- {}".format(
            avg_cost, torch.std(cost) / math.sqrt(len(cost))
        )
    )
    print(
        "\nValidation overall avg ratio to optimal: {} +- {}".format(
            avg_cr, torch.std(cr) / math.sqrt(len(cr))
        )
    )
    print("\nValidation competitive ratio", min_cr.item())

    return avg_cost, min_cr.item(), avg_cr


def eval_model(models, problem, opts):
    for j in range(len(models)):
        c, avg_crs, var_crs, min_cr, ratio = [], [], [], [], []
        for i in range(opts.eval_num):
            dataset = problem.make_dataset(
                u_size=opts.u_size,
                v_size=opts.u_size + i * 1,
                num_edges=opts.num_edges + (opts.u_size // 2) * i * 1,
                max_weight=opts.max_weight,
                num_samples=opts.val_size,
                distribution=opts.data_distribution,
            )
            cost, cr = rollout(models[j], dataset, opts)
            ratio.append(opts.u_size / (opts.u_size + i * 1))
            c.append(cost)
            min_cr.append(min(cr).item())
            var_crs.append(torch.std(cr) / math.sqrt(len(cr)))
            avg_crs.append(cr.mean())
        plt.plot(ratio, avg_crs)
    plt.xlabel("Ratio of U to V")
    plt.ylabel("Average Optimality Ratio")
    plt.savefig("graph1.png")
    return


def rollout_eval(models, dataset, opts):
    # Put in greedy evaluation mode!
    model = models[0]
    g = models[1]
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat, optimal):
        with torch.no_grad():
            cost, _, a = model(
                move_to(bat, opts.device),
                opts,
                baseline=None,
                return_pi=True,
                optimizer=None,
            )
            cost1, _, a1 = g(
                move_to(bat, opts.device),
                opts,
                baseline=None,
                return_pi=True,
                optimizer=None,
            )
        # print(-cost.data.flatten())
        jaccard = (a == a1).float().sum(1) / (
            2 * opts.v_size - (a == a1).float().sum(1)
        )
        num_agree = ((a == a1).float()).sum(0)
        count = torch.bincount(a[:, :20].flatten(), minlength=opts.u_size + 1)
        count1 = torch.bincount(a1[:, :20].flatten(), minlength=opts.u_size + 1)
        if (cost == cost1).all().item():
            w, p = 0, 0
        else:
            w, p = wilcoxon(-cost.squeeze(), -cost1.squeeze(), alternative="greater")
        # print(bat[-1])
        cr = (
            -cost.data.flatten()
            * opts.u_size
            / move_to(batch.y + (batch.y == 0).float(), opts.device)
        )
        # print(
        #     "\nBatch Competitive ratio: ", min(cr).item(),
        # )
        return (
            cost.data.cpu() * opts.u_size,
            cr,
            num_agree,
            count,
            count1,
            jaccard,
            [w, p],
        )

    cost = []
    crs = []
    n = []
    count_actions = []
    count_actions1 = []
    avg_jaccard = []
    wp = []
    for batch in tqdm(dataset):
        c, cr, num_agree, count, count1, j, wilcox = eval_model_bat(batch, None)
        cost.append(c)
        crs.append(cr)
        n.append(num_agree[None, :])
        count_actions.append(count[None, :])
        count_actions1.append(count1[None, :])
        avg_jaccard.append(j[None, :])
        wp.append(torch.tensor(wilcox)[None, :])
    return (
        torch.cat(cost, 0),
        torch.cat(crs, 0),
        torch.cat(n, 0).sum(0),
        torch.cat(count_actions, 0).sum(0),
        torch.cat(count_actions1, 0).sum(0),
        torch.cat(avg_jaccard, 0).mean(),
        torch.cat(wp, 0),
    )


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat, optimal):
        with torch.no_grad():
            cost, *_ = model(move_to(bat, opts.device), opts, None, None)

        # print(-cost.data.flatten())
        # print(bat[-1])
        # print(batch.y)
        cr = (-cost.data.flatten()) / move_to(
            batch.y + (batch.y == 0).float(), opts.device
        )
        # print(
        #     "\nBatch Competitive ratio: ", min(cr).item(),
        # )
        return cost.data.cpu(), cr

    cost = []
    crs = []
    for batch in tqdm(dataset):
        c, cr = eval_model_bat(batch, None)
        cost.append(c)
        crs.append(cr)
    return torch.cat(cost, 0), torch.cat(crs, 0)

    # return torch.cat(
    #     [
    #         eval_model_bat(bat)
    #         for bat in tqdm(
    #             DataLoader(dataset, batch_size=opts.eval_batch_size),
    #             disable=opts.no_progress_bar,
    #         )
    #     ],
    #     0,
    # )


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group["params"],
            max_norm
            if max_norm > 0
            else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2,
        )
        for group in param_groups
    ]
    grad_norms_clipped = (
        [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    )
    return grad_norms, grad_norms_clipped


def train_epoch(
    model,
    optimizers,
    baseline,
    lr_schedulers,
    epoch,
    val_dataset,
    training_dataloader,
    problem,
    tb_logger,
    opts,
):
    print(
        "Start train epoch {}, lr={} for run {}".format(
            epoch, optimizers[0].param_groups[0]["lr"], opts.run_name
        )
    )
    step = epoch * (opts.dataset_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.add_scalar("learnrate_pg0", optimizers[0].param_groups[0]["lr"], step)

    # Generate new training data for each epoch
    ## TODO: MODIFY SO THAT WE CAN ALSO USE A PRE-GENERATED DATASET
    # training_dataset = baseline.wrap_dataset(problem.make_dataset(opts))
    # training_dataloader = DataLoader(
    #     training_dataset, batch_size=opts.batch_size, num_workers=1
    # )

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")
    for batch_id, batch in enumerate(
        tqdm(training_dataloader, disable=opts.no_progress_bar)
    ):
        train_batch(
            model, optimizers, baseline, epoch, batch_id, step, batch, tb_logger, opts
        )

        step += 1

    epoch_duration = time.time() - start_time
    print(
        "Finished epoch {}, took {} s".format(
            epoch, time.strftime("%H:%M:%S", time.gmtime(epoch_duration))
        )
    )

    if (
        opts.checkpoint_epochs == 0 and (epoch == opts.n_epochs - 1) and not opts.tune
    ):  # TODO: This does not save both optimizers
        print("Saving model and state...")
        torch.save(
            {
                "model": get_inner_model(model).state_dict(),
                "optimizer": optimizers[0].state_dict(),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
                "baseline": baseline.state_dict(),
            },
            os.path.join(opts.save_dir, "latest-{}.pt".format(epoch)),
        )
    elif (
        not opts.tune
        and (opts.checkpoint_epochs != 0)
        and ((epoch % opts.checkpoint_epochs == 0) or (epoch == opts.n_epochs - 1))
    ):
        print("Saving model and state...")
        torch.save(
            {
                "model": get_inner_model(model).state_dict(),
                "optimizer": optimizers[0].state_dict(),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
                "baseline": baseline.state_dict(),
            },
            os.path.join(opts.save_dir, "epoch-{}.pt".format(epoch)),
        )

    avg_reward, min_cr, avg_cr = validate(model, val_dataset, opts)
    # avg_reward, min_cr, avg_cr = 0,0,0
    if not opts.no_tensorboard:
        tb_logger.add_scalar("val_avg_reward", -avg_reward, step)
        tb_logger.add_scalar("min_competitive_ratio", min_cr, step)

    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_schedulers[0].step()
    #    lr_schedulers[1].step()

    return avg_reward, min_cr, avg_cr


def train_n_step(cost, ll, x, optimizer, baseline):
    bl_val, bl_loss = baseline.eval(x, cost)

    # Calculate loss
    # print("\nCost: " , cost.item())
    reinforce_loss = ((cost.squeeze(1) - bl_val) * ll).mean()
    loss = reinforce_loss + bl_loss
    # print(loss.item())
    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return


def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend(
        [
            Line2D([0], [0], color="c", lw=4),
            Line2D([0], [0], color="b", lw=4),
            Line2D([0], [0], color="k", lw=4),
        ],
        ["max-gradient", "mean-gradient", "zero-gradient"],
    )
    plt.show()
    plt.savefig("grad.png")


def train_batch(
    model, optimizers, baseline, epoch, batch_id, step, batch, tb_logger, opts
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities

    cost, log_likelihood, e = model(x, opts, optimizers, baseline)
    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    # print("\nCost: " , cost.item())
    grad_norms = [[0, 0], [0, 0]]
    reinforce_loss = torch.tensor(0)
    loss = 0
    if not opts.n_step:
        reinforce_loss = ((cost.squeeze(1) - bl_val) * log_likelihood).mean()
        loss = reinforce_loss + bl_loss - opts.ent_rate * e

        # Perform backward pass and optimization step
        optimizers[0].zero_grad()
        loss.backward()

        # sum_grad = 0
        # for name, p in model.named_parameters():
        #     print(name, p.grad.norm())
        # print(sum_grad)
        # print(p.data)
        # Clip gradient norms and get (clipped) gradient norms for logging
        grad_norms = clip_grad_norms(optimizers[0].param_groups, opts.max_grad_norm)
        optimizers[0].step()
    # grad_norms1 = clip_grad_norms(optimizers[1].param_groups, opts.max_grad_norm)
    # optimizers[1].zero_grad()
    # optimizers[1].step() # Gradient update for node embedding init layer
    # Logging
    if step % int(opts.log_step) == 0:
        log_values(
            cost,
            grad_norms,
            epoch,
            batch_id,
            step,
            log_likelihood,
            reinforce_loss,
            bl_loss,
            tb_logger,
            opts,
        )
