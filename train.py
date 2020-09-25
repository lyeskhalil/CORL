import os
import time
from tqdm import tqdm
import torch
import math
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from policy.attention_model import set_decode_type
from log_utils import log_values
from functions import move_to


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


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

    return avg_cost, min_cr.item()


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


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device), opts)

        # print(-cost.data.flatten())
        # print(bat[-1])
        cr = -cost.data.flatten() / move_to(bat[-1], opts.device)
        # print(
        #     "\nBatch Competitive ratio: ", min(cr).item(),
        # )

        return cost.data.cpu(), cr

    cost = []
    crs = []
    for bat in tqdm(
        DataLoader(dataset, batch_size=opts.eval_batch_size),
        disable=opts.no_progress_bar,
    ):
        c, cr = eval_model_bat(bat)
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
    optimizer,
    baseline,
    lr_scheduler,
    epoch,
    val_dataset,
    problem,
    tb_logger,
    opts,
):
    print(
        "Start train epoch {}, lr={} for run {}".format(
            epoch, optimizer.param_groups[0]["lr"], opts.run_name
        )
    )
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value("learnrate_pg0", optimizer.param_groups[0]["lr"], step)

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(
        problem.make_dataset(
            u_size=opts.u_size,
            v_size=opts.v_size,
            num_edges=opts.num_edges,
            num_samples=opts.epoch_size,
            distribution=opts.data_distribution,
            max_weight=opts.max_weight,
        )
    )
    training_dataloader = DataLoader(
        training_dataset, batch_size=opts.batch_size, num_workers=1
    )

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(
        tqdm(training_dataloader, disable=opts.no_progress_bar)
    ):

        train_batch(
            model, optimizer, baseline, epoch, batch_id, step, batch, tb_logger, opts
        )

        step += 1

    epoch_duration = time.time() - start_time
    print(
        "Finished epoch {}, took {} s".format(
            epoch, time.strftime("%H:%M:%S", time.gmtime(epoch_duration))
        )
    )

    if opts.checkpoint_epochs == 0:
        print("Saving model and state...")
        torch.save(
            {
                "model": get_inner_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
                "baseline": baseline.state_dict(),
            },
            os.path.join(opts.save_dir, "latest-{}.pt".format(epoch)),
        )
    elif (epoch % opts.checkpoint_epochs == 0) or (epoch == opts.n_epochs - 1):
        print("Saving model and state...")
        torch.save(
            {
                "model": get_inner_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
                "baseline": baseline.state_dict(),
            },
            os.path.join(opts.save_dir, "epoch-{}.pt".format(epoch)),
        )

    avg_reward, min_cr = validate(model, val_dataset, opts)

    if not opts.no_tensorboard:
        tb_logger.log_value("val_avg_reward", avg_reward, step)
        tb_logger.log_value("min_competitive_ratio", min_cr, step)

    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()


def train_batch(
    model, optimizer, baseline, epoch, batch_id, step, batch, tb_logger, opts
):
    x, bl_val = baseline.unwrap_batch(batch)
    # print(x)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood = model(x, opts)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

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
