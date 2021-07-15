def log_values(
    cost,
    epoch,
    batch_id,
    step,
    log_likelihood,
    tb_logger,
    opts,
    grad_norms=None,
    batch_loss=None,
    reinforce_loss=None,
    bl_loss=None,
):
    avg_cost = cost.mean().item()
    if grad_norms is not None:
        grad_norms, grad_norms_clipped = grad_norms
        print("grad_norm: {}, clipped: {}".format(grad_norms[0], grad_norms_clipped[0]))
    # Log values to screen
    print(
        "epoch: {}, train_batch_id: {}, avg_cost: {}".format(epoch, batch_id, avg_cost)
    )

    # Log values to tensorboard
    if not opts.no_tensorboard:
        tb_logger.add_scalar("avg_cost", avg_cost, step)

        if opts.model == "ff-supervised" and batch_loss is not None:
            tb_logger.add_scalar("batch loss", batch_loss, step)
        else:
            if reinforce_loss is not None:
                tb_logger.add_scalar("actor_loss", reinforce_loss.item(), step)
            # tb_logger.add_scalar("nll", -sum(log_likelihood)/len(log_likelihood).item(), step)

            if grad_norms is not None:
                tb_logger.add_scalar("grad_norm", grad_norms[0], step)
                tb_logger.add_scalar("grad_norm_clipped", grad_norms_clipped[0], step)

            if opts.baseline == "critic":
                tb_logger.add_scalar("critic_loss", bl_loss.item(), step)
                tb_logger.add_scalar("critic_grad_norm", grad_norms[1], step)
                tb_logger.add_scalar(
                    "critic_grad_norm_clipped", grad_norms_clipped[1], step
                )
