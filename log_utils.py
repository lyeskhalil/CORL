def log_values(
    cost,
    grad_norms,
    epoch,
    batch_id,
    step,
    log_likelihood,
    reinforce_loss,
    bl_loss,
    tb_logger,
    batch_loss,
    opts,
):
    avg_cost = cost.mean().item()
    if grad_norms != None:
        grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    print(
        "epoch: {}, train_batch_id: {}, avg_cost: {}".format(epoch, batch_id, avg_cost)
    )

    print("grad_norm: {}, clipped: {}".format(grad_norms[0], grad_norms_clipped[0]))

    # Log values to tensorboard
    if not opts.no_tensorboard:
        tb_logger.add_scalar("avg_cost", avg_cost, step)

        if reinforce_loss != None:
            tb_logger.add_scalar("actor_loss", reinforce_loss.item(), step)
        tb_logger.add_scalar("nll", -log_likelihood.mean().item(), step)

        tb_logger.add_scalar("grad_norm", grad_norms[0], step)
        tb_logger.add_scalar("grad_norm_clipped", grad_norms_clipped[0], step)

        if opts.baseline == "critic" and opts.model != 'supervised':
            tb_logger.add_scalar("critic_loss", bl_loss.item(), step)
            tb_logger.add_scalar("critic_grad_norm", grad_norms[1], step)
            tb_logger.add_scalar(
                "critic_grad_norm_clipped", grad_norms_clipped[1], step
            )
        if opt.model == 'supervised' and batch_loss != None:
            tb_logger.add_scalar("batch loss", batch_loss, step)

