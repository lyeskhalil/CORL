import os
import time
import argparse
import torch


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Attention-based model for solving the Online Bipartite matching Problem with Reinforcement Learning"
    )

    # Data
    parser.add_argument(
        "--problem",
        type=str,
        default="obm",
        help="Problem: 'obm', 'e-obm', 'adwords' or 'displayads'",
    )
    parser.add_argument(
        "--graph_size", type=int, default=20, help="The size of the problem graph"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Number of instances per batch during training",
    )
    parser.add_argument(
        "--u_size", type=int, default=10, help="Number of nodes in U-set"
    )
    parser.add_argument(
        "--v_size", type=int, default=10, help="Number of nodes in the V-set"
    )
    parser.add_argument(
        "--graph_family",
        type=str,
        default="er",
        help="family of graphs to generate (er, ba, etc)",
    )
    parser.add_argument(
        "--num_edges",
        type=int,
        default=20,
        help="Number of edges in the Bipartite graph",
    )
#    parser.add_argument(
#        "--epoch_size",
#        type=int,
#        default=100,
#        help="Number of instances per epoch during training",
#    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=1000,
        help="Number of instances used for reporting validation performance",
    )
    parser.add_argument(
        "--val_dataset",
        type=str,
        default="dataset/val",
        help="Dataset file to use for validation",
    )

    parser.add_argument(
        "--train_dataset",
        type=str,
        default="dataset/train",
        help="Dataset file to use for training",
    )
    
    parser.add_argument(
        "--dataset_size", type=int, default=1000, help="Dataset size for training",
    )

    parser.add_argument(
        "--weight_distribution",
        type=str,
        default="uniform",
        help="Distribution of weights in graphs",
    )

   
    # Model
    parser.add_argument(
        "--model",
        default="attention",
        help="Model, 'attention' (default) or 'pointer or Feed forward'",
    )
    parser.add_argument(
        "--encoder",
        default="attention",
        help="Encoder, 'attention' (default) or 'mpnn'",
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=16, help="Dimension of input embedding"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=16,
        help="Dimension of hidden layers in Enc/Dec",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=2,
        help="Number of heads in Enc",
    )
    parser.add_argument(
        "--n_encode_layers",
        type=int,
        default=3,
        help="Number of layers in the encoder/critic network",
    )
    parser.add_argument(
        "--tanh_clipping",
        type=float,
        default=10.0,
        help="Clip the parameters to within +- this value using tanh. "
        "Set to 0 to not perform any clipping.",
    )
    parser.add_argument(
        "--normalization",
        default="batch",
        help="Normalization type, 'batch' (default) or 'instance'",
    )
    
    # Training
    parser.add_argument(
        "--lr_model",
        type=float,
        default=1e-3,
        help="Set the learning rate for the actor network",
    )
    parser.add_argument(
        "--lr_critic",
        type=float,
        default=1e-4,
        help="Set the learning rate for the critic network",
    )
    parser.add_argument(
        "--lr_decay", type=float, default=0.99, help="Learning rate decay per epoch"
    )

    parser.add_argument(
        "--n_epochs", type=int, default=1000, help="The number of epochs to train"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed to use")
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "--exp_beta",
        type=float,
        default=0.8,
        help="Exponential moving average baseline decay (default 0.8)",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Baseline to use: 'rollout', 'critic' or 'exponential'. Defaults to no baseline.",
    )
    parser.add_argument(
        "--bl_alpha",
        type=float,
        default=0.05,
        help="Significance in the t-test for updating rollout baseline",
    )
    parser.add_argument(
        "--bl_warmup_epochs",
        type=int,
        default=None,
        help="Number of epochs to warmup the baseline, default None means 1 for rollout (exponential "
        "used for warmup phase), 0 otherwise. Can only be used with rollout baseline.",
    )
    parser.add_argument(
        "--max_weight", type=int, default=4000, help="Maximum edge weight in the graph"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=10,
        help="Batch size to use during (baseline) evaluation",
    )
    parser.add_argument(
        "--checkpoint_encoder",
        action="store_true",
        help="Set to decrease memory usage by checkpointing encoder",
    )
    parser.add_argument(
        "--shrink_size",
        type=int,
        default=None,
        help="Shrink the batch size if at least this many instances in the batch are finished"
        " to save memory (default None means no shrinking)",
    )
    parser.add_argument(
        "--data_distribution",
        type=str,
        default=None,
        help="Data distribution to use during training, defaults and options depend on problem.",
    )
    parser.add_argument(
        "--weight_distribution_param",
        nargs="+",
        default=[5,4000],
        help="parameters of weight distribtion ",
    )

    parser.add_argument(
       "--graph_family_parameter",
        type=float,
        default=0.6,
        help="parameter of the graph family distribution",
    )
    
    # Evaluation
    
    parser.add_argument(
        "--eval_num",
        type=int,
        default=5,
        help="Number of U to V ratio's to evaluate the model on",
    )
    parser.add_argument(
        "--eval_size",
        type=int,
        default=10000,
        help="Number of examples in an evaluation dataset.",
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        help="path to folder containing all evaluation datasets",
    )
    parser.add_argument(
        "--eval_baselines", nargs="+", help="Different models to evaluate on",
        # Example: ["greedy", "greedy-rt"]
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Set this value to only evaluate model on a specific graph size",
    )
    parser.add_argument(
        "--eval_plot", action="store_true", help="plot results on test data",
    )
    parser.add_argument(
        "--eval_results_file", type=str, help="file that containes test results",
    )
    parser.add_argument(
        "--eval_range",
        nargs="+",
        help="evaluate model over a range of graph family parameters",
    )
   # parser.add_argument(
   #     "--eval_model_paths", nargs="+", help="paths to trained models files",
   # )
    parser.add_argument(
        "--load_path", help="Path to load model parameters and optimizer state from"
    )
    parser.add_argument(
        "--eval_ff_dir", type=str, help="path to the directory containing trained ff neural nets", 
    )
    parser.add_argument(
        "--eval_attention_dir", type=str, help="path to the directory containing trained attention models", 
    )
    parser.add_argument(
        "--eval_models", nargs="+", help="type of models to evaluate",
    )
    parser.add_argument(
        "--eval_set", nargs="+", help="Set of family parameters to evaluate models on",
    )
    
    parser.add_argument(
        "--eval_num_range",
        type=int,
        default=10,
        help="Number of grpah family parameter to evaluate model on over a specific range",
    )

#    parser.add_argument(
#        "--eval_family",
#        action="store_true",
#        help="Set this to true if you evaluating the model over a family of graphs",
#    )
    parser.add_argument(
        "--eval_output", type=str, help="path to output evaulation plots",
    )
    
    # Misc
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Set this to true if you want to tune the hyperparameters",
    )
    parser.add_argument(
        "--log_step", type=int, default=50, help="Log info every log_step steps"
    )
    parser.add_argument(
        "--log_dir",
        default="logs",
        help="Directory to write TensorBoard information to",
    )
    parser.add_argument("--run_name", default="run", help="Name to identify the run")
    parser.add_argument(
        "--output_dir", default="outputs", help="Directory to write output models to"
    )
    parser.add_argument(
        "--epoch_start",
        type=int,
        default=0,
        help="Start at epoch # (relevant for learning rate decay)",
    )
    parser.add_argument(
        "--checkpoint_epochs",
        type=int,
        default=0,
        help="Save checkpoint every n epochs (default 1), 0 to save no checkpoints",
    )
   
    parser.add_argument(
        "--load_path2",
        help="Path to load second model parameters and optimizer state from",
    )
    parser.add_argument("--resume", help="Resume from previous checkpoint file")
    parser.add_argument(
        "--no_tensorboard",
        action="store_true",
        help="Disable logging TensorBoard files",
    )
    parser.add_argument(
        "--no_progress_bar", action="store_true", help="Disable progress bar"
    )
 
    opts = parser.parse_args(args)
    
    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(
        opts.output_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name
    )
    if opts.bl_warmup_epochs is None:
        opts.bl_warmup_epochs = 1 if opts.baseline == "rollout" else 0
    assert (opts.bl_warmup_epochs == 0) or (opts.baseline == "rollout")
    assert (
        opts.dataset_size % opts.batch_size == 0
    ), "Epoch size must be integer multiple of batch size!"
    return opts
