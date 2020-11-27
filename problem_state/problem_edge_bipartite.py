from torch.utils.data import Dataset
import torch
import os
import pickle
from problem_state.edge_obm_state import StateEdgeBipartite
from data.generate_data import generate_edge_obm_data


class EdgeBipartite(object):

    NAME = "e-obm"

    # @staticmethod
    # def get_costs(dataset, pi):
    #     # TODO: MODIFY CODE SO IT WORKS WITH BIPARTITE INSTEAD OF TSP
    #     # Check that tours are valid, i.e. contain 0 to n -1
    #     assert (
    #         torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi)
    #         == pi.data.sort(1)[0]
    #     ).all(), "Invalid tour"
    #    # Gather dataset in order of tour
    #     d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))
    #     # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
    #     return (
    #         (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
    #         + (d[:, 0] - d[:, -1]).norm(p=2, dim=1),
    #         None,
    #     )

    @staticmethod
    def make_dataset(*args, **kwargs):
        return EdgeBipartiteDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateEdgeBipartite.initialize(*args, **kwargs)

    # @staticmethod
    # def beam_search(
    #     input,
    #     beam_size,
    #     expand_size=None,
    #     compress_mask=False,
    #     model=None,
    #     max_calc_batch_size=4096,
    # ):

    #     assert model is not None, "Provide model"

    #     fixed = model.precompute_fixed(input)

    #     def propose_expansions(beam):
    #         return model.propose_expansions(
    #             beam,
    #             fixed,
    #             expand_size,
    #             normalize=True,
    #             max_calc_batch_size=max_calc_batch_size,
    #         )

    #     state = Bipartite.make_state(
    #         input, visited_dtype=torch.int64 if compress_mask else torch.uint8
    #     )

    #     return beam_search(state, beam_size, propose_expansions)


class EdgeBipartiteDataset(Dataset):
    def __init__(self, dataset, size, problem, opts):
        super(EdgeBipartiteDataset, self).__init__()

        # self.data_set = dataset
        # self.optimal_size = torch.load("{}/optimal_match.pt".format(self.data_set))
        self.problem = problem
        if dataset is not None:
            self.optimal_size = torch.load("{}/optimal_match.pt".format(dataset))
            self.data_set = dataset
        else:
            # If no filename is specified generated data for edge obm probelm
            D, optimal_size = generate_edge_obm_data(
                opts.u_size,
                opts.v_size,
                opts.weight_distribution,
                opts.weight_distribution_param,
                opts.graph_family_parameter,
                opts.seed,
                opts.graph_family,
                opts.dataset_folder,
                size,
                opts.save_data
            )
            self.optimal_size = optimal_size
            self.data_set = D

        self.size = size

    def __getitem__(self, index):

        # Load data and get label
        if type(self.data_set) == str:
            X = torch.load("{}/graphs/{}.pt".format(self.data_set, index))
        else:
            X = self.data_set[index]
        Y = self.optimal_size[index]
        return X, Y

    def __len__(self):
        return self.size


# train_loader = torch.utils.data.DataLoader(
#              ConcatDataset(
#                  datasets.ImageFolder(traindir_A),
#                  datasets.ImageFolder(traindir_B)
#              ),
#              batch_size=args.batch_size, shuffle=True,
#              num_workers=args.workers, pin_memory=True)
