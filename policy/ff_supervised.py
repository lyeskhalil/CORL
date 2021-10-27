import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import DataParallel


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


def get_loss(log_p, y, optimizers, w, opts):
    # The cross entrophy loss of v_1, ..., v_{t-1} (this is a batch_size by 1 vector)
    # total_loss = torch.zeros(y.shape)

    # Calculate loss of v_t
    # print(log_p, y)
    loss_t = F.cross_entropy(log_p, y.long(), weight=w)

    # Update the loss for the whole graph
    # total_loss += loss_t
    loss = loss_t

    # Perform backward pass and optimization step
    # optimizers[0].zero_grad()
    # loss.backward()
    # optimizers[0].step()
    return loss


class SupervisedFFModel(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        problem,
        opts,
        tanh_clipping=None,
        mask_inner=None,
        mask_logits=None,
        n_encode_layers=None,
        normalization="batch",
        checkpoint_encoder=False,
        shrink_size=None,
        num_actions=4,
        n_heads=None,
        encoder=None,
    ):
        super(SupervisedFFModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.decode_type = None
        self.num_actions = self.num_actions = (
            5 * (opts.u_size + 1) + 8
            if opts.problem != "adwords"
            else 7 * (opts.u_size + 1) + 8
        )
        self.is_bipartite = problem.NAME == "bipartite"
        self.problem = problem
        self.shrink_size = None
        self.model_name = "ff-supervised"
        self.ff = nn.Sequential(
            nn.Linear(self.num_actions, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, opts.u_size + 1),
        )

    def forward(
        self,
        input,
        opt_match,
        opts,
        optimizer,
        training=False,
        baseline=None,
        return_pi=False,
    ):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        :param opt_match: (batch_size, U_size, V_size), the optimal matching of the graphs in the batch
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """
        _log_p, pi, cost, batch_loss = self._inner(
            input, opt_match, opts, optimizer, training
        )

        ll = self._calc_log_likelihood(_log_p, pi, None)

        return -cost, ll, pi, batch_loss

    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions
        # print(a[0, :])
        entropy = -(_log_p * _log_p.exp()).sum(2).sum(1).mean()
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        # if mask is not None:
        #     log_p[mask] = 0
        # if not (log_p > -10000).data.all():
        #     print(log_p)
        # assert (
        #     log_p > -10000
        # ).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        # print(_log_p)
        return log_p.sum(1), entropy

    def _inner(self, input, opt_match, opts, optimizer, training):

        outputs = []
        sequences = []
        # losses = []

        state = self.problem.make_state(input, opts.u_size, opts.v_size, opts)
        i = 1
        total_loss = 0
        while not (state.all_finished()):
            mask = state.get_mask()
            w = state.get_current_weights(mask)
            s, mask = state.get_curr_state(self.model_name)
            # s = w
            pi = self.ff(s)
            # Select the indices of the next nodes in the sequences, result (batch_size) long
            # if training:
            #     mask = torch.zeros(mask.shape, device=opts.device)
            selected, p = self._select_node(
                pi,
                mask.bool(),
            )  # Squeeze out steps dimension
            # entropy += torch.sum(p * (p.log()), dim=1)
            state = state.update((selected)[:, None])
            outputs.append(p)
            sequences.append(selected)

            # do backprop if in training mode
            if opts.problem != "adwords":
                none_node_w = torch.tensor(
                    [1.0 / (opts.v_size / opts.u_size)], device=opts.device
                ).float()
            else:
                none_node_w = torch.tensor([1.0], device=opts.device).float()
            w = torch.cat(
                [none_node_w, torch.ones(opts.u_size, device=opts.device).float()],
                dim=0,
            )
            # supervised learning
            y = opt_match[:, i - 1]
            # print('y: ', y)
            # print('selected: ', selected)
            loss = get_loss(pi, y, optimizer, w, opts)
            # print("Loss: ", loss)
            # keep track for logging
            total_loss += loss
            i += 1
        # Collected lists, return Tensor
        batch_loss = total_loss / state.v_size
        # print(batch_loss)
        if optimizer is not None and training:
            # print('epoch {} batch loss {}'.format(i, batch_loss))
            # print('outputs: ', outputs)
            # print('sequences: ', sequences)
            # print('optimal solution: ', opt_match)
            optimizer[0].zero_grad()
            batch_loss.backward()
            optimizer[0].step()
        return (
            torch.stack(outputs, 1),
            torch.stack(sequences, 1),
            state.size,
            batch_loss,
        )

    def _select_node(self, probs, mask):
        assert (probs == probs).all(), "Probs should not contain any nans"
        mask[:, 0] = False
        p = probs.clone()
        p[
            mask
        ] = (
            -1e6
        )  # Masking doesn't really make sense with supervised since input samples are independent, should only masking during testing.
        _, selected = p.max(1)
        return selected, p

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp
