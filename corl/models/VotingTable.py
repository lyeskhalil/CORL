import numpy as np
from fruit.buffers.table import LookupTable
from voting import Vote
from scipy.stats import rankdata


class VotinglookupTable(LookupTable):
    def __init__(
        self, environment, init_value=0.0, thresholds=None, voting_scheme=None
    ):
        super().__init__(environment=environment, init_value=init_value)
        self.thresholds = thresholds
        self.current_state_values = [
            [0.0 for x in range(self.num_of_objs)] for y in range(self.num_of_actions)
        ]
        self.voting_scheme = voting_scheme

    def get_thresholds(self):
        return self.thresholds

    def set_threshold(self, thresholds):
        self.thresholds = thresholds

    def select_greedy_action(self, state):
        self.get_action_values(state)
        return self.greedy_action(
            self.current_state_values, self.thresholds, self.voting_scheme
        )

    def get_action_values(self, state):
        for i in range(self.num_of_objs):
            for a in range(self.num_of_actions):
                self.current_state_values[a][i] = self.value_function[i][a][state]

    @staticmethod
    def greedy_action(action_values, thresholds, voting_scheme):
        action_values = np.array(action_values).T
        action_ranks = rankdata(action_values, axis=1, method="dense")
        votes = Vote(action_ranks)
        return votes.aggregate(voting_scheme)

        # TODO: USE VOTING SCHEMES TO SELECT BEST ACTION
        # linear_values = []
        # for i in range(len(action_values)):
        #     linear_values.append(np.sum(np.multiply(action_values[i], thresholds)))

        # greedy_action = 0
        # greedy_value = linear_values[0]
        # for i in range(len(linear_values)):
        #     if i > 0:
        #         if linear_values[i] > greedy_value:
        #             greedy_action = i
        # return greedy_action

        pass
