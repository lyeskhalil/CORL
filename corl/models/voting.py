import numpy as np
from itertools import combinations, permutations

class Vote:
    def __init__(self, votes):
        self.votes = votes

    def aggregate(self, rule):
        if rule == "plurality":
            return self.plurality(self.votes)
        elif rule == "kemeny":
            return self.kemeny(self.votes)

    def get_frequency(self, c):
        """
        returns a dict for the cth candidate with keys the rank and the value the occurance of that rank (how many voted for c as the first candidate etc...)
        Note: c is from 0 to m-1
        """
        unique, counts = np.unique(self.votes[:, c], return_counts=True)
        return dict(zip(unique, counts))

    def plurality(self, votes):
        """
        return plurality winner index
        """
        top_candidate = 0
        plurality = 0
        for i in range(self.votes.shape[1]):  # for each candidate
            f = self.get_frequency(i)
            if 1 in f:
                i_plurality = self.get_frequency(i)[1]  # number of top votes for i
            else:
                i_plurality = 0
            if i_plurality >= plurality:
                plurality = i_plurality
                top_candidate = i
        return top_candidate

    def make_table(self, votes):
        """
        create a table of preferences based on the votes. entry [i,j] represents the number of votes that prefer candidate i over j
        """
        pass


def generate_votes(n, m):
    """
    generates n ballots for m candidates
    """
    votes = np.empty((n, m))
    for i in range(0, n):
        votes[i] = np.random.choice(range(1, m + 1), replace=False, size=m)
    return votes

def kendalltau_dist(rank_a, rank_b):
    tau = 0
    n_candidates = len(rank_a)
    for i, j in combinations(range(n_candidates), 2):
        tau += (np.sign(rank_a[i] - rank_a[j]) ==
                -np.sign(rank_b[i] - rank_b[j]))
    return tau

def kemeny(ranks):
    min_dist = np.inf
    best_rank = None
    n_voters, n_candidates = ranks.shape
    for candidate_rank in permutations(range(n_candidates)):
        dist = np.sum(kendalltau_dist(candidate_rank, rank) for rank in ranks)
        if dist < min_dist:
            min_dist = dist
            best_rank = candidate_rank
    return best_rank[0]


if __name__ == "__main__":
    votes = generate_votes(10, 3)
    voting = Vote(votes)
    result = voting.aggregate("kemeny")
    print(result)
