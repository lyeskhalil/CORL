import numpy as np

class Vote():
    def __init__(self, votes):
        self.votes = votes
        
    def aggregate(self, rule):
        if rule == 'plurality':
            return plurality(self.votes)
        elif rule == 'kemeny':
            return kemeny(self.votes)
        


"""
generates n ballots for m candidates
"""
def generate_votes(n,m):
    votes = np.empty((n,m))
    for i in range(0,n):
        votes[i] = np.random.choice(range(1,m+1), replace=False, size = m)
    return votes



"""
returns a dict for the cth candidate with keys the rank and the value the occurance of that rank (how many voted for c as the first candidate etc...)
Note: c is from 0 to m-1
"""
def get_frequency(votes, c):
    unique, counts = np.unique(votes[:,c], return_counts=True)
    return dict(zip(unique, counts))
    
    
"""
return plurality winner index
"""
def plurality(votes):
    top_candidate = 0
    plurality = 0
    for i in range(0,votes.shape[1]): #for each candidate
        i_plurality = get_frequency(votes, i)[1] #number of top votes for i
        if i_plurality >= plurality:
            plurality = i_plurality
            top_candidate = i
    return top_candidate
    
    
    
"""
create a table of preferences based on the votes. entry [i,j] represents the number of votes that prefer candidate i over j
"""
def make_table(votes):
    pass






"""
computes the kemeny optimal ranking of the votes
"""
def kemeny(votes):
    pass




if __name__ == "__main__": 
    votes = generate_votes(10,3)
    voting = Vote(votes)
    result = voting.aggregate("plurality")
    print(result) 
