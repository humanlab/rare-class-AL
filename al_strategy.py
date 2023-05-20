import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import f1_score

#class for active learning strategy that takes probabilities, strategy, unannotated corpus and embedding model as inputs
#returns the indices of the most informative instances
class ALStrategy:
    def __init__(self, strategy, unannotated_corpus, embedding_model=None):
        self.strategy = strategy
        self.unannotated_corpus = unannotated_corpus
        self.embedding_model = embedding_model

    def get_indices(self, probs, batch_size):
        if self.strategy == 'random':
            return self.random(probs, batch_size)
        elif self.strategy == 'entropy':
            return self.entropy(probs, batch_size)
        elif self.strategy == 'contrastive_active_learning':
            return self.contrastive_active_learning(probs, batch_size)
        elif self.strategy == 'coreset':
            return self.coreset(probs, batch_size)
        elif self.strategy == 'prob_rare_class':
            return self.prob_rare_class(probs, batch_size)
   

    def random(self, probs, batch_size):
        indices = np.random.choice(len(probs), batch_size, replace=False)
        return indices
    
    def entropy(self, probs, batch_size):
        entropy = np.sum(-probs * np.log(probs), axis=1)
        indices = np.argsort(entropy)[:batch_size]
        return indices
    
    def contrastive_active_learning():
        pass

    def coreset():
        pass

    def prob_rare_class():
        pass



