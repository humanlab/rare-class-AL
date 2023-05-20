import numpy as np
import torch.nn.functional as F

#class for active learning strategy that takes probabilities, strategy, unannotated corpus and embedding model as inputs
#returns the indices of the most informative instances
class ALStrategy:
    def __init__(self, strategy, unannotated_corpus, rare_class=1, embeddings=None ):
        self.strategy = strategy
        self.unannotated_corpus = unannotated_corpus
        if embeddings is None and (strategy == 'contrastive_active_learning' or strategy == 'coreset'):
            raise ValueError('Embeddings must be provided for '+ strategy + ' strategy')
        self.embeddings = embeddings
        self.rare_class = rare_class #default is 1: for positive rare class

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
        indices = np.argsort(entropy,)[::-1][:batch_size] #descending order
        return indices
    
    def contrastive_active_learning(self, probs, batch_size):
        pass

    def coreset(self, probs, batch_size):
        pass

    def prob_rare_class(self, probs, batch_size):
        indices = np.argsort([p[self.rare_class] for p in probs],)[::-1][:batch_size]
        return indices



