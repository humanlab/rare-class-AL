import numpy as np
import torch.nn.functional as F
import torch
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

#class for active learning strategy that takes probabilities, strategy, unannotated corpus and embedding model as inputs
#returns the indices of the most informative instances

#create a parent class for ALStrategy and then create child classes for each strategy
#each child class will have a get_indices method that will return the indices of the most informative instances

class ALStrategy:
    def __init__(self, **kwargs):#strategy, unannotated_corpus, embeddings=None ):
        """self.strategy = kwargs.strategy
        if self.strategy == 'random':
            return RandomAL(**kwargs)
        elif self.strategy == 'entropy':
            return EntropyAL(**kwargs)
        elif self.strategy == 'contrastive_active_learning':
            return CAL(**kwargs)
        elif self.strategy == 'coreset':
            return CoreSetAL(**kwargs)
        elif self.strategy == 'prob_rare_class':
            return PRC(**kwargs)
        """
        self.annotated_corpus = kwargs['annotated_corpus']
        self.unannotated_corpus = kwargs['unannotated_corpus']
        self.batch_size = kwargs['batch_size']
        pass

    def select_indices(self, probs):
        raise NotImplementedError
    
    
class RandomAL(ALStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def select_indices(self):
        indices = np.random.choice(len(self.unannotated_corpus), self.batch_size, replace=False)
        return indices
    
class EntropyAL(ALStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self.model= kwargs['model']
        except KeyError:
            raise ValueError('model must be provided for entropy strategy')

    def select_indices(self):
        probs = self.model.get_probabilities(self.unannotated_corpus)
        entropy = np.sum(-probs * np.log(probs), axis=1)
        indices = np.argsort(entropy,)[::-1][:self.batch_size] #descending order
        return indices
    
class CAL(ALStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self.model= kwargs['model']
        except KeyError:
            raise ValueError('model must be provided for contrastive_active_learning strategy')

    def select_indices(self):

        train_output = self.model.predict(self.annotated_corpus)

        ids_unlabeled = [inst["id"] for inst in self.unannotated_corpus]
        embeds_unlabeled = self.model.get_embeddings(self.unannotated_corpus)
        logits_unlabeled = torch.from_numpy(train_output.predictions)

        # labeled_ids = [inst["id"] for inst in labeled_data]
        embeds_labeled = self.model.get_embeddings(self.annotated_corpus)
        logits_labeled = torch.from_numpy(train_output.predictions)
        labels_labeled = [inst["label"] for inst in self.annotated_corpus]

        # KNN
        neigh = KNeighborsClassifier(n_neighbors=100)
        neigh.fit(X=embeds_labeled, y=labels_labeled) # X: embeddings of labeled data, y: list of all labels of labeled data
        criterion = torch.nn.KLDivLoss(reduction="none") # https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html

        kl_scores = []
        num_adv = 0
        distances = []
        # candidate: unlabeled, neighbors: labeled
        for embeds_candidate, logits_candidate in tqdm(zip(embeds_unlabeled, logits_unlabeled)):
            distances_, neighbors = neigh.kneighbors(X=[embeds_candidate], return_distance=True)
            distances.append(distances_[0])
            # neighbors_labels = labels_labeled[neighbors[0]] # don't need it
            # calculate score
            logits_neighbors = [logits_labeled[neighbor] for neighbor in neighbors]
            pred_neighbors = [np.argmax(logits_labeled[n], axis=1) for n in neighbors] # predicted labels # checkkkkkk
            prob_neighbors = F.softmax(logits_labeled[neighbors], dim=-1) # checkkkkk
            pred_candidate = [np.argmax(logits_candidate)]
            # num_diff_pred = len(list(set(pred_neighbors).intersection(pred_candidate)))
            # print(num_diff_pred)    #isn't this always 0 or 1
            #
            # if num_diff_pred > 0:   num_adv += 1
            # args.ce = False
            uda_softmax_temp = 1
            logprob_candidate = F.log_softmax(logits_candidate / uda_softmax_temp, dim=-1) # checkkkk
            kl = np.array([torch.sum(criterion(logprob_candidate, prob_neighbor), dim=-1).numpy() for prob_neighbor in prob_neighbors])
            kl_scores.append(kl.mean())

        # annotations_per_iteration = 300
        #selected_inds = np.argpartition(kl_scores, -300)[-300:]
        score_pairs = sorted(zip([i for i in range(len(kl_scores))], kl_scores), key=lambda x: x[1], reverse=True)
        score_pairs = sorted(score_pairs[:self.batch_size], key=lambda x: x[0]) + score_pairs[self.batch_size:]
        selected_inds = [pair[0] for pair in score_pairs]
        sampled_ids = np.array(ids_unlabeled)[selected_inds]
        return sampled_ids


class CoreSetAL(ALStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self.model= kwargs['model']
        except KeyError:
            raise ValueError('model must be provided for contrastive_active_learning strategy')

    def select_indices(self):
        ids = [inst["id"] for inst in self.unannotated_corpus]
        #get the penultimate layer embeddings from trainer.model
        unlabeled_embeds = self.model.get_embeddings(self.unannotated_corpus)
        labeled_embeds = self.model.get_embeddings(self.annotated_corpus)
        #labeled_embeds = labeled_embeds.numpy()
        unlabeled_embeds.extend(labeled_embeds)
        all_embeds = np.asarray(unlabeled_embeds)

        #np.asarray(labeled_embeds.cpu() + unlabeled_embeds.cpu())
        #print(len(all_embeds))

        unlabeled_idxs = [False]*(len(self.unannotated_corpus))
        labeled_idxs = [True]*(len(self.annotated_corpus))
        unlabeled_idxs.extend(labeled_idxs)
        labeled_idxs = np.array(unlabeled_idxs.copy(),dtype=bool) # True if labeled False if unlabeled
        #init_labeled_idxs = labeled_idxs.copy() #keeping track of how it looks like initially
        del unlabeled_idxs


        dist_mat = np.matmul(all_embeds, all_embeds.transpose())
        sq = np.array(dist_mat.diagonal()).reshape(len(labeled_idxs), 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.transpose()
        dist_mat = np.sqrt(dist_mat)
        mat = dist_mat[~labeled_idxs, :][:, labeled_idxs]
        picked_ids = list()
        for i in tqdm(range(len(self.unannotated_corpus)), ncols=100):
            mat_min = mat.min(axis=1)
            q_idx_ = mat_min.argmax()
            q_idx = np.arange(len(labeled_idxs))[~labeled_idxs][q_idx_]
            labeled_idxs[q_idx] = 1
            picked_ids.append(ids[q_idx])
            mat = np.delete(mat, q_idx_, 0)
            mat = np.append(mat, dist_mat[~labeled_idxs, q_idx][:, None], axis=1)

            #print (np.arange(len(all_embeds))[(init_labeled_idxs ^ labeled_idxs)])
        
        """with open(args.name+"coreset_ranked_ids.txt", "w") as f:
            temp = sorted(picked_ids[:args.active_learning_sample_size], key = lambda x: int(x[2:]))
            for i in range(len(temp)):
                if i < args.active_learning_sample_size:
                    f.write(temp[i] +"\n")
                else:
                    f.write(picked_ids[i] + "\n")
        """
        return picked_ids[:self.batch_size]

class PRC(ALStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self.model= kwargs['model']
        except KeyError:
            raise ValueError('model must be provided for contrastive_active_learning strategy')
        
        #fix this for rare class calculated from the annotated corpus
        self.rare_class = self.annotated_corpus[0]['label'] #default is 1: for positive rare class
    
    def select_indices(self):
        probs = self.model.get_probabilities(self.unannotated_corpus)
        indices = np.argsort([p[self.model.rare_class] for p in probs],)[::-1][:self.batch_size]
        return indices
    

"""
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
        train_output = trainer.predict(labeled_data)

        ids_unlabeled = [inst["id"] for inst in unlabeled_data]
        embeds_unlabeled = trainer.extract_embeddings(unlabeled_data)
        logits_unlabeled = torch.from_numpy(test_output.predictions)

        # labeled_ids = [inst["id"] for inst in labeled_data]
        embeds_labeled = trainer.extract_embeddings(labeled_data)
        logits_labeled = torch.from_numpy(train_output.predictions)
        labels_labeled = [inst["label"] for inst in labeled_data]

        # KNN
        # look at lines 296~ in acquisition/cal.py
        neigh = KNeighborsClassifier(n_neighbors=100)
        neigh.fit(X=embeds_labeled, y=labels_labeled) # X: embeddings of labeled data, y: list of all labels of labeled data
        criterion = torch.nn.KLDivLoss(reduction="none") # https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html

        kl_scores = []
        num_adv = 0
        distances = []
        # candidate: unlabeled, neighbors: labeled
        for embeds_candidate, logits_candidate in tqdm(zip(embeds_unlabeled, logits_unlabeled)):
            distances_, neighbors = neigh.kneighbors(X=[embeds_candidate], return_distance=True)
            distances.append(distances_[0])
            # neighbors_labels = labels_labeled[neighbors[0]] # don't need it
            # calculate score
            logits_neighbors = [logits_labeled[neighbor] for neighbor in neighbors]
            pred_neighbors = [np.argmax(logits_labeled[n], axis=1) for n in neighbors] # predicted labels # checkkkkkk
            prob_neighbors = F.softmax(logits_labeled[neighbors], dim=-1) # checkkkkk
            pred_candidate = [np.argmax(logits_candidate)]
            # num_diff_pred = len(list(set(pred_neighbors).intersection(pred_candidate)))
            # print(num_diff_pred)    #isn't this always 0 or 1
            #
            # if num_diff_pred > 0:   num_adv += 1
            # args.ce = False
            uda_softmax_temp = 1
            logprob_candidate = F.log_softmax(logits_candidate / uda_softmax_temp, dim=-1) # checkkkk
            kl = np.array([torch.sum(criterion(logprob_candidate, prob_neighbor), dim=-1).numpy() for prob_neighbor in prob_neighbors])
            kl_scores.append(kl.mean())

        # annotations_per_iteration = 300
        #selected_inds = np.argpartition(kl_scores, -300)[-300:]
        score_pairs = sorted(zip([i for i in range(len(kl_scores))], kl_scores), key=lambda x: x[1], reverse=True)
        score_pairs = sorted(score_pairs[:args.active_learning_sample_size], key=lambda x: x[0]) + score_pairs[args.active_learning_sample_size:]
        selected_inds = [pair[0] for pair in score_pairs]
        sampled_ids = np.array(ids_unlabeled)[selected_inds]

        pass

        
        

    def coreset(self, probs, batch_size):
        ids = [inst["id"] for inst in unlabeled_data]
        #get the penultimate layer embeddings from trainer.model
        unlabeled_embeds = trainer.extract_embeddings(unlabeled_data)
        labeled_embeds = trainer.extract_embeddings(labeled_data)
        #labeled_embeds = labeled_embeds.numpy()
        unlabeled_embeds.extend(labeled_embeds)
        all_embeds = np.asarray(unlabeled_embeds)

        #np.asarray(labeled_embeds.cpu() + unlabeled_embeds.cpu())
        #print(len(all_embeds))

        unlabeled_idxs = [False]*(len(unlabeled_data))
        labeled_idxs = [True]*(len(labeled_data))
        unlabeled_idxs.extend(labeled_idxs)
        labeled_idxs = np.array(unlabeled_idxs.copy(),dtype=bool) # True if labeled False if unlabeled
        #init_labeled_idxs = labeled_idxs.copy() #keeping track of how it looks like initially
        del unlabeled_idxs


        dist_mat = np.matmul(all_embeds, all_embeds.transpose())
        sq = np.array(dist_mat.diagonal()).reshape(len(labeled_idxs), 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.transpose()
        dist_mat = np.sqrt(dist_mat)
        mat = dist_mat[~labeled_idxs, :][:, labeled_idxs]
        picked_ids = list()
        for i in tqdm(range(len(unlabeled_data)), ncols=100):
            mat_min = mat.min(axis=1)
            q_idx_ = mat_min.argmax()
            q_idx = np.arange(len(labeled_idxs))[~labeled_idxs][q_idx_]
            labeled_idxs[q_idx] = 1
            picked_ids.append(ids[q_idx])
            mat = np.delete(mat, q_idx_, 0)
            mat = np.append(mat, dist_mat[~labeled_idxs, q_idx][:, None], axis=1)

            #print (np.arange(len(all_embeds))[(init_labeled_idxs ^ labeled_idxs)])
        
        with open(args.name+"coreset_ranked_ids.txt", "w") as f:
            temp = sorted(picked_ids[:args.active_learning_sample_size], key = lambda x: int(x[2:]))
            for i in range(len(temp)):
                if i < args.active_learning_sample_size:
                    f.write(temp[i] +"\n")
                else:
                    f.write(picked_ids[i] + "\n")
        return picked_ids[:args.active_learning_sample_size]
    
    
    def prob_rare_class(self, probs, batch_size):
        indices = np.argsort([p[self.rare_class] for p in probs],)[::-1][:batch_size]
        return indices
"""


