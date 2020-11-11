
import os
import random
import numpy as np
import torch
import chaospy

class MNISTTask(object):
    '''
    Sample a few-shot learning task from the MNIST dataset
    Tasks are created by shuffling the labels
    Sample N-way k-shot train and val sets according to
     - split (dataset/meta level train or test)
     - N-way classification (sample this many chars)
     - k-shot (sample this many examples from each char class)
    '''

    def __init__(self, root, num_cls, num_inst, split='train'):
        self.dataset = 'mnist'
        self.root = root + '/' + split
        print (root)
        self.split = split
        self.num_cl = num_cls
        self.num_inst = num_inst
        all_ids = []
        for i in range(10):
            d = os.path.join(root, self.split, str(i))
            files = os.listdir(d)
            all_ids.append([ str(i) + '/' + f[:-4] for f in files])

        # To create a task, we randomly shuffle the labels
        self.label_map = dict(list(zip(list(range(10)), np.random.permutation(np.array(list(range(10)))))))
        
        # Choose num_inst ids from each of 10 classes
        self.train_ids = []
        self.val_ids = []
        for i in range(10):
            permutation = list(np.random.permutation(np.array(list(range(len(all_ids[i])))))[:num_inst*2])
            self.train_ids += [all_ids[i][j] for j in permutation[:num_inst]]
            
    
            self.train_labels = self.relabel(self.train_ids)
            self.val_ids += [all_ids[i][j] for j in permutation[num_inst:]]
            self.val_labels = self.relabel(self.val_ids)

    def relabel(self, img_ids):
        ''' Remap labels to new label map for this task '''
        orig_labels = [int(x[0]) for x in img_ids]
        return [self.label_map[x] for x in orig_labels]
        return np.array([self.label_map[x] for x in orig_labels])
        #print( np.array([self.label_map[x] for x in orig_labels]).shape)
