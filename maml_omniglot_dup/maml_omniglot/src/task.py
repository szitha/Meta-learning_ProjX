import os
import random
import numpy as np
import torch
import chaospy

class OmniglotTask(object):
    '''
    Sample a few-shot learning task from the Omniglot dataset
    Sample N-way k-shot train and val sets according to
     - split (dataset/meta level train or test)
     - N-way classification (sample this many chars)
     - k-shot (sample this many examples from each char class)
    Assuming that the validation set is the same size as the train set!
    '''

    def __init__(self, root, num_cls, num_inst, split='train'):
        self.dataset = 'omniglot'
        self.root = '{}/images_background'.format(root) if split == 'train' else '{}/images_evaluation'.format(root)
        #print (root)
        self.num_cl = num_cls
        self.num_inst = num_inst

        # Sample num_cls galaxies and num_inst instances of each
        languages = os.listdir(self.root)
         
        #print (languages)
        chars = []
        for l in languages:
            chars += [os.path.join(l, x) for x in os.listdir(os.path.join(self.root, l))]

        #print(chars)
        random.shuffle(chars)

        #random sampling number of galaxies
        #classes = random.sample(g_type, num_cls)
        #def Hammersley(num_cls,g_type):
        #    idxs = [];
        #    distribution = chaospy.J(chaospy.Uniform(1, len(g_type)))
        #    pul_sample = distribution.sample(num_cls, rule = "M")

        #     print(pul_sample)
        #    for i in pul_sample:
        #        idxs.append(int((i)))
        #
        #    indexes = np.unique(idxs, return_index=True)[1]
        #    unsort = [idxs[index] for index in sorted(indexes)]
        #
        #    sample=[g_type[j] for j in idxs];
        #    return sample
        #classes = Hammersley(num_cls, g_type)
        
        classes = chars[:num_cls]

        print(classes)

        labels = np.array(list(range(len(classes))))
        labels = dict(list(zip(classes, labels))) 
        instances = dict()


        # Now sample from the chosen classes to create class-balanced train and val sets
        self.train_ids = []
        self.val_ids = []

        for c in classes:
            # First get all isntances of that class
            temp = [os.path.join(c, x) for x in os.listdir(os.path.join(self.root, c))]
            instances[c] = random.sample(temp, len(temp))

            # Sample num_inst instances randomly each for train and val
            self.train_ids += instances[c][:num_inst]
            self.val_ids += instances[c][num_inst:num_inst*2]
        # Keep instances separated by class for class-balanced mini-batches
        self.train_labels = [labels[self.get_class(x)] for x in self.train_ids]
        self.val_labels = [labels[self.get_class(x)] for x in self.val_ids]
        

    def get_class(self, instance):
        return os.path.join(*instance.split('/')[:-1])
