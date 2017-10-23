from nets.dataset.classification import AbstractClassificationDataset
from copy import deepcopy
import csv
import numpy as np

UNKNOWN = '*UNKNOWN*'

def get_glove_fname(path, dim):
    return '{}vocab_embs{}'.format(path, dim)

class SentimentDataset(AbstractClassificationDataset):
    def __init__(self):
        super(SentimentDataset, self).__init__()
        self.dataset_name = 'Sentiment Analysis Dataset'
        self.embeddings = None
        self.vocab = None

    @staticmethod
    def tokenize(string):
        """ Tokenizes a string. """
        l = string.split(' ') # may do more in future
        new_l = []
        for word in l:
            for i,tok in enumerate(word.split('\'')): # split on apostrophes
                if tok != '':
                    if i > 0:
                        tok = '{}{}'.format('\'', tok)
                    new_l.append(tok)
        return new_l

    def get_vocabulary(self):
        """
        Gets the vocabulary over ALL tokens in the dataset (not just train).
        :return: set of all unique tokens in the dataset
        """
        if self.vocab:
            return deepcopy(self.vocab)
        self.vocab = set()
        for x, y, name in self.iter_all_data():
            for word_seq in x:
                self.vocab = self.vocab.union(set(word_seq))
        return deepcopy(self.vocab)

    def init_glove_for_vocab(self, dim=50, serialize_embeddings=True):
        assert(dim in [25, 50, 100, 200])

        vocab_set = self.get_vocabulary()
        embeddings = {}

        print('Getting glove data... May take some time...')
        glove_data_path = '/home/ml/kkenyo1/glove/glove.twitter.27B.{}d.txt'.format(dim)
        with open(glove_data_path, 'r') as f:
            for line in f.readlines():
                data = line.split()
                word = data[0]
                if word in vocab_set:
                    embeddings[word] = np.array(map(float, data[1:]))
                    vocab_set.remove(word)
        print('There are {} words without glove embeddings.'.format(len(vocab_set)))

        # save the embeddings with numpy for quick access
        if serialize_embeddings:
            np.save(get_glove_fname(self.get_path(), dim), np.array([embeddings]))

        return embeddings

    def get_glove_data(self, dim=50):
        self.embeddings = np.load(get_glove_fname(self.get_path(), dim)+'.npy')[0]
        return self.embeddings

    def _load_data_from_file(self, file_name):
        """
        :param file_name: string designating file name
        :return: x, y
        """
        file_name = self.get_path() + file_name
        x, y = super(SentimentDataset, self)._load_data_from_file(file_name)
        if (x is not None) and (y is not None):
            return x, y

        # assume all data comes in TSV form.
        x, y = [], []
        with open(file_name, 'rb') as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t')
            for row in reader:
                x.append(self.tokenize(row['sample']))
                y.append(int(row['label'])) # assumes labels are already integers
        return x, y

    def get_path(self):
        super(SentimentDataset, self).get_path()
        return ''.join([self.RAW_DATASETS_DIR, 'sentiment_analysis/'])

    def default_load(self, dataset_name=''):
        super(SentimentDataset, self).default_load()
        return self.load_all_data('{}_train.npz'.format(dataset_name),
                                  '{}_val.npz'.format(dataset_name),
                                  '{}_test.npz'.format(dataset_name))
