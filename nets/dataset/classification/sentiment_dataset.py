from nets.dataset.classification import AbstractClassificationDataset
from nets.util import word_embeddings as wemb
from copy import deepcopy
import csv
import numpy as np

UNKNOWN = '*UNKNOWN*'

class SentimentDataset(AbstractClassificationDataset):
    def __init__(self):
        super(SentimentDataset, self).__init__()
        self.name = 'Sentiment Analysis Dataset'
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
        return wemb.raw_load_and_extract_glove(self.get_vocabulary(), dim,
                                               self.get_path(), twitter=True)

    def get_glove_data(self, dim=50):
        self.embeddings = np.load(wemb.get_glove_fname(self.get_path(), dim)+'.npy')[0]
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

    def _get_default_set_fname(self, set_name):
        return 'senti_{}'.format(set_name)

    def get_path(self):
        super(SentimentDataset, self).get_path()
        return ''.join([self.RAW_DATASETS_DIR, 'sentiment_analysis/'])
