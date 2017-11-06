import numpy as np
from nets.dataset.classification import AbstractClassificationDataset
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split


class SyntheticDataset(AbstractClassificationDataset):
    """
    This class exists to build synthetic data.
    """
    def __init__(self, name, hyperparameters=None):
        super(SyntheticDataset, self).__init__()
        self.dataset_name = name

    def set_all_data(self, x, y):
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=0)
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.15, random_state=10)
        self._train_x = train_x
        self._train_y = train_y
        self._val_x = val_x
        self._val_y = val_y
        self._test_x = test_x
        self._test_y = test_y

    def _load_data_from_file(self, file_name, generate_data=False):
        """
        :param file_name: string designating file name or generative function
        :param generate_data: if true, we will generate data according to the fname
        :return: x, y
        """
        x, y = super(SyntheticDataset, self)._load_data_from_file(file_name)
        if (x is not None) and (y is not None) and (not generate_data):
            return x, y

        elif generate_data:
            if file_name == 'make_blobs':
                pass

            elif file_name == 'make_classification':
                pass
        else:
            # TODO: actually load synthetic data from a file...
            pass

        return x, y

    def get_path(self):
        super(SyntheticDataset, self).get_path()
        return ''.join([self.RAW_DATASETS_DIR, 'synthetic_data/'])

    def default_load(self):
        super(SyntheticDataset, self).default_load()

        # For now, until we decide on a solid synthetic dataset, we will use
        # this in a generative way.
        # return self.load_all_data('np_synth_train.npz',
        #                           'np_synth_val.npz',
        #                           'np_synth_test.npz')
