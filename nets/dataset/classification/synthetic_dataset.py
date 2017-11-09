import numpy as np
from nets.dataset.classification import AbstractClassificationDataset
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split


class SyntheticDataset(AbstractClassificationDataset):
    """
    This class exists to build synthetic data.
    """
    EASY = 'easy'
    MEDIUM = 'med'
    HARD = 'hard'

    DIFFICULTY_TO_SEP = {
        EASY: 3,
        MEDIUM: 2,
        HARD: 1,
    }

    DIFFICULTY_TO_CPC = {
        EASY: 2,
        MEDIUM: 2,
        HARD: 2,
    }

    DIFFICULTY_TO_STD = {
        EASY: 8,
        MEDIUM: 15,
        HARD: 20,
    }


    def __init__(self, name, difficulty='', blobs=True):
        super(SyntheticDataset, self).__init__()
        self.name = name
        self.difficulty = difficulty
        self.blobs = blobs

    @staticmethod
    def iter_all_difficulties():
        for diff in [SyntheticDataset.EASY, SyntheticDataset.MEDIUM, SyntheticDataset.HARD]:
            yield diff

    def make_dataset(self, num_classes):
        if self.blobs:
            num_samples = 5000
            num_features = 1000
            num_classes = num_classes
            cluster_std = self.DIFFICULTY_TO_STD[self.difficulty]

            x, y = make_blobs(
                n_samples=num_samples,
                n_features=num_features,
                centers=num_classes,
                cluster_std=cluster_std,
                random_state=8675309,
            )
        else:
            num_samples = 5000
            num_features = 1000
            num_redundant = 500
            num_informative = 50
            num_clusters_per_class = self.DIFFICULTY_TO_CPC[self.difficulty]
            class_sep = self.DIFFICULTY_TO_SEP[self.difficulty]

            x, y = make_classification(
                n_samples=num_samples,
                n_features=num_features,
                n_informative=num_informative,
                n_redundant=num_redundant,
                n_classes=num_classes,
                n_clusters_per_class=num_clusters_per_class,
                class_sep=class_sep,
                random_state=1848,
            )
        self.set_all_data(x, y)

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

    def _get_default_set_fname(self, set_name):
        return 'synth{}_{}'.format(self.name, set_name)

    def get_path(self):
        super(SyntheticDataset, self).get_path()
        return ''.join([self.RAW_DATASETS_DIR, 'synthetic_data/'])
