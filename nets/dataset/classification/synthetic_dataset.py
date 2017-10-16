from nets.dataset.classification import AbstractClassificationDataset
from sklearn.datasets import make_multilabel_classification, make_blobs
import numpy as np


class SyntheticDataset(AbstractClassificationDataset):
    """
    This class exists to build synthetic data.
    """
    def _load_data_from_file(self, file_name):
        super(SyntheticDataset, self)._load_data_from_file(file_name)

    def get_path(self):
        super(SyntheticDataset, self).get_path()

    def default_load(self):
        super(SyntheticDataset, self).default_load()
        return
