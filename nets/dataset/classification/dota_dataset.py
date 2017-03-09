from nets.dataset.classification import AbstractClassificationDataset
import numpy as np
import csv


class DotaDataset(AbstractClassificationDataset):

    # Overriding abstract method.
    def default_load(self):
        super(DotaDataset, self).default_load()
        return self.load_all_data('np_dota2_train.npz',
                                  'np_dota2_val.npz',
                                  'np_dota2_test.npz')

    # Overriding abstract method.
    def _load_data_from_file(self, file_name):
        x, y = super(DotaDataset, self)._load_data_from_file(file_name)

        # this occurs if the data is in numpy format
        if x is not None and y is not None:
            return x, y

        # Our data storage.
        x = []
        y = []
        sample_size = None
        # The data is stored as csv files, so read them!
        with open(self.get_path() + file_name, 'rb') as csvfile:
            dota_reader = csv.reader(csvfile)
            for row in dota_reader:
                x.append(self.sample_transform(row[1:]))
                y.append(self.class_transform(row[0]))

                if sample_size is None:
                    sample_size = len(x[0])
                else:
                    assert(len(x[-1]) == sample_size),\
                    'ERROR - samples have different sizes! {} vs {}!'.format(
                        sample_size, len(x[-1])
                    )

        return np.array(x, dtype=np.float64), np.array(y, dtype=int)

    def get_path(self):
        super(DotaDataset, self).get_path()
        return self.RAW_DATASETS_DIR + 'dota2_dataset/'

    # Overriding method.
    def class_transform(self, y_class):
        transformed = super(DotaDataset, self).class_transform(y_class)

        # the classes are encoded as <-1, 1> in the data, so turn it to <0, 1>
        return max(transformed, 0)

    # Overriding method.
    def sample_transform(self, sample_vec):
        #TODO: this doesn't actually do anything right now, may not need it to.

        v = super(DotaDataset, self).sample_transform(sample_vec)

        # These are the first three features which are game variables.
        non_binary_features = v[0:3]
        transformed_non_bin = non_binary_features

        # These are the characters chosen features.
        binary_features = v[3:]
        transformed_bin = binary_features

        # This stores our final feature vector.
        new_v = []

        # Done!
        new_v.extend(transformed_non_bin)
        new_v.extend(transformed_bin)
        return new_v

