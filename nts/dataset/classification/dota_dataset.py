from nts import AbstractClassificationDataset
import numpy as np
import csv


class DotaDataset(AbstractClassificationDataset):

    # Overriding abstract method.
    def _load_data_from_file(self, file_name):
        super(DotaDataset, self)._load_data_from_file()

        # Path to the file that will be put.
        data_dir = self.RAW_DATASETS_DIR + 'dota2_dataset/'

        # Our data storage.
        x = []
        y = []

        # The data is stored as csv files, so read them!
        with open(data_dir + file_name, 'rb') as csvfile:
            dota_reader = csv.reader(csvfile)
            for row in dota_reader:
                x.append(self.sample_transform(row[1:]))
                y.append(self.class_transform(row[0]))

        return np.array(x), np.array(y)

    # Overriding method.
    @staticmethod
    def class_transform(y_class):
        transformed = super(DotaDataset).class_transform(y_class)

        # the classes are encoded as <-1, 1> in the data, so turn it to <0, 1>
        return max(transformed, 0)

    # Overriding method.
    @staticmethod
    def sample_transform(sample_vec):
        #TODO: this doesn't actually do anything right now, may not need it to.

        v = super(DotaDataset).sample_transform(sample_vec)

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
