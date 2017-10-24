from nets import PROJECT_BASE_DIR
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import random


class AbstractDataset(object):
    """
    The base class for our test suite. This class represents the essential
    qualities of a ML dataset. It is only extended by
    AbstractClassificationDataset and AbstractClusteringDataset. By doing
    this we allow for an abstraction that is utilized in our test suite far
    downstream, and allows us to have a common representation of all standard
    ML datasets.

    The main assumption made is that the data is divided into a training and
    a test set, with the option of having a validation set as well. This
    facilitates an easy way to abstract the essential characteristics of a
    ML dataset, but is flexible enough to allow for abstract cross validation.

    This assumption also means that we assume all data is separated into three
    files, one for train, validation, and testing.
    """
    # Makes the class abstract.
    __metaclass__ = ABCMeta

    # Static constants.
    RAW_DATASETS_DIR = PROJECT_BASE_DIR + 'raw_datasets/'

    # Initializes the essential components of a dataset as attributes.
    def __init__(self):
        self._train_x = None
        self._train_y = None
        self._val_x = None
        self._val_y = None
        self._test_x = None
        self._test_y = None
        self.dataset_name = ''

    # Getter methods.
    def get_train_x(self):
        return self._train_x

    def get_train_y(self):
        return self._train_y

    def get_train_data(self):
        return self.get_train_x(), self.get_train_y()

    def get_val_x(self):
        return self._val_x

    def get_val_y(self):
        return self._val_y

    def get_test_x(self):
        return self._test_x

    def get_data_statistics(self):
        print('Statistics for dataset {}...'.format(self.dataset_name))
        for x, y, name in self.iter_all_data():
            unique_labels = set(y)
            print('Dataset is {}:'.format(name))
            print('\tX data is encoded as {}, with x[0] {}'.format(type(x), type(x[0])))
            print('\tY data is encoded as {}, with y[0] {}'.format(type(y), type(x[y])))
            print('\t- Number of unique labels: {}'.format(len(unique_labels)))
            print('\t- Total number of samples: {}'.format(len(x)))
            for label in unique_labels:
                l_count = len(y[y == label])
                print('\t- Total number of samples with label = {}: {}'.format(label, l_count))
            print('\t- Sample example samples:')
            for i in range(0, 5):
                print('\t\tSample {}: x=\"{}\", y=\"{}\"'.format(i, x[i], y[i]))
            print('')

    def iter_all_data(self):
        yield self._train_x, self._train_y, 'Training set'
        yield self._val_x, self._val_y, 'Validation set'
        yield self._test_x, self._test_y, 'Testing set'

    def iterate_train_minibatches(self, batchsize, epochs, shuffle=True):
        """
        Iterates minibatches, used mostly only by neural networks.
        :param batchsize: int - designates size of batch.
        :param epochs: int - number of epochs to train for.
        :param shuffle: bool - optional, says whether or not randomized shuffle.
        :return: tuple - yielded subset of the training data as a minibatch.
        """
        # TODO: verify that we probably don't need to copy - will be expensive.
        # ... Alternatively, if we copy, then continuously iterate over the
        # MBs until a stop condition is reached; e.g. number of epochs.
        x, y = deepcopy(self.get_train_data())

        for e in range(epochs):
            # Shuffle the data.
            if shuffle:
                c = list(zip(x, y))
                random.shuffle(c)
                x, y = zip(*c)

            # Iterate. Note that we cut off samples by rounding down.
            for i in xrange(len(x) / batchsize):
                start = i * batchsize
                end = (i+1) * batchsize
                yield x[start:end], y[start:end], e

    def iterate_cross_validation(self, k_folds=5, merge_train_val=False, normalize=False, random_seed=1917,
                                 verbose=True):
        """
        :param k_folds: int - designates number of folds for cross validation
        :param merge_train_val: bool - designates whether we should merge train
        and validation sets for testing
        :param normalize: bool - normalize data according to train statistics, if true
        :param random_seed: bool - set the random seed for CV, if None then no shuffle
        :param verbose: bool - print output to std-out
        :return: yields tuples (train_k_x, train_k_y, val_k_x, val_k_y)
        """
        kf = KFold(n_splits=k_folds, shuffle=not random_seed is None, random_state=random_seed)
        x = self._train_x + self._val_x if merge_train_val else self._train_x
        y = self._train_y + self._val_y if merge_train_val else self._train_y
        kf.get_n_splits(x)

        fold = 1
        for train_idx, val_idx in kf.split(x):
            if verbose: print('Currently testing fold {}...'.format(fold))
            x_train = x[train_idx]
            x_val = x[val_idx]

            if normalize:
                if verbose: print('\tNormalizing fold data...')
                s = StandardScaler()
                x_train = s.fit_transform(x_train)
                x_val = s.transform(x_val)

            fold += 1
            yield x_train, y[train_idx], x_val, y[val_idx]

    # Evaluation methods.
    @abstractmethod
    def evaluate(self, gold, predictions):
        """
        Evaluate results according to a metric specified in the subclass.
        This is for quick and easy overview of results without getting into
        detail of the specifics of performance.

        :param gold: an iterable of the gold standard or correct data.
        :param predictions: an iterable of a model's predictions.
        :return: dict - our results dictionary which will be results for each
                        metric being measured by the extending class.
        """
        assert len(gold) == len(predictions),\
            'Different number of gold samples than predicted! %d vs %d.'%(
                len(gold), len(predictions))

    def evaluate_train_predictions(self, train_pred):
        """
        Returns a triple of precision, recall, f1.
        """
        return self.evaluate(self._train_y, train_pred)

    def evaluate_val_predictions(self, val_pred):
        """
        Returns a triple of precision, recall, f1.
        """
        return self.evaluate(self._val_y, val_pred)

    def evaluate_test_predictions(self, test_pred):
        """
        Returns a triple of precision, recall, f1.
        """
        return self.evaluate(self._test_y, test_pred)

    @abstractmethod
    def _generate_results_report(self, gold, predictions):
        """
        This is for generating a detailed report on the results obtained by
        a model on this dataset. For classification this is trivial since it
        just calls sklearn's "classification_report". The aim is to provide
        detailed results analysis and serialization of the report in order to
        ensure complete clarity in which model obtained which results.

        :param gold: an iterable of the gold standard or correct data.
        :param predictions: an iterable of a model's predictions.
        :return: string - a string of our results.
        """
        assert len(gold) == len(predictions),\
            'Different number of samples than predictions! {} vs {}'\
                .format(len(gold), len(predictions))

    def _results_analysis(self, model_applied, data_subset_name, gold, preds):
        """
        This is an abstraction for the functions below since they would have
        almost the same header string for each data subset the model would be
        applied to.

        :param model_applied: string - denotes the name of the model applied to
        obtain these results. This may be an actual path to the model's config
        file in order to ensure absolute clarity in exactly what model with
        what parameters obtained these results.
        :param data_subset_name: string - simple string to denote data subset.
        :param gold: an iterable of the gold standard or correct data.
        :param preds: an iterable of a model's predictions.
        :return: string - a full results report.
        """

        header = 'Model {} got the following results on dataset {} - {}:\n\n'\
                 .format(model_applied, self.dataset_name, data_subset_name)
        analysis = self._generate_results_report(gold, preds)
        full_report = ''.join([header, analysis])

        # TODO: serialize results.
        """
        Serialize the results into a results directory. We want this results
        directory to be very explicit about which model obtained the results
        in the report. The best way to do this would be to have a model config
        file for each model applied, where the 'model_applied' variable denotes
        the directory location of that config file.
        """
        return full_report

    def train_results_analysis(self, model_applied, train_pred):
        return self._results_analysis(model_applied, 'TRAINING SET',
                                      self._train_y, train_pred)

    def val_results_analysis(self, model_applied, val_pred):
        return self._results_analysis(model_applied, 'VALIDATION SET',
                                      self._val_y, val_pred)

    def test_results_analysis(self, model_applied, test_pred):
        return self._results_analysis(model_applied, 'TEST SET',
                                      self._test_y, test_pred)


    # Initialization and data loading.
    @abstractmethod
    def _load_data_from_file(self, file_name):
        """
        This is the fundamental function that must be extended in order to
        use this class. Often times, an sub-class will simply be extending
        this as a csv-reader.

        :param file_name: string
        :return: x matrix of sample data, y vector of classes/clusters.
        """
        # Numpy data is all the same.
        if file_name.endswith('.npz'):
            try:
                data = np.load(self.get_path() + file_name)
            except IOError:
                data = np.load(file_name)
            return data['x'], data['y']

        # Otherwise the extending class has to implement the data loading.
        return None, None

    def __load_data(self, file_name):
        if file_name:
            x, y = self._load_data_from_file(file_name)
            assert len(x) == len(y),'Different number of samples than labels!' \
                                    ' {} vs {}'.format(len(x), len(y))
            return x, y
        else:
            return None, None

    def load_all_data(self, train_fname, val_fname='', test_fname='', get_path=False):
        """
        Loads all data.
        """
        path = self.get_path() if get_path else ''
        self._train_x, self._train_y = self.__load_data(path + train_fname)
        self._val_x, self._val_y = self.__load_data(path + val_fname)
        self._test_x, self._test_y = self.__load_data(path + test_fname)

    @abstractmethod
    def default_load(self):
        pass

    def split_train_into_val(self, proportion=0.2, random_seed=1871):
        """
        Split validation set off from the training set according to proportion.
        """
        new_train_x, val_x, new_train_y, val_y = \
            train_test_split(self._train_x, self._train_y,
                             test_size=proportion, shuffle=True, random_state=random_seed)

        self._train_x, self._train_y = new_train_x, new_train_y
        self._val_x, self._val_y = val_x, val_y

    # Transformation methods.
    def class_transform(self, y_class):
        """
        Transforms y sample string encoding into an integer.
        :param y_class: string - designates the class of a sample.
        :return: int - a transformed version of the input.
        """
        return int(y_class)

    def sample_transform(self, sample_vec):
        """
        Transforms input string feature vector into a float feature vector.
        :param sample_vec: iterable of strings
        :return: iterable of floats
        """
        return map(float, sample_vec)

    def get_normalized_data(self, get_val=True, get_test=True, reset_internals=False):
        """
        Normalizes the train, val, and test set data according to the structure
        of the training set data. Returns a 3-tuple.
        :param get_val: bool
        :param get_test: bool
        :param reset_internals: bool - if set to true it will reset all of the internal
        data to be the new normalized data
        :return: 3-tuple of xtrain, xval, xtest
        """
        s = StandardScaler()
        new_train = s.fit_transform(self.get_train_x())
        new_val = s.transform(self.get_val_x()) if get_val else None
        new_test = s.transform(self.get_test_x()) if get_test else None

        if reset_internals:
            self._train_x = new_train
            self._val_x = new_val
            self._test_x = new_test

        return new_train, new_val, new_test

    # Data serialization - we use numpy save.
    @abstractmethod
    def get_path(self):
        pass

    def __serialize_data(self, file_name, x, y):
        if file_name:
            np.savez(file_name, x=x, y=y)

    def serialize_all_data(self, train_fname, val_fname='', test_fname=''):
        path = self.get_path()
        self.__serialize_data(path + train_fname, self._train_x, self._train_y)
        self.__serialize_data(path + val_fname, self._val_x, self._val_y)
        self.__serialize_data(path + test_fname, self._test_x, self._test_y)

