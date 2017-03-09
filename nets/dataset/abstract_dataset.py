from nets import PROJECT_BASE_DIR
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
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
    def get_train_data(self):
        return self._train_x, self._train_y

    def get_val_x(self):
        return self._val_x

    def get_test_x(self):
        return self._test_x

    def iterate_train_minibatches(self, batchsize, shuffle=False):
        """
        Iterates minibatches, used mostly only by neural networks.
        :param batchsize: int - designates size of batch
        :param shuffle: bool - optional, says whether or not randomized shuffle.
        :return: tuple - yielded subset of the training data as a minibatch.
        """
        # Copy the train data so there is no problems later.
        x, y = deepcopy(self.get_train_data())

        # Shuffle the data, if desired.
        if shuffle:
            c = list(zip(x, y))
            random.shuffle(c)
            x, y = zip(*c)

        # Iterate. Note that we may cut off some samples by rounding down.
        for i in xrange(len(x) / batchsize):
            start = i * batchsize
            end = (i+1) * batchsize
            yield x[start:end], y[start:end]


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
            data = np.load(self.get_path() + file_name)
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

    def load_all_data(self, train_fname, val_fname='', test_fname=''):
        """
        Loads all data, note that only the filename, not the path, is needed.
        """
        self._train_x, self._train_y = self.__load_data(train_fname)
        self._val_x, self._val_y = self.__load_data(val_fname)
        self._test_x, self._test_y = self.__load_data(test_fname)

    @abstractmethod
    def default_load(self):
        pass

    def split_train_into_val(self, proportion=0.2, random_seed=1871):
        """
        If there is no validation set, split one off from the training set
         such that it is either of equal size to the test set or is 1/5 of the
         training set, whichever takes less data away from training.
        """
        assert (self._val_x is None and self._train_x is not None), \
            'Error - attempting to make validation set when it already exists!'

        # choose the proportion of data we will be taking from train set.
        prop_of_test_set = len(self._test_x) / float(len(self._train_x))
        proportion = min(proportion, prop_of_test_set)
        items_in_val = int(proportion * len(self._train_x))

        # seed and split away train set data into the val.
        random.seed(random_seed)
        indicies = range(len(self._train_x))
        random.shuffle(indicies)

        # set them
        new_train_x = self._train_x[indicies[items_in_val:]]
        new_train_y = self._train_y[indicies[items_in_val:]]
        val_x = self._train_x[indicies[:items_in_val]]
        val_y = self._train_y[indicies[:items_in_val]]

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

    def get_normalized_data(self, get_val=True, get_test=True):
        """
        Normalizes the train, val, and test set data according to the structure
        of the training set data. Returns a 3-tuple.
        :param get_val: bool
        :param get_test: bool
        :return: 3-tuple of xtrain, xval, xtest
        """
        s = StandardScaler()
        new_train = s.fit_transform(self.get_train_data()[0])
        new_val = s.transform(self.get_val_x()) if get_val else None
        new_test = s.transform(self.get_test_x()) if get_test else None
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

