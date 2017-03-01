from abc import ABCMeta, abstractmethod
import numpy as np

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

    # Initializes the essential components of a dataset as attributes.
    def __init__(self):
        self._train_x = None
        self._train_y =  None
        self._val_x = None
        self._val_y =  None
        self._test_x = None
        self._test_y =  None
        self.dataset_name = ''


    # Getter methods.
    def get_train_data(self):
        return self._train_x, self._train_y

    def get_val_x(self):
        return self._val_x

    def get_test_x(self):
        return self._test_x


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
        return self.evaluate(self._train_y, train_pred)

    def evaluate_val_predictions(self, val_pred):
        return self.evaluate(self._val_y, val_pred)

    def evaluate_test_predictions(self, test_pred):
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
            'Different number of gold samples than predicted! %d vs %d.'%(
                len(gold), len(predictions))

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

        header = 'Model {} got the following results on dataset {} - {}\n\n.'\
                 .format(model_applied, self.dataset_name, data_subset_name)
        analysis = self._generate_results_report(gold, predictions)
        full_report = ''.join([header, analysis])

        # TODO:
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
        if file_name.endswith('.npy'):
            data = np.load(file_name)
            return data[0], data[1]

        # Otherwise the extending class has to implement the data loading.
        pass

    def __load_data(self, file_name):
        if file_name:
            x, y = self._load_data_from_file(file_name)
            assert len(x) == len(y),\
                'Different number of samples than labels! %d vs %d'%(len(x), len(y))
            return x, y
        else:
            return None, None

    def load_all_data(self, train_fname, val_fname='', test_fname=''):
        self._train_x, self._train_y = self.__load_data(train_fname)
        self._val_x, self._val_y = self.__load_data(val_fname)
        self._test_x, self._test_y = self.__load_data(test_fname)


    # Data serialization - we use numpy save.
    def __serialize_data(self, file_name, x, y):
        if file_name:
            np.save(file_name, np.array([x, y]))

    def serialize_all_data(self, train_fname, val_fname='', test_fname=''):
        self.__serialize_data(train_fname, self._train_x, self._train_y)
        self.__serialize_data(val_fname, self._val_x, self._val_y)
        self.__serialize_data(test_fname, self._test_x, self._test_y)

