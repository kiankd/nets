from abc import ABCMeta, abstractmethod
from nets import AbstractDataset
from sklearn.metrics import classification_report, precision_recall_fscore_support

NUM_CLASSES = 'num_classes'

class AbstractClassificationDataset(AbstractDataset):

    # Makes the class abstract.
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_num_classes(self):
        pass

    # Overriding abstract method
    def evaluate(self, gold, predictions):
        """
        :param gold: an iterable of the gold standard or correct data.
        :param predictions: an iterable of a model's predictions.
        :return: tuple - a triple of floats, (prec, recall, f1)
        """
        super(AbstractClassificationDataset, self).evaluate(gold, predictions)
        precision, recall, fscore, _ = \
            precision_recall_fscore_support(gold, predictions,
                                            average='weighted')
        return precision, recall, fscore

    # Overriding abstract method
    def _generate_results_report(self, gold, predictions):
        super(AbstractClassificationDataset, self)._generate_results_report(
            gold, predictions)
        return classification_report(gold, predictions, digits=3)

