from abc import ABCMeta
from nts.dataset import AbstractDataset
from sklearn.metrics import classification_report, \
    precision_recall_fscore_support


class AbstractClassificationDataset(AbstractDataset):

    # Makes the class abstract.
    __metaclass__ = ABCMeta

    # Overriding abstract method
    def evaluate(self, gold, predictions):
        super(AbstractClassificationDataset, self).evaluate(gold, predictions)
        precision, recall, fscore, _ = \
            precision_recall_fscore_support(gold, predictions,
                                            average='weighted')
        return precision, recall, fscore

    # Overriding abstract method
    def _generate_results_report(self, gold, predictions):
        super(AbstractClassificationDataset, self)._generate_results_report()
        return classification_report(gold, predictions, digits=3)
