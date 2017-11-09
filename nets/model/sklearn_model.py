from nets import AbstractModel

class SKLearnModel(AbstractModel):
    def __init__(self, name, hyperparameters=None, model=None):
        """
        :param model: This is expected to be a SKlearn classification model,
        like LinearSVC or LogisticRegression, really just anything with:
            methods - fit(x,y) and predict(x)
        :param hyperparameters: note that no functionality for hyperparams
        is included at the current time for sklearn models.
        """
        super(SKLearnModel, self).__init__(name, hyperparameters)
        self.classifier = model

    def predict(self, x):
        super(SKLearnModel, self).predict(x)
        return self.classifier.predict(x)

    def train(self, x, y):
        super(SKLearnModel, self).train(x, y)
        self.classifier.fit(x, y)

