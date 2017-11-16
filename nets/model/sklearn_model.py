import inspect
from nets import AbstractModel

MODEL = 'model'

class SKLearnModel(AbstractModel):
    def __init__(self, name, params):
        """
        :param model: This is expected to be a SKlearn classification model,
        like LinearSVC or LogisticRegression, really just anything with:
            methods - fit(x,y) and predict(x)
        :param hyperparameters: note that no functionality for hyperparams
        is included at the current time for sklearn models.
        """
        super(SKLearnModel, self).__init__(name, params)

        self.params = params
        sklearn_params = {}
        for key in params:
            if key in inspect.getfullargspec(params[MODEL]).args:
                sklearn_params[key] = params[key]
        self.classifier = params[MODEL](**sklearn_params)

    def predict(self, x):
        super(SKLearnModel, self).predict(x)
        return self.classifier.predict(x)

    def train(self, x, y):
        super(SKLearnModel, self).train(x, y)
        self.classifier.fit(x, y)

