from abc import ABCMeta, abstractmethod


class AbstractModel(object):
    """
    This is the base class all models extend. It simply defines the base methods
    that are held common to all ML models. Models are initialized with a name
    and an optional set of hyperparameters, which is a dictionary.
    """
    # Makes the class abstract.
    __metaclass__ = ABCMeta

    def __init__(self, name, hyperparameters=None):
        self.model_name = name
        self.params = hyperparameters

    def get_full_name(self):
        """
        Gets the full name of a model based on the values of its hyper-params.
        :return: string
        """
        s = [self.model_name]
        try:
            for key,value in self.params.iteritems():
                if not self.is_default_parameter(key, value) and \
                                type(value) is not list:
                    s.append('_{}{}'.format(key, value))
        except AttributeError:
            pass
        return ''.join(s)

    def get_serializable_config(self):
        """
        Makes a complete description of model in a nice way.
        :return: string
        """
        s = ['Model name: ', self.model_name, '\n']
        for key, value in self.params.iteritems():
            s.append('\t{} : {}\n'.format(key, str(value)))
        s.append('\n')
        return ''.join(s)

    @staticmethod
    def is_default_parameter(param_name, value):
        """
        Boolean to be overridden which says whether or not a certain hyper-param
        setting is the default value.
        :param param_name: string - name of the hyper-parameter
        :param value: object - the value the hyperparameter takes
        :return: bool - whether or not this is a default setting
        """

        # TODO: ensure that this is not called when overridden!
        return False

    @abstractmethod
    def train(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass
