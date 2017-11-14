from nets.model.abstract_model import AbstractModel
from nets.model.sklearn_model import SKLearnModel
from nets.dataset.classification import SyntheticDataset, AbstractClassificationDataset
from nets.dataset.classification import synthetic_dataset as synth_params
from nets import util
from nets.util.experimenter import Experimenter, MODEL_START_KEY
from sklearn.svm import LinearSVC, SVC
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from itertools import product
from collections import OrderedDict

_d = {
    synth_params.DIFFICULTY: list(SyntheticDataset.iter_all_difficulties()),
    synth_params.NUM_CLASSES: [2, 5, 10],
    synth_params.MAKE_BLOBS: [False],
}
SYNTHETIC_DATA_PARAMS = OrderedDict(sorted(_d.items(), key=lambda t: t[0]))

_m = {
    'model': [LinearSVC()]
}
MODEL_PARAMS = OrderedDict(sorted(_m.items(), key=lambda t: t[0]))

RESULTS_HEADERS = [
    'train_acc',
    'val_acc',
    'test_acc',
]

# specific testing
def test_model_on_data(model, data):
    """
    :param model: AbstractModel
    :param data: AbstractClassificationDataset
    :return: triple of predictions for train, val, and test
    """
    model.train(*data.get_train_data())
    train_pred = model.predict(data.get_train_x())
    val_pred = model.predict(data.get_val_x())
    test_pred = model.predict(data.get_test_x())
    return train_pred, val_pred, test_pred

def cv_test_model_on_data(model, data, k=5):
    preds_golds = []
    for xtrain, ytrain, xval, yval in data.iterate_cross_validation(k_folds=k, merge_train_val=True):
        model.train(xtrain, ytrain)
        preds.append((model.predict(xval), yval))
    return preds_golds


################################
## global grid search testing ##
################################

def iter_all_datasets():
    keys = list(SYNTHETIC_DATA_PARAMS.keys())
    param_settings = product(*SYNTHETIC_DATA_PARAMS.values())

    for param_vals in param_settings:
        params = {key: value for key, value in zip(keys, param_vals)}
        d = SyntheticDataset(util.params_to_str(params), params)
        d.make_dataset()
        yield d

def iter_all_models():
    keys = list(MODEL_PARAMS.keys())
    param_settings = product(*MODEL_PARAMS.values())

    for param_vals in param_settings:
        params = {key: value for key, value in zip(keys, param_vals)}
        yield SKLearnModel(util.params_to_str(params), params)

def test_all_models(viz=False):
    exp = Experimenter('synthetic_results/sklearn_exps',
                       list(SYNTHETIC_DATA_PARAMS.keys()),
                       [MODEL_START_KEY] + list(MODEL_PARAMS.keys()) + RESULTS_HEADERS)

    for d in iter_all_datasets():
        if viz:
            d.visualize_data(util.str_to_save_name(d.name) + '/')

        for model in iter_all_models():
            print('Testing model {} on {}...'.format(model.get_full_name(), d.name))

            trainp, valp, testp = test_model_on_data(model, d)

            results = [
                d.evaluate_train_predictions(trainp)[-1],
                d.evaluate_val_predictions(valp)[-1],
                d.evaluate_test_predictions(testp)[-1],
            ]
            exp.add_result(d.get_param_vals(), model.get_param_vals(with_name=True) + results)

            # detailed results reports for each subset
            tr_res = d.train_results_analysis(model, trainp)
            v_res = d.val_results_analysis(model, valp)
            te_res = d.test_results_analysis(model, testp)

            res_name = ''.join([
                util.str_to_save_name(d.name),
                '_with_',
                util.str_to_save_name(model.model_name),
                '.txt',
            ])

            util.results_write('synthetic_results/' + res_name, [tr_res, v_res, te_res])
        exp.serialize()

if __name__ == '__main__':
    # test_all_models(viz=False)
    exp = Experimenter.load_results('synthetic_results/sklearn_exps.tsv')
    pass
