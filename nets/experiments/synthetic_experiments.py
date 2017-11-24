# quick fix
import sys
sys.path.append('/home/ml/kkenyo1/nets')
# TODO: fix this shit.

import torch
import numpy as np
from sys import argv
from nets import util
from nets.model.abstract_model import AbstractModel
from nets.model.sklearn_model import SKLearnModel
from nets.model.neural.basic_mlp import DENSE_DIMS, ACTIVATION
from nets.model.neural.neural_model import CORE, LAM1, LAM2, LAM3, LOSS_FUN
from nets.model.neural.neural_constructors import make_mlp, get_default_params
from nets.dataset.classification import SyntheticDataset, AbstractClassificationDataset
from nets.dataset.classification import synthetic_dataset as synth_params
from nets.util.experimenter import Experimenter, MODEL_START_KEY
from sklearn.svm import LinearSVC, SVC
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from itertools import product
from collections import OrderedDict

_d = {
    synth_params.DIFFICULTY: list(SyntheticDataset.iter_all_difficulties()),
    synth_params.NUM_CLASSES: [2, 5, 10],
    synth_params.MAKE_BLOBS: [False],
}
SYNTHETIC_DATA_PARAMS = OrderedDict(sorted(_d.items(), key=lambda t: t[0]))

_m = {
    'model': [
        # LinearSVC(),
        SVC,
        # LogisticRegression(),
        # LogisticRegressionCV(),
    ],
    'C': list(np.arange(0.05, 1, 0.05)) + [1, 2, 3, 4, 5],
    'kernel': ['rbf', 'linear']
}
MODEL_PARAMS = OrderedDict(sorted(_m.items(), key=lambda t: t[0]))
BATCH_SIZE = 'bsz'
RESULTS_HEADERS = [
    'train_acc',
    'val_acc',
    'test_acc',
]
EPOCHS = 'epochs'

# specific testing
def test_model_on_data(model, data, train=True):
    """
    :param model: AbstractModel
    :param data: AbstractClassificationDataset
    :return: triple of predictions for train, val, and test
    """
    if train:
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

def test_neural_model(model, dataset, bsz, learning_rate_decay=1, one_time=False):
    if model is None:
        nets_model = make_mlp(
            'Neural Model 1000-256-256-10_shuff',
            dataset.get_unique_labels(),
            input_dim=dataset.get_num_features(),
            num_classes=dataset.get_num_classes(),
            layer_dims=[256, 256],
            lr=1e-4,
        )
    else:
        nets_model = model

    print('GPU STATUS:')
    print(torch.cuda.is_available())
    nets_model.model.cuda()

    print('Model graph:')
    print(nets_model.model)

    best_val_acc = 0
    best_preds = []

    print('Testing minibatch iteration...')
    i = -1
    for samples, labels, epoch in dataset.iterate_train_minibatches(batch_size=bsz, epochs=model.params[EPOCHS], shuffle=True):
        if epoch > i:
            train_losses, train_accs, t_out_dist, t_mean_outs = \
                nets_model.predict_with_stats(dataset.get_train_x(), dataset.get_train_y())

            val_losses, val_accs, v_out_dist, v_mean_outs = \
                nets_model.predict_with_stats(dataset.get_val_x(), dataset.get_val_y())

            print(f'\nEpoch {epoch} results:')
            print('  Norm W    : {:>5}'.format(nets_model.get_w_norm()))
            print('  Norm Grad : {:>5}'.format(nets_model.get_grad_norm()))

            loss_idx = 1
            loss_names = {1: 'CCE (want to minimize to 0)',
                          2: 'Attractive Sample-to-Centroid (want to minimize to 0) cosine distance',
                          3: 'Repulsive Sample-to-Centroid (want to maximize to +1) cosine distance',
                          4: 'Centroid-to-Centroid Repulsive (want to maximize to +1) cosine distance'}
            for tloss, vloss in zip(train_losses, val_losses):
                print('  {}:'.format(loss_names[loss_idx]))
                print('    Train: {:>5}'.format(tloss.data[0] * (-1 if loss_idx > 2 else 1)))
                print('    Val  : {:>5}'.format(vloss.data[0] * (-1 if loss_idx > 2 else 1)))
                loss_idx += 1

            print('  Train Accuracies:')
            for name, acc in train_accs:
                print('    {:<10}:  {:>5}'.format(name, acc))
            # print('    Out dist  :  {}'.format(t_out_dist))
            # print('    Mean conf :  {}'.format(t_mean_outs))

            print('  Val Accuracies:')
            for name, acc in val_accs:
                print('    {:<10}:  {:>5}'.format(name, acc))

                if acc > best_val_acc:
                    best_val_acc = acc
                    best_preds = test_model_on_data(nets_model, dataset, train=False)

            # print('    Out dist  :  {}'.format(v_out_dist))
            # print('    Mean conf :  {}'.format(v_mean_outs))
            nets_model.update_lr(learning_rate_decay)
            i += 1

        nets_model.train(samples, labels)

    trainp, trainreps = best_preds[0]
    valp, valreps = best_preds[1]
    testp, testreps = best_preds[2]

    if one_time:
        # detailed results reports for each subset
        tr_res = dataset.train_results_analysis(nets_model, trainp)
        v_res = dataset.val_results_analysis(nets_model, valp)
        te_res = dataset.test_results_analysis(nets_model, testp)

        res_name = ''.join([
            util.str_to_save_name(dataset.name),
            '_with_',
            util.str_to_save_name(nets_model.model_name),
            '.txt',
        ])

        util.results_write(f'synthetic_results/{res_name}', [tr_res, v_res, te_res])

    return trainp, valp, testp


################################
## global grid search testing ##
################################

def iter_all_datasets(data_params):
    keys = list(data_params.keys())
    param_settings = product(*data_params.values())

    for param_vals in param_settings:
        params = {key: value for key, value in zip(keys, param_vals)}
        d = SyntheticDataset(util.params_to_str(params), params)
        d.make_dataset()
        yield d

def iter_all_models(model_params, dataset, sklearn=True):
    keys = list(model_params.keys())
    param_settings = product(*model_params.values())

    for param_vals in param_settings:
        grid_params = {key: value for key, value in zip(keys, param_vals)}

        if sklearn:
            yield SKLearnModel(util.params_to_str(grid_params), grid_params)
        else:
            n_params = get_default_params(
                input_dim=dataset.get_num_features(),
                num_classes=dataset.get_num_classes(),
                layer_dims=grid_params[DENSE_DIMS],
                lr=0.5e-4,
            )
            for hyper in grid_params:
                n_params[hyper] = grid_params[hyper]

            # print(n_params)
            yield make_mlp(
                name=util.params_to_str({**grid_params}),
                unique_labels=dataset.get_unique_labels(),
                params=n_params,
            )

def test_all_models(results_out, data_params, model_params, sklearn=False, viz=False, normalize=False):
    exp = Experimenter(f'synthetic_results/{results_out}',
                       list(data_params.keys()),
                       [MODEL_START_KEY] + list(model_params.keys()) + RESULTS_HEADERS)

    for d in iter_all_datasets(data_params):
        if viz:
            d.visualize_data(f'{util.str_to_save_name(d.name)}/')

        if normalize:
            d.get_normalized_data(reset_internals=True)

        for model in iter_all_models(model_params, d, sklearn=sklearn):
            print('\n\n-----------------------------------')
            print(f'Testing model {model.get_full_name()} on {d.name}...')

            if not sklearn:
                trainp, valp, testp = test_neural_model(model, d, model.params[BATCH_SIZE], one_time=False)
            else:
                trainp, valp, testp = test_model_on_data(model, d, train=sklearn)

            results = [
                d.evaluate_train_predictions(trainp)[-1],
                d.evaluate_val_predictions(valp)[-1],
                d.evaluate_test_predictions(testp)[-1],
            ]
            exp.add_result(d.get_param_vals(), model.get_param_vals(with_name=True, filtr=list(model_params.keys())) + results)

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

            util.results_write(f'synthetic_results/{res_name}', [tr_res, v_res, te_res])
        exp.serialize()

if __name__ == '__main__':
    data_params = {
        synth_params.DIFFICULTY: [synth_params.HARD],
        synth_params.NUM_CLASSES: [10],
        synth_params.MAKE_BLOBS: [False],
    }

    model_params = {
        ACTIVATION: [
            torch.nn.ReLU,
            # torch.nn.PReLU,
            # torch.nn.ReLU6,
            # torch.nn.LeakyReLU,
            # torch.nn.ELU,
            # torch.nn.Tanh,
        ],

        DENSE_DIMS: [
            # [128],
            [256],
            [512],
            [128, 128],
            [256, 256],
            [512, 512],
        ],

        CORE: [
            # {},
            # {LAM1: 1, LAM2: 0, LAM3: 0},
            # {LAM1: 0, LAM2: 1, LAM3: 0},
            {LAM1: 0, LAM2: 0, LAM3: 1},
            # {LAM1: 1, LAM2: 0, LAM3: 1},
            # {LAM1: 1, LAM2: 1, LAM3: 0},
            {LAM1: 1, LAM2: 1, LAM3: 1},
        ],

        BATCH_SIZE: [3400],
        EPOCHS: [500],

    }

    if 'neural' in argv:
        test_all_models('neural_prelu_CORE_log_softmax_tests', data_params, model_params, viz=False, sklearn=False)
    else:
        test_all_models('sklearn_svc2_tests', data_params, MODEL_PARAMS, viz=False, sklearn=True, normalize=True)


    # exp = Experimenter.load_results('synthetic_results/sklearn_exps.tsv')
    # test_neural_model()
