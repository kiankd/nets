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
from nets.util.constants import *
from nets.util.experimenter import Experimenter
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
RESULTS_HEADERS = [
    'train_acc',
    'val_acc',
    'test_acc',
]

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

def test_neural_model(nets_model, dataset, bsz, learning_rate_decay=1, one_time=False, only_accs=False):
    print('GPU STATUS:')
    print(torch.cuda.is_available())
    nets_model.model.cuda()

    print('Model graph:')
    print(nets_model.model)

    best_epoch = -1
    best_val_acc = 0
    best_preds = []

    tracking_results = []
    centroids_over_time = []

    print('Testing minibatch iteration...')
    i = -1
    for samples, labels, epoch in dataset.iterate_train_minibatches(batch_size=bsz, epochs=nets_model.params[EPOCHS], shuffle=True):
        if epoch > i:
            if only_accs and epoch > 0:
                train_accs = nets_model.fast_acc_pred(dataset.get_train_x(), dataset.get_train_y())
                val_accs = nets_model.fast_acc_pred(dataset.get_val_x(), dataset.get_val_y())
            else:
                train_losses, train_accs, t_out_dist, t_mean_outs, t_repr_norm = \
                    nets_model.predict_with_stats(dataset.get_train_x(), dataset.get_train_y())

                val_losses, val_accs, v_out_dist, v_mean_outs, v_repr_norm = \
                    nets_model.predict_with_stats(dataset.get_val_x(), dataset.get_val_y())

                # save results for tracking progress
                centroids_over_time.append(nets_model.model.centroid_matrix.data.type(torch.DoubleTensor).numpy())

                epoch_results = OrderedDict([
                    (EPOCH, epoch),
                    (WNORM, nets_model.get_w_norm()),
                    (GNORM, nets_model.get_grad_norm()),
                    (ANORM + TRAIN, t_repr_norm),
                    (ANORM + VAL, v_repr_norm),
                ])
                loss_keys = (CCE_LOSS, ATT_SC_LOSS, REP_SC_LOSS, REP_CC_LOSS,)
                for tloss, vloss, lkey in zip(train_losses, val_losses, loss_keys):
                    epoch_results[lkey + TRAIN] = tloss.data[0]
                    epoch_results[lkey + VAL] = vloss.data[0]

                epoch_results[F1_ACC + TRAIN] = train_accs[F1_ACC]
                epoch_results[F1_ACC + VAL] = val_accs[F1_ACC]

                tracking_results.append(epoch_results)

                print(f'\nEpoch {epoch} results:')

                print('  Norm W    : {:>5}'.format(epoch_results[WNORM]))
                print('  Norm Grad : {:>5}'.format(epoch_results[GNORM]))
                print('  Norm Train Repr : {:>5}'.format(t_repr_norm))
                print('  Norm Val Repr   : {:>5}'.format(v_repr_norm))

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
            for name, acc in train_accs.items():
                print('    {:<10}:  {:>5}'.format(name, acc))
            # print('    Out dist  :  {}'.format(t_out_dist))
            # print('    Mean conf :  {}'.format(t_mean_outs))

            print('  Val Accuracies:')
            for name, acc in val_accs.items():
                print('    {:<10}:  {:>5}'.format(name, acc))

                if acc > best_val_acc:
                    best_val_acc = acc
                    best_preds = test_model_on_data(nets_model, dataset, train=False)
                    best_epoch = epoch

            # print('    Out dist  :  {}'.format(v_out_dist))
            # print('    Mean conf :  {}'.format(v_mean_outs))
            nets_model.update_lr(learning_rate_decay)
            i += 1

        nets_model.train(samples, labels)

    print(f'\n{nets_model.model_name}\n---------\n  BEST EPOCH: {best_epoch}.\n  VAL-ACC: {best_val_acc}--------\n')
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

    return trainp, valp, testp, tracking_results, centroids_over_time


################################
## global grid search testing ##
################################

def init_nn_model(params, dataset):
    n_params = get_default_params(
        input_dim=dataset.get_num_features(),
        num_classes=dataset.get_num_classes(),
        layer_dims=params[DENSE_DIMS],
        lr=0.5e-4,
    )
    for hyper in params:
        n_params[hyper] = params[hyper]

    # print(n_params)
    return make_mlp(
        name=util.params_to_str({**params}),
        unique_labels=dataset.get_unique_labels(),
        params=n_params,
    )

def init_dataset(params):
    d = SyntheticDataset(util.params_to_str(params), params)
    d.make_dataset()
    return d

def iter_all_datasets(data_params):
    keys = list(data_params.keys())
    param_settings = product(*data_params.values())

    for param_vals in param_settings:
        params = {key: value for key, value in zip(keys, param_vals)}
        yield init_dataset(params)

def iter_all_models(model_params, dataset, sklearn=True):
    keys = list(model_params.keys())
    param_settings = product(*model_params.values())

    for param_vals in param_settings:
        grid_params = {key: value for key, value in zip(keys, param_vals)}

        if sklearn:
            yield SKLearnModel(util.params_to_str(grid_params), grid_params)
        else:
            yield init_nn_model(grid_params, dataset)

def test_all_models(results_out, data_params, model_params, sklearn=False, viz=False, normalize=False, fast=False, outdir=''):
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
                trainp, valp, testp, _, _ = test_neural_model(model, d, model.params[BATCH_SIZE], one_time=False, only_accs=fast)
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

            util.results_write(f'synthetic_results/{outdir}{res_name}', [tr_res, v_res, te_res])
        exp.serialize()

def track_model(data_params, nn_params, fprefix='', outdir=''):
    dataset = init_dataset(data_params)
    nn_model = init_nn_model(nn_params, dataset)

    constant_headers = list(data_params.keys()) + list(model_params.keys())
    constant_values = dataset.get_param_vals() + nn_model.get_param_vals(with_name=True, filtr=list(nn_params.keys()))
    tracking_headers = [MODEL_START_KEY] + list(EXP_TRACKING)
    exp = Experimenter(f'{outdir}{fprefix}_tracking_results', constant_headers, tracking_headers)

    trp, valp, testp, tracking_results, epoch_centroids = test_neural_model(nn_model, dataset, bsz=nn_model.params[BATCH_SIZE])

    np.save(f'{outdir}{fprefix}_centroids.npy', np.array(epoch_centroids))

    for results_dict in tracking_results:
        exp.add_result(constant_values, list(results_dict.values()))
    exp.serialize()

    # detailed results reports for each subset
    tr_res = dataset.train_results_analysis(nn_model, trp)
    v_res = dataset.val_results_analysis(nn_model, valp)
    te_res = dataset.test_results_analysis(nn_model, testp)

    res_name = ''.join([
        util.str_to_save_name(dataset.name),
        '_with_',
        util.str_to_save_name(nn_model.model_name),
        '.txt',
    ])

    util.results_write(f'{outdir}{res_name}', [tr_res, v_res, te_res])


if __name__ == '__main__':
    data_params = {
        synth_params.DIFFICULTY: [synth_params.HARD],
        synth_params.NUM_CLASSES: [10],
        synth_params.MAKE_BLOBS: [False],
    }

    model_params = {
        LEARNING_RATE: [
            # 0.0005,
            0.00065,
        ],

        ACTIVATION: [
            lambda: torch.nn.LeakyReLU(0.1),
        ],

        DENSE_DIMS: [
            [2048, 256, 4096],
            # [2048, 512, 4096],
        ],

        CORE: [
            # {},
            {LAM1: 1, LAM2: 1, LAM3: 0},
        ],

        BATCH_SIZE: [
            3400,
            # 100,
        ],
        EPOCHS: [100],

    }


    # results
    final_d_params = {
        synth_params.DIFFICULTY: synth_params.HARD,
        synth_params.NUM_CLASSES: 10,
        synth_params.MAKE_BLOBS: False,
    }

    best_corenet_params = {
        LEARNING_RATE: 0.00065,
        ACTIVATION: lambda: torch.nn.LeakyReLU(0.1),
        DENSE_DIMS: [2048, 256, 4096],
        CORE: {LAM1: 1, LAM2: 1, LAM3: 0},
        BATCH_SIZE: 3400,
        EPOCHS: 100,
    }

    best_ffnet_params = {
        LEARNING_RATE: 0.0005,
        ACTIVATION: lambda: torch.nn.LeakyReLU(0.1),
        DENSE_DIMS: [2048, 512, 4096],
        CORE: {},
        BATCH_SIZE: 100,
        EPOCHS: 100,
    }

    if 'neural' in argv:
        #test_all_models('final_tests', data_params, model_params, viz=False, sklearn=False, fast=False, outdir='final_tests/')
        track_model(final_d_params, best_corenet_params, fprefix='best_corenet', outdir='synthetic_results/final_tests3/')
        # track_model(final_d_params, best_ffnet_params, fprefix='best_ffnet', outdir='synthetic_results/final_tests2/')

    else:
        test_all_models('sklearn_svc2_tests', data_params, MODEL_PARAMS, viz=False, sklearn=True, normalize=True)


    # exp = Experimenter.load_results('synthetic_results/sklearn_exps.tsv')
    # test_neural_model()
