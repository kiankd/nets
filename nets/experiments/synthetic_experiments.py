from nets.model.abstract_model import AbstractModel
from nets.model.sklearn_model import SKLearnModel
from nets.dataset.classification import SyntheticDataset, AbstractClassificationDataset
from nets import util
from sklearn.svm import LinearSVC, SVC
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

def iter_sup_models():
    models = [
        SKLearnModel(name='Logistic Regression', model=LogisticRegression()),
        SKLearnModel(name='Linear SVM', model=LinearSVC()),
        # SKLearnModel(name='RBF SVM', model=SVC()),
    ]
    return models.__iter__()

def iter_by_diff_k(difficulty, num_classes, blob_settings=(True, False,)):
    datasets = []
    for use_blobs in blob_settings:
        s = 'blobs_' if use_blobs else 'make_'
        s += '{}_'.format(num_classes)
        d = SyntheticDataset(s + difficulty, difficulty=difficulty, blobs=use_blobs)
        d.make_dataset(num_classes)
        datasets.append(d)
    return datasets.__iter__()


# specific testing
def test_model_on_data(model, data):
    """
    :param model: AbstractModel
    :param data: AbstractClassificationDataset
    :return:
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

# big iter
def iter_all_datasets(blob_settings=(True, False,)):
    for diff in SyntheticDataset.iter_all_difficulties():
        for num_classes in [2, 5, 10]:
            for dataset in iter_by_diff_k(diff, num_classes, blob_settings=blob_settings):
                yield dataset

def test_all_models(viz=False):
    for d in iter_all_datasets():
        if viz:
            d.visualize_data(util.str_to_save_name(d.name) + '/')

        for model in iter_sup_models():
            print('Testing model {} on {}...'.format(model.get_full_name(), d.name))

            trainp, valp, testp = test_model_on_data(model, d)

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

if __name__ == '__main__':
    test_all_models(viz=False)
