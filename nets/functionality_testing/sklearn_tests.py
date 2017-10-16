from nets.dataset.classification import DotaDataset
from nets.model.sklearn_model import SKLearnModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC

# TODO: test these not with the dota dataset.

def test_model():
    d = DotaDataset()
    d.default_load()

    lr = SKLearnModel('Logistic Regression', model=LogisticRegression())
    linsvc = SKLearnModel('Linear SVC', model=LinearSVC())
    svc = SKLearnModel('SVC', model=SVC())

    train_x, train_y = d.get_train_data()
    val_x = d.get_val_x()

    norm_train_x, norm_val_x, _ = d.get_normalized_data(get_test=False)
    for model in [lr, linsvc, svc]:
        print 'Training model {}...'.format(model.model_name)
        model.train(train_x, train_y)
        val_pred = model.predict(val_x)
        # p,r,f1 = d.evaluate_val_predictions(val_pred)
        # print("  val recall:\t\t\t{:.3f}".format(r))
        # print("  val precision:\t\t{:.3f}".format(p))
        # print("  val f1:\t\t\t\t{:.3f}".format(f1))
        print d.val_results_analysis(model.model_name, val_pred)

        print '\nTraining model {} on normalized data...'.format(model.model_name)
        model.train(norm_train_x, train_y)
        val_pred = model.predict(norm_val_x)
        # p, r, f1 = d.evaluate_val_predictions(val_pred)
        # print("  val recall:\t\t\t{:.3f}".format(r))
        # print("  val precision:\t\t{:.3f}".format(p))
        # print("  val f1:\t\t\t\t{:.3f}".format(f1))
        print d.val_results_analysis(model.model_name, val_pred)
        print '\n'

if __name__ == '__main__':
    test_model()
