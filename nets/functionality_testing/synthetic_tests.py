# quick fix
import sys
sys.path.append('/home/ml/kkenyo1/nets')
# TODO: fix this shit.

from nets.dataset.classification.synthetic_dataset import SyntheticDataset
from sklearn.datasets import make_blobs, make_classification

def test_synthetic_data(d, x, y, save_dir):
    d.set_all_data(x, y)
    d.get_data_statistics()
    d.pca_visualize(save_dir)
    d.visualize_data(save_dir)

def make_testing():
    num_samples = 10000
    num_features = 10000
    num_redundant = 0
    num_informative = 50
    num_classes = 10
    num_clusters_per_class = 1
    class_sep = 1 # smaller = more difficult

    x, y = make_classification(
        n_samples=num_samples,
        n_features=num_features,
        n_informative=num_informative,
        n_redundant=num_redundant,
        n_classes=num_classes,
        n_clusters_per_class=num_clusters_per_class,
        class_sep=class_sep,
        random_state=100,
    )

    sepstr = str(class_sep).replace('.', 'e')
    name = 'MED make f{} fred{} k{} cpc{} csep{}'.format(num_features, num_redundant, num_classes,
                                                          num_clusters_per_class, sepstr)
    d = SyntheticDataset(name)
    save_dir = name.replace(' ', '_') + '/'
    test_synthetic_data(d, x, y, save_dir)

def blob_testing():
    num_samples = 1000
    num_features = 100
    num_classes = 10
    cluster_std = 5

    x, y = make_blobs(num_samples, n_features=num_features, centers=num_classes, random_state=0, cluster_std=cluster_std)
    d = SyntheticDataset('Blobs n{} f{} k{} std{}'.format(num_samples, num_features, num_classes, cluster_std))
    save_dir = 'n{}_f{}_k{}_std{}/'.format(num_samples, num_features, num_classes, cluster_std)

    test_synthetic_data(d, x, y, save_dir)

if __name__ == '__main__':
    make_testing()
