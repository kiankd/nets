import sys
sys.path.append('/home/ml/kkenyo1/nets')

import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from nets.util.constants import *
from nets.util.experimenter import Experimenter


FIG_SIZE = (20, 10)

def analyze_centroids(centroids_fname, cosine=True, noline=True):
    centroids_ot = np.load(centroids_fname)
    num_epochs = centroids_ot.shape[0]
    num_classes = centroids_ot.shape[1]
    num_feats = centroids_ot.shape[2]

    all_centroids = centroids_ot.reshape(num_epochs * num_classes, num_feats, order='F')
    if cosine:
        all_centroids = normalize(all_centroids)

    projectors = {
        'PCA': lambda x: PCA(n_components=2).fit_transform(x),
        'TNSE': lambda x: TSNE(n_components=2).fit_transform(x)
    }

    for name, projection_fun in projectors.items():
        print(f'Fitting {name}...')
        coordinates = projection_fun(all_centroids)
        print('Done fitting.')

        cent_coords_ot = coordinates.reshape((num_classes, num_epochs, 2))

        # make the figure
        plt.figure(figsize=FIG_SIZE)
        alphas = np.linspace(0.1, 1, num_epochs)
        for k, all_coords in enumerate(cent_coords_ot):
            xy = np.array(all_coords)
            if noline:
                for epoch, coord in enumerate(xy):
                    plt.plot(coord[0], coord[1], 'o', color=COLORS[k], alpha=alphas[epoch])
            else:
                plt.plot(xy[:,0], xy[:, 1], '-o', color=COLORS[k], label=f'Class {k}')

        if not noline:
            plt.legend()

        norm_str = '_cos' if cosine else ''
        plt.savefig(f'synthetic_results/final_tests3/centroid_mvt_{name}{norm_str}_noline.png', bbox_inches='tight')
    exit(0)

    # measuring change over time
    x = []
    dist_changes = defaultdict(lambda: [])
    prev_cents = None
    for epoch, centroids in enumerate(centroids_ot):
        if prev_cents is None:
            prev_cents = centroids
            continue

        class_centroid_changes = np.linalg.norm(centroids - prev_cents, axis=1)
        for k, delta in enumerate(class_centroid_changes):
            dist_changes[f'{k+1}'].append(delta)
        dist_changes['mean'].append(np.mean(class_centroid_changes))

        prev_cents = centroids
        x.append(epoch)


    plt.figure(figsize=FIG_SIZE)
    plt.plot(x, dist_changes['mean'], 'g-', linewidth=2, label='Mean', zorder=100)

    get_color = lambda alpha: (0, 0, 1, alpha)
    colors = list(map(get_color, np.linspace(0.2, 0.8, num_classes)))
    i = 0
    for key, y in dist_changes.items():
        if not key == 'mean':
            plt.plot(x, y, '--', color=colors[i], label=f'Class {key}')
            i += 1
    plt.xlabel('Epoch')
    plt.ylabel('Velocity of Centroid')
    plt.title('Centroid change over time')
    plt.legend()
    plt.savefig('synthetic_results/final_tests3/centroid_change.png', bbox_inches='tight')

# plotting of basic learning trends
def analyze_results(fname, title, outname):
    exp = Experimenter.load_results(fname)
    x = np.arange(0, 100)

    ### make CCE loss charts
    train_cce = np.array(exp.get_results(CCE_LOSS + TRAIN, astype=float))
    val_cce = np.array(exp.get_results(CCE_LOSS + VAL, astype=float))

    plt.figure(figsize=FIG_SIZE)
    plt.plot(x, train_cce, 'r-', label='Train set')
    plt.plot(x, val_cce, 'b-', label='Val set')
    plt.xlabel('Epoch')
    plt.ylabel('CCE Loss')
    suffix = ' - CCE Loss over Time'
    plt.title(title + suffix)
    plt.legend()
    plt.savefig(outname + '_cce', bbox_inches='tight')
    plt.clf()

    ### make F1-acc changing charts
    train_f1 = np.array(exp.get_results(F1_ACC + TRAIN, astype=float))
    val_f1 = np.array(exp.get_results(F1_ACC + VAL, astype=float))

    plt.figure(figsize=FIG_SIZE)
    plt.plot(x, train_f1, 'r-', label='Train set')
    plt.plot(x, val_f1, 'b-', label='Val set')
    plt.xlabel('Epoch')
    plt.ylabel('Weighted F1 Accuracy')
    plt.ylim(0, 1.1)
    suffix = ' - F1 Accuracy over Time'
    plt.title(title + suffix)
    plt.legend()
    plt.savefig(outname + '_f1acc', bbox_inches='tight')
    plt.clf()

    ### make CCE loss changing charts
    train_sc_att = np.array(exp.get_results(ATT_SC_LOSS + TRAIN, astype=float))
    val_sc_att = np.array(exp.get_results(ATT_SC_LOSS + VAL, astype=float))
    train_sc_rep = np.array(exp.get_results(REP_SC_LOSS + TRAIN, astype=lambda x: float(x)*-1))
    val_sc_rep = np.array(exp.get_results(REP_SC_LOSS + VAL, astype=lambda x: float(x)*-1))
    train_cc = np.array(exp.get_results(REP_CC_LOSS + TRAIN, astype=lambda x: float(x)*-1))
    val_cc = np.array(exp.get_results(REP_CC_LOSS + VAL, astype=lambda x: float(x)*-1))

    plt.figure(figsize=FIG_SIZE)
    plt.plot(x, train_sc_att, 'r-', label='Train cosd(s,C)')
    plt.plot(x, train_sc_rep, 'r--', label='Train cosd(s,~C)')
    plt.plot(x, train_cc, 'r:', label='Train cosd(C,~C)')
    plt.plot(x, val_sc_att, 'b-', label='Val cosd(s,C)')
    plt.plot(x, val_sc_rep, 'b--', label='Val cosd(s,~C)')
    plt.plot(x, val_cc, 'b:', label='Val cosd(C,~C)')

    plt.xlabel('Epoch')
    plt.ylabel('Mean Cosine Distance')
    suffix = ' - Cosine Distance Loss over Time'
    plt.title(title + suffix)
    plt.legend()
    plt.savefig(outname + '_Lcore', bbox_inches='tight')
    plt.clf()

    ### make wnorm and gnorm charts
    gnorm = np.array(exp.get_results(GNORM, astype=float))
    plt.figure(figsize=FIG_SIZE)
    plt.plot(x, gnorm, 'g-', label='Norm of Gradient')
    plt.xlabel('Epoch')
    plt.ylabel('Norm')
    suffix = ' - Gradient Norm over Time'
    plt.title(title + suffix)
    plt.legend()
    plt.savefig(outname + '_gnorm', bbox_inches='tight')
    plt.clf()

    wnorm = np.array(exp.get_results(WNORM, astype=float))
    plt.figure(figsize=FIG_SIZE)
    plt.plot(x, wnorm, '-', color='xkcd:orange', label='Norm of Weights')
    plt.xlabel('Epoch')
    plt.ylabel('Norm')
    suffix = ' - Weight Norm over Time'
    plt.title(title + suffix)
    plt.legend()
    plt.savefig(outname + '_wnorm', bbox_inches='tight')
    plt.clf()

    ### make activation norm changing charts
    train_anorm = np.array(exp.get_results(ANORM + TRAIN, astype=float))
    val_anorm = np.array(exp.get_results(ANORM + VAL, astype=float))

    plt.figure(figsize=FIG_SIZE)
    plt.plot(x, train_anorm, 'r-', label='Train set')
    plt.plot(x, val_anorm, 'b-', label='Val set')
    plt.xlabel('Epoch')
    plt.ylabel('Norm of Last Hidden Layer Activation')
    suffix = ' - Mean Norm of Latent Representations over Time'
    plt.title(title + suffix)
    plt.legend()
    plt.savefig(outname + '_hnorm', bbox_inches='tight')
    plt.clf()

if __name__ == '__main__':
    path = 'synthetic_results/final_tests2/'
    analyze_results(path + 'best_corenet_tracking_results.tsv', 'COREL-Net Results', path + 'corenet')
    # analyze_results(path + 'best_ffnet_tracking_results.tsv', 'FF Net Results', path + 'ffnet')
    exit(0)
    cent_path = 'synthetic_results/final_tests3/best_corenet_centroids.npy'
    analyze_centroids(cent_path)
