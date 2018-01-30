

# neural net constants
INPUT_DIM = 'input_dim'
DENSE_DIMS = 'dense_layers'
ACTIVATION = 'activation'
NUM_CLASSES = 'num_classes'
REPRESENTATION_LAYER = 'rep_layer'
BATCH_SIZE = 'bsz'
EPOCHS = 'epochs'

# parameter constants
LOSS_FUN = 'loss'
LEARNING_RATE = 'lr'
CLIP_NORM = 'cn'
WEIGHT_DECAY = 'decay'
CORE = 'core'
LAM1 = 'lam1'
LAM2 = 'lam2'
LAM3 = 'lam3'

# experimentation
MODEL_START_KEY = 'MODEL_NAME'
EPOCH = 'epoch'
WNORM = 'weight_norm'
GNORM = 'gradient_norm'
ANORM = 'activation_norm'
CCE_LOSS = 'cce_loss'
ATT_SC_LOSS = 'attractive_samp_cent_loss'
REP_SC_LOSS = 'repulsive_samp_cent_loss'
REP_CC_LOSS = 'repulisve_cent_cent_loss'
F1_ACC = 'f1_acc'

TRAIN = 'train'
VAL = 'val'

results_with_train_val = (
    ANORM,
    CCE_LOSS,
    ATT_SC_LOSS,
    REP_SC_LOSS,
    REP_CC_LOSS,
    F1_ACC,
)

EXP_TRACKING = [EPOCH, WNORM, GNORM] + [result + dset for result in results_with_train_val for dset in (TRAIN, VAL,) ]
EXP_TRACKING = tuple(EXP_TRACKING)


# mpl
COLORS = ['xkcd:purple', 'xkcd:green', 'xkcd:cyan', 'xkcd:yellow', 'xkcd:grey',
          'xkcd:red', 'xkcd:navy', 'xkcd:gold', 'xkcd:bright pink', 'xkcd:blue']
