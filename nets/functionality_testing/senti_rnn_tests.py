# quick fix
import sys
sys.path.append('/home/ml/kkenyo1/nets')
# TODO: fix this shit.

from nets.dataset.classification.sentiment_dataset import SentimentDataset
from nets.model.senti_birnn import SentiBiRNN
import torch
import numpy as np

def run_model_tests():
    # load data and get embedding
    print('Data loading test...')
    emb_size = 50
    dataset = SentimentDataset()
    dataset.default_load(dataset_name='rt')
    glove_embeddings = dataset.get_glove_data(dim=emb_size)
    glove_vocab = list(glove_embeddings.keys())

    print('Initialization test...')
    nets_model = SentiBiRNN(
        'Sentiment bi-directional RNN - 1',
        vocabulary=glove_vocab,
        embedding_dim=emb_size,
        encoder_hidden_dim=64,
        dense_hidden_dim=32,
        labels=[0, 1],
        dropout=0,
        learning_rate=1e-3,
        emb_dict=glove_embeddings,
    )

    print('GPU STATUS:')
    print(torch.cuda.is_available())
    nets_model.model.cuda()

    print('Model graph:')
    print(nets_model.model)

    print('Testing minibatch iteration...')
    i = -1
    for sample, label, epoch in dataset.iterate_train_minibatches(batchsize=1, epochs=10):
        if epoch > i:
            train_loss = nets_model.predict_loss_and_acc(dataset.get_train_x(), dataset.get_train_y())
            val_loss = nets_model.predict_loss_and_acc(dataset.get_val_x(), dataset.get_val_y())
            print('\nEpoch {} results:'.format(epoch))
            print('\tTrain loss: {}'.format(train_loss))
            print('\tVal loss:   {}'.format(val_loss))
            i += 1

        nets_model.train(sample[0], label[0])

if __name__ == '__main__':
    run_model_tests()
