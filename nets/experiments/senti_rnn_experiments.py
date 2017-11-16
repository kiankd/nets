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
    batch_size = 100

    dataset = SentimentDataset()
    dataset.default_load(dataset_name='rt')
    glove_embeddings = dataset.get_glove_data(dim=emb_size)
    glove_vocab = list(glove_embeddings.keys())

    print('Initialization test...')
    nets_model = SentiBiRNN(
        'Sentiment bi-directional RNN - 1',
        vocabulary=glove_vocab,
        embedding_dim=emb_size,
        encoder_hidden_dim=32,
        dense_hidden_dim=64,
        labels=[0, 1],
        dropout=0,
        clip_norm=1,
        weight_decay=0,
        learning_rate=1e-3,
        batch_size=batch_size,
        emb_dict=glove_embeddings,
    )

    print('GPU STATUS:')
    print(torch.cuda.is_available())
    nets_model.model.cuda()

    print('Model graph:')
    print(nets_model.model)

    print('Testing minibatch iteration...')
    i = -1
    for samples, labels, epoch in dataset.iterate_train_minibatches(batch_size=batch_size, epochs=100, shuffle=True):
        if epoch > i:
            train_loss, train_accs, t_out_dist, t_mean_outs = \
                nets_model.predict_with_stats(dataset.get_train_x(), dataset.get_train_y())

            val_loss, val_accs, v_out_dist, v_mean_outs = \
                nets_model.predict_with_stats(dataset.get_val_x(), dataset.get_val_y())

            print('\nEpoch {} results:'.format(epoch))
            print('  Norm W    : {:>5}'.format(nets_model.get_w_norm()))
            print('  Norm Grad : {:>5}'.format(nets_model.get_grad_norm()))
            print('  Train loss: {:>5}'.format(train_loss.data[0]))
            print('  Val loss  : {:>5}'.format(val_loss.data[0]))

            print('  Train Accuracies:')
            for name, acc in train_accs:
                print('    {:<10}:  {:>5}'.format(name, acc))
            print('    Out dist  :  {}'.format(t_out_dist))
            print('    Mean conf :  {}'.format(t_mean_outs))

            print('  Val Accuracies:')
            for name, acc in val_accs:
                print('    {:<10}:  {:>5}'.format(name, acc))
            print('    Out dist  :  {}'.format(v_out_dist))
            print('    Mean conf :  {}'.format(v_mean_outs))


            i += 1
        nets_model.train(samples, labels, batch_size=batch_size)


if __name__ == '__main__':
    run_model_tests()
