from nets.dataset.classification.sentiment_dataset import SentimentDataset
from sklearn.model_selection import train_test_split
import csv

def test_senti_data_extraction(wemb_dim, get_gloves=False):
    d = SentimentDataset()
    d.default_load(dataset_name='rt')
    d.get_vocabulary()
    if get_gloves:
        d.init_glove_for_vocab(dim=wemb_dim) # call to get new gloves
    d.get_glove_data(dim=wemb_dim)
    sample_emb = d.embeddings['the']
    print('Sanity check: embedding dimension = {}'.format(len(sample_emb)))
    print('Number of embeddings: {}'.format(len(d.embeddings)))
    print('Number of unique words: {}'.format(len(d.vocab)))
    print('Words w/o embeddings: {}'.format(len(d.vocab) - len(d.embeddings)))
    print('Example embedding for \'the\': {}'.format(sample_emb))
    print('')

    print('\n\n------------------------------------------\n\n')
    d.get_data_statistics()

    # manual testing
    # for f_end in ['train', 'val', 'test']:
    #     x, y = d._load_data_from_file('{}{}.tsv'.format('rt_', f_end))


# This converts original rt-polarity files to train, val, and test sets.
def convert_rt_polarity_to_tsv_and_split():
    def load(fname):
        with open(fname, 'r') as f:
            data = [line.strip('\n').replace('\t', '') for line in f.readlines()]
        return data

    d = SentimentDataset()
    pos = load(d.get_path() + 'orig/rt-polarity.pos')
    neg = load(d.get_path() + 'orig/rt-polarity.neg')
    y =  [1] * len(pos) + [0] * len(neg)

    # split data, test set is 20%, val set is 10% of the 80% of train, so it is 8% of entire dataset.
    x_train, x_test, y_train, y_test = train_test_split(pos + neg, y, test_size=0.2, random_state=1917)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=0) # also make val set

    print('Writing tsvs...')
    for x_data, y_data, name in [(x_train, y_train, 'train'),
                                 (x_val, y_val, 'val'),
                                 (x_test, y_test, 'test')]:
        with open(d.get_path() + 'rt_{}.tsv'.format(name), 'w') as tsvfile:
            field_names = ['sample', 'label']
            writer = csv.DictWriter(tsvfile, fieldnames=field_names, delimiter='\t')
            writer.writeheader()
            for sample, label in zip(x_data, y_data):
                writer.writerow({field_names[0]: sample, field_names[1]: label})

    print('Reloading...')
    d.load_all_data('rt_train.tsv', 'rt_val.tsv', 'rt_test.tsv', get_path=False)

    print('Serializing...')
    d.serialize_all_data('rt_train', 'rt_val', 'rt_test')

if __name__ == '__main__':
    # convert_rt_polarity_to_tsv_and_split() # this rewrites all the data!
    test_senti_data_extraction(wemb_dim=200, get_gloves=False)

