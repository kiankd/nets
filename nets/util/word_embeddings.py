# for loading and extracting word embeddings
from copy import deepcopy
import numpy as np

GLOVE_DATA_PATH = '/mnt/data/glove/'

def get_glove_fname(path, dim):
    return '{}vocab_embs{}'.format(path, dim)

def raw_load_and_extract_glove(vocab_set, dimensions, extract_dir, twitter=True):
    vocab = set(vocab_set)
    embeddings = {}

    print('Getting glove data {} dimensions, {} size of vocab... '
          'May take some time...'.format(dimensions, len(vocab)))

    glove_path = GLOVE_DATA_PATH
    if twitter:
        glove_path = GLOVE_DATA_PATH + 'glove.twitter.27B.{}d.txt'.format(dimensions)

    with open(glove_path, 'r') as f:
        for line in f.readlines():
            data = line.split()
            word = data[0]
            if word in vocab:
                embeddings[word] = np.array(map(float, data[1:]))
                vocab.remove(word)
    print('There are {} words without glove embeddings.'.format(len(vocab)))

    # save the embeddings with numpy for quick access
    if serialize_embeddings:
        np.save(get_glove_fname(extract_dir, dimensions), np.array([embeddings]))

    return embeddings
