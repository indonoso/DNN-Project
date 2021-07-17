import pickle
from torch import Tensor

def get_input_size(kind, config):
    network_input_size = 0
    if config[f'use_{kind}']:
        network_input_size = config[f'{kind}_embedding_size']
    return network_input_size


def get_embeddings(kind, config):
    if kind == 'word' or (kind == 'kg' and config['preprocess'][f'use_{kind}']):
        name = config['data']['processed'][f'{kind}_embeddings_path']
        size = config['preprocess'][f'{kind}_embedding_size']
        return load_embeddings(f'{name}-{size}.pickle')
    elif kind == 'pos' and config['preprocess'][f'use_{kind}']:
        return load_embeddings(config['data']['processed'][f'{kind}_embeddings_path'])
    else:
        return False


def load_embeddings(path):
    with open(path, 'rb') as f:
        embeddings = pickle.load(f)
    return Tensor(embeddings)
