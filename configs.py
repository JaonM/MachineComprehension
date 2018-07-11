"""
NN config
"""
import os

import torch

home = os.path.expanduser('.')

train_file = os.path.join(home, 'data', 'squad1.0', 'train-v1.1.json')
dev_file = os.path.join(home, 'data', 'squad1.0', 'dev-v1.1.json')

train_record_file = os.path.join(home, 'data', 'train.npz')
dev_record_file = os.path.join(home, 'data', 'dev.npz')

glove_emb_file = os.path.join(home, 'data', 'embedding', 'glove.840B.300d.txt')
fasttext_emb_file = os.path.join(home, 'data', 'embedding', 'wiki.en.vec')

glove_char_emb_file = os.path.join(home, 'data', '')

word_emb_file = os.path.join(home, 'data', 'word_emb.json')
char_emb_file = os.path.join(home, 'data', 'char_emb.json')

train_eval_file = os.path.join(home, 'data', 'train_eval.json')
dev_eval_file = os.path.join(home, 'data', 'dev_eval.json')

word2idx_file = os.path.join(home, 'data', 'word2idx.json')
char2idx_file = os.path.join(home, 'data', 'char2idx.json')

answer_file = os.path.join(home, 'data', 'answer.json')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = {
    'model': 'QANet',

    'QANet': {
        'word_pretrained': True,
        'char_pretrained': False,
        'word_embedding': 'glove',  # glove or fasttext
        'glove_emb_size': 300,
        'fasttext_emb_size': 300,
        'char_embedding': 'glove',
        'glove_char_emb_size': 200,
        'char_emb_size': 200,
        'word_emb_size': 300,

        'paragraph_limit': 300,
        'question_limit': 40,
        'answer_limit': 30,
        'char_limit': 15,
        'word_limit': -1,  # minimum required word

        'num_epoch': 30,
        'batch_size': 14,
        'learning_rate': 0.001,
        'early_stopping': 5,

        'char_dropout_rate': 0.05,
        'dropout_rate': 0.1,
        'connector_dim': 128,
    },

    'BIDAF': {
        'word_pretrained': True,
        'char_pretrained': False,
        'word_embedding': 'glove',  # glove or fasttext
        'glove_emb_size': 300,
        'fasttext_emb_size': 300,
        'char_embedding': 'glove',
        'glove_char_emb_size': 200,
        'char_emb_size': 200,
        'word_emb_size': 300,

        'paragraph_limit': 300,
        'question_limit': 40,
        'answer_limit': 30,
        'char_limit': 16,
        'word_limit': -1,  # minimum required word

        'num_epoch': 30,
        'batch_size': 8,
        'learning_rate': 0.001,
        'early_stopping': 5,

        'char_dropout_rate': 0.05,
        'dropout_rate': 0.1,
    },

    'ensemble': {

    }
}
