"""
preprocess file for squad data-set
modified from https://github.com/HKUST-KnowComp/R-Net/blob/master/prepro.py
"""
import codecs
import ujson as json
from collections import Counter

import numpy as np
import spacy
from tqdm import tqdm

import configs
from configs import config

nlp = spacy.blank('en')


def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_index(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def process_file(filename, data_type, word_counter, char_counter):
    """

    :param filename:
    :param data_type: data kind train or dev
    :param word_counter: word counter
    :param char_counter: char counter
    :return:
    """
    print('processing {} data...'.format(data_type))
    train_data = []
    eval_data = {}
    total = 0  # total data count
    with codecs.open(filename, 'r',encoding='utf-8') as f:
        source = json.load(f)
        for article in tqdm(source['data']):
            for paragraph in article['paragraphs']:
                context = paragraph["context"].replace("''", '"').replace("``", '"')
                context_tokens = word_tokenize(context)
                context_chars = [list(token) for token in context_tokens]
                spans = convert_index(context, context_tokens)
                for token in context_tokens:
                    word_counter[token] += len(paragraph['qas'])
                    for char in token:
                        char_counter[char] += len(paragraph['qas'])
                for qa in paragraph['qas']:

                    ques = qa['question'].replace("''", '" ').replace("``", '" ')
                    print(ques)
                    ques_tokens = word_tokenize(ques)
                    ques_chars = [list(token) for token in ques_tokens]
                    for token in ques_tokens:
                        word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1
                    y1s, y2s = [], []
                    answer_texts = []
                    for answer in qa['answers']:
                        answer_text = answer['text']
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_span = []
                        for idx, span in enumerate(spans):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                        y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1)
                        y2s.append(y2)
                    data = {'context_tokens': context_tokens, 'context_chars': context_chars,
                            'question_tokens': ques_tokens, 'question_chars': ques_chars, 'id': total,
                            'y1s': y1s, 'y2s': y2s}

                    train_data.append(data)
                    eval_data[total] = {'context': context, 'spans': spans, 'answers': answer_texts, 'uuid': qa['id']}
                    total += 1
        print('total {} questions'.format(len(train_data)))
    return train_data, eval_data


def get_embedding(counter, emb_type, limit=-1, emb_file=None, emb_size=None):
    """
    get word or char embedding from file
    :param counter: word or char counter
    :param emb_type: word or char
    :param limit: minimum word/char count in embedding
    :param emb_file: embedding file
    :param emb_size: embedding size
    :return: embedding matrix and token2index dict
    """
    print('Creating {} embedding...'.format(emb_type))
    emb_index = dict()
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert emb_size is not None
        with codecs.open(emb_file, encoding='utf-8') as f:
            for line in tqdm(f):
                array = line.split()
                word = ''.join(array[0:-emb_size])
                vector = list(map(float, array[-emb_size:]))
                if word in filtered_elements and counter[word] > limit:
                    emb_index[word] = vector
            print(
                '{}/{} tokens have corresponding {} embedding'.format(len(emb_index), len(filtered_elements), emb_type))
    else:
        assert emb_size is not None
        for token in filtered_elements:
            emb_index[token] = [np.random.normal(scale=0.1) for _ in range(emb_size)]
        print('{} tokens have corresponding embedding'.format(len(filtered_elements)))

    NULL = 'NULL'
    OOV = 'OOV'
    token2idx = {token: idx for idx, token in enumerate(emb_index.keys(), 2)}
    token2idx[NULL] = 0
    token2idx[OOV] = 1
    emb_index[NULL] = [0. for _ in range(emb_size)]
    emb_index[OOV] = [0. for _ in range(emb_size)]
    idx2emb = {idx: emb_index[token] for token, idx in token2idx.items()}
    emb_mat = [idx2emb[idx] for idx in range(len(idx2emb))]
    return emb_mat, token2idx


def build_features(config, data_set, data_type, out_file, word2idx, char2idx):
    """
    turn character into indexes
    :param config: config file
    :param data_set: date-set
    :param data_type: train or dev data-set
    :param out_file: store file
    :param word2idx: word2index dict
    :param char2idx: char2index dict
    :return:
    """
    print('Creating {} features...'.format(data_type))
    paragraph_limit = config['paragraph_limit']  # maximum paragraph word count
    question_limit = config['question_limit']  # maximum question word count
    # answer_limit = config['answer_limit']  # maximum answer word count
    char_limit = config['char_limit']  # maximum char count in each word

    total = 0
    filter_total = 0

    context_idxes = []  # store paragraphs shape=(n_sample,paragraph_limit)
    context_char_idxes = []  # store paragraphs in char shape=(n_sample,paragraph_limit,char_limit)
    question_idxes = []  # store questions shape =(n_sample,question_limit)
    question_char_idxes = []  # store questions shape=(n_sample,question_limit,char_limit)
    y1s = []  # start answer index
    y2s = []  # end answer index
    ids = []  # index id

    def _filter(data):
        return len(data['context_tokens']) > paragraph_limit or len(data['question_tokens']) > question_limit
               # (data['y2s'][0] - data['y1s'][0]) > answer_limit

    def _get_word_idx(word):
        for each in (word, word.lower(), word.upper(), word.capitalize()):
            if each in word2idx:
                return word2idx[each]
        return 1

    def _get_char_idx(char):
        if char in char2idx:
            return char2idx[char]
        return 1

    for n, data in enumerate(data_set):
        print('processing {} data...'.format(n))
        total += 1

        if _filter(data):
            continue
        filter_total += 1

        context_idx = np.zeros([paragraph_limit], dtype='int32')
        context_char_idx = np.zeros([paragraph_limit, char_limit], dtype='int32')
        question_idx = np.zeros([question_limit], dtype='int32')
        question_char_idx = np.zeros([question_limit, char_limit], dtype='int32')

        for idx, token in enumerate(data['context_tokens']):
            context_idx[idx] = _get_word_idx(token)
        context_idxes.append(context_idx)

        for idx, token in enumerate(data['question_tokens']):
            question_idx[idx] = _get_word_idx(token)
        question_idxes.append(question_idx)

        for i, token in enumerate(data['context_chars']):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idx[i, j] = _get_char_idx(char)
        context_char_idxes.append(context_char_idx)

        for i, token in enumerate(data['question_chars']):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                question_char_idx[i, j] = _get_char_idx(char)
        question_char_idxes.append(question_char_idx)

        start, end = data['y1s'][0], data['y2s'][0]
        ids.append(data['id'])
        y1s.append(start)
        y2s.append(end)
    np.savez(out_file, context_idxes=np.array(context_idxes), context_char_idxes=np.array(context_char_idxes),
             question_idxes=np.array(question_idxes), question_char_idxes=np.array(question_char_idxes),
             y1s=np.array(y1s), y2s=np.array(y2s), ids=np.array(ids))
    print('Created {} / {} features in total'.format(filter_total, total))


def save(filename, obj, message=None):
    with codecs.open(filename, 'w','utf-8') as f:
        if message is not None:
            print('Saving {}...'.format(message))
        json.dump(obj, f)


def preproc(config):
    word_counter, char_counter = Counter(), Counter()

    train_file = configs.train_file
    dev_file = configs.dev_file
    train_data, train_eval = process_file(train_file, 'train', word_counter, char_counter)
    dev_data, dev_eval = process_file(dev_file, 'dev', word_counter, char_counter)

    word_limit = config['word_limit']
    if config['word_pretrained']:
        if config['word_embedding'] == 'glove':
            word_emb_mat, word2idx = get_embedding(word_counter, 'word', word_limit, configs.glove_emb_file,
                                                   config['glove_emb_size'])
        elif config['word_embedding'] == 'fasttext':
            word_emb_mat, word2idx = get_embedding(word_counter, 'word', word_limit, configs.fasttext_emb_file,
                                                   config['fasttext_emb_size'])
        else:
            print('No {} word embedding,use no pretrained embedding'.format(config['word_embedding']))
            word_emb_mat, word2idx = get_embedding(word_counter, 'word', word_limit, None, emb_size=300)
    else:
        word_emb_mat, word2idx = get_embedding(word_counter, 'word', word_limit, None, emb_size=300)

    # char_limit = config['char_limit']
    if config['char_pretrained']:
        if config['char_embedding'] == 'glove':
            char_emb_mat, char2idx = get_embedding(char_counter, 'char', -1, configs.glove_char_emb_file,
                                                   config['glove_char_emb_size'])
        else:
            print('No {} char embedding,use no pretrained embedding'.format(config['char_embedding']))
            char_emb_mat, char2idx = get_embedding(char_counter, 'char', -1, None, emb_size=config['char_emb_size'])
    else:
        char_emb_mat, char2idx = get_embedding(char_counter, 'char', -1, None, emb_size=config['char_emb_size'])

    # build features
    build_features(config, train_data, 'train', configs.train_record_file, word2idx, char2idx)
    build_features(config, dev_data, 'dev', configs.dev_record_file, word2idx, char2idx)

    save(configs.word_emb_file, word_emb_mat, message="word embedding")
    save(configs.char_emb_file, char_emb_mat, message="char embedding")
    save(configs.train_eval_file, train_eval, message="train eval")
    save(configs.dev_eval_file, dev_eval, message="dev eval")
    # save(config.test_eval_file, test_eval, message="test eval")
    save(configs.word2idx_file, word2idx, message="word dictionary")
    save(configs.char2idx_file, char2idx, message="char dictionary")


if __name__ == '__main__':
    preproc(config)
