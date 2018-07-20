"""
project entrance
"""
import codecs
import os
import ujson as json
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from models.bidaf import BIDAF
import configs
from configs import config
from configs import device
from evaluate import evaluate
from evaluate import idx2tokens
from models.qanet import QANet
from models.ensemble import OutputEnsemble


class SquadDataset(Dataset):

    def __init__(self, npz_file):
        self.data = np.load(npz_file)
        self.context_idxes = torch.from_numpy(self.data['context_idxes']).long()
        self.context_char_idxes = torch.from_numpy(self.data['context_char_idxes']).long()
        self.question_idxes = torch.from_numpy(self.data['question_idxes']).long()
        self.question_char_idxes = torch.from_numpy(self.data['question_char_idxes']).long()
        self.ids = torch.from_numpy(self.data['ids']).long()
        self.y1s = torch.from_numpy(self.data['y1s']).long()
        self.y2s = torch.from_numpy(self.data['y2s']).long()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.context_idxes[idx], self.context_char_idxes[idx], self.question_idxes[idx], \
               self.question_char_idxes[idx], self.ids[idx], self.y1s[idx], self.y2s[idx]


def train(model, optimizer, data, _config):
    """
    train process for each batch
    :param model:
    :param optimizer:
    :param data:
    :return: mini-batch loss
    """
    # optimizer.zero_grad()
    model.zero_grad()
    context_word_idxes, context_char_idxes, question_word_idxes, question_char_idxes, ids, y1s, y2s = data
    context_word_idxes, context_char_idxes, question_word_idxes, question_char_idxes = context_word_idxes.to(
        device), context_char_idxes.to(device), question_word_idxes.to(device), question_char_idxes.to(device)
    y1s, y2s = y1s.to(device), y2s.to(device)
    start, end = model(context_word_idxes, context_char_idxes, question_word_idxes, question_char_idxes)
    # criterion = nn.NLLLoss()
    start_loss = F.nll_loss(start, y1s, size_average=True)
    end_loss = F.nll_loss(end, y2s, size_average=True)
    loss = (start_loss + end_loss) / 2
    # loss.cuda()
    loss.backward()

    optimizer.step()
    torch.nn.utils.clip_grad_norm_(model.parameters(), _config['grad_clip'])
    # print('loss here', loss)

    return loss


def test(model, data, eval_file, answer_dict):
    with torch.no_grad():
        context_word_idxes, context_char_idxes, question_word_idxes, question_char_idxes, ids, y1s, y2s = data
        context_word_idxes, context_char_idxes, question_word_idxes, question_char_idxes = context_word_idxes.to(
            device), context_char_idxes.to(device), question_word_idxes.to(device), question_char_idxes.to(device)
        y1s, y2s = y1s.to(device), y2s.to(device)
        start, end = model(context_word_idxes, context_char_idxes, question_word_idxes, question_char_idxes)
        # criterion = nn.NLLLoss()
        start_loss = F.nll_loss(start, y1s, size_average=True)
        end_loss = F.nll_loss(end, y2s, size_average=True)
        loss = (start_loss + end_loss) / 2
        # loss.cuda()
        start_idxes, end_idxes = QANet.generate_answer_idxes(start, end)
        answers = idx2tokens(eval_file, ids, start_idxes, end_idxes)
        answer_dict.update(answers)
        return loss


def train_qanet():
    _config = config['QANet']
    with codecs.open(configs.word_emb_file, 'r', 'utf-8') as f:
        word_mat = np.array(json.load(f), dtype='float32')
    with codecs.open(configs.char_emb_file, 'r', 'utf-8') as f:
        char_mat = np.array(json.load(f), dtype='float32')
    # with codecs.open(configs.train_eval_file, 'r', 'utf-8') as f:
    #     train_eval_file = json.load(f)
    with codecs.open(configs.dev_eval_file, 'r', 'utf-8') as f:
        dev_eval_file = json.load(f)

    train_dataset = SquadDataset(configs.train_record_file)
    dev_dataset = SquadDataset(configs.dev_record_file)
    train_loader = DataLoader(dataset=train_dataset, batch_size=_config['batch_size'], shuffle=True,
                              collate_fn=collate_fn)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=_config['batch_size'], shuffle=True, collate_fn=collate_fn)

    print('Start Building Model')
    model = QANet(word_mat, char_mat).to(device)
    print(model)

    params = list(filter(lambda param: param.requires_grad, model.parameters()))
    optimizer = optim.Adam(params=params, lr=_config['learning_rate'], weight_decay=3e-7)
    # optimizer = optim.SGD(params=params, lr=_config['learning_rate'], momentum=0.9, weight_decay=3e-7)
    best_f1 = 0
    best_em = 0
    patience = 0  # early stop patience
    epoch_index = 0
    for epoch in range(_config['num_epoch']):

        answer_dict = dict()

        for step, data in enumerate(train_loader):
            loss = train(model, optimizer, data, _config)
            # dev_losses = []
            # for _step, _data in enumerate(dev_loader):
            #     print('test dev step', _step)
            #     loss = test(model, _data, dev_eval_file, answer_dict)
            #     dev_losses.append(loss.item())
            print('{} step,training loss is {} ...'.format(step, loss))
            # print('{} step dev loss is {}...'.format(step, np.mean(dev_losses)))
        # test the dev file
        dev_losses = []
        for step, data in enumerate(dev_loader):
            loss = test(model, data, dev_eval_file, answer_dict)
            dev_losses.append(loss.item())

        print('{} epoch dev loss is {}...'.format(epoch_index, np.mean(dev_losses)))
        dev_losses.clear()
        dev_file = codecs.open(configs.dev_file, 'r', 'utf-8')
        dev_file = json.load(dev_file)
        metrics = evaluate(dev_file['data'], answer_dict)
        f1, em = metrics['f1'], metrics['exact_match']

        if f1 < best_f1 and em < best_em:
            patience += 1
            if patience > _config['early_stopping']:
                break
        else:
            best_em = max(em, best_em)
            best_f1 = max(f1, best_f1)
            patience = 0

        # save answers
        f = codecs.open(configs.answer_file, 'w', 'utf-8')
        json.dump(answer_dict, f)
        epoch_index += 1
    print('best dev f1 is {},best dev em is {}..'.format(best_f1, best_em))
    home = os.path.expanduser('.')
    save_file = os.path.join(home, 'output', 'qanet.mod')
    torch.save(model, save_file)


def train_bidaf():
    _config = config['BIDAF']
    with codecs.open(configs.word_emb_file, 'r', 'utf-8') as f:
        word_mat = np.array(json.load(f), dtype='float32')
    with codecs.open(configs.char_emb_file, 'r', 'utf-8') as f:
        char_mat = np.array(json.load(f), dtype='float32')
    # with codecs.open(configs.train_eval_file, 'r', 'utf-8') as f:
    #     train_eval_file = json.load(f)
    with codecs.open(configs.dev_eval_file, 'r', 'utf-8') as f:
        dev_eval_file = json.load(f)

    train_dataset = SquadDataset(configs.train_record_file)
    dev_dataset = SquadDataset(configs.dev_record_file)
    train_loader = DataLoader(dataset=train_dataset, batch_size=_config['batch_size'], shuffle=True,
                              collate_fn=collate_fn)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=_config['batch_size'], shuffle=True, collate_fn=collate_fn)

    print('Start Building Model')
    model = BIDAF(word_mat, char_mat).to(device)

    params = list(filter(lambda param: param.requires_grad, model.parameters()))
    optimizer = optim.Adam(params=params, lr=_config['learning_rate'], weight_decay=3e-7)

    best_f1 = 0
    best_em = 0
    patience = 0  # early stop patience

    epoch_index = 0
    for epoch in range(_config['num_epoch']):
        dev_losses = []
        for step, data in enumerate(train_loader):
            loss = train(model, optimizer, data, _config)
            print('{} step,training loss is {} ...'.format(step, loss))
        answer_dict = dict()
        # test the dev file
        for step, data in enumerate(dev_loader):
            loss = test(model, data, dev_eval_file, answer_dict)
            dev_losses.append(loss)
            print('predicting {} step dev data...'.format(step))
        print('{} epoch dev loss is {}...'.format(epoch_index, np.mean(dev_losses)))
        dev_losses.clear()
        dev_file = codecs.open(configs.dev_file, 'r', 'utf-8')
        dev_file = json.load(dev_file)
        metrics = evaluate(dev_file['data'], answer_dict)
        f1, em = metrics['f1'], metrics['exact_match']

        if f1 < best_f1 and em < best_em:
            patience += 1
            if patience > _config['early_stopping']:
                break
        else:
            best_em = max(em, best_em)
            best_f1 = max(f1, best_f1)
            patience = 0

        # save answers
        f = codecs.open(configs.answer_file, 'w', 'utf-8')
        json.dump(answer_dict, f)
        epoch_index += 1
    print('best dev f1 is {},best dev em is {}..'.format(best_f1, best_em))
    home = os.path.expanduser('.')
    save_file = os.path.join(home, 'output', 'bidaf.mod')
    torch.save(model, save_file)


def train_ensemble():
    _config = config['ensemble']
    with codecs.open(configs.word_emb_file, 'r', 'utf-8') as f:
        word_mat = np.array(json.load(f), dtype='float32')
    with codecs.open(configs.char_emb_file, 'r', 'utf-8') as f:
        char_mat = np.array(json.load(f), dtype='float32')
    # with codecs.open(configs.train_eval_file, 'r', 'utf-8') as f:
    #     train_eval_file = json.load(f)
    with codecs.open(configs.dev_eval_file, 'r', 'utf-8') as f:
        dev_eval_file = json.load(f)

    train_dataset = SquadDataset(configs.train_record_file)
    dev_dataset = SquadDataset(configs.dev_record_file)
    train_loader = DataLoader(dataset=train_dataset, batch_size=_config['batch_size'], shuffle=True,
                              collate_fn=collate_fn)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=_config['batch_size'], shuffle=True, collate_fn=collate_fn)

    print('Start Building Model')
    model = OutputEnsemble(word_mat, char_mat).to(device)

    params = list(filter(lambda param: param.requires_grad, model.parameters()))
    optimizer = optim.Adam(params=params, lr=_config['learning_rate'], weight_decay=3e-7)

    best_f1 = 0
    best_em = 0
    patience = 0  # early stop patience

    epoch_index = 0
    for epoch in range(_config['num_epoch']):
        dev_losses = []
        for step, data in enumerate(train_loader):
            loss = train(model, optimizer, data, _config)
            print('{} step,training loss is {} ...'.format(step, loss))
        answer_dict = dict()
        # test the dev file
        for step, data in enumerate(dev_loader):
            loss = test(model, data, dev_eval_file, answer_dict)
            dev_losses.append(loss)
            print('predicting {} step dev data...'.format(step))
        print('{} epoch dev loss is {}...'.format(epoch_index, np.mean(dev_losses)))
        dev_losses.clear()
        dev_file = codecs.open(configs.dev_file, 'r', 'utf-8')
        dev_file = json.load(dev_file)
        metrics = evaluate(dev_file['data'], answer_dict)
        f1, em = metrics['f1'], metrics['exact_match']

        if f1 < best_f1 and em < best_em:
            patience += 1
            if patience > _config['early_stopping']:
                break
        else:
            best_em = max(em, best_em)
            best_f1 = max(f1, best_f1)
            patience = 0

        # save answers
        f = codecs.open(configs.answer_file, 'w', 'utf-8')
        json.dump(answer_dict, f)
        epoch_index += 1
    print('best dev f1 is {},best dev em is {}..'.format(best_f1, best_em))
    home = os.path.expanduser('.')
    save_file = os.path.join(home, 'output', 'qanet_bidaf_ensemble.mod')
    torch.save(model, save_file)


def collate_fn(batch):
    """load batch_size data"""
    context_idxes, context_char_idxes, question_idxes, question_char_idxes, ids, y1s, y2s = zip(*batch)
    context_idxes = list(map(lambda x: x.numpy(), list(context_idxes)))
    context_idxes = torch.Tensor(context_idxes).long()
    context_char_idxes = list(map(lambda x: x.numpy(), context_char_idxes))
    context_char_idxes = torch.Tensor(context_char_idxes).long()
    question_idxes = list(map(lambda x: x.numpy(), question_idxes))
    question_idxes = torch.Tensor(question_idxes).long()
    question_char_idxes = list(map(lambda x: x.numpy(), question_char_idxes))
    question_char_idxes = torch.Tensor(question_char_idxes).long()
    ids = list(map(lambda x: x.item(), ids))
    ids = torch.Tensor(ids).long()
    # start_answers = []
    # end_answers = []
    # for value in y1s:
    #     start = np.zeros(config['paragraph_limit'])
    #     start[value.item()] = 1
    #     start_answers.append(torch.Tensor(start))
    # for value in y2s:
    #     end = np.zeros(config['paragraph_limit'])
    #     end[value.item()] = 1
    #     end_answers.append(torch.Tensor(end))
    start_answers = list(map(lambda x: x.item(), y1s))
    start_answers = torch.Tensor(start_answers).long()
    end_answers = list(map(lambda x: x.item(), y2s))
    end_answers = torch.Tensor(end_answers).long()
    return context_idxes, context_char_idxes, question_idxes, question_char_idxes, ids, start_answers, end_answers


if __name__ == '__main__':
    if config['model'] == 'QANet':
        train_qanet()
    elif config['model'] == 'BIDAF':
        train_bidaf()
    elif config['model'] == 'ensemble':
        train_ensemble()
