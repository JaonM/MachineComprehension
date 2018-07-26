"""
Bidirectional Attention Flow Machine Comprehension
https://arxiv.org/abs/1611.01603
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from configs import device
import gc
import numpy as np
from configs import config

config = config['BIDAF']


class Highway(nn.Module):
    """
    Highway Network
    Input shape=(batch_size,length,dim)
    Output shape=(batch_size,length,dim)
    """

    def __init__(self, layer_num, dim=config['word_emb_size'] + config['char_emb_size']):
        super(Highway, self).__init__()
        self.layer_num = layer_num
        self.linear = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.layer_num)])
        self.gate = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.layer_num)])

    def forward(self, x):
        for i in range(self.layer_num):
            gate = F.sigmoid(self.gate[i](x))
            nonlinear = F.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        return x


class Embedding(nn.Module):
    """
    Embedding Layer

    Input shape: word_emb=(batch_size,length,word_emb_size) char_emb=(batch_size,length,char_limit,char_emb_size)
    Output shape: shape=(batch_size,length,word_emb_size+char_emb_size)
    """

    def __init__(self):
        super(Embedding, self).__init__()
        self.highway = Highway(2)
        # self.conv = nn.Conv2d(length, config['char_emb_size'], 5)
        # self.max_pool = nn.MaxPool2d(5)
        # self.char_emb = torch.Tensor(np.zeros(shape=(config['batch_size'],length,config['char_emb_size'])))

    def forward(self, word_emb, char_emb):
        # char_emb = self.conv(char_emb)
        # char_emb = self.max_pool(char_emb)
        # print(char_emb.size())
        char_emb, _ = torch.max(char_emb, dim=2)
        x = torch.cat([word_emb, char_emb], dim=2)
        x = self.highway(x)
        return x


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class ContextualEmbedding(nn.Module):
    """
    LSTM contextual embedding

    Input shape=(batch_size,length,word_emb_size+char_emb_size)
    Output shape=(batch_size,length,2*(word_emb_size+char_emb_size))
    """

    def __init__(self, input_dim):
        super(ContextualEmbedding, self).__init__()
        self.lstm = nn.LSTM(input_dim, input_dim)

    def forward(self, x):
        x = x.transpose(0, 1)
        x = F.dropout(x, config['dropout_rate'], training=self.training)

        out, hidden = self.lstm(x)

        x_backward = flip(x, 0)
        x_backward = F.dropout(x_backward, config['dropout_rate'], training=self.training)

        out_back, hidden = self.lstm(x_backward)
        x = torch.cat([out, out_back], dim=2)
        x = x.transpose(0, 1)
        return x


class AttentionFlowLayer(nn.Module):
    """
        context-query attention in paper

        Input shape: Context=(batch_size,context_length,d) Query=(batch_size,query_length,d)
        Output shape: out = (batch_size,context,8*d)
        """

    def __init__(self):
        super(AttentionFlowLayer, self).__init__()
        self.W0 = nn.Parameter(torch.randn(6 * (config['word_emb_size'] + config['char_emb_size'])))

    def forward(self, c, q):
        shape = (c.size(0), c.size(1), q.size(1), c.size(2))
        ct = c.unsqueeze(2).expand(shape)
        qt = q.unsqueeze(1).expand(shape)
        cq = torch.mul(ct, qt)
        S = torch.cat([ct, qt, cq], dim=3)
        S = torch.matmul(S,self.W0)
        S1 = F.softmax(S, dim=1)
        A = torch.bmm(S1, q)
        S2 = F.softmax(S, dim=2)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), c)

        out = torch.cat([c, A, torch.mul(c, A), torch.mul(c, B)], dim=2)
        out = F.dropout(out,config['dropout_rate'],training=self.training)
        del S, S1, S2
        gc.collect()
        return out


class ModelingLayer(nn.Module):
    """
    Bidirectional RNN

    Input shape=(batch_size,context_length,6*(word_emb_size+char_emb_size))
    Output shape=(batch_size,context_length,2*(word_emb_size+char_emb_size))
    """

    def __init__(self):
        super(ModelingLayer, self).__init__()
        self.lstm = nn.LSTM(8 * (config['word_emb_size'] + config['char_emb_size']),
                            (config['word_emb_size'] + config['char_emb_size']),
                            bidirectional=True)

    def forward(self, x):
        x = x.transpose(0, 1)
        x = F.dropout(x, config['dropout_rate'], training=self.training)
        out, hidden = self.lstm(x)

        return out.transpose(0, 1)


class Output(nn.Module):
    """
    Output layer

    Input: G = (batch_size,context_length,8*d) M=(batch_size,context_length,2*d)
    Output: start,end = (batch_size,context_length)
    """

    def __init__(self):
        super(Output, self).__init__()
        self.lstm = nn.LSTM(2 * (config['word_emb_size'] + config['char_emb_size']),
                            2 * (config['word_emb_size'] + config['char_emb_size']))
        self.w1 = nn.Parameter(torch.randn(10 * (config['word_emb_size'] + config['char_emb_size']), 1))
        self.w2 = nn.Parameter(torch.randn(10 * (config['word_emb_size'] + config['char_emb_size']), 1))

    def forward(self, G, M):
        start = torch.cat([G, M], dim=2)
        start = torch.matmul(start, self.w1)
        start = F.dropout(start, config['dropout_rate'], training=self.training)
        start = F.log_softmax(start, dim=1)

        M = M.transpose(0, 1)
        M2, hidden = self.lstm(M)
        M2 = M2.transpose(0, 1)
        end = torch.cat([G, M2], dim=2)
        end = torch.matmul(end, self.w2)
        end = F.dropout(end, config['dropout_rate'], training=self.training)
        end = F.log_softmax(end, dim=1)
        return start.squeeze(2), end.squeeze(2)


class BIDAF(nn.Module):
    """
    Bidirectional Attention Flow
    """

    def __init__(self, word_mat, char_mat):
        super(BIDAF, self).__init__()
        if config['char_pretrained']:
            self.char_emb = nn.Embedding.from_pretrained(torch.Tensor(char_mat), freeze=True)
        else:
            self.char_emb = nn.Embedding.from_pretrained(torch.Tensor(char_mat), freeze=False)
        if config['word_pretrained']:
            self.word_emb = nn.Embedding.from_pretrained(torch.Tensor(word_mat), freeze=False)
        else:
            self.word_emb = nn.Embedding.from_pretrained(torch.Tensor(word_mat), freeze=True)
        self.context_emb = Embedding()
        self.question_emb = Embedding()
        self.context_contextual_emb = ContextualEmbedding(config['word_emb_size'] + config['char_emb_size'])
        self.question_contextual_emb = ContextualEmbedding(config['word_emb_size'] + config['char_emb_size'])
        self.attention_layer = AttentionFlowLayer()
        self.modeling_layer = ModelingLayer()
        self.output = Output()

    def forward(self, context_word_idxes, context_char_idxes, question_word_idxes, question_char_idxes):
        context_emb = self.context_emb(self.word_emb(context_word_idxes), self.char_emb(context_char_idxes))
        question_emb = self.context_emb(self.word_emb(question_word_idxes), self.char_emb(question_char_idxes))

        context_emb = self.context_contextual_emb(context_emb)
        question_emb = self.question_contextual_emb(question_emb)

        G = self.attention_layer(context_emb, question_emb)
        M = self.modeling_layer(G)
        start, end = self.output(G, M)
        return start, end


if __name__ == '__main__':
    a = torch.rand(8, 40, 300)
    b = torch.rand(8, 40, 16, 200)

    x = Embedding()(a, b)
    x = ContextualEmbedding(500)(x)
    print('question', x.size())

    c = torch.rand(8, 300, 300)
    d = torch.rand(8, 300, 16, 200)

    y = Embedding()(c, d)
    y = ContextualEmbedding(500)(y)
    print('context', y.size())

    G = AttentionFlowLayer()(y, x)

    M = ModelingLayer()(G)

    start, end = Output()(G, M)
    print(start)
    print(end.size())

    # conv=nn.Conv1d(config['char_limit'], config['char_emb_size'], 5)
    # x=torch.rand(40,16,200)
    # x = conv(x)
    # print(x.size())
