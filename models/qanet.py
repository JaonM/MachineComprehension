"""
QANet
https://arxiv.org/abs/1804.09541
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from configs import config
from configs import device

config = config['QANet']


class Highway(nn.Module):
    """
    Input shape=(batch_size,dim,dim)
    Output shape=(batch_size,dim,dim)
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


class PositionEncoder(nn.Module):
    def __init__(self, length, d_pos_vec):
        super(PositionEncoder, self).__init__()
        # freqs = torch.Tensor(
        #     [10000 ** (-i / (config['word_emb_size'] + config['char_emb_size'])) if i % 2 == 0 else -10000 ** (
        #                 (1 - i) / (config['word_emb_size'] + config['char_emb_size']))
        #      for i in range(config['word_emb_size'] + config['char_emb_size'])]).unsqueeze(dim=1)
        # phases = torch.Tensor([0 if i % 2 == 0 else math.pi / 2 for i in
        #                        range(config['word_emb_size'] + config['char_emb_size'])]).unsqueeze(
        #     dim=1)
        # pos = torch.arange(length).repeat(config['word_emb_size'] + config['char_emb_size'], 1)
        # self.pos_encoding = nn.Parameter(torch.sin(torch.add(torch.mul(pos, freqs), phases)), requires_grad=False)

        ''' Init the sinusoid position encoding table '''
        # d_pos_vec = config['word_emb_size'] + config['char_emb_size']
        position_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)] for pos in range(length)])

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        self.pos_encoding = torch.from_numpy(position_enc).type(torch.FloatTensor).to(device)
        # self.pos_encoding = torch.from_numpy(position_enc).type(torch.FloatTensor)

    def forward(self, x):
        # pos = self.pos_encoding.transpose(0, 1)
        x = x + self.pos_encoding
        return x


class DepthwiseSeparableConv(nn.Module):
    """
    Input: shape=(batch_size,input_channels,context_length)
    Output: shape=(batch_size,output_channels,context_length)
    """

    def __init__(self, input_channels, output_channels, kernel_size, bias=True):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise_conv = nn.Conv1d(input_channels, input_channels, kernel_size=kernel_size, groups=input_channels,
                                        padding=kernel_size // 2, bias=bias)
        self.pointwise_conv = nn.Conv1d(input_channels, output_channels, kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        # return self.pointwise_conv(self.depthwise_conv(x))
        return x


class Embedding(nn.Module):
    """
    word and char embedding

    Input shape: word_emb=(batch_size,sentence_length,emb_size) char_emb=(batch_size,sentence_length,word_length,emb_size)
    Output shape: y= (batch_size,sentence_length,word_emb_size+char_emb_size)
    """

    def __init__(self):
        super(Embedding, self).__init__()
        self.highway = Highway(2)

    def forward(self, word_emb, char_emb):
        char_emb, _ = torch.max(char_emb, 2)

        char_emb = F.dropout(char_emb, config['char_dropout_rate'], training=self.training)

        word_emb = F.dropout(word_emb, config['dropout_rate'], training=self.training)

        emb = torch.cat([word_emb, char_emb], dim=2)
        emb = self.highway(emb)

        return emb


class SelfAttention(nn.Module):
    """
    Self Attention layer to capture global interactions between text

    Input shape: x = (batch_size,sentence_length,d)
    Output shape: x = (batch_size,sentence_length,d)
    """

    def __init__(self, num_head):
        super(SelfAttention, self).__init__()
        d = config['connector_dim']
        self.d_k, self.d_v = d // num_head, d // num_head
        self.num_head = num_head
        w_o = torch.empty(self.d_v * num_head, d)
        w_qs = [torch.empty(d, self.d_k) for _ in range(num_head)]
        w_ks = [torch.empty(d, self.d_k) for _ in range(num_head)]
        w_vs = [torch.empty(d, self.d_v) for _ in range(num_head)]

        nn.init.kaiming_uniform_(w_o)
        for i in range(num_head):
            nn.init.xavier_uniform_(w_qs[i])
            nn.init.xavier_uniform_(w_ks[i])
            nn.init.xavier_uniform_(w_vs[i])

        self.w_o = nn.Parameter(w_o)
        self.w_qs = nn.ParameterList([nn.Parameter(x) for x in w_qs])
        self.w_ks = nn.ParameterList([nn.Parameter(x) for x in w_ks])
        self.w_vs = nn.ParameterList([nn.Parameter(x) for x in w_vs])

    def forward(self, x):
        """

        :param x: x shape=(batch_size,sentence_length,config['connector_dim'])
        :return:
        """
        _w_qs, _w_ks, _w_vs = [], [], []
        scale = 1 / math.sqrt(self.d_k)
        for i in range(self.num_head):
            _w_qs.append(torch.matmul(x, self.w_qs[i]))
            _w_ks.append(torch.matmul(x, self.w_ks[i]))
            _w_vs.append(torch.matmul(x, self.w_vs[i]))

        heads = []
        # compute scaled doc-product attention for each head
        for i in range(self.num_head):
            qk_mul = torch.bmm(_w_qs[i], _w_ks[i].transpose(1, 2))
            qk_mul = torch.mul(qk_mul, scale)
            qk_mul = F.softmax(qk_mul, dim=1)
            out = torch.bmm(qk_mul, _w_vs[i])
            heads.append(out)
        del _w_qs, _w_ks, _w_vs
        gc.collect()
        head = torch.cat(heads, dim=2)
        head = torch.matmul(head, self.w_o)
        return head


class BlockEncoder(nn.Module):
    """
    BlockEncoder for embedding layer and context-query attention layer

    Input shape: input = (batch_size,length,input_dim)
    Output shape: output = (batch_size,length,d)
    """

    def __init__(self, input_dim, conv_num, ch_num, length, kernel_size):
        super(BlockEncoder, self).__init__()
        self.pos = PositionEncoder(length, input_dim)
        # self.shape_convert_fc = nn.Linear(config['word_emb_size'] + config['char_emb_size'], ch_num)
        self.first_conv = DepthwiseSeparableConv(input_dim, config['connector_dim'], kernel_size)
        self.convs = nn.ModuleList([DepthwiseSeparableConv(ch_num, ch_num, kernel_size) for _ in range(conv_num)])
        self.conv_norm = nn.LayerNorm([ch_num, length])
        self.layer_norm = nn.LayerNorm([length, ch_num])
        self.attention = SelfAttention(num_head=8)
        self.fc = nn.Linear(ch_num, ch_num)
        self.conv_num = conv_num

    def forward(self, x):
        x = self.pos(x)
        x = x.permute(0, 2, 1)
        # x = self.shape_convert_fc(x)
        # x = F.relu(x)
        # x = F.dropout(x, config['dropout_rate'])
        # print('origin size', x.size())
        x = self.first_conv(x)
        # print('first conv size', x.size())
        # x = F.relu(x)
        res = x
        for i in range(self.conv_num):
            pl = i / self.conv_num * (1 - 0.9)  # dropout rate decay
            x = self.conv_norm(x)
            x = self.convs[i](x)
            x = F.relu(x)
            x = res + x
            x = F.dropout(x, pl, training=self.training)
            res = x

        x = x.permute(0, 2, 1)
        res = x
        x = self.layer_norm(x)
        x = self.attention(x)
        x = res + x
        x = F.dropout(x, config['dropout_rate'], training=self.training)
        res = x
        x = self.layer_norm(x)
        x = self.fc(x)
        x = F.relu(x, config['dropout_rate'])
        x = res + x
        del res
        gc.collect()
        x = F.dropout(x, config['dropout_rate'], training=self.training)
        return x


class ContextQueryAttention(nn.Module):
    """
    context-query attention in paper

    Input shape: Context=(batch_size,context_length,d) Query=(batch_size,query_length,d)
    Output shape: out = (batch_size,context,3*d)
    """

    def __init__(self):
        super(ContextQueryAttention, self).__init__()
        # w = [torch.empty(config['question_limit'], 3 * config['connector_dim']) for _ in
        #      range(config['paragraph_limit'])]
        lim = 1 / config['connector_dim']
        # for i in range(config['paragraph_limit']):
        w = torch.empty(3 * config['connector_dim'])
        nn.init.uniform_(w, -math.sqrt(lim), math.sqrt(lim))
        # self.W0 = nn.ParameterList(nn.Parameter(x) for x in w)
        self.W0 = nn.Parameter(w)

    def forward(self, c, q):
        shape = (c.size(0), c.size(1), q.size(1), c.size(2))
        ct = c.unsqueeze(2).expand(shape)
        qt = q.unsqueeze(1).expand(shape)
        cq = torch.mul(ct, qt)
        S = torch.cat([ct, qt, cq], dim=3)
        S = torch.matmul(S, self.W0)
        S1 = F.softmax(S, dim=2)
        A = torch.bmm(S1, q)
        S2 = F.softmax(S, dim=1)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), c)

        out = torch.cat([c, A, torch.mul(c, A), torch.mul(c, B)], dim=2)
        del S, S1, S2
        gc.collect()
        return out


class Output(nn.Module):
    """
    Output layer

    Input shape: input=(batch_size,context_length,2*d)
    Output shape: output=(batch_size,context_length)
    """

    def __init__(self):
        super(Output, self).__init__()
        self.fc = nn.Linear(config['paragraph_limit'] * config['connector_dim'] * 2, config['paragraph_limit'])

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.relu(x)
        # x = F.dropout(x, config['dropout_rate'], training=self.training)
        return F.log_softmax(x, dim=1)


class QANet(nn.Module):
    """
    QANet model
    Input shape: Context=(batch_size,context_length,emb_size),Question=(batch_size,question_length,emb_size)
    Output shape: answer_start = (batch_size,context_length), answer_end = (batch_size,context_length)
    """

    def __init__(self, word_mat, char_mat):
        super(QANet, self).__init__()
        if config['char_pretrained']:
            self.char_emb = nn.Embedding.from_pretrained(torch.Tensor(char_mat))
        else:
            self.char_emb = nn.Embedding.from_pretrained(torch.Tensor(char_mat), freeze=False)
        if config['word_pretrained']:
            self.word_emb = nn.Embedding.from_pretrained(torch.Tensor(word_mat))
        else:
            self.word_emb = nn.Embedding.from_pretrained(torch.Tensor(word_mat),freeze=True)
        self.context_emb = Embedding()
        self.question_emb = Embedding()
        self.context_emb_encoder = BlockEncoder(config['word_emb_size'] + config['char_emb_size'], 4,
                                                config['connector_dim'],
                                                config['paragraph_limit'], 7)
        self.question_emb_encoder = BlockEncoder(config['word_emb_size'] + config['char_emb_size'], 4,
                                                 config['connector_dim'],
                                                 config['question_limit'], 7)
        self.context_query_attention = ContextQueryAttention()
        self.resizer = nn.Linear(4 * config['connector_dim'],
                                 config['connector_dim'])  # resize the context query output
        self.M0 = nn.ModuleList(
            [BlockEncoder(config['connector_dim'], 2, config['connector_dim'], config['paragraph_limit'],
                          5) for _ in range(7)])
        # self.M1 = self.M0
        # self.M2 = self.M0
        self.start = Output()
        self.end = Output()

    def forward(self, context_word_idxes, context_char_idxes, ques_word_idxes, ques_char_idxes):
        context_emb = self.context_emb(self.word_emb(context_word_idxes), self.char_emb(context_char_idxes))
        question_emb = self.question_emb(self.word_emb(ques_word_idxes), self.char_emb(ques_char_idxes))
        context_enc = self.context_emb_encoder(context_emb)
        question_enc = self.question_emb_encoder(question_emb)
        context_question_att = self.context_query_attention(context_enc, question_enc)
        # print(context_question_att.size())
        m0 = self.resizer(context_question_att)
        for i in range(len(self.M0)):
            m0 = self.M0[i](m0)
        m1 = m0
        for i in range(len(self.M0)):
            m1 = self.M0[i](m1)
        m2 = m1
        for i in range(len(self.M0)):
            m2 = self.M0[i](m2)

        start = torch.cat([m0, m1], dim=2)
        end = torch.cat([m0, m2], dim=2)
        start = self.start(start)
        end = self.end(end)
        del m0, m1, m2
        gc.collect()
        return start, end

    @staticmethod
    def generate_answer_idxes(start, end):
        """
        generate answer indexes in context span
        :param start: (batch_size,context_length)
        :param end: (batch_size,context_length)
        :return: start answer list,end answer list
        """
        start_anwsers = list()
        end_answers = list()
        batch_size = start.size(0)
        start = start.detach().cpu().numpy()
        end = end.detach().cpu().numpy()
        for batch in range(batch_size):
            max_p = start[batch][0] * end[batch][0]
            start_max_idx = 0
            end_max_idx = 0
            for i in range(1, len(end[batch])):
                start_max_p = start[batch][:i].max()
                _start_max_idx = start[batch][:i].argmax()
                if max_p >= start_max_p * end[batch][i]:
                    start_max_idx = _start_max_idx
                    end_max_idx = i
                    max_p = start_max_p * end[batch][i]
            start_anwsers.append(start_max_idx)
            end_answers.append(end_max_idx)
        return start_anwsers, end_answers


if __name__ == '__main__':
    a = torch.rand(8, 40, 300)
    b = torch.rand(8, 40, 16, 200)

    x = Embedding()(a, b)
    x = BlockEncoder(500, 4, 128, 40, 7)(x)
    print('question', x.size())

    c = torch.rand(8, 300, 300)
    d = torch.rand(8, 300, 16, 200)

    y = Embedding()(c, d)
    y = BlockEncoder(500, 4, 128, 300, 7)(y)
    print('context', y.size())

    res = ContextQueryAttention()(y, x)
    print(res.size())
    # a, b = torch.rand(32, 400), torch.rand(32, 400)
    # start, end = QANet.generate_answer_idxes(a, b)
    # print(start[0])
    # print(end[0])
