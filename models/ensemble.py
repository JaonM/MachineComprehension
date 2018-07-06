"""
ensemble model
"""
import torch.nn as nn
import torch
from models.bidaf import BIDAF
from models.qanet import QANet


class OutputEnsemble(nn.Module):

    def __init__(self, word_mat, char_mat):
        super(OutputEnsemble, self).__init__()
        self.bidaf = BIDAF(word_mat, char_mat)
        self.qanet = QANet(word_mat, char_mat)

    def forward(self, context_word_idxes, context_char_idxes, question_word_idxes, question_char_idxes):
        qa_out = self.qanet(context_word_idxes, context_char_idxes, question_word_idxes, question_char_idxes)
        bi_out = self.bidaf(context_word_idxes, context_char_idxes, question_word_idxes, question_char_idxes)
        out = torch.add(qa_out, 1, bi_out)
        out = torch.div(out, 2)
        return out
