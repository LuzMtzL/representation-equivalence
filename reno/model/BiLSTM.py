import torch
import torch.nn as nn

from .lstm import lstm
import reno

log = reno.utils.get_logger()


class BiLSTM(nn.Module):

    def __init__(self, args):
        super(BiLSTM, self).__init__()
        u_dim = args.input_dim
        h_dim = args.hidden_size
        tag_size = 1

        self.device = args.device

        self.rnn_mod = lstm(u_dim, h_dim, tag_size, args)

    def forward(self, features, indexes, context, lens):
        y_hat = self.rnn_mod(features)

        return y_hat, torch.zeros_like(y_hat)

    def get_loss(self, features, labels, indexes, context, lens):
        loss = self.rnn_mod.get_loss(features, labels)

        return loss, None, None, None