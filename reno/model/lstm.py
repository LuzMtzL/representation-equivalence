import torch
import torch.nn as nn
import torch.nn.functional as F

import reno

log = reno.utils.get_logger()


class lstm(nn.Module):
    def __init__(self, input_dim, hidden_size, tag_size, args):
        super(lstm, self).__init__()
        self.rnn1 = nn.LSTM(input_dim, hidden_size, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(args.drop_rate)
        self.rnn2 = nn.LSTM(hidden_size*2, hidden_size, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(hidden_size*2, tag_size, bias=True)

    def get_prob(self, x):
        hidden, _ = self.rnn1(x)
        hidden = self.drop(hidden)
        hidden, _ = self.rnn2(hidden)

        return hidden

    def forward(self, x):
        hidden = self.get_prob(x)
        y_hat = self.lin(hidden)

        return y_hat

    def get_loss(self, x, label_tensor):
        hidden = self.get_prob(x)
        y_hat = self.lin(hidden)
        loss = self.concordance(torch.flatten(label_tensor), torch.flatten(y_hat))

        return loss

    def concordance(self, true_labels, predicted_values):
        true_labels = true_labels.float()
        predicted_values = predicted_values.float()
        cov = torch.sum((true_labels - torch.mean(true_labels)) * (predicted_values - torch.mean(predicted_values))) / (
                    len(true_labels) - 1)
        numerator = 2 * cov
        denominator = torch.var(true_labels, unbiased=True) + torch.var(predicted_values, unbiased=True) + \
                      ((torch.mean(true_labels) - torch.mean(predicted_values)) * (
                                  torch.mean(true_labels) - torch.mean(predicted_values)))
        loss = 1 - numerator / denominator
        return loss
