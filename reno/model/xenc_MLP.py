import torch
import torch.nn as nn
import torch.nn.functional as F

import reno

log = reno.utils.get_logger()


class xenc_MLP(nn.Module):
    def __init__(self, input_dim, hidden_size, tag_size, args):
        super(xenc_MLP, self).__init__()
        self.lin1 = nn.Linear(input_dim, hidden_size, bias=True)
        self.drop1 = nn.Dropout(args.drop_rate)
        self.lin2 = nn.Linear(hidden_size*2, hidden_size, bias=True)
        self.drop2 = nn.Dropout(args.drop_rate)
        self.lin3 = nn.Linear(hidden_size, tag_size, bias=True)
        self.pos_type = args.positional_encoding

    def get_encoding(self, x, p):
        hidden = self.drop1(F.relu(self.lin1(x)))
        if self.pos_type == "det":
            hidden = torch.cat((hidden, p), dim=2)
            hidden = self.drop2(F.relu(self.lin2(hidden)))

        return hidden

    def forward(self, x, p):
        hidden = self.get_encoding(x, p)
        x_hat = self.lin3(hidden)

        return x_hat
