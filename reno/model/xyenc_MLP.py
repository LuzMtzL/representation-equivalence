import torch
import torch.nn as nn
import torch.nn.functional as F

import reno

log = reno.utils.get_logger()


class xyenc_MLP(nn.Module):
    def __init__(self, input_dim, hidden_size, tag_size, args):
        super(xyenc_MLP, self).__init__()
        self.lin1 = nn.Linear(input_dim, hidden_size, bias=True)
        self.drop1 = nn.Dropout(args.drop_rate)
        self.lin2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.drop2 = nn.Dropout(args.drop_rate)
        self.lin3 = nn.Linear(hidden_size, tag_size, bias=True)

    def forward(self, x_hat, y):
        y_mat = torch.reshape(y, (y.size(0), y.size(1), 1))
        hidden = torch.cat((x_hat, y_mat), dim=2)
        hidden = self.drop1(F.relu(self.lin1(hidden)))
        hidden = self.drop2(F.relu(self.lin2(hidden)))
        r = self.lin3(hidden)

        return r
