import torch
import torch.nn as nn
import torch.nn.functional as F

import reno

log = reno.utils.get_logger()


class dec_MLP(nn.Module):
    def __init__(self, input_dim, hidden_size, tag_size, args):
        super(dec_MLP, self).__init__()
        self.lin1 = nn.Linear(input_dim, hidden_size, bias=True)
        self.drop1 = nn.Dropout(args.drop_rate)
        self.lin2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.drop2 = nn.Dropout(args.drop_rate)
        self.lin3 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.drop3 = nn.Dropout(args.drop_rate)
        self.lin4 = nn.Linear(hidden_size, tag_size, bias=True)

        self.loss_weight = args.loss_weight

    def get_prob(self, x_hat, r):
        batch_size, num_target, r_dim = r.size(0), x_hat.size(1), r.size(1)
        r_mat = torch.reshape(r,(batch_size,1,r_dim))
        r_mat = r_mat.repeat(1, num_target, 1)
        hidden = torch.cat((x_hat, r_mat), dim=2)
        hidden = self.drop1(F.relu(self.lin1(hidden)))
        hidden = self.drop2(F.relu(self.lin2(hidden)))
        hidden = self.drop3(F.relu(self.lin3(hidden)))

        return hidden

    def forward(self, x_hat, r):
        hidden = self.get_prob(x_hat, r)
        stats = self.lin4(hidden)
        y_hat = stats[:, :, 0]
        y_var = stats[:, :, 1]

        return y_hat, y_var

    def get_loss(self, x_hat, r, label_tensor, var_tensor):
        hidden = self.get_prob(x_hat, r)
        stats = self.lin4(hidden)
        y_hat = stats[:, :, 0]
        y_var = stats[:, :, 1]
        loss_CCC = self.concordance(torch.flatten(label_tensor), torch.flatten(y_hat))
        # loss_var = self.variance_loss(torch.flatten(label_tensor), torch.flatten(y_hat), torch.flatten(y_var))
        loss_var = self.concordance(torch.flatten(var_tensor), torch.flatten(y_var))
        loss = self.loss_weight*loss_CCC + (1-self.loss_weight)*loss_var
        return loss, y_hat, y_var

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
    
    def variance_loss(self, true_labels, predicted_values, predicted_var):
        true_labels = true_labels.float()
        predicted_values = predicted_values.float()
        diff = torch.square(true_labels - predicted_values)
        loss = F.mse_loss(predicted_var, diff)
        return loss
