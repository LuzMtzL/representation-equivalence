import torch
import torch.nn as nn
import torch.nn.functional as F

from .xenc_MLP import xenc_MLP
from .xyenc_MLP import xyenc_MLP
from .dec_MLP import dec_MLP
import reno

log = reno.utils.get_logger()


class CNP_MLP_Mean(nn.Module):

    def __init__(self, args):
        super(CNP_MLP_Mean, self).__init__()
        u_dim = args.input_dim
        hx_dim = args.hidden_size*2
        x_dim = args.hidden_size
        r_dim = args.r_dim
        g_dim = args.r_dim
        label_size = 1
        tag_size = 2
        self.pos_type = args.positional_encoding
        self.atten = args.enc_atten

        self.wp = args.wp
        self.wf = args.wf
        self.device = args.device
        self.num_target = args.num_target
        self.context_noise_std = args.context_noise_std
        
        self.pos_size = hx_dim
        if self.pos_type == "det":
            self.pos_enc = self.getPositionEncoding(512)
        else:
            self.pos_enc = torch.zeros((512, self.pos_size))

        self.xenc = xenc_MLP(u_dim, hx_dim, x_dim, args)
        self.xyenc = xyenc_MLP(x_dim+label_size, r_dim, r_dim, args)
        self.dec = dec_MLP(x_dim+r_dim, r_dim, tag_size, args)
    
    def update_PositionEncoding(self, seq_len):
        self.pos_enc = self.getPositionEncoding(seq_len)
        
    def getPositionEncoding(self, seq_len, n=torch.tensor(10000)):
        D = self.pos_size
        P = torch.zeros((seq_len, D))
        for k in range(seq_len):
            for i in range(int(D/2)):
                denominator = n.pow(2*i/D)
                P[k, 2*i] = torch.sin(k/denominator)
                P[k, 2*i+1] = torch.cos(k/denominator)
        return P

    def get_rep(self, features, context_points, context, lens):
        max_len = features.size(1)
        if max_len > self.pos_enc.size(0):
            if self.pos_type == "det":
                self.update_PositionEncoding(max_len)
            else:
                self.pos_enc = torch.zeros((max_len, self.pos_size))
        pos_enc = torch.stack([self.pos_enc[:max_len] for x in lens]).to(self.device)

        x_hat_ALL = self.xenc(features, pos_enc)
        batch_size, context_size, x_dim = x_hat_ALL.size(0), context_points.size(1), x_hat_ALL.size(2)
        x_hat_context = torch.zeros((batch_size, context_size, x_dim)).to(self.device)
        y_context = torch.zeros((batch_size, context_size)).to(self.device)
        for i in range(batch_size):
            x_hat_context[i, :, :] = x_hat_ALL[i, context_points[i, :], :]
            y_context[i, :] = context[i, :] + self.context_noise_std*torch.randn(context_size).to(self.device)
        r_context = self.xyenc(x_hat_context, y_context)
        r = torch.mean(r_context, dim=1)

        return r, x_hat_ALL

    def forward(self, features, indexes, context, lens):
        r, x_hat_ALL = self.get_rep(features, indexes, context, lens)
        y_hat, y_hat_var = self.dec(x_hat_ALL, r)

        return y_hat, y_hat_var

    def get_loss(self, features, y_ALL, context_points, context, len_tensor):
        r, x_hat_ALL = self.get_rep(features, context_points, context, len_tensor)
        batch_size, context_size, x_dim = x_hat_ALL.size(0), context_points.size(1), x_hat_ALL.size(2)
        target_size = int(self.num_target + context_size)
        x_hat_target = torch.zeros((batch_size, target_size, x_dim)).to(self.device)
        y_target = torch.zeros((batch_size, target_size)).to(self.device)
        indx_target = torch.zeros((batch_size, target_size)).to(self.device)
        for i in range(batch_size):
            perm = torch.randperm(len_tensor[i])
            idx = perm[:target_size]
            idx[:context_size] = context_points[i, :]
            x_hat_target[i, :, :] = x_hat_ALL[i, idx, :]
            y_target[i, :] = y_ALL[i, idx]
            indx_target[i, :] = idx
        loss, y_hat, y_var = self.dec.get_loss(x_hat_target, r, y_target)

        return loss, y_hat, y_var, indx_target