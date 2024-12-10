import copy
import time

import numpy as np
import torch
from tqdm import tqdm
from sklearn import metrics

import reno

log = reno.utils.get_logger()


class Coach:

    def __init__(self, train_loader, test_loader, model, opt, args, model_file, pred_fldr):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.opt = opt
        self.args = args
        self.model_file = model_file
        self.pred_fldr = pred_fldr
        self.start_epoch = 1

    def load_ckpt(self, ckpt):
        self.start_epoch = ckpt["epoch"] + 1
        state = ckpt["state"]
        self.model.load_state_dict(state)

    def train(self):
        log.debug(self.model)

        # Train
        for epoch in range(self.start_epoch, self.args.epochs + 1):
            torch.cuda.reset_peak_memory_stats(device=None)
            mem_start = torch.cuda.max_memory_allocated(device=None)
            time_start = time.time()
            self.train_epoch(epoch)

            state = copy.deepcopy(self.model.state_dict())
            self.save_model(epoch, state, self.model_file)
            time_end = time.time()
            mem_end = torch.cuda.max_memory_allocated(device=None)
            log.info("[Time elapsed] [sec {}]".format(time_end-time_start))
            log.info("[GPU memory usage] [max bytes {}]".format(mem_end-mem_start))

        # The best
        log.info("")
        test_ccc, test_corr, test_mse, y_hat, y_var, names = self.evaluate(test=True)
        log.info("[Test set] [ccc {:.4f}] [corr {:.4f}] [mse {:.4f}]".format(test_ccc, test_corr, test_mse))

        return

    def train_epoch(self, epoch):
        start_time = time.time()
        epoch_loss = 0
        self.model.train()
        for features, labels, indexes, context, lens, names in tqdm(self.train_loader):
            self.model.zero_grad()
            features = features.to(self.args.device)
            labels = labels.to(self.args.device)
            indexes = indexes.to(self.args.device)
            context = context.to(self.args.device)
            lens = lens.to(self.args.device)
            
            ccc_loss, y_hat, y_var, indxx = self.model.get_loss(features, labels, indexes, context, lens)

            epoch_loss += ccc_loss.item()
            ccc_loss.backward()
            self.opt.step()

        end_time = time.time()
        log.info("")
        log.info("[Epoch %d] [Loss: %f] [Time: %f]" %
                 (epoch, epoch_loss/len(self.train_loader), end_time - start_time))

    def evaluate(self, test=False):
        data_loader = self.test_loader if test else self.dev_loader
        self.model.eval()
        with torch.no_grad():
            golds = []
            preds = []
            lengths = []
            indx = []
            ctxt = []
            names = []
            for features, labels, indexes, context, lens, name in data_loader:
                features = features.to(self.args.device)
                indexes = indexes.to(self.args.device)
                context = context.to(self.args.device)
                lens = lens.to(self.args.device)

                [y_hat, y_var] = self.model(features, indexes, context, lens)
                
                for pred, l, lab, i, x, n in zip(y_hat, lens, labels, indexes, context, name):
                    lengths.append(l.detach().to('cpu'))
                    preds.append(pred.detach().to('cpu'))
                    golds.append(lab)
                    indx.append(i.detach().to('cpu'))
                    ctxt.append(x.detach().to('cpu'))
                    names.append(n)

            golds = [lab[:l] for lab, l in zip(golds, lengths)]
            preds = [pred[:l] for pred, l in zip(preds, lengths)]
            indices = [i for i in indx]
            observations = [obs for obs in ctxt]
            if self.args.save_pred:
                with open(self.pred_fldr + '/test_names.txt', 'w') as f:
                    for line in names:
                        f.write(line + '\n')
                with open(self.pred_fldr + '/test_golds.txt', 'w') as f:
                    for line in golds:
                        for l in line:
                            f.write(str(l.numpy()) + ' ')
                        f.write('\n')
                with open(self.pred_fldr + '/test_preds.txt', 'w') as f:
                    for line in preds:
                        for l in line:
                            f.write(str(l.numpy()) + ' ')
                        f.write('\n')
                with open(self.pred_fldr + '/test_indices.txt', 'w') as f:
                    for line in indices:
                        for l in line:
                            f.write(str(l.numpy()) + ' ')
                        f.write('\n')
                with open(self.pred_fldr + '/test_context.txt', 'w') as f:
                    for line in observations:
                        for l in line:
                            f.write(str(l.numpy()) + ' ')
                        f.write('\n')
            golds = torch.cat(golds, dim=0).numpy()
            preds = torch.cat(preds, dim=0).numpy()
            ccc, corr = self.evaluation_metrics(golds, preds)
            mse = metrics.mean_squared_error(golds, preds)
        return ccc, corr[0, 1], mse, y_hat, y_var, names

    def evaluation_metrics(self, true_value, predicted_value):
        corr_coeff = np.corrcoef(true_value.transpose(), predicted_value.transpose())
        ccc = 2*predicted_value.std()*true_value.std()*corr_coeff[0, 1]/(predicted_value.var() + true_value.var() +
                                                                        (predicted_value.mean() - true_value.mean())**2)
        return ccc, corr_coeff

    def save_model(self, epoch, state, model_file):
        # Save.
        checkpoint = {
            "epoch": epoch,
            "state": state,
        }
        torch.save(checkpoint, model_file)