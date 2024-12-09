import math
import random
import torch
import numpy as np
import reno.utils as utils
from scipy.signal import find_peaks
import os


class DatasetDetSampl:

    def __init__(self, funcs, args, test=False):
        self.funcs = funcs
        self.batch_size = args.batch_size
        self.num_batches = math.ceil(len(self.funcs) / self.batch_size)
        self.feat_size = args.input_dim
        self.pos_size = args.hidden_size*2
        self.sampling = args.sampling
        self.min_frames = args.min_samples
        self.max_frames = args.max_samples
        self.test = test
        self.num_target = args.num_target
        self.K = args.K

    def __len__(self):
        return len(self.funcs)

    def __getitem__(self, index):
        func = self.funcs[index]
        cur_len = 2*self.K + 1
        N = len(func.Input_Freq)
        # make input and output sequences
        inSeq = []
        outSeq = []
        for k in range(-1*self.K, self.K + 1):
            x_k = k/(2*self.K+1)

            A = func.Input_Amp*np.cos(func.Input_Freq*x_k)
            B = (1-func.Input_Amp)*np.sin(func.Input_Freq*x_k)
            inSeq.append(np.sum(A+B)/N)

            A = func.Output_Amp*np.cos(func.Output_Freq*x_k)
            B = (1-func.Output_Amp)*np.sin(func.Output_Freq*x_k)
            outSeq.append(np.sum(A+B)/N)


        features = torch.reshape(torch.tensor(inSeq).float(), (cur_len,1))
        label_tensor = np.array(outSeq)
        
        index = np.array(self.sampleFrames(label_tensor, func.name), dtype='int_')

        context_tensor = label_tensor[index]

        # context_tensor = torch.tensor(context_tensor).float()
        label_tensor = torch.tensor(label_tensor).float()
        return features, label_tensor, index, context_tensor, func.name

    def sampleFrames(self, label_trace, name):
        index = []
        if self.sampling == "uniform_const":
            conv_num = name.split('_')
            conv_num = conv_num[1]
            conv_num = int(conv_num)
            len_trace = len(label_trace)
            if self.min_frames < len_trace-self.num_target:
                jump_len = np.floor(len_trace/self.min_frames)
                st = conv_num % jump_len
                for i in range(self.min_frames):
                    index.append(st + i*jump_len)
            else:
                jump_len = np.floor(len_trace/self.num_target)
                st = conv_num % jump_len
                index = np.arange(len_trace).tolist()
                for i in range(self.num_target):
                    index.remove(st + i*jump_len)
        elif self.sampling == "uniform":
            len_trace = len(label_trace)
            if self.min_frames < len_trace-self.num_target:
                jump_len = np.floor(len_trace/self.min_frames)
                st = random.randint(0, jump_len-1)
                for i in range(self.min_frames):
                    index.append(st + i*jump_len)
            else:
                jump_len = np.floor(len_trace/self.num_target)
                st = random.randint(0, jump_len-1)
                index = np.arange(len_trace).tolist()
                for i in range(self.num_target):
                    index.remove(st + i*jump_len)
        elif self.sampling == "max":
            len_trace = len(label_trace)
            dist = np.floor(len_trace/(self.min_frames+1))
            enough = False
            while not enough:
                tmp_indx, _ = find_peaks(label_trace, distance=dist)
                if len(tmp_indx) < self.min_frames:
                    dist = dist - 5
                else:
                    enough = True
            trace = label_trace[tmp_indx]
            srt = np.argsort(-1*trace)
            tmp_indx = tmp_indx[srt]
            for i in range(self.min_frames):
                index.append(tmp_indx[i])
        elif self.sampling == "min":
            len_trace = len(label_trace)
            dist = np.floor(len_trace/(self.min_frames+1))
            enough = False
            while not enough:
                tmp_indx, _ = find_peaks(-1*label_trace, distance=dist)
                if len(tmp_indx) < self.min_frames:
                    dist = dist - 5
                else:
                    enough = True
            trace = label_trace[tmp_indx]
            srt = np.argsort(trace)
            tmp_indx = tmp_indx[srt]
            for i in range(self.min_frames):
                index.append(tmp_indx[i])
        else:
            raise ValueError("Not a valid sampling option.")
        return index

    def shuffle(self):
        random.shuffle(self.funcs)


class DatasetDetSampl_old:

    def __init__(self, convs, args, feat_mean, feat_std):
        self.convs = convs
        self.batch_size = args.batch_size
        self.num_batches = math.ceil(len(self.convs) / self.batch_size)
        self.emo = args.emo
        self.feat_size = args.input_dim
        self.pos_size = args.hidden_size*2
        self.sampling = args.sampling
        self.min_frames = args.min_samples
        self.max_frames = args.max_samples
        self.feat_mean = feat_mean
        self.feat_std = feat_std
        self.chunk = args.chunk
        self.predictions = args.predictions

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        batch = self.raw_batch(index)
        return self.padding(batch)

    def raw_batch(self, index):
        assert index < self.num_batches, "batch_indx %d > %d" % (index, self.num_batches)
        batch = self.convs[index*self.batch_size: (index+1)*self.batch_size]
        return batch

    def padding(self, convs):
        batch_size = len(convs)
        len_tensor = torch.tensor([len(c.speakers) for c in convs]).long()
        mx_len = torch.max(len_tensor).item()
        feature_tensor = torch.zeros((batch_size, mx_len, self.feat_size))
        position_tensor = torch.zeros((batch_size, mx_len, self.pos_size))
        speaker_tensor = torch.zeros((batch_size, mx_len)).long()
        index_tensor = torch.zeros((batch_size, self.max_frames)).long()
        label_tensor = torch.zeros((batch_size, mx_len)).long()
        context_tensor = torch.zeros((batch_size, mx_len,)).long()
        for i, c in enumerate(convs):
            cur_len = len(c.speakers)
            if self.chunk:
            # if True:
                tmp = [torch.from_numpy(t).float() for t in c.features]
            else:
                features = utils.getFeat(c.feature_file, self.feat_mean, self.feat_std, cur_len)
                tmp = [torch.from_numpy(t).float() for t in features]
            tmp = torch.stack(tmp)
            feature_tensor[i, :cur_len, :] = tmp
            speaker_tensor[i, :cur_len] = torch.tensor([int(s) for s in c.speakers])
            if self.emo == "A":
                label_tensor[i, :cur_len] = torch.tensor([float(s) for s in c.arousal])
                if self.predictions:
                    context_tensor[i, :cur_len] = torch.tensor([float(s) for s in c.pred_arousal])
                index = self.sampleFrames(c.arousal)
            elif self.emo == "V":
                label_tensor[i, :cur_len] = torch.tensor([float(s) for s in c.valence])
                if self.predictions:
                    context_tensor[i, :cur_len] = torch.tensor([float(s) for s in c.pred_valence])
                index = self.sampleFrames(c.valence)
            elif self.emo == "D":
                label_tensor[i, :cur_len] = torch.tensor([float(s) for s in c.dominance])
                if self.predictions:
                    context_tensor[i, :cur_len] = torch.tensor([float(s) for s in c.pred_dominance])
                index = self.sampleFrames(c.dominance)
            else:
                raise ValueError("Not a valid emotional dimension.")
            index_tensor[i, :len(index)] = torch.tensor([int(ix) for ix in index])
            position_tensor[i, :cur_len, :] = utils.getPositionEncoding(cur_len, self.pos_size)
            if not self.predictions:
                context_tensor = label_tensor
        
        data = {
            "len_tensor": len_tensor,
            "feature_tensor": feature_tensor,
            "position_tensor": position_tensor,
            "speaker_tensor": speaker_tensor,
            "label_tensor": label_tensor,
            "index_tensor": index_tensor,
            "context_label_tensor": context_tensor
        }
        return data

    def sampleFrames(self, label_trace):
        index = []
        if self.sampling == "uniform":
            len_trace = len(label_trace)
            jump_len = np.floor(len_trace/self.min_frames)
            st = random.randint(0, jump_len)
            for i in range(self.min_frames):
                index.append(st + i*jump_len)
        elif self.sampling == "max":
            len_trace = len(label_trace)
            dist = np.floor(len_trace/(self.min_frames+1))
            enough = False
            while not enough:
                tmp_indx, _ = find_peaks(label_trace, distance=dist)
                if len(tmp_indx) < self.min_frames:
                    dist = dist - 5
                else:
                    enough = True
            trace = label_trace[tmp_indx]
            srt = np.argsort(-1*trace)
            tmp_indx = tmp_indx[srt]
            for i in range(self.min_frames):
                index.append(tmp_indx[i])
        elif self.sampling == "min":
            len_trace = len(label_trace)
            dist = np.floor(len_trace/(self.min_frames+1))
            enough = False
            while not enough:
                tmp_indx, _ = find_peaks(-1*label_trace, distance=dist)
                if len(tmp_indx) < self.min_frames:
                    dist = dist - 5
                else:
                    enough = True
            trace = label_trace[tmp_indx]
            srt = np.argsort(trace)
            tmp_indx = tmp_indx[srt]
            for i in range(self.min_frames):
                index.append(tmp_indx[i])
        else:
            raise ValueError("Not a valid sampling option.")
        return index

    def shuffle(self):
        random.shuffle(self.convs)
