import argparse

import torch
import torch.utils

import reno
import numpy as np
from tqdm import tqdm
import os
CUDA_LAUNCH_BLOCKING="1"

log = reno.utils.get_logger()

def pad_collate(batch):
    (data, labels, indexes, context, names) = zip(*batch)

    #sanity check, make sure they have the same length
    # for x, y in zip(data, labels):
    #     assert x.size(1) == y.size(1)
    lens = torch.tensor([len(x) for x in data]).long()
    data_pad = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)
    labels_pad = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)

    indx_len_list = [len(x) for x in indexes]
    indx_len = min(indx_len_list)
    if indx_len == max([indx_len_list]):
        indexes = torch.tensor(np.array(indexes), dtype=torch.long)
        context = torch.tensor(np.array(context)).float()
    else:
        indexes = torch.tensor(np.array([x[0:indx_len] for x in indexes])).long()
        context = torch.tensor(np.array([x[0:indx_len] for x in context])).float()

    return data_pad, labels_pad, indexes, context, lens, names

def main(args):
    reno.utils.set_seed(args.seed)
    # args.chunk = True

    # load data
    log.debug("Loading data from '%s'." % args.data)
    data = reno.utils.load_pkl(args.data)
    log.info("Loaded data.")

    # get data sets

    log.debug("Building model...")
    model_name = args.model + "_" + str(args.K) + "_" + args.sampling + "_" + str(args.min_samples)
    if args.model_name == "":
        model_file = "./save/model_" + model_name + ".pt"
    else:
        model_file = "./save/model_" + args.model_name + ".pt"
    

    if args.min_samples > args.max_samples:
        args.max_samples = args.min_samples  
    if args.num_target > 2*args.K+1:
        args.num_target = 2*args.K+1 - args.min_samples
  
    trainset = reno.DatasetDetSampl(data["train"], args, test=False)
    testset = reno.DatasetDetSampl(data["test"], args, test=True)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size = args.batch_size, collate_fn=pad_collate,
                                            num_workers=4, pin_memory=True, shuffle = True, prefetch_factor=4)
    test_loader = torch.utils.data.DataLoader(testset, batch_size = args.batch_size, collate_fn=pad_collate,
                                            num_workers=4, pin_memory=True, shuffle = False, prefetch_factor=4)

    if args.model == "CNP_MLP_Mean":
        model = reno.CNP_MLP_Mean(args).to(args.device)
    elif args.model == "BiLSTM":
        model = reno.BiLSTM(args).to(args.device)
    else:
        model = reno.CNP_MLP_Mean(args).to(args.device)
        
    opt = reno.Optim(args.learning_rate, args.max_grad_value, args.weight_decay, args.momentum)
    opt.set_parameters(model.parameters(), args.optimizer)

    coach = reno.Coach(train_loader, test_loader, model, opt, args, model_file, './predictions/'+model_name)
    if not os.path.isdir('./predictions/'+model_name):
        os.mkdir('./predictions/'+model_name)
    if not args.from_begin:
    # if 0:
        ckpt = torch.load(model_file, weights_only=False)
        coach.load_ckpt(ckpt)

    # Train.
    log.info("Start training...")
    ret = coach.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument("--K", type=int, default=30,
                        help="K of input and output frame sequences.")
    parser.add_argument("--data", type=str, default="scripts/data.sh",
                        help="Path to data")
    parser.add_argument("--save_pred", action="store_true",
                        help="If the model predictions will be saved.")
    # CNP_MLP_Mean
    parser.add_argument("--model", type=str, default="CNP_MLP_Mean",
                        choices=["CNP_MLP_Mean", "BiLSTM"],
                        help="Model type.")
    # uniform, max, min, minmax
    parser.add_argument("--sampling", type=str, default="uniform",
                        choices=["uniform"],
                        help="Trace sampling type.")

    # frame-level wav2vec: min:100, max:100, target:1000
    # chunk-level (1/0.2 sec) wav2vec: min:5, max:5, target:50
    parser.add_argument("--min_samples", type=int, default=5,
                        help="Minimum number of trace samples.")
    parser.add_argument("--max_samples", type=int, default=5,
                        help="Maximum number of trace samples.")
    parser.add_argument("--num_target", type=int, default=50,
                        help="Number of target points during training.")
    parser.add_argument("--input_dim", type=int, default=1,
                        help="Dimension of input (features).")

    # Training parameters
    parser.add_argument("--positional_encoding", type=str, default="det",
                        choices=["det", "none"],
                        help="Type of positional encoding for the CNP.")
    parser.add_argument("--from_begin", action="store_true",
                        help="Training from begin.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Computing device.")
    parser.add_argument("--epochs", default=1, type=int,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", default=50, type=int,
                        help="Batch size.")
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["sgd", "rmsprop", "adam"],
                        help="Name of optimizer.")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-8,
                        help="Weight decay.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum (for SGD).")
    parser.add_argument("--max_grad_value", default=-1, type=float,
                        help="""If the norm of the gradient vector exceeds this,
                        normalize it to have the norm equal to max_grad_norm""")
    parser.add_argument("--drop_rate", type=float, default=0.5,
                        help="Dropout rate.")

    # Model parameters
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="Hidden size of input encoding.")
    parser.add_argument("--r_dim", type=int, default=32,
                        help="Size of r in the NP.")
    parser.add_argument("--rnn", type=str, default="lstm",
                        choices=["lstm", "gru"], help="Type of RNN cell.")

    # others
    parser.add_argument("--seed", type=int, default=24,
                        help="Random seed.")
    parser.add_argument("--model_name", type=str, default="",
                        help="File name for saving model.")
    args = parser.parse_args()
    log.debug(args)
    # args.from_begin = True
    main(args)
