import argparse
import numpy as np
import random
import reno

# Goal: generate 1,000 training functions, and 300 testing functions
# Each function will be given two vectors, each vector of size N
#   Vector F will have the frequencies of each sin and cos: vector of floats between 0 and 30
#   Vector A will have the weight of the cos: vector of floats between 0 and 1
# f(x)= 1/N SUM{n=1 to N} A[n]cos(F[n]x) + (1-A[n])sin(F[n]x)
# This will be a 30-bandlimited function with maximum amplitude of 1

def generateFunction(name, N=50, max_freq=30, max_A=1):
    # Generate Fs
    F1 = np.random.rand(N)*max_freq
    F2 = np.random.rand(N)*max_freq
    # Generate As
    A1 = np.random.rand(N)*max_A
    A2 = np.random.rand(N)*max_A
    func = reno.BandFunc(name,F1,A1,F2,A2)
    return func

def main(args):
    reno.utils.set_seed(args.seed)

    train = []
    for i in range(args.num_train):
        N = random.randint(1,args.max_sin)
        train.append(generateFunction("trainfunc_"+str(i),N, args.max_freq, args.max_amp))
    
    test = []
    for i in range(args.num_test):
        N = random.randint(1,args.max_sin)
        test.append(generateFunction("testfunc_"+str(i),N, args.max_freq, args.max_amp))

    data = {"train": train, "test": test}

    # save all data
    reno.utils.save_pkl(data, args.data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="data_generation.py")
    parser.add_argument("--data", type=str, default="scripts/data.sh",
                        help="Path to data")
    parser.add_argument("--max_sin", type=int, default=100,
                        help="Maximum number of sinusoids to add together to make a function.")
    parser.add_argument("--max_freq", type=float, default=30,
                        help="Maximum frequency of bandlimited functions.")
    parser.add_argument("--max_amp", type=float, default=1,
                        help="Maximum amplitude of functions.")
    parser.add_argument("--seed", type=int, default=24,
                        help="Random seed.")
    parser.add_argument("--num_train", type=int, default=1000,
                        help="Number of functions for training.")
    parser.add_argument("--num_test", type=int, default=300,
                        help="Number of functions for testing.")
    args = parser.parse_args()

    main(args)

