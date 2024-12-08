import numpy as np
import random

# Goal: generate 1,000 training functions, and 300 testing functions
# Each function will be given two vectors, each vector of size N
#   Vector F will have the frequencies of each sin and cos: vector of floats between 0 and 30
#   Vector A will have the weight of the cos: vector of floats between 0 and 1
# f(x)= 1/N SUM{n=1 to N} A[n]cos(F[n]x) + (1-A[n])sin(F[n]x)
# This will be a 30-bandlimited function with maximum amplitude of 1

def generateFunction(max_N=100, max_freq=30, max_A=1):
    # Get random N
    N = random.randint(1,100)
    # Generate F
    F = np.random.rand(N)*30
    # Generate A
    A = np.random.rand(N)
    return N, F, A

