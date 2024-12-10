import numpy as np
import argparse
from scipy import special

def main(args):
    K = args.train_K
    uTfx = []
    names = []
    with open(args.pred_fldr + "/test_names.txt") as f:
        lines = f.readlines()
        for line in lines:
            names.append(line.strip())
    with open(args.pred_fldr + "/test_preds.txt") as f:
        lines = f.readlines()
        for line in lines:
            tmp = []
            for val in line.strip().split():
                tmp.append(float(val))
            uTfx.append(tmp)
    train_uTfx = {names[i]: uTfx[i] for i in range(len(names))}

    X = []
    Y = []
    for K in range(args.start_K,args.last_K+1,args.step_K):
        test_fldr = args.pred_fldr.replace(str(args.train_K),str(K))
        uTfx = []
        names = []
        with open(test_fldr + "/test_names.txt") as f:
            lines = f.readlines()
            for line in lines:
                names.append(line.strip())
        with open(test_fldr + "/test_preds.txt") as f:
            lines = f.readlines()
            for line in lines:
                tmp = []
                for val in line.strip().split():
                    tmp.append(float(val))
                uTfx.append(tmp)
        test_uTfx = {names[i]: uTfx[i] for i in range(len(names))}
        Errors = []
        for name in names:
            if K < args.train_K:
                # Convert train sequence to test resolution
                uTfx = train_uTfx[name]
                uTfx_ref = test_uTfx[name]
                K1 = K
                K2 = args.train_K
            if K >= args.train_K:
                # Convert test sequence to train resolution
                uTfx_ref = train_uTfx[name]
                uTfx = test_uTfx[name]
                K2 = K
                K1 = args.train_K
            # synthesize then discretize
            TTuTfx = []
            for k in range(-1*K1,K1+1):
                xk = k/(2*K1+1)
                sum = 0
                for m in range(-1*K2,K2):
                    xm = m/(2*K2+1)
                    cm = uTfx[m+K2]
                    dm = special.diric(xk-xm,K2)
                    sum += cm*dm
                TTuTfx.append(sum)
            err = np.linalg.norm(np.array(uTfx_ref)-np.array(TTuTfx))
            Errors.append(err)
        RepresentationEquivalenceError = np.mean(Errors)
        X.append(2*K+1)
        Y.append(RepresentationEquivalenceError)
        
    with open('./results/' + args.pred_fldr.split('/')[-1] + '.txt') as f:
        f.write('Resolution;Representation Equivalence Error\n')
        for res, err in zip(X,Y):
            f.write(f"{res};{err}\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plot_results.py")
    parser.add_argument("--pred_fldr", type=str, default="predictions/CNP_MLP_Mean_30_uniform_5",
                        help="Path to folder with predictions.")
    parser.add_argument("--train_K", type=int, default=30,
                        help="K of frame sequence used for training.")
    parser.add_argument("--start_K", type=int, default=5,
                        help="First K of frame sequence used for testing.")
    parser.add_argument("--last_K", type=int, default=100,
                        help="Last K of frame sequence used for testing.")
    parser.add_argument("--step_K", type=int, default=5,
                        help="Step of counter for K of frame sequence used for testing.")
    args = parser.parse_args()

    main(args)