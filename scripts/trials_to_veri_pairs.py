import numpy as np
import sys
import os

def load_n_col(file, numpy=False):
    data = []
    with open(file) as fp:
        for line in fp:
            data.append(line.strip().split(' '))
    columns = list(zip(*data))
    if numpy:
        columns = [np.array(list(i)) for i in columns]
    else:
        columns = [list(i) for i in columns]
    return columns

if __name__ == "__main__":
    infile = sys.argv[1]
    outfile = sys.argv[2]

    utt0s, utt1s, labs = load_n_col(infile)

    targs = ['1' if l == 'target' else '0' for l in labs]

    with open(outfile, 'w+') as fp:
        for u0, u1, t in zip(utt0s, utt1s, targs):
            line = '{} {} {}\n'.format(t, u0, u1)
            fp.write(line)