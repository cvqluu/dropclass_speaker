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

    labs, utt0s, utt1s = load_n_col(infile)

    targs = ['target' if l == '1' else 'nontarget' for l in labs]

    with open(outfile, 'w+') as fp:
        for u0, u1, t in zip(utt0s, utt1s, targs):
            line = '{} {} {}\n'.format(u0, u1, t)
            fp.write(line)