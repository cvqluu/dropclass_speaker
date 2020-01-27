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

def write_trimmed(col0, col1, col_include, outfile):
    # write 2col with only col0 entries in col_include
    with open(outfile, 'w+') as wp:
        for a, b in zip(col0, col1):
            if a in col_include:
                line = '{} {}\n'.format(a, b)
                wp.write(line)

if __name__ == "__main__":
    datapath = sys.argv[1]

    fscp = os.path.join(datapath, 'feats.scp')
    vscp = os.path.join(datapath, 'vad.scp')
    verip_path = os.path.join(datapath, 'veri_pairs')

    fscp_trim = os.path.join(datapath, 'feats_trimmed.scp')
    vscp_trim = os.path.join(datapath, 'vad_trimmed.scp')

    _, u0, u1 = load_n_col(verip_path)
    trimmed_utts = list(set(u0 + u1))

    f0, f1 = load_n_col(fscp)
    v0, v1 = load_n_col(vscp)

    write_trimmed(f0, f1, trimmed_utts, fscp_trim)
    write_trimmed(v0, v1, trimmed_utts, vscp_trim)
