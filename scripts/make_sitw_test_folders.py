import os
import sys
import numpy as np
from collections import OrderedDict

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

def odict_from_2_col(file, numpy=False):
    col0, col1 = load_n_col(file, numpy=numpy)
    return OrderedDict({c0: c1 for c0, c1 in zip(col0, col1)})

def convert_trial_labs(trial_labs):
    targs = ['1' if l == 'target' else '0' for l in trial_labs]
    return targs

def write_value_from_utt(uttlist, valuedict, file):
    with open(file, 'w+') as fp:
        for utt in uttlist:
            value = valuedict[utt]
            line = '{} {}\n'.format(utt, value)
            fp.write(line)

if __name__ == "__main__":
    base_sitw_data_dir = sys.argv[1]

    for ds in ['dev', 'eval']:
        lstfile = os.path.join(base_sitw_data_dir, 'sitw_{}_test/trials/core-core.lst'.format(ds))
        enrolldir = os.path.join(base_sitw_data_dir, 'sitw_{}_enroll'.format(ds))
        testdir = os.path.join(base_sitw_data_dir, 'sitw_{}_test'.format(ds))

        combined_data_dir = os.path.join(base_sitw_data_dir, 'sitw_{}_combined'.format(ds))
        os.makedirs(combined_data_dir, exist_ok=True)

        u0, u1, lab = load_n_col(lstfile)

        feats_enroll = odict_from_2_col(os.path.join(enrolldir, 'feats.scp'))
        vad_enroll = odict_from_2_col(os.path.join(enrolldir, 'vad.scp'))
        spk2utt_enroll = odict_from_2_col(os.path.join(enrolldir, 'spk2utt'))

        feats_test = odict_from_2_col(os.path.join(testdir, 'feats.scp'))
        vad_test = odict_from_2_col(os.path.join(testdir, 'vad.scp'))

        enroll_utts = [spk2utt_enroll[i] for i in u0]
        test_utts = u1
        targs = convert_trial_labs(lab)

        veri_pairs = os.path.join(combined_data_dir, 'veri_pairs')

        with open(veri_pairs, 'w+') as fp:
            for l, enr, test in zip(targs, enroll_utts, test_utts):
                line = '{} {} {}\n'.format(l, enr, test)
                fp.write(line)

        combined_utts = sorted(list(set(enroll_utts + test_utts)))

        feats = dict(list(feats_enroll.items()) + list(feats_test.items()))
        vad = dict(list(vad_enroll.items()) + list(vad_test.items()))

        feats_trimmed = os.path.join(combined_data_dir, 'feats_trimmed.scp')
        vad_trimmed = os.path.join(combined_data_dir, 'vad_trimmed.scp')

        write_value_from_utt(combined_utts, feats, feats_trimmed)
        write_value_from_utt(combined_utts, vad, vad_trimmed)