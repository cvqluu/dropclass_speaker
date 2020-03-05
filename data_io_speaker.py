import os
import random
import asyncio
import itertools
from joblib import Parallel, delayed
from multiprocessing import Pool
from collections import OrderedDict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from kaldi_io import read_mat


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

def load_one_tomany(file, numpy=False):
    one = []
    many = []
    with open(file) as fp:
        for line in fp:
            line = line.strip().split(' ', 1)
            one.append(line[0])
            m = line[1].split(' ')
            many.append(np.array(m) if numpy else m)
    if numpy:
        one = np.array(one)
    return one, many

def train_transform(feats, seqlen):
    leeway = feats.shape[0] - seqlen
    startslice = np.random.randint(0, int(leeway)) if leeway > 0  else 0
    feats = feats[startslice:startslice+seqlen] if leeway > 0 else np.pad(feats, [(0,-leeway), (0,0)], 'constant')
    return torch.FloatTensor(feats)

async def get_item_train(instructions):
    fpath = instructions[0]
    seqlen = instructions[1]
    raw_feats = read_mat(fpath)
    feats = train_transform(raw_feats, seqlen)
    return feats

async def get_item_test(filepath):
    raw_feats = read_mat(filepath)
    return torch.FloatTensor(raw_feats)

def async_map(coroutine_func, iterable):
    loop = asyncio.get_event_loop()
    future = asyncio.gather(*(coroutine_func(param) for param in iterable))
    return loop.run_until_complete(future)


class SpeakerDataset(Dataset):

    def __init__(self, data_base_path, asynchr=True, num_workers=3):
        self.data_base_path = data_base_path
        self.num_workers = num_workers
        utt2spk_path = os.path.join(data_base_path, 'utt2spk')
        spk2utt_path = os.path.join(data_base_path, 'spk2utt')
        feats_scp_path = os.path.join(data_base_path, 'feats.scp')

        assert os.path.isfile(utt2spk_path)
        assert os.path.isfile(feats_scp_path)
        assert os.path.isfile(spk2utt_path)

        self.utts, self.uspkrs = load_n_col(utt2spk_path)
        self.utt_fpath_dict = odict_from_2_col(feats_scp_path)
        
        self.label_enc = LabelEncoder()

        self.spkrs, self.spkutts = load_one_tomany(spk2utt_path)
        self.spkrs = self.label_enc.fit_transform(self.spkrs)
        self.spk_utt_dict = OrderedDict({k:v for k,v in zip(self.spkrs, self.spkutts)})
        
        self.uspkrs = self.label_enc.transform(self.uspkrs)
        self.utt_spkr_dict = OrderedDict({k:v for k,v in zip(self.utts, self.uspkrs)})

        self.utt_list = list(self.utt_fpath_dict.keys())
        self.first_batch = True

        self.num_classes = len(self.label_enc.classes_)
        self.asynchr = asynchr

        self.allowed_classes = self.spkrs.copy() # classes the data can be drawn from
        self.idpool = self.allowed_classes.copy()
        self.ignored = []


    def __len__(self):
        return len(self.utt_list)

    @staticmethod
    def get_item(instructions):
        fpath = instructions[0]
        seqlen = instructions[1]
        feats = read_mat(fpath)
        feats = train_transform(feats, seqlen)
        return feats

    def get_batches(self, batch_size=256, max_seq_len=400):
        # with Parallel(n_jobs=self.num_workers) as parallel:
        assert batch_size < len(self.allowed_classes) #Metric learning assumption large num classes
        lens = [max_seq_len for _ in range(batch_size)]
        while True:
            if len(self.idpool) <= batch_size:
                batch_ids = np.array(self.idpool)
                self.idpool = self.allowed_classes.copy()
                rem_ids = np.random.choice(self.idpool, size=batch_size-len(batch_ids), replace=False)
                batch_ids = np.concatenate([batch_ids, rem_ids])
                self.idpool = list(set(self.idpool) - set(rem_ids))
            else:
                batch_ids = np.random.choice(self.idpool, size=batch_size, replace=False)
                self.idpool = list(set(self.idpool) - set(batch_ids))

            batch_fpaths = []
            for i in batch_ids:
                utt = np.random.choice(self.spk_utt_dict[i])
                batch_fpaths.append(self.utt_fpath_dict[utt])

            if self.asynchr:
                batch_feats = async_map(get_item_train, zip(batch_fpaths, lens))
            else:
                batch_feats = [self.get_item(a) for a in zip(batch_fpaths, lens)]
            # batch_feats = parallel(delayed(self.get_item)(a) for a in zip(batch_fpaths, lens))

            yield torch.stack(batch_feats), list(batch_ids)
    
    def set_remaining_classes(self, remaining:list):
        self.allowed_classes = sorted(list(set(remaining)))
        self.ignored = sorted(set(np.arange(self.num_classes)) - set(remaining))
        self.idpool = self.allowed_classes.copy()

    def set_ignored_classes(self, ignored:list):
        self.ignored = sorted(list(set(ignored)))
        self.allowed_classes = sorted(set(np.arange(self.num_classes)) - set(ignored))
        self.idpool = self.allowed_classes.copy()

    def set_remaining_classes_comb(self, remaining:list, combined_class_label):
        remaining.append(combined_class_label)
        self.allowed_classes = sorted(list(set(remaining)))
        self.ignored = sorted(set(np.arange(self.num_classes)) - set(remaining))
        self.idpool = self.allowed_classes.copy()        
        for ig in self.ignored:
            # modify self.spk_utt_dict[combined_class_label] to contain all the ignored ids utterances
            self.spk_utt_dict[combined_class_label] += self.spk_utt_dict[ig]
        self.spk_utt_dict[combined_class_label] = list(set(self.spk_utt_dict[combined_class_label]))

    def get_batches_example(self, num_classes=40, egs_per_cls=42, max_seq_len=350):
        # this is only for the plot in the paper
        batch_size = num_classes
        lens = [max_seq_len for _ in range(batch_size)]
        self.idpool = np.random.choice(self.allowed_classes, size=num_classes, replace=False)
        for i in range(egs_per_cls):
            if len(self.idpool) <= batch_size:
                batch_ids = np.array(self.idpool)
                self.idpool = self.allowed_classes.copy()
                rem_ids = np.random.choice(self.idpool, size=batch_size-len(batch_ids), replace=False)
                batch_ids = np.concatenate([batch_ids, rem_ids])
                self.idpool = list(set(self.idpool) - set(rem_ids))
            else:
                batch_ids = np.random.choice(self.idpool, size=batch_size, replace=False)
                self.idpool = list(set(self.idpool) - set(batch_ids))

            batch_fpaths = []
            for i in batch_ids:
                utt = np.random.choice(self.spk_utt_dict[i])
                batch_fpaths.append(self.utt_fpath_dict[utt])

            if self.asynchr:
                batch_feats = async_map(get_item_train, zip(batch_fpaths, lens))
            else:
                batch_feats = [self.get_item(a) for a in zip(batch_fpaths, lens)]
            # batch_feats = parallel(delayed(self.get_item)(a) for a in zip(batch_fpaths, lens))

            yield torch.stack(batch_feats), list(batch_ids)

class SpeakerTestDataset(Dataset):
    
    def __init__(self, data_base_path, asynchr=True):
        self.data_base_path = data_base_path
        feats_scp_path = os.path.join(data_base_path, 'feats.scp')
        verilist_path = os.path.join(data_base_path, 'veri_pairs')
        utt2spk_path = os.path.join(data_base_path, 'utt2spk')


        assert os.path.isfile(verilist_path)

        if os.path.isfile(feats_scp_path):
            self.utt_fpath_dict = odict_from_2_col(feats_scp_path)

        self.veri_labs, self.veri_0, self.veri_1 = load_n_col(verilist_path, numpy=True)
        self.utt2spk_dict = odict_from_2_col(utt2spk_path)
        self.enrol_utts = list(set(self.veri_0))
        self.veri_utts = list(set(np.concatenate([self.veri_0, self.veri_1])))
        self.veri_labs = self.veri_labs.astype(int)
        self.init_uniform()

    def init_uniform(self):
        # current undersampling, TODO: oversample option instead.
        self.enrol_uspkrs = [self.utt2spk_dict[i] for i in self.enrol_utts]
        self.utts_per_espk = Counter(self.enrol_uspkrs)
        self.min_utts_spk = self.utts_per_espk[min(self.utts_per_espk, key=self.utts_per_espk.get)]
        counts = {}
        self.uniform_enrol_utts = []
        for utt, spk in zip(self.enrol_utts, self.enrol_uspkrs):
            if spk not in counts:
                counts[spk] = 1
                self.uniform_enrol_utts.append(utt)
            else:
                if counts[spk] >= self.min_utts_spk:
                    continue
                else:
                    counts[spk] += 1
                    self.uniform_enrol_utts.append(utt)

    def __len__(self):
        return len(self.veri_labs)

    def __getitem__(self, idx):
        utt = self.veri_utts[idx]
        fpath = self.utt_fpath_dict[utt]
        feats = torch.FloatTensor(read_mat(fpath))
        return feats, utt

    def get_batches(self, batch_size=256, max_seq_len=400):
        lens = [max_seq_len for _ in range(batch_size)]
        num_batches = int(np.ceil(len(self.enrol_utts) / batch_size))
        for i in range(num_batches):
            start = i * batch_size
            batch_utts = self.enrol_utts[start:start+batch_size]
            batch_fpaths = [self.utt_fpath_dict[utt] for utt in batch_utts]
            batch_feats = async_map(get_item_train, zip(batch_fpaths, lens))
            yield torch.stack(batch_feats)
    
    def get_batches_uniform(self, batch_size=256, max_seq_len=400):
        lens = [max_seq_len for _ in range(batch_size)]
        num_batches = int(np.ceil(len(self.uniform_enrol_utts) / batch_size))
        for i in range(num_batches):
            start = i * batch_size
            batch_utts = self.uniform_enrol_utts[start:start+batch_size]
            batch_fpaths = [self.utt_fpath_dict[utt] for utt in batch_utts]
            batch_feats = async_map(get_item_train, zip(batch_fpaths, lens))
            yield torch.stack(batch_feats)

class SpeakerEvalDataset(Dataset):
    
    def __init__(self, data_base_path, asynchr=True):
        self.data_base_path = data_base_path
        feats_scp_path = os.path.join(data_base_path, 'feats.scp')
        model_enrollment_path = os.path.join(data_base_path, 'model_enrollment.txt')
        eval_veri_pairs_path = os.path.join(data_base_path, 'trials.txt')

        if os.path.isfile(feats_scp_path):
            self.utt_fpath_dict = odict_from_2_col(feats_scp_path)

        self.models, self.enr_utts = load_one_tomany(model_enrollment_path)
        if self.models[0] == 'model-id':
            self.models, self.enr_utts = self.models[1:], self.enr_utts[1:]
        assert len(self.models) == len(set(self.models))
        self.model_enr_utt_dict = OrderedDict({k:v for k,v in zip(self.models, self.enr_utts)})

        self.all_enrol_utts = list(itertools.chain.from_iterable(self.enr_utts))

        self.models_eval, self.eval_utts = load_n_col(eval_veri_pairs_path)
        if self.models_eval[0] == 'model-id':
            self.models_eval, self.eval_utts = self.models_eval[1:], self.eval_utts[1:]
        
        assert set(self.models_eval) == set(self.models)

        self.model_eval_utt_dict = OrderedDict({})
        for m, ev_utt in zip(self.models_eval, self.eval_utts):
            if m not in self.model_eval_utt_dict:
                self.model_eval_utt_dict[m] = []
            self.model_eval_utt_dict[m].append(ev_utt)
        
        self.models = list(self.model_eval_utt_dict.keys())
        
        
    def __len__(self):
        return len(self.models)


    def __getitem__(self, idx):
        '''
        Returns enrolment utterances and eval utterances for a specific model
        '''
        model = self.models[idx]
        enrol_utts = self.model_enr_utt_dict[model]
        eval_utts = self.model_eval_utt_dict[model]

        enrol_feats = async_map(get_item_test, enrol_utts)
        eval_feats = async_map(get_item_test, eval_utts)

        return enrol_utts, enrol_feats, eval_utts, eval_feats
