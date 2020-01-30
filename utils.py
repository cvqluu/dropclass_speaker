from operator import itemgetter
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import pairwise_distances, roc_curve, accuracy_score
from sklearn.metrics.pairwise import paired_distances
import numpy as np

import torch
import torch.nn as nn

from data_io_speaker import async_map, get_item_train

class SpeakerRecognitionMetrics:
    '''
    This doesn't need to be a class [remnant of old structuring]. 
    To be reworked
    '''

    def __init__(self, distance_measure=None):
        if not distance_measure:
            distance_measure = 'cosine'
        self.distance_measure = distance_measure

    def get_labels_scores(self, vectors, labels):
        labels = labels[:, np.newaxis]
        pair_labels = pairwise_distances(labels, metric='hamming').astype(int).flatten()
        pair_scores = pairwise_distances(vectors, metric=self.distance_measure).flatten()
        return pair_labels, pair_scores

    def get_roc(self, vectors, labels):
        pair_labels, pair_scores = self.get_labels_scores(vectors, labels)
        fpr, tpr, threshold = roc_curve(pair_labels, pair_scores, pos_label=1, drop_intermediate=False)
        # fnr = 1. - tpr
        return fpr, tpr, threshold

    def get_eer(self, vectors, labels):
        fpr, tpr, _ = self.get_roc(vectors, labels)
        # fnr = 1 - self.tpr
        # eer = self.fpr[np.nanargmin(np.absolute((fnr - self.fpr)))]
        eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        return eer

    def eer_from_pairs(self, pair_labels, pair_scores):
        self.fpr, self.tpr, self.thresholds = roc_curve(pair_labels, pair_scores, pos_label=1, drop_intermediate=False)
        fnr = 1 - self.tpr
        eer = self.fpr[np.nanargmin(np.absolute((fnr - self.fpr)))]
        return eer

    def eer_from_ers(self, fpr, tpr):
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        return eer

    def scores_from_pairs(self, vecs0, vecs1):
        return paired_distances(vecs0, vecs1, metric=self.distance_measure)

    def compute_min_dcf(self, fpr, tpr, thresholds, p_target=0.01, c_miss=1, c_fa=1):
        #adapted from compute_min_dcf.py in kaldi sid
        # thresholds, fpr, tpr = list(zip(*sorted(zip(thresholds, fpr, tpr))))
        incr_score_indices = np.argsort(thresholds, kind="mergesort")
        thresholds = thresholds[incr_score_indices]
        fpr = fpr[incr_score_indices]
        tpr = tpr[incr_score_indices]

        fnr = 1. - tpr
        min_c_det = float("inf")
        for i in range(0, len(fnr)):
            c_det = c_miss * fnr[i] * p_target + c_fa * fpr[i] * (1 - p_target)
            if c_det < min_c_det:
                min_c_det = c_det

        c_def = min(c_miss * p_target, c_fa * (1 - p_target))
        min_dcf = min_c_det / c_def
        return min_dcf


def drop_classes(dataset, classifier, num_drop=1000):
    ''' Drop classes randomly from the dataset '''
    assert (num_drop < dataset.num_classes), "Can't drop all classes"
    classes_to_drop = list(np.random.choice(dataset.num_classes, size=num_drop, replace=False))
    classifier.set_ignored_classes(classes_to_drop)
    dataset.set_ignored_classes(classes_to_drop)
    return dataset, classifier

def drop_per_batch(batch_spkrs:list, classifier):
    ''' Drop classes that arent in the batch '''
    batch_spkr_set = list(set(batch_spkrs))
    classifier.set_remaining_classes(batch_spkr_set)
    return classifier


def drop_adapt_onlydata(adapt_probs, train_dataset, num_drop=500):
    '''
    Drop least probable classes from training dataset from aggregated probs
    Only drops data, no classifier
    '''
    assert num_drop < len(adapt_probs)
    top_cls = adapt_probs.argsort()[::-1][:-num_drop]
    train_dataset.set_remaining_classes(top_cls)
    return train_dataset


def drop_adapt(adapt_probs, classifier, train_dataset, num_drop=500):
    '''
    Drop least probable classes from training dataset from aggregated probs
    returns train_dataset and classifier
    '''
    assert len(adapt_probs) == len(classifier.rem_classes)
    assert num_drop < len(adapt_probs)
    top_cls = adapt_probs.argsort()[::-1][:-num_drop]
    top_cls_orig_label = classifier.get_orig_labels(top_cls)
    classifier.set_remaining_classes(top_cls_orig_label)
    train_dataset.set_remaining_classes(top_cls_orig_label)
    return train_dataset, classifier

def drop_adapt_combine(adapt_probs, classifier, train_dataset, num_drop=500):
    '''
    Combine least probable classes from training dataset from aggregated probs into a single class
    returns train_dataset and classifier
    '''
    assert len(adapt_probs) == len(classifier.rem_classes)
    assert num_drop < len(adapt_probs)
    
    top_cls = adapt_probs.argsort()[::-1][:-num_drop]
    top_cls_orig_label = classifier.get_orig_labels(top_cls)
    if classifier.combined_class_label is None:
        # select the least probable remaining class to combine into
        classifier.combined_class_label = top_cls_orig_label[-1]
    classifier.set_remaining_classes_comb(top_cls_orig_label)
    train_dataset.set_remaining_classes_comb(top_cls_orig_label, classifier.combined_class_label)
    return train_dataset, classifier

def drop_adapt_random(classifier, train_dataset, num_drop=500):
    '''
    Drop random classes from training dataset
    returns train_dataset and classifier
    '''
    top_cls = np.random.choice(np.arange(len(classifier.rem_classes)), size=(len(classifier.rem_classes) - num_drop), replace=False)
    orig_label = classifier.get_orig_labels(top_cls)
    classifier.set_remaining_classes(orig_label)
    train_dataset.set_remaining_classes(orig_label)
    return train_dataset, classifier

def aggregate_probs(adapt_dataset, generator, classifier, device, batch_size=500, max_seq_len=350, uniform=False):
    ''' 
    Aggregate class probs across adapt_dataset (type: SpeakerTestDataset)
    return array of shape (num_remaining_classes,)
    '''
    generator.eval()
    classifier.eval()
    with torch.no_grad():
        full_probs = []
        if uniform:
            dloader = adapt_dataset.get_batches_uniform(batch_size=batch_size, max_seq_len=max_seq_len)
        else:
            dloader = adapt_dataset.get_batches(batch_size=batch_size, max_seq_len=max_seq_len)
        for feats in dloader:
            feats = feats.to(device)
            embeds = generator(feats)
            probs = torch.softmax(classifier.forward_drop(embeds), dim=1)
            full_probs.append(probs.cpu().numpy())
        full_probs = np.concatenate(full_probs)
        del feats, embeds, probs
    generator.train()
    classifier.train()
    return np.sum(full_probs, axis=0)

'''
Following methods based on code from 
https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/master/util/utils.py 
'''

def warm_up_lr(batch, num_batch_warm_up, init_lr, optimizer):
    for params in optimizer.param_groups:
        params['lr'] = batch * init_lr / num_batch_warm_up

def schedule_lr(optimizer, factor=0.1):
    for params in optimizer.param_groups:
        params['lr'] *= factor
    print(optimizer)

def separate_resnet_bn_paras(modules):
    all_parameters = modules.parameters()
    paras_only_bn = []

    for pname, p in modules.named_parameters():
        if pname.find('bn') >= 0:
            paras_only_bn.append(p)
            
    paras_only_bn_id = list(map(id, paras_only_bn))
    paras_wo_bn = list(filter(lambda p: id(p) not in paras_only_bn_id, all_parameters))
    
    return paras_only_bn, paras_wo_bn