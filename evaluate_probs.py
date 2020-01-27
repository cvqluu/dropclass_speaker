import argparse
import configparser
import glob
import json
import os
import pickle
import random
import shutil
import time
from collections import OrderedDict
from pprint import pprint

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import uvloop
from data_io_speaker import SpeakerDataset, SpeakerTestDataset
from loss_functions import (AdaCos, AMSMLoss, DisturbLabelLoss, L2SoftMax,
                            LabelSmoothingLoss, XVecHead, SoftMax, ArcFace, SphereFace)
from models_speaker import ETDNN, FTDNN, XTDNN
from test_model_speaker import test, test_nosil
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import (SpeakerRecognitionMetrics, aggregate_probs, drop_adapt, drop_adapt_combine,
                   drop_classes, drop_per_batch, schedule_lr)
from train_speaker import parse_config

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate probs of SV model')
    parser.add_argument('--cfg', type=str, default='./configs/example_speaker.cfg')
    args = parser.parse_args()
    assert os.path.isfile(args.cfg)
    args._start_time = time.ctime()
    return args

def evaluate_train(ds_train):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print('='*30)
    print('USE_CUDA SET TO: {}'.format(use_cuda))
    print('CUDA AVAILABLE?: {}'.format(torch.cuda.is_available()))
    print('='*30)
    device = torch.device("cuda" if use_cuda else "cpu")

    writer = SummaryWriter(comment=os.path.basename(args.cfg))
    num_classes = ds_train.num_classes

    if args.model_type == 'XTDNN':
        generator = XTDNN()
    if args.model_type == 'ETDNN':
        generator = ETDNN()
    if args.model_type == 'FTDNN':
        generator = FTDNN()

    if args.loss_type == 'adm':
        classifier = AMSMLoss(512, num_classes)
    if args.loss_type == 'adacos':
        classifier = AdaCos(512, num_classes)
    if args.loss_type == 'l2softmax':
        classifier = L2SoftMax(512, num_classes)
    if args.loss_type == 'softmax':
        classifier = SoftMax(512, num_classes)
    if args.loss_type == 'xvec':
        classifier = XVecHead(512, num_classes)
    if args.loss_type == 'arcface':
        classifier = ArcFace(512, num_classes)
    if args.loss_type == 'sphereface':
        classifier = SphereFace(512, num_classes)


    generator.train()
    classifier.train()

    generator = generator.to(device)
    classifier = classifier.to(device)


    model_str = os.path.join(args.model_dir, '{}_adapt_start.pt')
    for model, modelstr in [(generator, 'g'), (classifier, 'c')]:
        model_path = model_str.format(modelstr)
        assert os.path.isfile(model_path), "Couldn't find [g|c]_adapt_start.pt models in {}".format(args.model_dir)
        model.load_state_dict(torch.load(model_path))

    if args.multi_gpu:
        dpp_generator = nn.DataParallel(generator).to(device)

    data_generator = ds_train.get_batches(batch_size=args.batch_size, max_seq_len=args.max_seq_len)

    if args.use_dropclass:
        classifier.drop()
    else:
        classifier.nodrop()

    if args.model_type == 'FTDNN':
        drop_indexes = np.linspace(0, 1, args.num_iterations)
        drop_sch = ([0, 0.5, 1], [0, 0.5, 0])
        drop_schedule = np.interp(drop_indexes, drop_sch[0], drop_sch[1])
    
    if TEST_SET:
        full_probs = aggregate_probs(ds_adapt, generator, classifier, device,
                                            batch_size=300, max_seq_len=args.max_seq_len, uniform=True)
        np.save(os.path.join(args.model_dir, 'test_probs_uniform.npy'), full_probs)
    else:
        with torch.no_grad():
            for i in tqdm(range(300)):
                full_probs = []
                for feats, iden in ds_train.get_batches_example(num_classes=40, egs_per_cls=42, max_seq_len=350):
                    feats = feats.to(device)
                    embeds = generator(feats)
                    probs = torch.softmax(classifier(embeds), dim=1)
                    full_probs.append(probs.cpu().numpy())
                full_probs = np.sum(np.concatenate(full_probs), axis=0)
                np.save(os.path.join(args.model_dir, 'train_probs_{}.npy'.format(i)), full_probs)
        

if __name__ == "__main__":
    TEST_SET = False
    args = parse_args()
    args = parse_config(args)
    pprint(vars(args))
    uvloop.install()
    ds_train = SpeakerDataset(args.train_data)
    if args.test_data_vc1:
        ds_test_vc1 = SpeakerTestDataset(args.test_data_vc1)
    if args.test_data_sitw:
        ds_test_sitw = SpeakerTestDataset(args.test_data_sitw)
    if args.use_dropadapt:
        if args.ds_adapt == 'vc':
            ds_adapt = ds_test_vc1
        if args.ds_adapt == 'sitw':
            ds_adapt = ds_test_sitw
    evaluate_train(ds_train)
    