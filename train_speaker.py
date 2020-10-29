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
from utils import (SpeakerRecognitionMetrics, aggregate_probs, drop_adapt, drop_adapt_combine, drop_adapt_random, drop_adapt_onlydata,
                   drop_classes, drop_per_batch, schedule_lr)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def parse_args():
    parser = argparse.ArgumentParser(description='Train SV model')
    parser.add_argument('--cfg', type=str, default='./configs/example_speaker.cfg')
    parser.add_argument('--resume-checkpoint', type=int, default=0)
    args = parser.parse_args()
    assert os.path.isfile(args.cfg)
    args._start_time = time.ctime()
    return args

def parse_config(args):
    config = configparser.ConfigParser()
    config.read(args.cfg)

    args.train_data = config['Datasets']['train']
    assert args.train_data
    args.test_data_vc1 = config['Datasets'].get('test_vc1')
    print('VC1 dataset: {}'.format(args.test_data_vc1))
    args.test_data_sitw = config['Datasets'].get('test_sitw')
    print('SITW dataset: {}'.format(args.test_data_sitw))

    args.model_type = config['Model'].get('model_type', fallback='XTDNN')
    assert args.model_type in ['XTDNN', 'ETDNN', 'FTDNN']

    args.loss_type = config['Optim'].get('loss_type', fallback='adacos')
    assert args.loss_type in ['l2softmax', 'adm', 'adacos', 'xvec', 'arcface', 'sphereface', 'softmax']
    args.label_smooth_type = config['Optim'].get('label_smooth_type', fallback='None')
    assert args.label_smooth_type in ['None', 'disturb', 'uniform']
    args.label_smooth_prob = config['Optim'].getfloat('label_smooth_prob', fallback=0.1)

    args.lr = config['Hyperparams'].getfloat('lr', fallback=0.2)
    args.batch_size = config['Hyperparams'].getint('batch_size', fallback=400)
    args.max_seq_len = config['Hyperparams'].getint('max_seq_len', fallback=400)
    args.no_cuda = config['Hyperparams'].getboolean('no_cuda', fallback=False)
    args.seed = config['Hyperparams'].getint('seed', fallback=123)
    args.num_iterations = config['Hyperparams'].getint('num_iterations', fallback=50000)
    args.momentum = config['Hyperparams'].getfloat('momentum', fallback=0.9)
    args.scheduler_steps = np.array(json.loads(config.get('Hyperparams', 'scheduler_steps'))).astype(int)
    args.scheduler_lambda = config['Hyperparams'].getfloat('scheduler_lambda', fallback=0.5)
    args.multi_gpu = config['Hyperparams'].getboolean('multi_gpu', fallback=False)
    args.classifier_lr_mult = config['Hyperparams'].getfloat('classifier_lr_mult', fallback=1.)

    args.model_dir = config['Outputs']['model_dir']
    args.log_file = os.path.join(args.model_dir, 'train.log')
    args.checkpoint_interval = config['Outputs'].getint('checkpoint_interval')
    args.results_pkl = os.path.join(args.model_dir, 'results.p')

    args.use_dropclass = config['Dropclass'].getboolean('use_dropclass', fallback=False)
    args.its_per_drop = config['Dropclass'].getint('its_per_drop', fallback=1000)
    args.num_drop = config['Dropclass'].getint('num_drop', fallback=2000)
    args.drop_per_batch = config['Dropclass'].getboolean('drop_per_batch', fallback=False)
    args.reset_affine_each_it = config['Dropclass'].getboolean('reset_affine_each_it', fallback=False)

    args.use_dropadapt = config['Dropclass'].getboolean('use_dropadapt', fallback=False)
    args.ds_adapt = config['Dropclass'].get('ds_adapt', fallback='vc')
    assert args.ds_adapt in ['vc', 'sitw']
    args.dropadapt_combine = config['Dropclass'].getboolean('dropadapt_combine', fallback=True)
    args.dropadapt_uniform_agg = config['Dropclass'].getboolean('dropadapt_uniform_agg', fallback=False)
    args.dropadapt_random = config['Dropclass'].getboolean('dropadapt_random', fallback=False)
    args.dropadapt_onlydata = config['Dropclass'].getboolean('dropadapt_onlydata', fallback=False)
    return args


def train(ds_train, ds_adapt, args):
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

    if args.resume_checkpoint != 0:
        model_str = os.path.join(args.model_dir, '{}_{}.pt')
        for model, modelstr in [(generator, 'g'), (classifier, 'c')]:
            model.load_state_dict(torch.load(model_str.format(modelstr, args.resume_checkpoint)))

    if args.use_dropadapt and args.use_dropclass:
        model_str = os.path.join(args.model_dir, '{}_adapt_start.pt')
        for model, modelstr in [(generator, 'g'), (classifier, 'c')]:
            model_path = model_str.format(modelstr)
            assert os.path.isfile(model_path), "Couldn't find [g|c]_adapt_start.pt models in {}".format(args.model_dir)
            model.load_state_dict(torch.load(model_path))

    optimizer = torch.optim.SGD([{'params': generator.parameters(), 'lr': args.lr}, 
                                    {'params': classifier.parameters(), 'lr': args.lr * args.classifier_lr_mult},
                                ],
                                    momentum=args.momentum)

    if args.label_smooth_type == 'None':
        criterion = nn.CrossEntropyLoss()
    if args.label_smooth_type == 'disturb':
        criterion = DisturbLabelLoss(device, disturb_prob=args.label_smooth_prob)
    if args.label_smooth_type == 'uniform':
        criterion = LabelSmoothingLoss(smoothing=args.label_smooth_prob)

    iterations = 0

    total_loss = 0
    running_loss = [np.nan for _ in range(500)]

    best_vc1_eer = (-1, 1.0)
    best_sitw_eer = (-1, 1.0)

    if os.path.isfile(args.results_pkl):
        rpkl = pickle.load(open(args.results_pkl, "rb"))
        if args.test_data_vc1:
            v1eers = [(rpkl[key]['vc1_eer'], i) for i, key in enumerate(rpkl)]
            bestvc1 = min(v1eers)
            best_vc1_eer = (bestvc1[1], bestvc1[0])
        if args.test_data_sitw:
            sitweers = [(rpkl[key]['sitw_eer'], i) for i, key in enumerate(rpkl)]
            bestsitw = min(sitweers)
            best_sitw_eer = (bestsitw[1], bestsitw[0])
    else:
        rpkl = OrderedDict({})

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

    for iterations in range(1, args.num_iterations + 1):
        if iterations > args.num_iterations:
            break
        if iterations in args.scheduler_steps:
            schedule_lr(optimizer, factor=args.scheduler_lambda)
        if iterations <= args.resume_checkpoint:
            print('Skipping iteration {}'.format(iterations))
            print('Skipping iteration {}'.format(iterations), file=open(args.log_file, "a"))
            continue

        if args.model_type == 'FTDNN':
            generator.set_dropout_alpha(drop_schedule[iterations-1])

        if args.use_dropclass and not args.drop_per_batch and not args.use_dropadapt:
            if iterations % args.its_per_drop == 0 or iterations == 1:
                ds_train, classifier = drop_classes(ds_train, classifier, num_drop=args.num_drop)
                if args.reset_affine_each_it:
                    classifier.reset_parameters()

        if args.use_dropclass and args.use_dropadapt:
            if iterations % args.its_per_drop == 0 or iterations == 2:
                # this feeds one batch in to 'reserve' CUDA memory, having iterations == 1 fails
                if args.dropadapt_random:
                    ds_train, classifier = drop_adapt_random(classifier, ds_train, num_drop=args.num_drop)
                else:
                    with torch.no_grad():
                        print('------ [{}/{}] classes remaining'.format(len(classifier.rem_classes), classifier.n_classes))
                        print('------ Aggregating training class probs on {}'.format(args.ds_adapt))
                        full_probs = aggregate_probs(ds_adapt, generator, classifier, device,
                                                        batch_size=300, max_seq_len=args.max_seq_len, uniform=args.dropadapt_uniform_agg)
                        np.save(os.path.join(args.model_dir, 'probs_{}.npy'.format(iterations)), full_probs)
                        print('------ Dropping ~{} more classes from the next {} training steps'.format(args.num_drop, args.its_per_drop))
                        if args.dropadapt_combine:
                            print('------ Combining least probable classes into one...')
                            ds_train, classifier = drop_adapt_combine(full_probs, classifier, ds_train, num_drop=args.num_drop)
                        else:
                            if args.dropadapt_onlydata:
                                ds_train = drop_adapt_onlydata(full_probs, ds_train, num_drop=args.num_drop)
                            else:
                                ds_train, classifier = drop_adapt(full_probs, classifier, ds_train, num_drop=args.num_drop)
                        print('------ [{}/{}] classes remaining'.format(len(classifier.rem_classes), classifier.n_classes))
                        np.save(os.path.join(args.model_dir, 'remclasses_{}.npy'.format(iterations)), classifier.rem_classes)
                        del full_probs

        feats, iden = next(data_generator)

        if args.drop_per_batch and args.use_dropclass:
            classifier = drop_per_batch(iden, classifier)
            if args.reset_affine_each_it:
                classifier.reset_parameters()

        feats = feats.to(device)

        if args.use_dropclass:
            iden = classifier.get_mini_labels(iden).to(device)
        else:
            iden = torch.LongTensor(iden).to(device)

        if args.multi_gpu:
            embeds = dpp_generator(feats)
        else:
            embeds = generator(feats)

        if args.loss_type == 'softmax':
            preds = classifier(embeds)
        else:
            preds = classifier(embeds, iden)

        loss = criterion(preds, iden)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.model_type == 'FTDNN':
            generator.step_ftdnn_layers()

        running_loss.pop(0)
        running_loss.append(loss.item())
        rmean_loss = np.nanmean(np.array(running_loss))

        if iterations % 10 == 0:
            msg = "{}: {}: [{}/{}] \t C-Loss:{:.4f}, AvgLoss:{:.4f}, lr: {}, bs: {}".format(
                                                                                args.model_dir,
                                                                                time.ctime(),
                                                                                iterations,
                                                                                args.num_iterations,
                                                                                loss.item(),
                                                                                rmean_loss, 
                                                                                get_lr(optimizer), 
                                                                                len(feats))
            print(msg)
            print(msg, file=open(args.log_file, "a"))

        writer.add_scalar('class loss', loss.item(), iterations)
        writer.add_scalar('Avg loss', rmean_loss, iterations)

        if iterations % args.checkpoint_interval == 0:
            for model, modelstr in [(generator, 'g'), (classifier, 'c')]:
                model.eval().cpu()
                cp_filename = "{}_{}.pt".format(modelstr, iterations)
                cp_model_path = os.path.join(args.model_dir, cp_filename)
                torch.save(model.state_dict(), cp_model_path)
                model.to(device).train()

            rpkl[iterations] = {}

            if args.test_data_vc1:
                vc1_eer = test(generator, ds_test_vc1, device)
                print('EER on VoxCeleb1: {}'.format(vc1_eer))
                print('EER on Voxceleb1: {}'.format(vc1_eer), file=open(args.log_file, "a"))
                writer.add_scalar('vc1_eer', vc1_eer, iterations)
                if vc1_eer < best_vc1_eer[1]:
                    best_vc1_eer = (iterations, vc1_eer)
                print('Best VC1 EER: {}'.format(best_vc1_eer))
                print('Best VC1 EER: {}'.format(best_vc1_eer), file=open(args.log_file, "a"))
                rpkl[iterations]['vc1_eer'] = vc1_eer

            if args.test_data_sitw:
                sitw_eer = test_nosil(generator, ds_test_sitw, device)
                print('EER on SITW(DEV): {}'.format(sitw_eer))
                print('EER on SITW(DEV): {}'.format(sitw_eer), file=open(args.log_file, "a"))
                writer.add_scalar('sitw_eer', sitw_eer, iterations)
                if sitw_eer < best_sitw_eer[1]:
                    best_sitw_eer = (iterations, sitw_eer)
                print('Best SITW(DEV) EER: {}'.format(best_sitw_eer))
                print('Best SITW(DEV) EER: {}'.format(best_sitw_eer), file=open(args.log_file, "a"))
                rpkl[iterations]['sitw_eer'] = sitw_eer
            
            pickle.dump(rpkl, open(args.results_pkl, "wb"))

    # ---- Final model saving -----
    for model, modelstr in [(generator, 'g'), (classifier, 'c')]:
        model.eval().cpu()
        cp_filename = "final_{}_{}.pt".format(modelstr, iterations)
        cp_model_path = os.path.join(args.model_dir, cp_filename)
        torch.save(model.state_dict(), cp_model_path)

if __name__ == "__main__":
    args = parse_args()
    args = parse_config(args)
    os.makedirs(args.model_dir, exist_ok=True)
    if args.resume_checkpoint == 0:
        shutil.copy(args.cfg, os.path.join(args.model_dir, 'experiment_settings.cfg'))
    else:
        shutil.copy(args.cfg, os.path.join(args.model_dir, 'experiment_settings_resume.cfg'))
    if os.path.exists(args.log_file):
        os.remove(args.log_file)
    pprint(vars(args))
    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    uvloop.install()
    ds_train = SpeakerDataset(args.train_data)
    if args.test_data_vc1:
        ds_test_vc1 = SpeakerTestDataset(args.test_data_vc1)
    if args.test_data_sitw:
        ds_test_sitw = SpeakerTestDataset(args.test_data_sitw)
    if args.use_dropadapt:
        assert args.use_dropclass
        if args.ds_adapt == 'vc':
            ds_adapt = ds_test_vc1
        if args.ds_adapt == 'sitw':
            ds_adapt = ds_test_sitw
    if args.use_dropclass:
        assert not (args.use_dropadapt and args.drop_per_batch)
    train(ds_train, ds_adapt, args)
