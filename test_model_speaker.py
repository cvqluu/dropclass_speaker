import argparse
import configparser
import os
import pickle
import sys
from collections import OrderedDict

import numpy as np
from sklearn.metrics import roc_curve
from sklearn.preprocessing import normalize
from tqdm import tqdm
from math import log10, floor

import torch
import uvloop
from data_io_speaker import SpeakerTestDataset
from kaldiio import ReadHelper
from models_speaker import ETDNN, FTDNN, XTDNN
from utils import SpeakerRecognitionMetrics


def parse_args():
    parser = argparse.ArgumentParser(description='Test SV Model')
    parser.add_argument('--cfg', type=str, default='./configs/example_speaker.cfg')
    parser.add_argument('--best', action='store_true', default=False, help='Use best model')
    parser.add_argument('--checkpoint', type=int, default=-1, # which model to use, overidden by 'best'
                            help='Use model checkpoint, default -1 uses final model')
    args = parser.parse_args()
    assert os.path.isfile(args.cfg)
    return args

def parse_config(args):
    config = configparser.ConfigParser()
    config.read(args.cfg)

    args.test_data_vc1 = config['Datasets'].get('test_vc1')
    print('VC1 dataset: {}'.format(args.test_data_vc1))
    args.test_data_sitw = config['Datasets'].get('test_sitw')
    print('SITW dataset dev: {}'.format(args.test_data_sitw))
    args.test_data_sitw_eval = config['Datasets'].get('test_sitw_eval')
    print('SITW dataset eval: {}'.format(args.test_data_sitw_eval))

    args.no_cuda = config['Hyperparams'].getboolean('no_cuda', fallback=False)

    args.model_type = config['Model'].get('model_type', fallback='XTDNN')
    assert args.model_type in ['XTDNN', 'ETDNN', 'FTDNN']

    args.num_iterations = config['Hyperparams'].getint('num_iterations', fallback=50000)

    args.model_dir = config['Outputs']['model_dir']
    return args


def test(generator, ds_test, device, mindcf=False):
    generator.eval()
    all_embeds = []
    all_utts = []
    num_examples = len(ds_test.veri_utts)

    with torch.no_grad():
        for i in tqdm(range(num_examples)):
            feats, utt = ds_test.__getitem__(i)
            feats = feats.unsqueeze(0).to(device)
            embeds = generator(feats)
            all_embeds.append(embeds.cpu().numpy())
            all_utts.append(utt)

    metric = SpeakerRecognitionMetrics(distance_measure='cosine')
    all_embeds = np.vstack(all_embeds)
    all_embeds = normalize(all_embeds, axis=1)
    all_utts = np.array(all_utts)

    print(all_embeds.shape, len(ds_test.veri_utts))
    utt_embed = OrderedDict({k:v for k, v in zip(all_utts, all_embeds)})

    emb0 = np.array([utt_embed[utt] for utt in ds_test.veri_0])
    emb1 = np.array([utt_embed[utt] for utt in ds_test.veri_1])

    scores = metric.scores_from_pairs(emb0, emb1)
    fpr, tpr, thresholds = roc_curve(1 - ds_test.veri_labs, scores, pos_label=1, drop_intermediate=False)
    eer = metric.eer_from_ers(fpr, tpr)
    generator.train()
    if mindcf:
        mindcf1 = metric.compute_min_dcf(fpr, tpr, thresholds, p_target=0.01)
        mindcf2 = metric.compute_min_dcf(fpr, tpr, thresholds, p_target=0.001)
        return eer, mindcf1, mindcf2
    else:
        return eer

def test_nosil(generator, ds_test, device, mindcf=False):
    generator.eval()
    all_embeds = []
    all_utts = []
    num_examples = len(ds_test.veri_utts)

    with torch.no_grad():
        with ReadHelper('ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:{0}/feats_trimmed.scp ark:- | select-voiced-frames ark:- scp:{0}/vad_trimmed.scp ark:- |'.format(ds_test.data_base_path)) as reader:
            for key, feat in tqdm(reader, total=num_examples):
                if key in ds_test.veri_utts:
                    all_utts.append(key)
                    feats = torch.FloatTensor(feat).unsqueeze(0).to(device)
                    embeds = generator(feats)
                    all_embeds.append(embeds.cpu().numpy())

    metric = SpeakerRecognitionMetrics(distance_measure='cosine')
    all_embeds = np.vstack(all_embeds)
    all_embeds = normalize(all_embeds, axis=1)
    all_utts = np.array(all_utts)

    print(all_embeds.shape, len(ds_test.veri_utts))
    utt_embed = OrderedDict({k:v for k, v in zip(all_utts, all_embeds)})

    emb0 = np.array([utt_embed[utt] for utt in ds_test.veri_0])
    emb1 = np.array([utt_embed[utt] for utt in ds_test.veri_1])

    scores = metric.scores_from_pairs(emb0, emb1)
    fpr, tpr, thresholds = roc_curve(1 - ds_test.veri_labs, scores, pos_label=1, drop_intermediate=False)
    eer = metric.eer_from_ers(fpr, tpr)
    generator.train()
    if mindcf:
        mindcf1 = metric.compute_min_dcf(fpr, tpr, thresholds, p_target=0.01)
        mindcf2 = metric.compute_min_dcf(fpr, tpr, thresholds, p_target=0.001)
        return eer, mindcf1, mindcf2
    else:
        return eer


def round_sig(x, sig=2):
    return round(x, sig-int(floor(log10(abs(x))))-1)

if __name__ == "__main__":
    args = parse_args()
    args = parse_config(args)
    uvloop.install()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print('='*30)
    print('USE_CUDA SET TO: {}'.format(use_cuda))
    print('CUDA AVAILABLE?: {}'.format(torch.cuda.is_available()))
    print('='*30)
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.checkpoint == -1:
        g_path = os.path.join(args.model_dir, "final_g_{}.pt".format(args.num_iterations))
        g_path_sitw = g_path
        g_path_vc1 = g_path
    else:
        g_path = os.path.join(args.model_dir, "g_{}.pt".format(args.checkpoint))
        g_path_sitw = g_path
        g_path_vc1 = g_path

    if args.model_type == 'XTDNN':
        generator = XTDNN()
    if args.model_type == 'ETDNN':
        generator = ETDNN()
    if args.model_type == 'FTDNN':
        generator = FTDNN()
    
    if args.best:
        args.results_pkl = os.path.join(args.model_dir, 'results.p')
        rpkl = pickle.load(open(args.results_pkl, "rb"))
        
        if args.test_data_vc1:
            v1eers = [(rpkl[key]['vc1_eer'], key) for key in rpkl]
            best_vc1_cp = min(v1eers)[1]
            g_path_vc1 = os.path.join(args.model_dir, "g_{}.pt".format(best_vc1_cp))
            print('Best VC1 Model: {}'.format(g_path_vc1))


        if args.test_data_sitw:
            sitweers = [(rpkl[key]['sitw_eer'], key) for key in rpkl]
            best_sitw_cp = min(sitweers)[1]
            g_path_sitw = os.path.join(args.model_dir, "g_{}.pt".format(best_sitw_cp))
            print('Best SITW Model: {}'.format(g_path_sitw))

    if args.test_data_vc1:
        ds_test_vc1 = SpeakerTestDataset(args.test_data_vc1)
        generator.load_state_dict(torch.load(g_path_vc1))
        generator = generator.to(device)
        vc1_eer, vc1_mdcf1, vc1_mdcf2 = test(generator, ds_test_vc1, device, mindcf=True)
        print("="*60)
        print('VC1:: \t EER: {}, minDCF(p=0.01): {}, minDCF(p=0.001): {}'.format(round_sig(vc1_eer, 3), round_sig(vc1_mdcf1, 3), round_sig(vc1_mdcf2, 3)))
        print("="*60)

    if args.test_data_sitw:
        ds_test_sitw = SpeakerTestDataset(args.test_data_sitw)
        generator.load_state_dict(torch.load(g_path_sitw))
        generator = generator.to(device)
        sitw_eer, sitw_mdcf1, sitw_mdcf2 = test_nosil(generator, ds_test_sitw, device, mindcf=True)
        print("="*60)
        print('SITW(dev):: \t EER: {}, minDCF(p=0.01): {}, minDCF(p=0.001): {}'.format(round_sig(sitw_eer, 3), round_sig(sitw_mdcf1, 3), round_sig(sitw_mdcf2, 3)))
        print("="*60)

    if args.test_data_sitw_eval:
        ds_test_sitw_eval = SpeakerTestDataset(args.test_data_sitw_eval)
        generator.load_state_dict(torch.load(g_path_sitw))
        generator = generator.to(device)
        sitw_eval_eer, sitw_eval_mdcf1, sitw_eval_mdcf2 = test_nosil(generator, ds_test_sitw_eval, device, mindcf=True)
        print("="*60)
        print('SITW(eval):: \t EER: {}, minDCF(p=0.01): {}, minDCF(p=0.001): {}'.format(round_sig(sitw_eval_eer, 3), round_sig(sitw_eval_mdcf1, 3), round_sig(sitw_eval_mdcf2, 3)))
        print("="*60)
