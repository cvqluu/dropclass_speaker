#!/bin/bash
# Copyright    2017   Johns Hopkins University (Author: Daniel Povey)
#              2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#              2018   Ewald Enzinger
#              2018   David Snyder
# Apache 2.0.
#
# This is an x-vector-based recipe for Speakers in the Wild (SITW).
# It is based on "X-vectors: Robust DNN Embeddings for Speaker Recognition"
# by Snyder et al.  The recipe uses augmented VoxCeleb 1 and 2 for training.
# The augmentation consists of MUSAN noises, music, and babble and
# reverberation from the Room Impulse Response and Noise Database.  Note that
# there are 60 speakers in VoxCeleb 1 that overlap with our evaluation
# dataset, SITW.  The recipe removes those 60 speakers prior to training.
# See ../README.txt for more info on data required.  The results are reported
# in terms of EER and minDCF, and are inline in the comments below.

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc


sitw_root=/PATH/TO/SITW
sitw_dev_trials_core=data/sitw_dev_test/trials/core-core.lst
sitw_eval_trials_core=data/sitw_eval_test/trials/core-core.lst

stage=0

if [ $stage -le 0 ]; then
  # Prepare Speakers in the Wild.  This is our evaluation dataset.
  local/make_sitw.sh $sitw_root data
fi

if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in sitw_eval_enroll sitw_eval_test sitw_dev_enroll sitw_dev_test; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 20 --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 20 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
  done
fi

if [ $stage -le 2 ]; then
    # removes silence frames and performs cmvn
   for name in sitw_eval_enroll sitw_eval_test sitw_dev_enroll sitw_dev_test; do
        local/nnet3/xvector/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd" \
        data/${name} data/${name}_nosil exp/${name}_nosil
    utils/fix_data_dir.sh data/${name}_nosil
  done
fi

if [ $stage -le 3 ]; then
    # combine dev portions
    # copy trials over
    # utils/combine_data.sh data/sitw_dev_combined data/sitw_dev_enroll_nosil data/sitw_dev_test_nosil
    # cp -r data/sitw_dev_test/trials data/sitw_dev_combined
fi