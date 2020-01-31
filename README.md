# DropClass and DropAdapt: Dropping classes for deep speaker representation learning

This repository presents the code for the paper submitted to Speaker Odyssey 2020:  

**"DropClass and DropAdapt: Dropping classes for deep speaker representation learning"**

These methods around the concept of 'dropping' classes periodically throughout training. 

## DropClass

The overall DropClass method is described as follows:

1. Randomly choose a set of classes to drop
2. Exclude training examples belonging to these classes
3. Mask out the corresponding rows in the final affine transform s.t. they do not contribute to the classification softmax
4. Train for a number of iterations on this subset classification task
5. Go back to step 1. and choose a new set of classes to drop and repeat periodically until training is finished.

Other than these steps, training can be performed as standard. This technique can be seen somewhat as dropout on the final affine layer. This method is also shown in the diagram below:

![model_fig](figures/dc_diag.png?raw=true "dc_diag")

**Motivation:** Diversify the learning objective that the network receives during training. Classification of a set number of classes is a static condition, whereas representation learning requires generalization to any number of unseen classes (ideally learning the complete manifold of faces/speakers for example). By varying the set of classes during training, the idea is to encourage discrimination between subsets, and encourage the network to be agnostic about any single classification objective.

Ideally, the resulting network should have some of the benefits shown from multi-task training techniques - also inspired by meta learning approaches.

## DropAdapt

This method is for adapting a trained model to a set of enrolment speakers in an unsupervised manner. Can be described like so:

1. Fully train a model
2. Calculate average class probabilities for the enrollment utterances.
3. Rank these probabilities, drop classes with lob probability from the training data, and the final affine matrix.
4. Train for a number of iterations on the reduced classification problem.
5. Go back to step 2 and repeat.

**Motivation:** Class distribution from train to test is mismatched (See Figure 2 in the paper). This suggests the model is trained on a distribution of speakers which is not seen at test time. Solution: fine tune a model by correctively oversampling the apparently underrepresented classes. The alternative interpretation is that the low probability predicted classes are not important to distinguish between for a chosen test set - thus fine tune on a classification problem which you hypothesize is more important in the test set.

## Requirements

For all: torch, uvloop, scikit-learn

Kaldi, kaldi_io, kaldiio


# Data Preparation

The primary speaker datasets used here are VoxCeleb (1+2) and SITW. The only portion used for training is VoxCeleb 2 (train portion), to ensure no overlap with SITW, in addition to allowing evaluation on the extended and hard VoxCeleb verification lists (VoxCeleb-E, VoxCeleb-H) which are drawn from VoxCeleb 1.

## VoxCeleb

The VoxCeleb data preparation step is nearly identical to the [VoxCeleb recipe](https://github.com/kaldi-asr/kaldi/tree/master/egs/voxceleb/v2/run.sh) in Kaldi. The modified version is found in `scripts/run_vc_dataprep.sh`

To run this, modify the variables at the top of the file to point to the location of VoxCeleb 1, 2, and MUSAN corpora, and then run the following, with `$KALDI_ROOT` referring to the location of your Kaldi installation.

```sh
mv scripts/run_vc_dataprep.sh $KALDI_ROOT/egs/voxceleb/v2/
cd $KALDI_ROOT/egs/voxceleb/v2
source path.sh
./run_vc_dataprep.sh
```

Running this dataprep recipe does the following:
* Makes Kaldi data folder for VoxCeleb 2 (just train portion)
* Makes Kaldi data folder for VoxCeleb 1 (train+test portion)
* Makes MFCCs for each dataset
* Augments the VoxCeleb 2 train portion with MUSAN and RIR_NOISES data, in addition to removing silence frames.
* Removes silence frames from VoxCeleb 1

The Kaldi data folder `$KALDI_ROOT/egs/voxceleb/v2/data/train_combined_no_sil` is the end result of the preparation and will be fed later into this repository's Data Loader. If done correctly, the resulting train dataset should have 5994 speakers (5994 lines in `spk2utt`).

## Speakers In The Wild (SITW)

Similar to the VoxCeleb data prep, the SITW data prep script can be found in `scripts/run_sitw_dataprep.sh`. After making similar modifications to the variables at the top:

```sh
mv scripts/run_sitw_dataprep.sh $KALDI_ROOT/egs/sitw/v2/
cd $KALDI_ROOT/egs/sitw/v2
source path.sh
./run_sitw_dataprep.sh
```

## Additional necessary data prep

For speaker datasets intended to be used as evaluation/test datasets, there must also be a file called `veri_pairs` within these data folders. This is similar to a `trials` file used by Kaldi which lists the pairs of utterances that are to be compared, along with the true label of whether or not they belong to the same speaker.

The format of this `veri_pairs` file is as follows:

```
1 <utterance_a> <utterance_b>
0 <utterance_a> <utterance_c>
```

where 1 indicates both utterances have the same identity, and 0 indicates a different identity. To obtain the primary verification list for VoxCeleb, the following code can be run:

```sh
wget http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt
python scripts/vctxt_to_veripairs.py veri_test.txt $KALDI_ROOT/egs/voxceleb/v2/data/voxceleb1_nosil/veri_pairs
```

For SITW, the dev and eval core-core lists are merged into a kaldi-like data folder for each, which contains the features and `veri_pairs` file needed for evaluation. This is performed from the repository root like so.

```sh
python scripts/trials_to_veri_pairs.py $KALDI_ROOT/egs/sitw/v2/data
```

# Training a Model

The training for face or speaker representation learning are handled within `train_<speaker|face>.py`. Both these scripts are run with a `.cfg` file as input like so:

```sh
python train_speaker.py --cfg config/example_speaker.cfg
```

In order to resume an experiment from an existing checkpoint interval:

```sh
python train_speaker.py --cfg config/example_speaker.cfg --resume-checkpoint 50000
```

When this resuming is possible and the documentation of a `.cfg` files will be described below.

# Configuration files

The overall pipeline for training a speaker/face representation network has two main components, which are referred to in this repo as a `generator` and a `classifier`. The generator is the network which actually produces the embedding:

```
input (Acoustic features/image) -> generator (i.e. TDNNs) -> embedding  
```

Acting on this embedding is the `classifier`:

```
embedding -> classifier (i.e. feed-forward NN projected to num_classes with Softmax) -> class prediction
```

In a classic scenario, this classifier is usually a feed forward network which projects to the number of classes, and trained using Cross Entropy Loss. This repo includes some alternate options such as angular penalty losses (such as CosFace and AdaCos which have been taken from [this repo](https://github.com/4uiiurz1/pytorch-adacos)).

## Speaker

An example `.cfg` file for speaker training is provided below and in `configs/example_speaker.cfg`:

```ini
[Datasets]
train = $KALDI_PATH/egs/voxceleb/v2/data/train_combined_no_sil
test_vc1 = $KALDI_PATH/egs/voxceleb/v2/data/voxceleb1_nosil #OPTIONAL
test_sitw = $KALDI_PATH/egs/sitw/v2/data/sitw_dev_combined #OPTIONAL
```

These are the locations of the datasets. `test_vc1` and/or `test_sitw` are **OPTIONAL** fields. If they are not included in the config file, no evaluation is done during training.

```ini
[Model]
#allowed model_type : ['XTDNN', 'ETDNN' 'FTDNN']
model_type = XTDNN
```

`XTDNN` refers to the original x-vector architecture, and is identical up until the embedding layer. `ETDNN` is the Extended TDNN architecture seen in more recent architectures (also up until the embedding layer). `FTDNN` is the Factorized TDNN x-vector arch. The models can be viewed in `models_speaker.py`

```ini
[Optim]
#allowed loss_type values: ['l2softmax', 'adm', 'adacos', 'xvec']
loss_type = adm
#allowed smooth_type values: ['None', 'disturb', 'uniform']
label_smooth_type = None
label_smooth_prob = 0.1
```

The `loss_type` field dictates the architecture of the `classifier` network as described above. 

* `l2softmax` is a simple projection to the number of classes with both embeddings and weight matrices L2 normalized.
* `adm` is the additive margin Softmax loss/CosFace presented in [REF]
* `adacos` is the adaptive cosine penalty loss presented in [REF]
* `xvec` is a feed forward network with one hidden layer, choosing this option and `XTDNN` as the model type is almost identical to the original x-vector architecture

Two kinds of label smoothing have also been implemented:

* `uniform` which is the standard scenario, in which the true class label is penalized from 1 by the value `label_smooth_prob` and the rest of the probability mass is uniformly distributed to the other labels.
* `disturb` is the label smoothing shown in DisturbLabel [REF], in which each example has a probability of being randomly assigned the wrong label, with probability `label_smooth_prob`.

The implementation of these can be seen in `loss_functions.py`.

```ini
[Hyperparams]
lr = 0.2
batch_size = 500 # must be less than num_classes
max_seq_len = 350
no_cuda = False
seed = 1234
num_iterations = 120000 # total num batches to train
momentum = 0.5
scheduler_steps = [50000, 60000, 70000, 80000, 90000, 100000, 110000]
scheduler_lambda = 0.5 # multiplies lr by this value at each step above
multi_gpu = False # dataparallel
```

Most of these configurable hyper-parameters are fairly self-explanatory.

**Note**: The way the Data Loader is implemented in this repo is to force each batch to have one example from each class. For each batch, `batch_size` number of classes is sampled and then removed from the pool of training classes, and a random single example belonging to each class is chosen to be in the batch. These classes are sampled until there are no remaining classes in the pool, at which point the pool is replenished and the already sampled classes are cycled in again. This is repeated until the end of training.

```ini
[Outputs]
model_dir = exp/example_exp_speaker # place where models are stored
checkpoint_interval = 500 # Interval to save models and also evaluate
```

The `model_dir` is the folder in which models are stored. At every `checkpoint_interval` iterations, both the `generator` and `classifier` will be stored as a `.pt` model inside this folder. Each model has the form: `g_<iterations>.pt`, `c_<iterations>.pt`. This is relevant to the above section of how to resume from a previous checkpoint. For example, to resume from the 1000th iteration, both `g_1000.pt, c_1000.pt` must exist in `model_dir`. 


```ini
[Dropclass]
use_dropclass = False # enables DropClass
its_per_drop = 500 # Iterations to run before randomly selecting new classes to drop
num_drop = 2000 # Number of classes to drop per iteration
drop_per_batch = False # Drops all classes that aren't in the batch (overwrites above 2)
use_dropadapt = # enables DropAdapt
ds_adapt = vc #choose which out of [sitw, vc] you want to calculate $p_{average}$ on
dropadapt_combine = False # Combine all dropped classes into one or not
dropadapt_uniform_agg = False # Calculate p_avg using a uniform dist of speakers
dropadapt_random = False # Drop random classes for dropadapt
dropadapt_onlydata = False # Drop only the low prob classes, leave them in the output layer
```

Here, the options for the central idea of these experiments is configured.

# Results

See paper (to be put on arXiv soon...)

# Issues and TODOs

* Actually test the recipes
* Update example config with all the implemented options

# References

Implementations for AdaCos and some of the other angular penalty loss functions taken from https://github.com/4uiiurz1/pytorch-adacos

Some other simple utils taken from https://github.com/ZhaoJ9014/face.evoLVe.PyTorch 