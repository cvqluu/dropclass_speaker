import math
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
AdaCos and Ad margin loss taken from https://github.com/4uiiurz1/pytorch-adacos
'''

class DropClassBase(nn.Module):

    def __init__(self, num_classes):
        '''
        DropClass class which other classifier heads should inherit from

        This is to package the useful wrapper scripts for which classes to include/ignore

        The class has two main modes, called via .drop() and .nodrop(), which sets which method will be
        called by .forward()

        forward_drop defines the ordinary behaviour
        forward_nodrop defines the behaviour in which only the remaining class columns are used
        '''
        super(DropClassBase, self).__init__()
        self.n_classes = num_classes
        self.dropmode = False # Default is the normal behaviour
        self.set_ignored_classes([])
        self.combined_class_label = None

    def forward(self, input, label=None):
        '''
        input: (batch_size, num_features): FloatTensor
        label (optional): (batch_size): LongTensor
        '''
        if self.dropmode:
            if label is not None:
                assert (torch.max(label) < len(self.rem_classes)), 'Contains label out of range of allowed classes: Have they been converted?'
            return self.forward_drop(input, label=label)
        else:
            return self.forward_nodrop(input, label=label)

    def drop(self):
        self.dropmode = True
    
    def nodrop(self):
        self.dropmode = False

    def forward_drop(self, input, label=None):
        raise NotImplementedError

    def forward_nodrop(self, input, label=None):
        raise NotImplementedError

    def set_ignored_classes(self, ignored:list):
        if len(ignored) != 0:
            assert min(ignored) >= 0
            assert max(ignored) < self.n_classes
        self.ignored = sorted(list(set(ignored)))
        self.rem_classes = sorted(set(np.arange(self.n_classes)) - set(ignored))
        self.ldict = OrderedDict({k:v for v, k in enumerate(self.rem_classes)}) #mapping of original label to new index
        self.idict = OrderedDict({k:v for k, v in enumerate(self.rem_classes)}) #mapping of remaining indexes to original label

    def set_remaining_classes(self, remaining:list):
        assert min(remaining) >= 0
        assert max(remaining) < self.n_classes
        self.rem_classes = sorted(set(remaining))
        self.ignored = sorted(set(np.arange(self.n_classes)) - set(remaining))
        self.ldict = OrderedDict({k:v for v, k in enumerate(self.rem_classes)}) #mapping of original label to new index
        self.idict = OrderedDict({k:v for k, v in enumerate(self.rem_classes)}) #mapping of remaining indexes to original label

    def get_mini_labels(self, label:list):
        # convert list of labels into new indexes for ignored classes
        mini_labels = torch.LongTensor(list(map(lambda x: self.ldict[x], label)))
        return mini_labels

    def get_orig_labels(self, label:list):
        # convert list of mini_labels into original class labels
        # assert not self.combined_class_label, 'Combined classes means original labels not recoverable'
        orig_labels = list(map(lambda x: self.idict[x], label))
        return orig_labels

    def set_remaining_classes_comb(self, remaining:list):
        # remaining must not include the combined class
        assert self.combined_class_label is not None, 'combined_class_label has not been set'
        assert min(remaining) >= 0
        assert max(remaining) < self.n_classes
        remaining.append(self.combined_class_label)
        self.rem_classes = sorted(set(remaining))
        self.ignored = sorted(set(np.arange(self.n_classes)) - set(remaining)) # not really ignored, just combined
        self.ldict = OrderedDict({k:v for v, k in enumerate(self.rem_classes)})
        for k in self.ignored:
            self.ldict[k] = self.combined_class_label # set all ignored classes to the combined class label
        self.idict = OrderedDict({k:v for k, v in enumerate(self.rem_classes)}) # not the original mapping for comb classes

class DropAffine(DropClassBase):

    def __init__(self, num_features, num_classes):
        super(DropAffine, self).__init__(num_classes)
        self.fc = nn.Linear(num_features, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()
        
    def forward_nodrop(self, input, label=None):
        W = self.fc.weight
        b = self.fc.bias
        logits = F.linear(input, W, b)
        return logits

    def forward_drop(self, input, label=None):
        W = self.fc.weight[self.rem_classes]
        b = self.fc.bias[self.rem_classes]
        logits = F.linear(input, W, b)
        return logits

class L2SoftMax(DropClassBase):

    def __init__(self, num_features, num_classes):
        super(L2SoftMax, self).__init__(num_classes)
        self.W = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
    
    def forward_nodrop(self, input, label=None):
        x = F.normalize(input)
        W = F.normalize(self.W)
        logits = F.linear(x, W)
        return logits

    def forward_drop(self, input, label=None):
        x = F.normalize(input)
        W = F.normalize(self.W[self.rem_classes])
        logits = F.linear(x, W)
        return logits

class SoftMax(DropClassBase):

    def __init__(self, num_features, num_classes):
        super(SoftMax, self).__init__(num_classes)
        self.W = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
    
    def forward_nodrop(self, input, label=None):
        x = input
        W = self.W
        logits = F.linear(x, W)
        return logits

    def forward_drop(self, input, label=None):
        x = input
        W = self.W[self.rem_classes]
        logits = F.linear(x, W)
        return logits

class XVecHead(DropClassBase):

    def __init__(self, num_features, num_classes, hidden_features=None):
        super(XVecHead, self).__init__(num_classes)
        hidden_features = num_features if not hidden_features else hidden_features
        self.fc_hidden = nn.Linear(num_features, hidden_features)
        self.nl = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(hidden_features)
        self.fc = nn.Linear(hidden_features, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()

    def forward_nodrop(self, input, label=None):
        input = self.fc_hidden(input)
        input = self.nl(input)
        input = self.bn(input)
        W = self.fc.weight
        b = self.fc.bias
        logits = F.linear(input, W, b)
        return logits

    def forward_drop(self, input, label=None):
        input = self.fc_hidden(input)
        input = self.nl(input)
        input = self.bn(input)
        W = self.fc.weight[self.rem_classes]
        b = self.fc.bias[self.rem_classes]
        logits = F.linear(input, W, b)
        return logits


class AMSMLoss(DropClassBase):

    def __init__(self, num_features, num_classes, s=30.0, m=0.4):
        super(AMSMLoss, self).__init__(num_classes)
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)

    def forward_nodrop(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # add margin
        target_logits = logits - self.m
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        return output

    def forward_drop(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W[self.rem_classes])
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # add margin
        target_logits = logits - self.m
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        return output


class SphereFace(DropClassBase):

    def __init__(self, num_features, num_classes, s=30.0, m=1.35):
        super(SphereFace, self).__init__(num_classes)
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)

    def forward_nodrop(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(self.m * theta)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        return output
    
    def forward_drop(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W[self.rem_classes])
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(self.m * theta)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        return output

class ArcFace(DropClassBase):

    def __init__(self, num_features, num_classes, s=30.0, m=0.50):
        super(ArcFace, self).__init__(num_classes)
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)

    def forward_nodrop(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        return output
    
    def forward_drop(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W[self.rem_classes])
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        return output

class AdaCos(DropClassBase):

    def __init__(self, num_features, num_classes, m=0.50):
        super(AdaCos, self).__init__(num_classes)
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)
        self.m = m
        self.W = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)

    def forward_nodrop(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
        output = self.s * logits

        return output

    def forward_drop(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W[self.rem_classes])
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
        output = self.s * logits

        return output




class DisturbLabelLoss(nn.Module):

    def __init__(self, device, disturb_prob=0.1):
        super(DisturbLabelLoss, self).__init__()
        self.disturb_prob = disturb_prob
        self.ce = nn.CrossEntropyLoss()
        self.device = device

    def forward(self, pred, target):
        with torch.no_grad():
            disturb_indexes = torch.rand(len(pred)) < self.disturb_prob
            target[disturb_indexes] = torch.randint(pred.shape[-1], (int(disturb_indexes.sum()),)).to(self.device)
        return self.ce(pred, target)

    
class LabelSmoothingLoss(nn.Module):

    def __init__(self, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.shape[-1] - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))