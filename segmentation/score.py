from __future__ import division
import caffe
import numpy as np
import os
import sys
from datetime import datetime
sys.path.append('/home/yuzhile/toolboxes/utils')
from transformer import zoom
from h5_tools import read_h5_data
import cv2
#def zoom(feat_map,zoom_factor=1):
#    '''
#    zoom the feature map
#    inputs
#    - feat_map: np.array with(c,h,w)
#    returns
#    - zoom_map
#    '''
#    c,h,w = feat_map.shape
#    out_w = (w-1)*zoom_factor + 1
#    out_h = (h-1)*zoom_factor + 1
#    trans_map = feat_map.transpose((1,2,0))
#    
#    #zoom_map = cv2.resize(trans_map,None,fx=zoom_factor,fy=zoom_factor)
#    zoom_map = cv2.resize(trans_map,(out_h,out_w))
#    return zoom_map.transpose((2,0,1))

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(pd_label, gt_label,zoom_factor=8):
    '''
    given the predict label and gt_label, we compute the accuracy hist
    inputs;
    - pd_label:np.array with (n,c,h,w)
    - gt_label:np.array_with (n,c,h,w)
    returns:
    - hist: the confusin matrix
    '''
    print 'pd shape',pd_label.shape
    print 'gt shape',gt_label.shape 
    n_cl = pd_label.shape[1]
    print 'class number',n_cl
    n_samples = pd_label.shape[0]
    hist = np.zeros((n_cl, n_cl))

    loss = 0
    for i in xrange(n_samples):
        print 'handling sameple {}.'.format(i)
        zoom_pd_label = zoom(pd_label[i],zoom_factor)
        # recover the source label by mask
                # do not need transpose
#        zoom_pd_label = zoom_pd_label.transpose((0,2,1))
        hist += fast_hist(gt_label[i][0].flatten(),
                                zoom_pd_label.argmax(0).flatten(),
                                n_cl)

        # compute the loss as well
    return hist


def do_seg_tests(hf_file, iter,pd_name='fc8_voc12', gt_name='label',label_mask=None, ignore_label=-1,zoom_factor = None):
    '''
    inputs
    - hf_file: h5 file,which saves the extarcted value.
    - pd_name: dataset name, which contains the predict result from parrots,
        is (n,c,h,w)
    - gt_name: dataset name, which contains the groundtruth label,
      with(n,1,h,w)
    - label_mask: dataset name, which contains the label mask, with (n,1,h,w). If it's given, 
        combining with label can recover the turel label
    - ignore_label: the ignore label indicate
    '''
    if label_mask != None:
        assert ignore_label != -1, ' ignore_label is -1, should give ligeal ignnore label'
        print ' recover the true label from label mask and label'
        values = read_h5_data(hf_file,[pd_name,gt_name,label_mask])
        # recover the ture label
        ignore_mask = 1-values[2]
        values[1][ignore_mask.astype(bool)] = ignore_label
    else:
        values = read_h5_data(hf_file,[pd_name,gt_name])
    if zoom_factor == None:
        zoom_factor = 8
    hist = compute_hist(values[0],values[1],zoom_factor)
    # overall accuracy
    acc = np.diag(hist).sum() / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'overall accuracy', acc
    # per-class accuracy
    acc = np.diag(hist) / hist.sum(1)
    print '>>>', datetime.now(), 'Iteration', iter, 'mean accuracy', np.nanmean(acc)
    print 'acc',acc
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print '>>>', datetime.now(), 'Iteration', iter, 'mean IU', np.nanmean(iu)
    print 'iu',iu
    freq = hist.sum(1) / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'fwavacc', \
            (freq[freq > 0] * iu[freq > 0]).sum()
    return hist
