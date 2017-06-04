import numpy as np
from pickle import load
import os
import csv
import scipy.io
import scipy.signal
from sklearn.model_selection import train_test_split


root_dir = '/Users/Torben/Documents/cai-3_dataset/data/'

fnames = ['data.1.train.preprocessed.mat',
          'data.2.train.preprocessed.mat',
          'data.3.train.preprocessed.mat',
          'data.4.train.preprocessed.mat',
          'data.5.train.preprocessed.mat'
         ]
location = [0,0,0,1,0]
brain_state = [0,0,0,1,2]
indicator = [0,0,1,0,1]

calcium_ind=0
spike_ind=2;
fps_ind=3


location_map = ['V1','V1','V1','Retina','V1']
brain_state_map = ['AN','AN','AN','ex_vivo','AWK']
indicator_map = ['OGB','OGB','Gcamp','OGB','Gcamp']

only_these = [
          'data.3.train.preprocessed.mat',
          'data.5.train.preprocessed.mat' # save for validation
         ]

def load_data(root_dir, norm_calcium=False):
    """
    Loads all data. Returns all unprocessed features of data.
    """

    this_calcium=[]
    these_spikes=[]
    this_fps = []
    this_loc = []
    this_bs = []
    this_ind = []
    n_cells = 0
    for f in xrange(len(fnames)):
        if fnames[f] in only_these:
            raw_data = scipy.io.loadmat(root_dir+fnames[f])
            cells = raw_data['data'][0]

            for c in xrange(len(cells)):
                n_cells +=1
                calcium = cells[c][0][0][calcium_ind][0]
                spikes = cells[c][0][0][spike_ind][0]
                fps = cells[c][0][0][fps_ind][0]
                if norm_calcium:
                    this_calcium.append((calcium - np.median(calcium))/np.std(calcium))
                else:
                    this_calcium.append(calcium)
                these_spikes.append(spikes)
                this_fps.append(fps)
                this_loc.append([location[f]]*calcium.shape[0])
                this_bs.append([brain_state[f]]*calcium.shape[0])
                this_ind.append([indicator[f]]*calcium.shape[0])

    print str(n_cells) + ' cells loaded'

    all_calcium = np.concatenate(this_calcium,0)
    all_spikes = np.concatenate(these_spikes,0)
    all_loc = np.concatenate(this_loc,0)
    all_bs = np.concatenate(this_bs,0)
    all_ind = np.concatenate(this_ind,0)

    return all_calcium, all_spikes, all_loc, all_bs, all_ind





def build_offset_features(raw_dat,n_offsets, forward=True, backward=True, verbose=False):
    '''
    Builds a feature matrix of imaging data n points away
    '''

    output_mat = np.zeros(raw_dat.shape);
    if forward:
        forward_mat = np.zeros((n_offsets,raw_dat.shape[0]))

        for n in xrange(1,n_offsets+1):
            if verbose:
                print 'Offsetting by ' + str(n)
            forward_mat[n-1,:] = np.roll(raw_dat,n)

        output_mat=np.vstack([output_mat,forward_mat])

    if backward:
        backward_mat = np.zeros((n_offsets,raw_dat.shape[0]))

        for n in xrange(1,n_offsets+1):
            if verbose:
                print 'Offsetting by -' + str(n)
            backward_mat[n-1,:] = np.roll(raw_dat,-1*n)

        output_mat=np.vstack([output_mat,backward_mat])

    return output_mat[1:,:]


def build_derivative_features(raw_dat):
    pre_diff=np.zeros((1,raw_dat.shape[0]))
    post_diff=np.zeros((1,raw_dat.shape[0]))
    diff2=np.zeros((1,raw_dat.shape[0]))

    pre_diff[0,0:-1] = np.diff(raw_dat)
    post_diff[0,1:] = np.diff(raw_dat)

    diff2[0,1:-1] = np.diff(np.diff(raw_dat))

    return np.vstack([pre_diff,post_diff,diff2])


def build_smoothed_features(raw_dat,n_smooths=5):

    smooth_mat=np.zeros((n_smooths,raw_dat.shape[0]))
    this_medfilt=3
    for n in xrange(n_smooths):
        smooth_mat[n,:]=scipy.signal.medfilt(raw_dat,this_medfilt)
        this_medfilt+=2
    return smooth_mat


def gkern(kernlen=21, nsig=3):
    """
    NOT FUNCTIONAL
    Returns a 1D Gaussian kernel array. If you want to convolve spikes with gaussian.
    """

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))

    return kern1d

def build_feature_mat(all_calcium, all_loc, all_ind, all_bs):
    """
    Calls methods above to organize and process data.

    """

    offset_mat = build_offset_features(all_calcium,20)
    derivative_mat = build_derivative_features(all_calcium)
    smooth_mat = build_smoothed_features(all_calcium,11)
    complete_feature_mat = np.vstack([all_calcium,smooth_mat,offset_mat,derivative_mat,all_loc,all_bs,all_ind])


    return complete_feature_mat


def build_train_test_sets(complete_feature_mat,all_spikes, ceil_spikes=True, subsample=None, test_size=.2):

    if ceil_spikes:
        all_spikes[all_spikes>1]=1

    if subsample:
        sample_inds = np.random.choice(complete_feature_mat.shape[1], subsample)
    else:
        sample_inds = xrange(len(complete_feature_mat.shape[1]))

    calcium = complete_feature_mat[:,sample_inds]
    spikes = all_spikes[sample_inds]

    X_train, X_test, y_train, y_test = train_test_split(calcium.T, spikes.T, test_size=test_size, random_state=16)

    return X_train, X_test, y_train, y_test

