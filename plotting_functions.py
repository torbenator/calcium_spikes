import matplotlib.pyplot as plt
import numpy as np
import re



def plot_feature_mat(complete_feature_mat, inds = [0,1000],srate=100):
    """
    tool to visualize feature matrix
    """
    x_axis = np.linspace(inds[0]/srate,inds[1]/srate,inds[1] - inds[0])
    fig = plt.figure(figsize=(20,5))
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(x_axis,complete_feature_mat[:,inds[0]:inds[1]].T);
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ax1.set_title('Feature Matrix of Calcium Imaging Data')
    ax1.set_ylabel('Feature Value')
    ax1.set_xlabel('Time (Seconds)')
    return fig



def plot_dat_seg(calcium,spikes=None, other_feature=None, other_feature_label='other feature', inds=[0,1000], fig_title=None):
    srate=100

    x_axis = np.linspace(inds[0]/srate,inds[1]/srate,inds[1] - inds[0])
    fig = plt.figure(figsize=(20,5))
    ax1 = fig.add_subplot(1,1,1)

    ax1.plot(x_axis,calcium[inds[0]:inds[1]],'y',label='Calcium')
    if not spikes == None:
        ax1.plot(x_axis,spikes[inds[0]:inds[1]],'k',alpha=0.5, label='Spikes')
    if not other_feature == None:
        ax1.plot(x_axis,other_feature[inds[0]:inds[1]],'b.',label='other feature')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Calcium')
    if not fig_title == None:
        ax1.set_title(fig_title)

    ax1.legend()
    return fig



def plot_all_accuracies(all_accuracies,figtitle=None):

    """
    plots (either train or test) output of calculate_all_accuracy
    """

    n_thresh = all_accuracies.shape[1]
    thresh_vals = np.linspace(0,1,n_thresh)
    fig = plt.figure(figsize=(20,5))
    ax1 = fig.add_subplot(1,1,1)
    #ax1.plot(true_positives,false_positives)
    ax1.plot(thresh_vals,all_accuracies[0,:],'b',label='True Positives')
    ax1.plot(thresh_vals,all_accuracies[1,:],'r',label='False Positives')
    ax1.plot(thresh_vals,all_accuracies[2,:],'g',label='True Negatives')
    ax1.plot(thresh_vals,all_accuracies[3,:],'c',label='False Negatives')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Percent Accuracy')
    ax1.legend()
    return fig


def natural_sort(unsorted_features):
    """
    helper function for feature importance helper function
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(unsorted_features, key = alphanum_key)


def organize_feature_importance(model):
    """
    helper function to plot feature importance scores.
    """
    sorted_features = natural_sort(model.feature_names)
    weight_scores = model.get_score(importance_type='weight')
    gain_scores = model.get_score(importance_type='gain')
    cover_scores = model.get_score(importance_type='cover')
    bar_array = np.zeros((3,len(sorted_features)))
    for f in xrange(len(sorted_features)):
        if sorted_features[f] in weight_scores.keys():
            bar_array[0,f] = weight_scores[sorted_features[f]]
        if sorted_features[f] in gain_scores.keys():
            bar_array[1,f] = gain_scores[sorted_features[f]]
        if sorted_features[f] in cover_scores.keys():
            bar_array[2,f] = cover_scores[sorted_features[f]]
    return bar_array, sorted_features


def plot_feature_importance(model):

    """
    plots weight, gain, and cover feature importance scores.
    """

    bar_array, features = organize_feature_importance(model)
    fig = plt.figure(figsize=(20,20))
    ax1 = fig.add_subplot(3,1,1)
    ax1.bar(xrange(len(features)),bar_array[0,:], color='k', align="center")
    ax1.set_xticks(xrange(len(features)))
    ax1.set_xticklabels(features)
    ax1.set_title('Weight Scores')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ax1.set_xlim([-1,bar_array.shape[1]+1])

    ax1 = fig.add_subplot(3,1,2)
    ax1.bar(xrange(len(features)),bar_array[1,:], color='k', align="center")
    ax1.set_xticks(xrange(len(features)))
    ax1.set_xticklabels(features)
    ax1.set_title('Gain Scores')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ax1.set_xlim([-1,bar_array.shape[1]+1])

    ax1 = fig.add_subplot(3,1,3)
    ax1.bar(xrange(len(features)),bar_array[2,:], color='k', align="center")
    ax1.set_xticks(xrange(len(features)))
    ax1.set_xticklabels(features)
    ax1.set_title('Cover Scores')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ax1.set_xlim([-1,bar_array.shape[1]+1])

    return fig