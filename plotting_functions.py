import matplotlib.pyplot as plt
import numpy as np



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
        ax1.plot(x_axis,spikes[inds[0]:inds[1]],'k',label='Spikes')
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