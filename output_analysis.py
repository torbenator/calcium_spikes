import numpy as np


def calculate_accuracy_metrics(y_train,y_test, Yr, Yt, myceil=0.5, verbose=True):

    """
    Calculates accuracy metrics.

    """
    if np.ndim(Yr)>1:
        Yr = np.argmax(Yr,1)
        Yt = np.argmax(Yt,1)
    else:
        Yr = Yr>myceil
        Yt = Yt>myceil


    train_corr = np.corrcoef(Yr,y_train)[0,1]
    test_corr = np.corrcoef(Yt,y_test)[0,1]

    spike_bins = list(set(y_train))
    true_pos_train = np.zeros(len(spike_bins))
    false_pos_train = np.zeros(len(spike_bins))
    true_neg_train = np.zeros(len(spike_bins))
    false_neg_train = np.zeros(len(spike_bins))
    n_train = np.zeros(len(spike_bins))

    true_pos_test = np.zeros(len(spike_bins))
    false_pos_test = np.zeros(len(spike_bins))
    true_neg_test = np.zeros(len(spike_bins))
    false_neg_test = np.zeros(len(spike_bins))
    n_test = np.zeros(len(spike_bins))

    train_scores = np.zeros((4, len(spike_bins)))
    test_scores = np.zeros((4, len(spike_bins)))

    for f in spike_bins:
        true_pos_train[f] = len(np.intersect1d(np.where(Yr == f)[0],np.where(y_train == f)[0]))
        false_pos_train[f] = len(np.intersect1d(np.where(Yr == f)[0],np.where(y_train != f)[0]))
        true_neg_train[f] = len(np.intersect1d(np.where(Yr == 0)[0],np.where(y_train == 0)[0]))
        false_neg_train[f] = len(np.intersect1d(np.where(Yr != 0)[0],np.where(y_train == 0)[0]))
        n_train[f] = float(len(np.where(y_train == f)[0]))

        train_scores[0,f] = true_pos_train[f]/n_train[f]
        train_scores[1,f] = false_pos_train[f]/(len(y_train)-n_train[f])
        train_scores[2,f] = true_neg_train[f]/len(y_train)
        train_scores[3,f] = false_neg_train[f]/(len(y_train)-n_train[f])

        true_pos_test[f] = len(np.intersect1d(np.where(Yt == f)[0],np.where(y_test == f)[0]))
        false_pos_test[f] = len(np.intersect1d(np.where(Yt == f)[0],np.where(y_test != f)[0]))
        true_neg_test[f] = len(np.intersect1d(np.where(Yt == 0)[0],np.where(y_test == 0)[0]))
        false_neg_test[f] = len(np.intersect1d(np.where(Yt != 0)[0],np.where(y_test == 0)[0]))
        n_test[f] = float(len(np.where(y_test == f)[0]))

        test_scores[0,f] = true_pos_test[f]/n_test[f]
        test_scores[1,f] = false_pos_test[f]/(len(y_test)-n_test[f])
        test_scores[2,f] = true_neg_test[f]/len(y_test)
        test_scores[3,f] = false_neg_test[f]/(len(y_test)-n_test[f])

    if verbose:
        print 'Train correlation: ' + str(np.round(train_corr,3))
        print 'Test correlation: ' + str(np.round(test_corr,3))
        print '='*10

        print 'Training Accuracy:'
        for i in xrange(len(n_train)):
            print str(i) + ' spikes: ' + str(int(true_pos_train[i])) + ' True Positive (' + str(np.round(train_scores[0,i],4)) + '%). ' + str(int(false_pos_train[i])) + ' False Positive (' + str(np.round(train_scores[1,i],4)) + '%). ' + str(int(true_neg_train[i])) + ' True Negative (' + str(np.round(train_scores[2,i],4)) + '%). ' + str(int(false_neg_train[i])) + ' False Negative (' + str(np.round(train_scores[3,i],4)) + '%). ' + str(int(n_train[i])) + ' Total. '

        print 'Testing Accuracy:'
        for i in xrange(len(n_test)):
            print str(i) + ' spikes: ' + str(int(true_pos_test[i])) + ' True Positive (' + str(np.round(test_scores[0,i],4)) + '%). ' + str(int(false_pos_test[i])) + ' False Positive (' + str(np.round(test_scores[1,i],4)) + '%). ' + str(int(true_neg_test[i])) + ' True Negative (' + str(np.round(test_scores[2,i],4)) + '%). ' + str(int(false_neg_test[i])) + ' False Negative (' + str(np.round(test_scores[3,i],4)) + '%). ' + str(int(n_test[i])) + ' Total. '

    train_metrics = [true_pos_train,false_pos_train,n_train, true_neg_train, false_neg_train, train_corr]
    test_metrics = [true_pos_test,false_pos_test,n_test, true_neg_test, false_neg_test, test_corr]
    return train_metrics, test_metrics, [train_scores, test_scores]


def calculate_all_accuracy(y_train, y_test, Yr, Yt, n_thresh=100):
    """
    Assumes binary.
    Calculates the accuracy scores for n_thresh thresholds.

    """

    thresh_vals = np.linspace(0,1,n_thresh)

    train_measures = np.zeros((4,len(thresh_vals)))
    test_measures = np.zeros((4,len(thresh_vals)))


    for t in xrange(len(thresh_vals)):
        train_metrics, test_metrics, [train_scores, test_scores] = calculate_accuracy_metrics(y_train,y_test,
                                                                 Yr, Yt,
                                                                 myceil=thresh_vals[t], verbose=False)

        train_measures[:,t] = train_scores[:,1]
        test_measures[:,t] = test_scores[:,1]
    return train_measures, test_measures