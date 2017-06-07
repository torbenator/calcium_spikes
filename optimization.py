import csv
import numpy as np
import itertools

import boosting
import output_analysis


binary_params = {
		'objective': "binary:logistic",
		'eval_metric':"error",
		'eta': 1, #step size shrinkage. larger--> more conservative / less overfitting
		'alpha':0.01, #l1 regularization
		'lambda':0.01, #l2 regularizaion
		'gamma':3, # default = 0, minimum loss reduction to further partitian on a leaf node. larger-->more conservative
	    'max_depth': 5,
		'seed': 16,
		'silent': 1,
		'colsample_bytree':.5
		}

binary_opt_params_dict = {
		'eta' : [0.5,1],
		'alpha' : [0,0.01,0.1],
		'lambda' : [0,0.01,0.1],
		'gamma' : [0,1,3,5],
		}


def _create_param_dicts(param_dict):
	"""
	makes dictionary of parameters to optimize
	"""
	all_dicts = []
	sorted_keys = sorted(param_dict)
	combinations = list(itertools.product(*(param_dict[key] for key in sorted_keys)))
	for c in xrange(len(combinations)):
		new_dict = dict()
		for i, key in enumerate(sorted_keys):
			new_dict[key]=combinations[c][i]
		all_dicts.append(new_dict)

	return all_dicts


def write_output(metrics, fname,out_dir='./'):
	with open(out_dir+fname, 'wb') as csvfile:
		mywriter = csv.writer(csvfile, delimiter=',')
		mywriter.writerow(['spike_bin', 'true_positive','false_positive','true negative','false negative'])
		for spike_bin in xrange(len(metrics[0])):
			dat = [str(i[spike_bin]) for i in metrics[:-1]]
			dat.insert(0,str(spike_bin))
			mywriter.writerow(dat)
		mywriter.writerow(['correlation: ', str(metrics[-1])])



def optimize_binary(X_train,X_test,y_train,y_test,params=None, opt_params_dict=None, verbose=True, out_dir='./'):

	"""
	evaluates choice hyperparameters to optimize the way XGBoost fits.
	"""

	if params == None:
		params = binary_params
	if opt_params_dict == None:
		opt_params_dict = binary_opt_params_dict


	all_param_combos = _create_param_dicts(opt_params_dict)

	for e, this_combo in enumerate(all_param_combos):

		# temporary variable
		these_params = params

		for k,v in this_combo.items():
			these_params[k]=v

		fname = ''.join([str(k)+'_'+str(v) + '|' for k,v in these_params.items()]) + '.csv'

		if verbose:
			print 'Run ' + str(e) + '/' + str(len(all_param_combos))
			print 'running params: ' + str(these_params.items())
			print 'file: ' + fname

		Yr, Yt, model = boosting.binary_booster(X_train, X_test, y_train, y_test, params=params)
		train_metrics, test_metrics, [train_scores, test_scores] = output_analysis.calculate_accuracy_metrics(y_train,y_test, Yr, Yt, myceil=0.5, verbose=verbose)

		train_fname = 'train_'+fname
		write_output(train_metrics, train_fname, out_dir=out_dir)

		test_fname = 'test_'+fname
		write_output(test_metrics, test_fname, out_dir=out_dir)



