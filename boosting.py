import numpy as np
import xgboost as xgb

num_round = 200

binary_params = {'objective': "binary:logistic",
               'eval_metric':"error",
                'eta': .5, #step size shrinkage. larger--> more conservative / less overfitting
                'alpha' : 0, #l1 regularizaion
                'lambda':0.1, #l2 regularizaion
                'gamma':1, # default = 0, minimum loss reduction to further partitian on a leaf node. larger-->more conservative
                #'max_depth': 5,
                'seed': 16,
                'silent': 1,
                'missing': '-999.0',
                #'colsample_bytree':.5
                }


def binary_booster(X_train, X_test, y_train, y_test, params=binary_params):
    """
    Trains a binary (spikes are 1 or 0) gradient boosted trees classifier.
    """

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    dtrain_y = xgb.DMatrix(X_train)
    model = xgb.train(params, dtrain, num_round)

    Yr = model.predict(dtrain_y)
    Yt = model.predict(dtest)

    return Yr, Yt, model


multivar_params = {'objective': 'multi:softprob',
               #'num_class' : num_class,
               'eta': 1, #step size shrinkage. larger--> more conservative / less overfitting
               #'alpha':0.01, #l1 regularization
               'lambda':0.1, #l2 regularizaion
               'gamma':1, # default = 0, minimum loss reduction to further partitian on a leaf node. larger-->more conservative
               #'max_depth': 5,
               'seed': 16,
               'silent': 1,
               'missing': '-999.0',
               #'colsample_bytree':.5
                }


def multivariate_booster(X_train, X_test, y_train, y_test, params=multivar_params):
    """
    Trains a multivariate (spikes are 0:inf) gradient boosted trees classifier.
    """

    num_class = len(set(y_train))
    multivar_params['num_class'] = num_class

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    dtrain_y = xgb.DMatrix(X_train)
    model = xgb.train(params, dtrain, num_round)

    Yr = model.predict(dtrain_y)
    Yt = model.predict(dtest)

    return Yr, Yt, model


