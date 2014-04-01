'''
Build Email Classification Model

'''

import app.common as cpm 
from config import pkl_dir, graph_dir

import numpy as np
import matplotlib.pyplot as plt
from pylab import cm
import seaborn 

from scipy.stats.kde import gaussian_kde
from sklearn.preprocessing import Binarizer
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.grid_search import GridSearchCV

from time import time
import os 

# X is features / bag of words per email
# Y is target true and false per email

# Build Model


def build_model(model, X_train, y_train):
    start = time()
    model.fit(X_train, y_train)
    print "Train model in %0.2fs." % (time() - start)
    return model

###########################################################
# Grid Search


def grid_search(model, params, X_train, y_train,  n_jobs=1, k_fold=3):
    start = time()
    # Confirm param vals in list form otherwise grid search throws error

    clf = GridSearchCV(model, params, cv=k_fold, n_jobs=n_jobs)
    clf.fit(X_train, y_train)
    
    print "Grid search model in %0.2fs." % (time()-start)

    print 'Best Estimators'
    print clf.best_estimator_
    print 'Best params'
    print clf.best_params_
    print 'Best Scores'
    print clf.best_score_

    # print 'Grid Scores'    
    # for params, mean_score, scores in clf.grid_scores_:
    #     print"%0.3f (+/-%0.03f) for %r" %(mean_score, scores.std(), params)
    
    return clf

'''

# Grid Search stuff

fluff = [LogisticRegression(), 
        MultinomialNB(), 
        SVC(), 
        RandomForestClassifier(),
        GradientBoostingClassifier()]

params = [{ 'penalty': ['l1','l2'], 'C': np.logspace(0, 9, 10), 'fit_intercept': [True, False], 'class_weight':['auto']},
    {'alpha': np.logspace(0, 6, 7), 'fit_prior':[True, False]},
    { 'C': np.logspace(0, 9, 10), 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'shrinking':[True,False], 'probability':[True, False], 'class_weight':['auto']},
    {'n_estimators': np.arange(1, 200, 10), 'criterion':['gini', 'entropy'], 'bootstrap':[True,False], 'max_features': ['auto','sqrt','log2',None], 'oob_score':[True, False]},
    {'learning_rate':np.logspace(0.1, 1, 5), 'n_estimators': np.arange(50, 250, 25)}
]
Params to possibly use if want to rerun grid search

params_round_2 = {
'LogisticRegression':{'C': np.linspace(approx_best_c/10., approx_best_c*10., 200)},
'MultinomialNB': {'alpha': np.linspace(approx_best_alpha/10., approx_best_alpha*10., 200)},
'SVC': {'C': np.linspace(svc_approx_params['C']/10., svc_approx_params['C']*10., 200) }
'RandomForestClassifier': {'n_estimators': np.arange(approx_best_n-50, approx_best_n+50, 1)},
'GradientBoostingClassifier': {'learning_rate': np.linspace(approx_best_learn/10., approx_best_learn*10., 100), 'n_estimators': np.arange(approx_best_n-50, approx_best_n+50, 1)}
}
'''


###########################################################
# Cross Validation

def cross_validate(model_name, model, X, y):
    print 
    print '%s Cross Validation Scores:' % model_name
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')

    print 'Min Score = %f\nMean Score = %f\nMax Score = %f' % (cv_scores.min(), cv_scores.mean(), cv_scores.max())

def plot_cross_val(model_name, model, X_test, y_test, save, graph_fn='cross_val.png'):
    print 'Jeeves Cross Validation Scores Plot:' 
    cv_scores = cross_val_score(model, X_test, y_test, cv=3, scoring='roc_auc')
    
    plt.figure(figsize=(20,10))

    _ = plt.hist(cv_scores, range=(0, 1), bins=30, alpha=0.2)

    X_vals = np.linspace(0, 1, 1000)
    smoothed = gaussian_kde(cv_scores).evaluate(X_vals)

    plt.plot(X_vals, smoothed, label='Smoothed Distribution')
    top = np.max(smoothed)

    plt.vlines([np.mean(cv_scores)], 0, top, color='r', label='Mean Test Score')
    plt.vlines([np.median(cv_scores)], 0, top, color='b', linestyles='dashed',
           label='Median Test Score')
    plt.legend(loc='best')

    _ = plt.title('Cross Validated Test Scores Distribution')
    if save:
        plt.savefig(os.path.join(graph_dir, graph_fn))


###########################################################
# Feature Names

def get_feature_names(vectorizer=None, vec_fn='final_vec.pkl'):
    if vectorizer == None:
        vectorizer = cpm.unpickle(os.path.join(pkl_dir, vec_fn))

    return vectorizer.get_feature_names()


def rank_features(feature_names, class_model_coef, start_point=-30):
    print 'Email Top %d Features' % -start_point
    feature_names = np.asarray(feature_names)
    for rank, feat_index in enumerate(np.argsort(class_model_coef, axis=0)[start_point:], start=1):
        print '%i: %s' % (rank, feature_names[feat_index])


def request_feature_rank(model_name, model, feature_names=get_feature_names()):
    if model_name == 'RandomForest' or model_name == 'GradientBoost':
        rank_features(feature_names, model.feature_importances_)
    else:
        rank_features(feature_names, model.coef_[0])

###########################################################

def setup_model(model_name, model, X_train, y_train, feature_names, save=False):
    model = build_model(model, X_train, y_train)
    
    cross_validate(model_name, model, X_train, y_train)

    print 
    
    request_feature_rank(model_name, model)

    if save:
        model_fn='final_'+ model_name + '_.pkl'
        cpm.pickle(model, os.path.join(pkl_dir, model_fn))
        print "New classifier saved."

    return model

def main(model_names, save=False, train_fn='train_split.pkl', model_fn='final_model.pkl', new_data_split=False, random=11):

    if new_data_split:  
        X_train, X_test, y_train, y_test = cpm.create_datasets(save=save, vectorize=True, random=random)
    else:
        X_train, X_test, y_train, y_test = cpm.unpickle(os.path.join(pkl_dir, train_fn))

    X_train = X_train.todense()
    
    #model_names = ['LogisticRegression', 'MultinomialNB', 'SVC', 'RandomForest', 'GradientBoost']

    class_model = { 
        'LogisticRegression': LogisticRegression(C=10000.0 , penalty='l2', class_weight='auto', fit_intercept=True), 
        'MultinomialNB': MultinomialNB(alpha=0.100000, fit_prior=True),
        'SVC': SVC(C=10000.0, kernel='linear', shrinking=True, probability=True, class_weight='auto'),
        'RandomForest': RandomForestClassifier(n_estimators=121, criterion='entropy', bootstrap=True, oob_score=True, max_features='auto'),
        'GradientBoost': GradientBoostingClassifier()
    }

    model_results = []

    for model in model_names:
        model_results.append(setup_model(model, class_model[model], X_train, y_train, save))

    return model_results


if __name__ == '__main__':
    main()