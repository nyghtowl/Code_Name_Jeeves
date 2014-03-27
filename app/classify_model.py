'''
Email Classification Model Testing

'''

import app.common as cpm 
import app.feature_model as fm
from config import pkl_dir, graph_dir

import numpy as np
import matplotlib.pyplot as plt
import seaborn 

from scipy.stats.kde import gaussian_kde
from sklearn.preprocessing import Binarizer
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, classification_report
from sklearn.grid_search import GridSearchCV

from time import time
import os 

# X is features / bag of words per email
# Y is target true and false per email


def build_model(model, X_train, y_train):
    start = time()
    model.fit(X_train, y_train)
    print "Train model in %0.2fs." % (time() - start)
    return model


def model_eval(model, X_test, y_test, save=False):
    # The mean square error - how much the values can fluctuate - need for logistic
    # print('Root Mean Squared Error: %.2f' % ((np.mean((model.predict(X_test) - y_test) ** 2))**(0.5)))

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # R squared / proportion of variance explained by model: 1 is perfect prediction
    print 'SKLearn Accuracy Score: %.5f' % model.score(X_test, y_test)

    print "Classification Report:"
    print classification_report(y_test, y_pred)

    labels = [True, False]
    cm = confusion_matrix(y_test, y_pred, labels)
    print 'Confusion Matrix:'
    print cm
    cm_plot = plot_confusion_matrix(cm, labels, save)
    plot_roc_curve(y_test, y_proba, save)


def plot_confusion_matrix(cm, labels, save):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Email Confusion Matrix (True = email needs location)', fontsize=16)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels, fontsize=13)
    ax.set_yticklabels([''] + labels, fontsize=13)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    if save:
        # os.path.join(graph_dir, ''
        plt.savefig('./graph_dir/cfm.png')
    plt.show()

def plot_roc_curve(y_test, y_proba, save):
    # Receiver operating characteristic - if line is close to top left its a good model fit
    fpr, tpr, thresholds = roc_curve(y_test, y_proba[:,1])
    roc_auc = auc(fpr, tpr) # same as roc_auc_score method
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot((0, 1), (0, 1), 'k--')  # random predictions curve
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.title('Email Classifier ROC Curve', fontsize=16)
    plt.xlabel('False Positive Rate or (1 - Specifity)', fontsize=14)
    plt.ylabel('True Positive Rate or (Sensitivity)', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(loc='lower right')
    if save:
        # os.path.join(graph_dir, ''
        plt.savefig('./graph_dir/roc_plot.png')


# Add precision recall curve - similar to roc curve - shows how sold true is in comparison to true and false positives

def get_cv_scores(model, X_test, y_test):
    return cross_val_score(model, X_test, y_test, cv=3, scoring='roc_auc')

def plot_cross_val(model, X_test, y_test, save):
    cv_scores = get_cv_scores(model, X_test, y_test)
    print 'Min Score = %f\nMean Score = %f\nMax Score = %f' % \
    (cv_scores.min(), cv_scores.mean(), cv_scores.max())
    plt.figure(figsize=(20,10))
    _ = plt.hist(cv_scores, range=(0, 1), bins=30, alpha=0.2)
    x_vals = np.linspace(0, 1, 1000)
    smoothed = gaussian_kde(cv_scores).evaluate(x_vals)
    plt.plot(x_vals, smoothed, label='Smoothed Distribution')
    top = np.max(smoothed)
    plt.vlines([np.mean(cv_scores)], 0, top, color='r', label='Mean Test Score')
    plt.vlines([np.median(cv_scores)], 0, top, color='b', linestyles='dashed',
           label='Median Test Score')
    plt.legend(loc='best')
    _ = plt.title('Cross Validated Test Scores Distribution')
    if save:
        # os.path.join(graph_dir, ''
        plt.savefig('./graph_dir/cross_val.png')


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

def rank_features(feature_names, class_model, axis=1,start_point=-20):
    feature_names = np.asarray(feature_names)
    for rank, feat_index in enumerate(np.argsort(class_model.coef_[0], axis=0)[start_point:], start=1):
        print '%i: %s' % (rank, feature_names[feat_index])

def main(model, data=None, save=False, model_fn='final_model.pkl', data_fn='pd_dataframe.pkl', vec_fn='final_vec.pkl'):
    if data is None:
        data = cpm.unpickle(os.path.join(pkl_dir, data_fn))

    vectorizer = cpm.unpickle(os.path.join(pkl_dir, vec_fn))

    if save:
        X_raw, y = cpm.unpickle(os.path.join(pkl_dir, 'x_y_data.pkl'))
        if X_raw.empty:
            X_raw, y = cpm.define_x_y_data(data, True)
        X_vect = fm.apply_feature_vector(vectorizer, X_raw)
        X_train, X_test, y_train, y_test = cpm.create_datasets(X_vect, y, True)
        # Current size is train = 662 and test = 284
    else:
        X_train, X_test, y_train, y_test = cpm.unpickle(os.path.join(pkl_dir, 'train_split.pkl'))
  
    # cpm.print_test_train_shape(X_train, X_test, y_train, y_test)


    if model == 'RandomForestClassifier':
    # or model == 'GradientBoostingClassifier':
        X_train = np.nan_to_num(X_train)
        X_test = np.nan_to_num(X_test)
    #     X_train = X_train.todense()
    #     X_test = X_test.todense()


    class_model = { 
        'LogisticRegression': LogisticRegression(C=10000.0, penalty='l2', class_weight='auto', fit_intercept=True), 
        'MultinomialNB': MultinomialNB(alpha=0.100000, fit_prior=True),
        'SVC': SVC(C=10000.0, kernel='linear', shrinking=True, probability=True, class_weight='auto'),
        'RandomForestClassifier': RandomForestClassifier(n_estimators=121, criterion='entropy', bootstrap=True, oob_score=True, max_features='auto'),
        'GradientBoostingClassifier': GradientBoostingClassifier()
    }
    '''
 
    models = [LogisticRegression(), MultinomialNB(), SVC(), RandomForestClassifier(), GradientBoostingClassifier()]

    params = {
        'LogisticRegression': { 'penalty': ['l1','l2'], 'C': np.logspace(0, 9, 10), 'fit_intercept': [True, False], 'class_weight':['auto']},
        'MultinomialNB':  {'alpha': np.logspace(0, 6, 7), 'fit_prior':[True, False]},
        'SVC': { 'C': np.logspace(0, 9, 10), 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'shrinking':[True,False], 'probability':[True, False], 'class_weight':['auto']},
        'RandomForestClassifier': {'n_estimators': np.arange(1, 200, 10), 'criterion':['gini', 'entropy'], 'bootstrap':[True,False], 'max_features': ['auto','sqrt','log2',None], 'oob_score':[True, False]},
        'GradientBoostingClassifier': {'learning_rate':np.logspace(0.1, 1, 5), 'n_estimators': np.arange(50, 250, 25)}
    }

    params_round_2 = {
    'LogisticRegression':{'C': np.linspace(approx_best_c/10., approx_best_c*10., 200)},
    'MultinomialNB': {'alpha': np.linspace(approx_best_alpha/10., approx_best_alpha*10., 200)},
    'SVC': {'C': np.linspace(svc_approx_params['C']/10., svc_approx_params['C']*10., 200) }
    'RandomForestClassifier': {'n_estimators': np.arange(approx_best_n-50, approx_best_n+50, 1)},
    'GradientBoostingClassifier': {'learning_rate': np.linspace(approx_best_learn/10., approx_best_learn*10., 100), 'n_estimators': np.arange(approx_best_n-50, approx_best_n+50, 1)}
    }
    '''


    # TO DO - store each model and then run against each other

    print "Model:", model

    # TO DO - grid search again
    clf = build_model(class_model[model], X_train, y_train)

    model_eval(clf, X_test, y_test)

    # TO DO - work on making Random work with this

    #rank_features(vectorizer.get_feature_names(), clf)

    if save:
        print "New classifier saved."
        cpm.pickle(model, os.path.join(pkl_dir, model_fn))

    return clf


if __name__ == '__main__':


    main()