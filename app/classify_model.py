'''
Email Classification Model Testing

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
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, classification_report
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

# Evaluate Model

def model_eval(model_name, model, X_test, y_test, y_pred, feature_names, save=False):
    # The mean square error - how much the values can fluctuate - need for logistic
    # print('Root Mean Squared Error: %.2f' % ((np.mean((model.predict(X_test) - y_test) ** 2))**(0.5)))

    # R squared / proportion of variance explained by model: 1 is perfect prediction
    print '%s Accuracy Score: %.5f' % (model_name, model.score(X_test, y_test))
    print
    print "%s Classification Report:" % model_name
    print classification_report(y_test, y_pred)
    print 
    create_confusion_matrix(model_name, y_test, y_pred, save)
    print 
    print '%s Cross Validation Scores:' % model_name
    cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='roc_auc')

    print 'Min Score = %f\nMean Score = %f\nMax Score = %f' % (cv_scores.min(), cv_scores.mean(), cv_scores.max())

    print
    if model_name == 'RandomForest' or model_name == 'GradientBoost':
        rank_features(model_name, feature_names, model.feature_importances_)
    else:
        try:
            rank_features(model_name, feature_names, model.coef_[0])
        except:
            pass


#    plot_cross_val(model_name, model, X_test, y_test, save)

def create_confusion_matrix(model_name, y_test, y_pred, save=False):
    labels = [True, False]
    conf_matrix = confusion_matrix(y_test, y_pred, labels)
    print 'Jeeves Confusion Matrix:'
    print conf_matrix
    print
    #cm_plot = plot_confusion_matrix(model_name, conf_matrix, labels, save)
    return conf_matrix

def plot_confusion_matrix(model_name, conf_matrix, labels, save, graph_fn='cfm.png', cmap=cm.cubehelix_r):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_matrix, cmap=cmap)
    fig.colorbar(cax)
    plt.title('Jeeves Confusion Matrix (True = email needs location)', fontsize=16)
    ax.set_xticklabels([''] + labels, fontsize=13)
    ax.set_yticklabels([''] + labels, fontsize=13)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    if save:
        plt.savefig(os.path.join(graph_dir, graph_fn))
    plt.show()

def rank_features(model_name, feature_names, class_model_coef, start_point=-30):
    print '%s Top 20 Features' % model_name
    feature_names = np.asarray(feature_names)
    for rank, feat_index in enumerate(np.argsort(class_model_coef, axis=0)[start_point:], start=1):
        print '%i: %s' % (rank, feature_names[feat_index])

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
        # plt.savefig('./graph_dir/cross_val.png')

def plot_roc_curve(model_names, models, X_test, y_test, save=False, graph_fn='roc_plot.png'):
    for i, model in enumerate(models):
        y_proba = model.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_proba[:,1])
        roc_auc = auc(fpr, tpr) # same as roc_auc_score method
        plt.plot(fpr, tpr, label='ROC curve %s (area = %0.3f)' % (model_names[i], roc_auc))

    plt.plot((0, 1), (0, 1), 'k--')  # random predictions curve
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.title('Jeeves Classifier ROC Curve', fontsize=16)
    plt.xlabel('False Positive Rate or (1 - Specifity)', fontsize=14)
    plt.ylabel('True Positive Rate or (Sensitivity)', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(loc='lower right')
    if save:
        plt.savefig(os.path.join(graph_dir, graph_fn))
        # plt.savefig('./graph_dir/roc_plot.png')


# Add precision recall curve - similar to roc curve - shows how sold true is in comparison to true and false positives

def run_model(model_name, model, data_set, feature_names, save=False):
    X_train, X_test, y_train, y_test = data_set

    model = build_model(model, X_train, y_train)
    
    y_pred = model.predict(X_test)

    print "Model:", model_name

    model_eval(model_name, model, X_test, y_test, y_pred, feature_names, save)

    model_fn='final_'+ model_name + '_.pkl'

    if save:
        print "New classifier saved."
        cpm.pickle(model, os.path.join(pkl_dir, model_fn))

    return model

def main(model_names, save=False, train_fn='train_split.pkl', model_fn='final_model.pkl', new_data_split=False):

    if new_data_split: # 
        X, y = cpm.get_x_y_data()
        X_train, X_test, y_train, y_test = cpm.create_datasets(X, y, True)
    else:
        X_train, X_test, y_train, y_test = cpm.unpickle(os.path.join(pkl_dir, train_fn))

    X_train = X_train.todense()
    X_test = X_test.todense()
    
    # Current size is train = 662 and test = 284 & features = 5766
    # cpm.print_test_train_shape(X_train, X_test, y_train, y_test)

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
        model_results.append(run_model(model, class_model[model], [X_train, X_test, y_train, y_test], cpm.get_feature_names(), save))

    plot_roc_curve(model_names[:len(model_results)], model_results, X_test, y_test, True)


    return model_results


if __name__ == '__main__':


    main()