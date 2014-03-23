'''
Email Classification Model

'''
# Need to replace this by pushing and pulling data from postgres or other persistant data source
from feature_model import main as feature_main

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats.kde import gaussian_kde
from sklearn.preprocessing import Binarizer
from sklearn.cross_validation import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, classification_report
from sklearn.grid_search import GridSearchCV

import cPickle # same as pickle but faster and implemented in C

from time import time
import os 

# X is bag of words per email
# Y is target true and false per email

def create_datasets(X, y, split_size=.30): # regularization sprint - cross val hyper
    #X[:,np.newaxis] ?
    return train_test_split(X, y, test_size=split_size)

def print_test_train_shape(X_train, X_test, y_train, y_test):

    print 'X_train:', shape(X_train)
    print 'X_test:', shape(X_test)
    print 'y_train:', shape(y_train)
    print 'y_test:', shape(y_test)

def build_model(model, X_train, y_train):
    start = time()
    model.fit(X_train, y_train)
    print "Train model in %0.2fs." % (start - time())
    return model

def model_eval(model, X_test, y_test):
    # The mean square error - how much the values can fluctuate - need for logistic
    # print('Root Mean Squared Error: %.2f' % ((np.mean((model.predict(X_test) - y_test) ** 2))**(0.5)))

    # R squared / proportion of variance explained by model: 1 is perfect prediction

    # These are the same as score on the model
    # print 'Straight Accuracy Calc:', (float(([y_val == y_pred[i] for i, y_val in np.ndenumerate(y_test)]).count(True)) / len(y_pred))
    # print 'SKLearn Prediction Accuracy Score: %.5f' % accuracy_score(y_test, y_pred) 

    y_pred = model.predict(X_test)

    print 'SKLearn Accuracy Score: %.5f' % model.score(X_test, y_test)

    print "Classification Report:"
    print classification_report(y_test, y_pred)

    labels = [True, False]
    cm = confusion_matrix(y_test, y_pred, labels)
    print 'Confusion Matrix:'
    print cm
    plot_confusion_matrix(cm, labels)
    plot_roc_curve(y_test, y_pred)

def plot_confusion_matrix(cm, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Email Classifier Confusion Matrix (identifying email that needs a meeting')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_roc_curve(y_test, y_pred):
    # Receiver operating characteristic - if line is close to top left its a good model fit
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr) # same as roc_auc_score method
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot((0, 1), (0, 1), 'k--')  # random predictions curve
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc='lower right')

# Add precision recall curve - similar to roc curve - shows how sold true is in comparison to true and false positives

def get_cv_scores(model, X_test, y_test):
    return cross_val_score(model, X_test, y_test, cv=3, scoring='roc_auc')

def plot_cross_val(model, X_test, y_test):
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


def grid_search(model, params, X_train, y_train,  n_jobs=-1, k_fold=3):
    start = time()
    # Confirm param vals in list form otherwise grid search throws error
    #params = {k: (v if type(v) == list else [v]) for k, v in params.items() }

    clf = GridSearchCV(model, params, cv=k_fold, n_jobs=n_jobs)
    clf.fit(X_train, y_train)
    
    print "Grid search model in %0.2fs." % (start - time())

    print 'Best Estimators'
    print clf.best_estimator_
    print 'Best params'
    print clf.best_params_
    print 'Best Scores'
    print clf.best_score_

    # print 'Grid Scores'    
    # for params, mean_score, scores in clf.grid_scores_:
    #     print"%0.3f (+/-%0.03f) for %r" %(mean_score, scores.std(), params)
    
    return clf.best_params_, clf.best_estimator_


def pickle_stuff(stuff, filename):
    # use pkl for the filename and write in binary
    with open(filename, 'wb') as f:
        cPickle.dump(stuff, f)

def unpickle_stuff(filename):
    with open(filename, 'rb') as f:
        return cPickle.load(f)

def main():
    X, y = feature_main()
    # model = LogisticRegression()

    # X_train, x_test, y_train, y_test = create_datasets(X, y)

    # model = build_model(model, X_train, y_train)

    # model_directory = 'model_pkl'
    # if not os.path.exists(model_directory):
    #     os.makedirs(model_directory)

    # model_path = os.path.join(model_directory, filename)

    # pickle_stuff(model, model_path)
    
    return X, y

if __name__ == '__main__':
    main()