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
from sklearn.metrics import confusion_matrix, roc_curve, auc
# X is bag of words per email
# Y is target true and false per email

def create_datasets(X, y, split_size=.30):
    #X[:,np.newaxis] ?
    return train_test_split(X, y, test_size=split_size)

def build_model(model, X_train, y_train):
    #return model.fit_transform(X_train, y_train) - why is this not working?
    model.fit(X_train, y_train)
    return model

def predict_eval(model, X_test, y_test):
    # The mean square error - how much the values can fluctuate
    # print('Root Mean Squared Error: %.2f' % ((np.mean((model.predict(X_test) - y_test) ** 2))**(0.5)))

    # R squared / proportion of variance explained by model: 1 is perfect prediction
    y_pred = model.predict(X_test)
    print 'Straight Accuracy Calc:', (float(([y_val == y_pred[i] for i, y_val in np.ndenumerate(y_test)]).count(True)) / len(y_pred))
    print 'SKLearn Accuracy Score: %.5f' % model.score(X_test, y_test)
    print 'Confusion Matrix:', '\n', confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_pred)

def plot_roc_curve(y_test, y_pred):
    # Receiver operating characteristic - if line is close to top left its a good model fit
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot((0, 1), (0, 1), 'k--')  # random predictions curve
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc='lower right')

def get_cv_scores(model, X_test, y_test):
    return cross_val_score(model, X_test, y_test, cv=10, scoring='roc_auc')

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

def main():
    X, y = feature_main()
    return X, y

if __name__ == '__main__':
    main()