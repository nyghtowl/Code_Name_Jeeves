
'''
Evaluate Email Classification Model

'''

import app.common as cpm 
from config import pkl_dir, graph_dir

import numpy as np
import matplotlib.pyplot as plt
from pylab import cm


from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, classification_report

from time import time
import os 

# Evaluate Model

def model_eval(model_name, model, X_test, y_test, save=False):
    # The mean square error - how much the values can fluctuate - need for logistic
    # print('Root Mean Squared Error: %.2f' % ((np.mean((model.predict(X_test) - y_test) ** 2))**(0.5)))

    # R squared / proportion of variance explained by model: 1 is perfect prediction
    print "Evaluate %s Performance:" %model_name
    print

    y_pred = model.predict(X_test)

    print '%s Accuracy Score: %.5f' % (model_name, model.score(X_test, y_test))
    print
    print "%s Classification Report:" % model_name
    print classification_report(y_test, y_pred)
    print 
    create_confusion_matrix(model_name, y_test, y_pred, save)

def create_confusion_matrix(model_name, y_test, y_pred, save=False):
    labels = [True, False]
    conf_matrix = confusion_matrix(y_test, y_pred, labels)
    print 'Jeeves Confusion Matrix:'
    print conf_matrix
    print
    cm_plot = plot_confusion_matrix(model_name, conf_matrix, labels, save)
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


# Add precision recall curve - similar to roc curve - shows how sold true is in comparison to true and false positives


def main(model_name, X_test=None, y_test=None, models=None, save=False, train_fn='train_split.pkl', model_fn='final_model.pkl'):

    #model_names = ['LogisticRegression', 'MultinomialNB', 'SVC', 'RandomForest', 'GradientBoost']
    if models is None:
        models = [cpm.unpickle(os.path.join(pkl_dir, model_fn))]
    
    if X_test is None:
        X_train, X_test, y_train, y_test = cpm.unpickle(os.path.join(pkl_dir, train_fn))

    X_test = X_test.todense()

    for i, model in enumerate(models):
        model_eval(model_name[i], model, X_test, y_test, save)

    plot_roc_curve(model_name[:len(models)], models, X_test, y_test, save)

if __name__ == '__main__':
    main()