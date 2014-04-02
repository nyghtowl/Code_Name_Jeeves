from pprint import pprint
from time import time
import numpy as np
import app.common as cpm 
import app.classify_model as class_m
import app.feature_model as feature_m
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

def main():
    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    ###############################################################################
    # define pipelines combining a text feature extractor & classifier


    clf_models = [LogisticRegression()]
    #[LogisticRegression(), MultinomialNB(), SVC(), RandomForestClassifier(), GradientBoostingClassifier()]

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=feature_m.nltk_tokenizer, analyzer='word', strip_accents='unicode')),
        ('tfidf', TfidfTransformer()),
        ('clf', clf_models[0]),
    ])


    pipeline = Pipeline([
        ('tfvect', TfidfVectorizer(tokenizer=feature_m.nltk_tokenizer, analyzer='word', strip_accents='unicode')),
        ('clf', clf_models[0]),
    ])


    # uncommenting more parameters will give better exploring power but will
    # increase processing time in a combinatorial way
    parameters = {
        # 'vect__max_df': (0.5, 0.75, 1.0),
        'vect__min_df': [.5, 1.0, 1.5, 2.0, 2.5],
        'vect__max_features': [None, 5000, 10000, 50000],
        'vect__ngram_range': [(1, 1), (1, 2), (1,3)],  # unigrams or bigrams or trigrams
        'vect__max_features': [None, 5000, 10000, 20000],
        'vect__lowercase': [True, False],

        'tfidf__use_idf': [True, False],
        'tfidf__norm': ['l1', 'l2'],
        'tfidf__smooth_idf': [True, False],

        'tfvect__min_df': [0.5, 1.0, 1.5, 2.0, 2.5],
        'tfvect__max_features': [None, 5000, 10000, 50000],
        'tfvect__ngram_range': [(1, 1), (1, 2), (1,3)],  # unigrams or bigrams or trigrams
        'tfvect__max_features': [None, 5000, 10000, 20000],
        'tfvect__lowercase': [True, False],

        'tfvect__use_idf': [True, False],
        'tfvect__norm': ['l1', 'l2'],
        'tfvect__smooth_idf': [True, False],

        # 'clf__alpha': [0.00001, 0.000001],
        # 'clf_loss': ['str', 'hinge', 'log', 'modeified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive']
        #'clf__n_iter': [10, 50, 80],

        'clf__C': np.logspace(0, 9, 10), 
        'clf__penalty': ['l1', 'l2'],

        'clf__fit_intercept': [True, False], 
        'clf__class_weight':['auto']

    }

    '''
    params = {
        'LogisticRegression': 
            { 
            'penalty': ['l1','l2'], 
            'C': np.logspace(0, 9, 10), 
            'fit_intercept': [True, False],
            'class_weight':['auto']
            },

        'MultinomialNB':  {'alpha': np.logspace(0, 6, 7), 'fit_prior':[True, False]},
        'SVC': { 'C': np.logspace(0, 9, 10), 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'shrinking':[True,False], 'probability':[True, False], 'class_weight':['auto']},
        'RandomForestClassifier': {'n_estimators': np.arange(1, 200, 10), 'criterion':['gini', 'entropy'], 'bootstrap':[True,False], 'max_features': ['auto','sqrt','log2',None], 'oob_score':[True, False]},
        'GradientBoostingClassifier': {'learning_rate':np.logspace(0.1, 1, 5), 'n_estimators': np.arange(50, 250, 25)}
    }

    '''


    X, y = class_m.main()
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(X, y)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

if __name__ == "__main__":
    main()