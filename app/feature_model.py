'''
Jeeves Project

Engineer and store features

create cpt tables and leverage naive bayes sql for this section

'''

# Need to replace this by pushing and pulling data from postgres or other persistant data source

import common as cpm

from sklearn.decomposition import NMF, RandomizedPCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import train_test_split
import vincent
vincent.core.initialize_notebook()

from time import time
import pandas as pd

# Create Bag of Words / Features
def init_vectorizer(vectorizer_object):
    # tokens = nltk.word_tokenize(article)
    #vectorizer = vectorizer_object(min_df=2, strip_accents='unicode', max_features=5000, analyzer='word',token_pattern=r'\\w{1,}',ngram_range=(1, 2))
    return vectorizer_object(min_df=2, strip_accents='unicode', max_features=10000, analyzer='word',ngram_range=(1, 1), stop_words='english')

def fit_vectorizer(features, vectorizer):
    start = time()
    vectorizer.fit(features)    
    print "Fit vectorizer in %0.3fs." % (start - time())
    return vectorizer

def apply_features(data, feature_set):
    start = time()
    transformed_x = feature_set.transform(data)
    print "Trasformed bag in %0.3fs." % (start - time())
    return transformed_x

#    print data['body'].isnull().sum()

def create_vec_model(vectorizer=TfidfVectorizer, data=None):
    if not data:
        data = cpm.load_emails_pd()
        data = data.body
    print 'data shape', data.shape
    vec_model = init_vectorizer(vectorizer)
    feature_set = fit_vectorizer(data, vec_model)
    return feature_set

def main():
    # call get_features_names on vectorizer model to get them
    feature_set = create_vec_model()
    cpm.pickle(feature_set, './model_pkl/final_vec.pkl')
    return feature_set

 

if __name__ == '__main__':
    main()