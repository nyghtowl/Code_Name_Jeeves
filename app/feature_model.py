'''
Jeeves Project

Engineer and store features

create cpt tables and leverage naive bayes sql for this section

'''

# Need to replace this by pushing and pulling data from postgres or other persistant data source

import common as cpm


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from time import time

def stem_words(word):
    lmtzr = WordNetLemmatizer()
    return lmtzr.lemmatize(word)

def nltk_tokenizer(raw):
    tokens = word_tokenize(raw)
    return [stem_words(word) for word in tokens]

# Shape words - do this during tokenization
# POS tagging ? 

# Capture if date in body of text - True or False and add to feature set about doc...

# Need to test vectorizer parameters in grid search

# Create Bag of Words / Features
def init_vectorizer(vectorizer_object):
    # doesn't include punctuation
    # norm='l1' = normalized token frequencies
    # potentially good to add
    return vectorizer_object(min_df=2, strip_accents='unicode', max_features=10000, analyzer='word',ngram_range=(1, 3), stop_words='english', lowercase=True, norm='l1', tokenizer=nltk_tokenizer)

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

def create_vec_model(vectorizer=TfidfVectorizer, data=None):
    if not data:
        data = cpm.load_emails_pd()
        data = data.body
    print 'data shape', data.shape
    vec_model = init_vectorizer(vectorizer)
    feature_set = fit_vectorizer(data, vec_model)
    return feature_set

def main(safe=False):
    # call get_features_names on vectorizer model to get them
    feature_set = create_vec_model()
    if save == True:
        cpm.pickle(feature_set, './model_pkl/final_vec.pkl')
    
    return feature_set


if __name__ == '__main__':
    main()