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

#from nltk.stem.snowball import EnglishStemmer
#from nltk.corpus import words - can build out feature model to include all english words
#english_corpus = set(w.lower() for w in words.words())
#import SnowballTokenizer

from time import time
import string
import re


def drop_punc(word):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    return regex.sub('', word)

def change_num(word):
    return re.sub(r'[0-9]+', r'd', word)

def stem_words(word):
    lmtzr = WordNetLemmatizer()
    return lmtzr.lemmatize(word)

def nltk_tokenizer(raw):
    tokens = word_tokenize(raw)
    return [stem_words(change_num(drop_punc(word.strip()))) for word in tokens]

# Shape words - do this during tokenization
# POS tagging ? 

# Capture if date in body of text - True or False and add to feature set about doc...

# Need to test vectorizer parameters in grid search

# Create Bag of Words / Features
def init_vectorizer(vectorizer_object):
    # doesn't include punctuation
    # norm='l1' = normalized token frequencies
    # potentially good to add
    return vectorizer_object(min_df=2, strip_accents='unicode', max_features=10000, analyzer='word',ngram_range=(1, 3), stop_words='english', lowercase=True, norm='l1', tokenizer=nltk_tokenizer, use_idf=True)

def fit_vectorizer(features, vectorizer):
    start = time()
    vectorizer.fit(features)    
    print "Fit vectorizer in %0.3fs." % (time() - start)
    return vectorizer

def apply_features(vectorizer, data=None):
    start = time()
    if not data:
        data = cpm.load_emails_pd()
        data = data.body
    feature_set = vectorizer.transform(data)
    print "Trasformed features in %0.3fs." % (time() - start)
    return feature_set

def create_vec_model(vectorizer=TfidfVectorizer, data=None):
    if not data:
        data = cpm.load_emails_pd()
        data = data.body
    print 'data shape', data.shape
    init_vect_model = init_vectorizer(vectorizer)
    fit_vect_model = fit_vectorizer(data, init_vect_model)
    return fit_vect_model, init_vect_model


def main(save=False):
    vectorizer, vec_model = create_vec_model()
    feature_set = apply_features(vectorizer)
    feature_names = vectorizer.get_feature_names()
    if save == True:
        cpm.pickle(vectorizer, './model_pkl/final_vec.pkl')
    
    return vectorizer, feature_set, feature_names


if __name__ == '__main__':
    main()