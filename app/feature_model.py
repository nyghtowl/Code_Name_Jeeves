'''
Find & Engineer Features

create cpt tables and leverage naive bayes sql for this section

'''

import common as cpm
import numpy as np

from config import pkl_dir

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

#from nltk.stem.snowball import EnglishStemmer
#from nltk.corpus import words - can build out feature model to include all english words
#english_corpus = set(w.lower() for w in words.words())
#import SnowballTokenizer

import dateutil.parser as parser
from time import time
import datetime
import string
import re
import os

# POS tagging ? 

def drop_punc(word):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    return regex.sub('', word)

def change_num(word):
    return re.sub(r'[0-9]+', r'd', word)

def remove_wspace(word):
    return re.sub(r'^\s+|\s+\Z', r'', word)

def stem_words(word):
    lmtzr = WordNetLemmatizer()
    return lmtzr.lemmatize(word)

def nltk_tokenizer(raw):
    tokens = word_tokenize(raw)
    return [stem_words(change_num(drop_punc(remove_wspace(word)))) for word in tokens]

# Capture if date in body of text - True or False and add to feature set about doc...

def create_vec_model(vectorizer, data):
    start = time()
    print 'data shape', data.shape
    fit_vect_model = vectorizer.fit(data)    
    print "Fit vectorizer in %0.3fs." % (time() - start)
    return fit_vect_model

def apply_feature_vector(vectorizer, data):
    start = time()
    feature_set = vectorizer.transform(data)
    print "Trasformed features in %0.3fs." % (time() - start)
    return feature_set

def check_for_date(text):
    today = datetime.datetime.now().date()
    for word in text.split():
        try:
            if parser.parse(word, fuzzy=True).date() < today or val.date() > today:
                return True
        except:
            pass
    return False


def create_date_feature(data):
    result = []
    for text in data:
        result.append(check_for_date(text))
    return np.array(result)


def main(save=False, X=None, vec_fn='final_vec.pkl', data_fn='pd_dataframe.pkl'):

    vectorizer_model = TfidfVectorizer(min_df=2, strip_accents='unicode', max_features=10000, analyzer='word',ngram_range=(1, 3), stop_words='english', lowercase=True, norm='l1', tokenizer=nltk_tokenizer, use_idf=True)

    if X is None:
        X, y = cpm.get_x_y_data()
    
    vectorizer = create_vec_model(vectorizer_model, X)

    if save:
        print "New vectorizer saved."
        cpm.pickle(vectorizer, os.path.join(pkl_dir, vec_fn))
    
    return vectorizer


if __name__ == '__main__':
    main()