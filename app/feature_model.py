'''
Jeeves Project

Build and store features - create cpt tables and leverage naive bayes sql for this section

'''

from config import conn, db
from data_eda import main as eda_main

from sklearn import decomposition
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import train_test_split

from time import time

# Create Bag of Words / Features
def create_vectorizer(vectorizer_object):
    # tokens = nltk.word_tokenize(article)
    #vectorizer = vectorizer_object(min_df=2, strip_accents='unicode', max_features=5000, analyzer='word',token_pattern=r'\\w{1,}',ngram_range=(1, 2))
    vectorizer = vectorizer_object(min_df=2, strip_accents='unicode', max_features=5000, analyzer='word',ngram_range=(1, 1))
    return vectorizer

def create_bag(features, vectorizer):
    start = time()
    bag_words = vectorizer.fit_transform(features)
    feature_names = vectorizer.get_feature_names()
    print "done in %0.3fs." % (start - time())
    return bag_words, feature_names

# Feature Decomposition - NMF

def fit_nmf(k, bag):
    start = time()
    nmf = decomposition.NMF(n_components=k).fit(bag)
    print "done in %0.3fs." % (start - time())
    return nmf  

def nmf_transform(nmf):
    W = nmf.transform(bag_tf) # apps by k
    H = nmf.components_ # words by k
    print W.shape, H.shape
    return W, H

def top_vals(matrix, labels, num):
    for idx, row in enumerate(matrix):    
        print("Category #%d:" % idx)
        print("| ".join([labels[i] for i in row.argsort()[:-num-1:-1]]))
        print()

def review_nmf(H, W, feature_names, data):
    print "H"
    print top_vals(H, feature_names, 3)
    print "W"
    print top_vals(W.T, data, 3)


# def main():
#     # cv = create_vectorizer(CountVectorizer)
#     tf = create_vectorizer(TfidfVectorizer)

#     # bag_cv, feature_names_cv = create_bag(data['app description'], cv)
#     bag_tf, feature_names_tf = create_bag(data['app description'], tf

    # nmf = fit_nmf(7, bag.todense())

# Feature Decomposition - PCA

# Feature Decomposition - SVD


# table_data = {
#                 "priors": ["label_id integer, p_label float, FOREIGN KEY(label_id) REFERENCES label(rowid)", "p_label"],
#                 "cpts": ["word_id integer, label_id integer, p_word_label float, FOREIGN KEY(word_id) REFERENCES word(rowid), FOREIGN KEY(label_id) REFERENCES label(rowid)", "label_id"],
#         }
