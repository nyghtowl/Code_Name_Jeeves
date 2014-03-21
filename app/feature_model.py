'''
Jeeves Project

Build and store features - create cpt tables and leverage naive bayes sql for this section

'''

from config import conn, db
# Need to replace this by pushing and pulling data from postgres or other persistant data source

from data_eda import main as eda_main

from sklearn.decomposition import NMF, RandomizedPCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import train_test_split
import vincent
vincent.core.initialize_notebook()

from time import time

# Create Bag of Words / Features
def create_vectorizer(vectorizer_object):
    # tokens = nltk.word_tokenize(article)
    #vectorizer = vectorizer_object(min_df=2, strip_accents='unicode', max_features=5000, analyzer='word',token_pattern=r'\\w{1,}',ngram_range=(1, 2))
    vectorizer = vectorizer_object(min_df=2, strip_accents='unicode', max_features=10000, analyzer='word',ngram_range=(1, 1), stop_words='english')
    return vectorizer

def create_bag(features, vectorizer):
    start = time()
    bag_words = vectorizer.fit_transform(features)
    feature_names = vectorizer.get_feature_names()
    print "done in %0.3fs." % (start - time())
    return bag_words, feature_names

# Feature Decomposition 

def create_decomp(decomp_model, k):
    #NMF, RandomizedPCA
    return decomp_model(n_components=k)

def transform_features(decomp_model, data):
    start = time()
    result = decomp_model.fit_transform(data.todense())
    print "done in %0.3fs." % (start - time())
    return result, decomp_model
    # Are these different?


# NMF Assess
def top_vals(matrix, labels, num):
    for idx, row in enumerate(matrix):    
        print("Category #%d:" % idx)
        print("| ".join([labels[i] for i in row.argsort()[:-num-1:-1]]))
        print()

def get_top_vals(H, W, feature_names, data, num):
    print "H - Words"
    print top_vals(H, feature_names, num)
    print "W - Data Source"
    print top_vals(W.T, data, num)

def print_cloud(data, feature_names):
    # Good for analyzing bag of words
    for i in range(data.shape[0]):
        word_dict = {feature_names[idx]: data[i][idx]*100 for idx in data[i].argsort()[:-20:-1]}
        cloud = vincent.Word(word_dict)
        print "Cloud", i
        cloud.width = 400
        cloud.height = 400
        cloud.padding = 0
        cloud.display()


# PCA Assess
def plot_scatter(x_pca, y_train):
    colors = ['r', 'b']
    markers = ['o', 's']
    plt.figure(figsize=(20,10))

    plt.xlim(-0.05, 0.45)
    plt.ylim(-0.2, 0.8)
    for i, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(x_pca[y_train == i, 0], 
                    x_pca[y_train == i, 1],
                    c=c, 
                    marker=m, 
                    label=i, 
                    alpha=0.7, 
                    s=200)

    _ = plt.legend(loc='best')

def main():
    vectorizer = TfidfVectorizer
    data = eda_main()
#    print data['body'].isnull().sum()

    vect = create_vectorizer(vectorizer)
    bag, feature_labels = create_bag(data.body, vect)
    return bag, data['target']

#     # bag_cv, feature_names_cv = create_bag(data['app description'], cv)
#     bag_tf, feature_names_tf = create_bag(data['app description'], tf

    # nmf = fit_nmf(7, bag.todense())


# table_data = {
#                 "priors": ["label_id integer, p_label float, FOREIGN KEY(label_id) REFERENCES label(rowid)", "p_label"],
#                 "cpts": ["word_id integer, label_id integer, p_word_label float, FOREIGN KEY(word_id) REFERENCES word(rowid), FOREIGN KEY(label_id) REFERENCES label(rowid)", "label_id"],
#         }
