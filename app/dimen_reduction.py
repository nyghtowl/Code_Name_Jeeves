# Dimensionality Reduction Methods 

from sklearn.decomposition import NMF, RandomizedPCA, TruncatedSVD

import vincent
vincent.core.initialize_notebook()

def create_decomp(decomp_model, k):
    #NMF, RandomizedPCA
    return decomp_model(n_components=k)

def transform_features(decomp_model, data):
    start = time()
    result = decomp_model.fit_transform(data.todense())
    print "done in %0.3fs." % (time() - start)
    return result, decomp_model


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
    pass


#     # bag_cv, feature_names_cv = create_bag(data['app description'], cv)
#     bag_tf, feature_names_tf = create_bag(data['app description'], tf

    # nmf = fit_nmf(7, bag.todense())


# table_data = {
#                 "priors": ["label_id integer, p_label float, FOREIGN KEY(label_id) REFERENCES label(rowid)", "p_label"],
#                 "cpts": ["word_id integer, label_id integer, p_word_label float, FOREIGN KEY(word_id) REFERENCES word(rowid), FOREIGN KEY(label_id) REFERENCES label(rowid)", "label_id"],
#         }

if __name__ == '__main__':
    main()