'''
Email Classification Model

'''

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

# X is bag of words per email
# Y is target true and false for each email

def create_datasets(X, y, split_size=.30):
    #X[:,np.newaxis] ?
    return train_test_split(X, y, test_size=split_size)

def build_model(model, X_train, y_train):
    #return model.fit_transform(X_train, y_train) - why is this not working?
    model.fit(X_train, y_train)
    return model

def predict_eval(model, X_test, y_test):
    # The mean square error - how much the values can fluctuate
    print('Root Mean Squared Error: %.2f' % ((np.mean((model.predict(X_test) - y_test) ** 2))**(0.5)))

    # R squared / proportion of variance explained by model: 1 is perfect prediction
    print('R squared: %.2f' % model.score(X_test, y_test))

def main():
    pass

if __name__ == '__main__':
    main()