'''
Common Project Methods

'''

from config import TEST_GMAIL_ID, TEST_GMAIL_PWD, connect_db, pkl_dir
from sklearn.cross_validation import train_test_split
import pandas
import datetime
import gmail
import re
import os

import cPickle # same as pickle but faster and implemented in C

# Pickle

def pickle(stuff, filename):
    # use pkl for the filename and write in binary
    with open(filename, 'wb') as f:
        cPickle.dump(stuff, f)

def unpickle(filename):
    with open(filename, 'rb') as f:
        return cPickle.load(f)

###########################################################

# Gmail

def get_emails(box, date=datetime.datetime.now()):
    gmail_conn = gmail.login(TEST_GMAIL_ID, TEST_GMAIL_PWD)
    emails = gmail_conn.label(box).mail(after=date)
    results = []
    print "num emails:", len(emails)

    for email in emails:
        # Call fetch to pull in email attributes
        email.fetch()
        results.append(email)

    gmail_conn.logout()
    print "Logged in?", gmail_conn.logged_in

    return results

def clean_raw_txt(text):
    # sample string literal issues found 0xc2, 0x20, \xa9, 0x80
    if text:
        # Dropping everything after On since its a repeat of threads
        text = re.sub(r'(On)\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s(PM|AM)(,)+', r'', text)
        text = text.replace('\r', '').replace('\n', ' ')
        text = re.sub(r'[\x80-\xff]', r'', text)
        text = re.sub(r'\s+', ' ', text)
    else:
        text = 'd' # prevent empyt values
    return text

# Load data from sql to pandas - add limit if there is too much data or only want to look at subset

###########################################################
# Pandas DataFrame

def _load_data(table, cols):
    with connect_db() as db:
        db.execute('''SELECT * from %s''' % table)
        return pandas.DataFrame(db.fetchall(), columns=cols)

def load_emails_pd(table, save=False, df_fn='pd_dataframe.pkl'):
    index_col = 'message_id'
    table_cols = ['message_id', 'thread_id', 'to_email', 'from_email', 'cc', 'date', 'starred', 'subject', 'body', 'sub_body', 'email_owner', 'box','target']
    raw_df = _load_data(table, table_cols)
    raw_df = raw_df.set_index(index_col, drop=True, verify_integrity=True)
    raw_df['weekday'] = raw_df['date'].apply(lambda d: d.weekday())
    raw_df['body'] = raw_df['body'].fillna(value='empty')
    #    print data['body'].isnull().sum()
    if save:
        pickle(raw_df, os.path.join(pkl_dir, df_fn))
    return raw_df

###########################################################
# Data Split

# Would be good to split text, cross val and train when there is more data

def define_x_y_data(data=None, save=False, x_col='body', y_col='target', data_fn='pd_dataframe.pkl', x_y_fn= 'x_y_data.pkl', vec_fn='final_vec.pkl'):
    if data is None:
        data = unpickle(os.path.join(pkl_dir, data_fn))
    vectorizer = unpickle(os.path.join(pkl_dir, vec_fn))

    X, y = vectorizer.transform(data[x_col]), data[y_col] 
    
    if save:
        pickle([X,y], os.path.join(pkl_dir, x_y_fn))
    
    return X, y

def get_x_y_data(data=None, x_y_fn= 'x_y_data.pkl', data_fn='pd_dataframe.pkl'):
    if data is None:
        data = unpickle(os.path.join(pkl_dir, data_fn))

    X, y = unpickle(os.path.join(pkl_dir, x_y_fn))

    if X.empty:
        print "New x y split saved."
        X, y = cpm.define_x_y_data(data, True)
    
    return X, y

# random of 9 provides dataset that is 93% accuracy on Logistic Regression
def create_datasets(X=None, y=None, random=None, split_size=.30, save=False, train_split_fn= 'train_split.pkl'):
    if X is None:
        X, y = get_x_y_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=random)
    if save:
        pickle([X_train, X_test, y_train, y_test], os.path.join(pkl_dir, train_split_fn))
    return X_train, X_test, y_train, y_test 


def print_test_train_shape(X_train, X_test, y_train, y_test):

    print 'X_train:', X_train.shape
    print 'X_test:', X_test.shape
    print 'y_train:', y_train.shape
    print 'y_test:', y_test.shape


def get_feature_names(vectorizer=None, vec_fn='final_vec.pkl'):
    if vectorizer == None:
        vectorizer = unpickle(os.path.join(pkl_dir, vec_fn))

    return vectorizer.get_feature_names()


if __name__ == '__main__':
    main()