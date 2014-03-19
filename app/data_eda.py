'''
Jeeves Project

Main Working File

'''

import pandas as pd
import numpy as np
import seaborn
import vincent
vincent.core.initialize_notebook()

import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

from time import time


import gmaillib

account = gmaillib.account('username', 'password') # initialize
account.inbox(start=0, amount=10) # get messages from inbox and just a range
#account.get_all_messages() # alternative
#account.get_inbox_count()
#account.filter('from:foo@bar.com') - pass email that can be filtered

def main():
    pass

if __name__ == '__main__':
    main()