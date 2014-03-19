'''
Data EDA file - using pandas to look at data

'''
from config import conn, db

import pandas as pd
import numpy as np
import seaborn
# import vincent
# vincent.core.initialize_notebook()

import matplotlib.pyplot as plt

from time import time

# Load data from sql

def load_data():
    db.execute("SELECT to_email, subject from raw_data")
    return pd.DataFrame(db.fetchall(), columns=['to_email', 'subject'])


# Explore Data

def eda(data):
    print "Summary Stats", data.describe()
    print "Shape", data.shape
    print "Top 5 rows", data.head()
    print "Bottom 5 rows", data.tail()
    print "Missing Values", ((data.shape[0] * data.shape[1]) - data.count().sum())

def plot_hist(data, bin=10):
    #range(min,max+binwidth,binwidth)
    plt.hist(data, bins=bin)

def print_cloud(data):
    # Good for analyzing bag of words
    for i in range(data.shape[0]):
        word_dict = {feature_names[idx]: data[i][idx]*100 for idx in data[i].argsort()[:-20:-1]}
        cloud = vincent.Word(word_dict)
        print "Cloud", i
        cloud.display()


def main():
    df = load_data()
    return df

if __name__ == '__main__':
    main()