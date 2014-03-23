'''
Data EDA file - using pandas to look at data

'''

import pandas as pd
import numpy as np
import seaborn

import matplotlib.pyplot as plt


# Explore Data

def eda(df):
    print "Summary Stats", df.describe()
    print "Shape", df.shape

    print "# email threads", len(df.thread_id.unique())
    print "# email threads that meet conditions", len(df[df.target == True].thread_id.unique())

    print "Top 5 rows", df.head()
    print "Bottom 5 rows", df.tail()


def plot_line(df, data_title, data_xlabel, data_ylabel, save_name=None):
    data_ln = df.plot()
    data_ln.set_xticks(rotation=70)
    data_ln.set_title(data_title, fontsize=14)
    data_ln.set_xlabel(data_xlabel)
    data_ln.set_ylabel(data_ylabel)    
    if save_name:
        data_ln.savefig(save_name)
    #data_ln.show()

def plot_hist(df, data_title, data_xlabel, data_ylabel, bin=7, save_name=None):
    #temp['weekday'] = data[data['target']== True]
    # ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']
    hist_ex = df.hist(bins=bin)
#    hist_ex.set_xticklabels(xlabels)
    hist_ex.set_title(data_title, fontsize=14)
    hist_ex.set_xlabel(data_xlabel)
    hist_ex.set_ylabel(data_ylabel)
    if save_name:
        hist_ex.savefig(save_name)
    #hist_ex.show()

def add_col(df, col_name, val):
    df[col_name] = val
    return df

def del_col(df, col_name):
    del df[col_name]

def del_row(df, row_name):
    del df[row_name, :]


def main():
    pass

if __name__ == '__main__':
    main()