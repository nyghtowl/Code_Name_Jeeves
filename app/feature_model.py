'''
Jeeves Project

Build and store features - create cpt tables and leverage naive bayes sql for this section

'''
from config import conn, db

from time import time




# table_data = {
#                 "priors": ["label_id integer, p_label float, FOREIGN KEY(label_id) REFERENCES label(rowid)", "p_label"],
#                 "cpts": ["word_id integer, label_id integer, p_word_label float, FOREIGN KEY(word_id) REFERENCES word(rowid), FOREIGN KEY(label_id) REFERENCES label(rowid)", "label_id"],
#             }


#conn.close()