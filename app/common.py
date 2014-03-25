'''
Common Project Methods

'''

from config import TEST_GMAIL_ID, TEST_GMAIL_PWD, connect_db
import pandas
import datetime
import gmail
import re

import cPickle # same as pickle but faster and implemented in C

# Context manager for context - with database 

def pickle(stuff, filename):
    # use pkl for the filename and write in binary
    with open(filename, 'wb') as f:
        cPickle.dump(stuff, f)

def unpickle(filename):
    with open(filename, 'rb') as f:
        return cPickle.load(f)

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
        # Consider dropping everything after On since its a repeat <(On).+>
        text = re.sub(r'(On)\s\S+\s\S+\s\S+\s\S+\s\S+\s\S+\s(PM|AM)(,)', r'', text)
        text = text.replace('\r', '').replace('\n', ' ')
        text = re.sub(r'[\x80-\xff]', r'', text)
        text = re.sub(r'\s+', ' ', text)
    return text

# Load data from sql to pandas - add limit if there is too much data or only want to look at subset

def load_data(table):
    with connect_db() as db:
        db.execute('''SELECT * from %s''' % table)
        return pandas.DataFrame(db.fetchall(), columns=['message_id', 'thread_id', 'to_email', 'from_email', 'cc', 'date', 'subject', 'starred', 'body', 'target'])

def load_emails_pd():
    index_col = 'message_id'
    raw_df = load_data('raw_data')
    raw_df = raw_df.set_index(index_col, drop=True, verify_integrity=True)
    raw_df['weekday'] = raw_df['date'].apply(lambda d: d.weekday())
    raw_df['body'] = raw_df['body'].fillna(value='empty')
    #    print data['body'].isnull().sum()
    return raw_df

if __name__ == '__main__':
    main()