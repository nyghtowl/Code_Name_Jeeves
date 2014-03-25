'''
Creating Postgres Structures

Storing raw(ish) email data

'''

from config import connect_db
import psycopg2
import datetime
import re
import sys
import app.common as cpm
from time import time


# Building the structure

def create_table(table_name, values):
    with connect_db() as db:
        db.execute('''create table %s(%s)''' % (table_name, values))

def create_index(table_name, idx_col):
    with connect_db() as db:
        db.execute('''create index id_%s on %s (%s);''' % (table_name, table_name, idx_col)) 

def drop_table(name):
    with connect_db() as db:
        db.execute('''DROP TABLE IF EXISTS %s;''' % (name))

def build_tables(table_data):
    for idx, (name, params) in enumerate(table_data.iteritems()):
        create_table(name, params[0])
        create_index(name, params[1])

# Getting and storing the raw data

def store_email(email, box, email_owner):
    # Use nvarchar in case storing non-english data
    target, starred = False, False

    # TO DO - make table name a variable
    store_e = '''
        insert into raw_data_2 
        (message_id, thread_id, to_email, from_email, cc, date, starred, subject, body, sub_body, email_owner, box, target)
        values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    '''
    # in psycopg - var placeholder must be %s even with int or dates or other types

    # prep text for storage
    body = cpm.clean_raw_txt(email.body)
    subject = cpm.clean_raw_txt(email.subject)
    sub_body = subject + ' ' + body

    # Marks taget colum
    if 'Jeeves' in email.labels and '\\Starred' in email.labels:
        target = starred = True
    elif '\\Starred' in email.labels:
        starred = True

    # Could apply executemany with query and list of values but still need to do fetch first and apply changes
    with connect_db() as db:
        try:
            db.execute(store_e, (email.message_id, email.thread_id, email.to, email.fr, email.cc, email.sent_at, starred, subject, body, sub_body, email_owner, box, target))
        except psycopg2.IntegrityError:
        # if exists then skip loading it
            with open('../not_needed/load_errors.txt', 'a') as f:
                print "Problem loading email"
                f.write(email.message_id, email_fr, body)

def get_data(box, email_owner, date):
    start = time()
    emails = cpm.get_emails(box, date)
    end = time()
    print "Pulled gmail in %0.2fs." % (end - start)
    for email in emails:
        store_email(email, box, email_owner)
    print "Stored emails in %0.2fs." % (time() - end)

def print_table_size(table):
    with connect_db() as db:
        db.execute('''SELECT count(*) from %s''' % table)
        print db.fetchone()

def main(box, email_owner, date=datetime.datetime.now()):
    table_data = {
                "raw_data_2": [
                "message_id varchar(255) not null, \
                thread_id varchar(255), \
                to_email text, \
                from_email varchar(255), \
                cc varchar(255), \
                date timestamp, \
                starred boolean, \
                subject text, \
                body text, \
                sub_body text, \
                email_owner varchar(255),\
                box varchar(255), \
                target boolean, \
                PRIMARY KEY(message_id)", "message_id"]
            }
    # would be good to check if table exists and only create if it doesn't
    with connect_db() as db:
        db.execute('''select * from information_schema.tables where table_name=%s''', ('raw_data_2',))
        if not bool(db.rowcount):
            build_tables(table_data)

    get_data(box, email_owner, date)
    print_table_size('raw_data_2')

if __name__ == '__main__':
    main()