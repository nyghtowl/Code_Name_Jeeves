'''
Creating Postgres Structures

Storing raw(ish) email data

'''

from config import connect_db
import psycopg2
import datetime
import re
import pdb
import app.common as cpm


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
    # Use nvarchar in case storing non-english data

    for idx, (name, params) in enumerate(table_data.iteritems()):
        create_table(name, params[0])
        create_index(name, params[1])

# Getting and storing the raw data

def store_email(email):
    target, starred = False, False
    store_e = '''
        insert into raw_data 
        (message_id, thread_id, to_email, from_email, cc, date, subject, starred, body, target)
        values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    '''
    # in psycopg - var placeholder must be %s even with int or dates or other types

    if 'Jeeves' in email.labels and '\\Starred' in email.labels:
        target = starred = True
    elif '\\Starred' in email.labels:
        starred = True

    # Could apply executemany with query and list of values but still need to do fetch first and apply changes
    try:
        with connect_db() as db:
            db.execute(store_e, (email.message_id, email.thread_id, email.to, email.fr, email.cc, email.sent_at, email.subject, starred, cpm.clean_raw_text(email.body), target))
    except psycopg2.IntegrityError:
    # if exists then skip loading it
        pass

def store_postgres(box, date=datetime.datetime.now()):
    emails = cpm.get_emails(box, date)
    for email in emails:
        store_email(email)

def main():
    table_data = {
                "raw_data": [
                "message_id varchar(255) not null, \
                thread_id varchar(255), \
                to_email text, \
                from_email varchar(255), \
                cc varchar(255), \
                date timestamp, \
                subject text, \
                starred boolean, \
                body text, \
                target boolean, \
                PRIMARY KEY(message_id)", "message_id"]
            }
    build_tables(table_data)
    store_database('INBOX', datetime.date(2013, 5, 1))
    store_database('Career', datetime.date(2013, 5, 1))
    store_database('Hackday_Group', datetime.date(2013, 5, 1))
# conn.close()

if __name__ == '__main__':
    main()