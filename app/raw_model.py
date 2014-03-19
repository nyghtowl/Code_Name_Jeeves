from config import conn, db, TEST_GMAIL_ID, TEST_GMAIL_PWD
import psycopg2
import gmail
import datetime
import re
import pdb

'''
Creating and storing raw(ish) data.
'''

# Building the structure

def create_table(table_name, values):
    db.execute('''create table %s(%s)''' % (table_name, values))

def create_index(table_name, idx_col):
    db.execute('''create index id_%s on %s (%s);''' % (table_name, table_name, idx_col)) 

def drop_table(name):
    db.execute('''DROP TABLE IF EXISTS %s;''' % (name))

def build_tables(table_data):
    # Use nvarchar in case storing non-english data

    for idx, (name, params) in enumerate(table_data.iteritems()):
        create_table(name, params[0])
        create_index(name, params[1])

# Getting and Storing the Data

def clean_body(body):
    # sample string literal issues found 0xc2, 0x20, \xa9, 0x80
    if body:
        body = body.replace('\r', '').replace('\n', ' ')
        body = re.sub(r'[\x80-\xff]', r'', body)
        body = re.sub(r'\s+', ' ', body)
    return body

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
        db.execute(store_e, (email.message_id, email.thread_id, email.to, email.fr, email.cc, email.sent_at, email.subject, starred, clean_body(email.body), target))
    except psycopg2.IntegrityError:
    # if exists then skip loading it
        pass

def get_emails(box, date=datetime.datetime.now()):
    gmail_conn = gmail.login(TEST_GMAIL_ID, TEST_GMAIL_PWD)
    emails = gmail_conn.label(box).mail(after=date)

    print "num emails:", len(emails)
    for email in emails:
        # Call fetch to pull in email attributes
        email.fetch()
        store_email(email)

    gmail_conn.logout()
    print "Logged in?", gmail_conn.logged_in

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
    get_emails('INBOX', datetime.date(2014, 3, 16))
    get_emails('Career')
    get_emails('Hackday_Group')
# conn.close()