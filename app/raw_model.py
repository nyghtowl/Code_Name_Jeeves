from config import conn, db, TEST_GMAIL_ID, TEST_GMAIL_PWD
import gmail
import datetime

'''
Creating and storing raw(ish) data.
'''

# Building the structure

table_data = {
                "raw_data": [
                "message_id varchar(255) not null,  thread_id varchar(255), \
                to_email varchar(255), \
                from_email varchar(255), \
                cc varchar(255), \
                date timestamp, \
                subject varchar(255), \
                body text, \
                target boolean, \
                PRIMARY KEY(message_id)", "message_id"]
            }

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
    return body.replace('\r', '').replace('\xa9', '')

def store_email(email):
    target = False
    store_e = '''
        insert into mw_prof_emails 
        (message_id, thread_id, to_email, from_email, cc, date, subject, body)
        values (%s, %s, %s, %s, %s, %s, %s, %s, %s);
    '''
    # in psycopg - var placeholder must be %s even with int or dates or other types
    if 'Jeeves' in email.labels and '\\Starred' in email.labels:
        target = True
    db.execute(store_e, (email.message_id, email.thread_id, email.to, email.fr, email.cc, email.sent_at, email.subject, email.labels, clean_body(email.body), target))

def get_emails():
    gmail_conn = gmail.login(TEST_GMAIL_ID, TEST_GMAIL_PWD)
    emails = gmail_conn.inbox().mail(after=datetime.date(2014, 3, 1))
    for email in emails:
        # Call fetch to pull in email attributes
        store_email(email.fetch())

    gmail_conn.logout()
    print "Logged in?", gmail_conn.logged_in

# conn.close()