from config import conn, db, TEST_GMAIL_ID, TEST_GMAIL_PWD
import gmail
import datetime

'''
Creating table for a data dump
'''

def create_table(table_name, values):
    db.execute('''create table %s(%s)''' % (table_name, values))

    # conn.commit()

def create_index(table_name, idx_col):
    db.execute('''create index id_%s on %s (%s);''' % (table_name, table_name, idx_col)) 
    # conn.commit() 

def build_tables():
    # Use nvarchar in case storing non-english data
    table_data = {
                    "mw_prof_emails": ["message_id int not null,  thread_id int, to_email varchar(255), from_email varchar(120), cc varchar(255), date varchar(64),subject varchar(120), body text, PRIMARY KEY(message_id)", "message_id"]
                }

    for idx, (name, params) in enumerate(table_data.iteritems()):
        create_table(name, params[0])
        create_index(name, params[1])

def drop_table(name):
    db.execute('''DROP TABLE IF EXISTS %s;''' % (name))
    # conn.commit()

def store_email(email):
    store_e = '''
        insert into mw_prof_emails 
        (message_id, thread_id, to_email, from_email, cc, date, subject, body)
        values (%s, %s, %s, %s, %s, %s, %s, %s);
    '''
    # in psycopg - var placeholder must be %s even with int or dates or other types
    db.execute(store_e, (int(email.message_id), int(email.thread_id), email.to, email.fr, email.cc, email.headers['Received'], email.subject, email.body))

# may be able to pull from other folders
def get_emails():
    gmail_conn = gmail.login(TEST_GMAIL_ID, TEST_GMAIL_PWD)
    emails = gmail_conn.inbox().mail(before=datetime.date(2014, 1, 1))
    for email in emails:
        email.fetch()
        store_email(email)

    gmail_conn.logout()
    print "Check logged out.", gmail_conn.logged_in

# conn.close()