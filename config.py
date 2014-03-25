# Config
# -*- coding: utf-8 -*-

from contextlib import contextmanager

import os

import psycopg2
import sys


from signal import signal, SIGPIPE, SIG_DFL 
from twilio.rest import TwilioRestClient
#Ignore SIG_PIPE and don't throw exceptions on it... (http://docs.python.org/library/signal.html)
signal(SIGPIPE,SIG_DFL) 



SECRET_KEY = os.environ.get('key')
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')
TWILIO_NUM = os.environ.get('TWILIO_NUM')
TWILIO_APP_SID = os.environ.get('TWILIO_APP_SID')
MY_NUM = os.environ.get('MY_NUM')
TEST_GMAIL_ID = os.environ.get('test_gmail_id_1')
TEST_GMAIL_PWD = os.environ.get('test_gmailpwd_1')
TEST_GMAIL_ID_2 = os.environ.get('test_gmail_id_2')
TEST_GMAIL_PWD_2 = os.environ.get('test_gmailpwd_2')
PSQL_USER = os.environ.get('psql_user')


twilio_client = TwilioRestClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Connect to postgres
@contextmanager
def connect_db():
    try:
        conn = psycopg2.connect(database='jeeves_db', user=PSQL_USER) 
        db = conn.cursor()
        conn.set_session(autocommit=True)
        yield db
    except psycopg2.DatabaseError, e:
        print 'Error %s' % e    
        sys.exit(1)
    
    finally:
        if conn:
            conn.close()



# # Stores migrate data files
# SQLALCHEMY_MIGRATE_REPO = os.path.join(basedir, 'jeeves_db_repository')

# # Threashold for slow loading (in seconds)
# DATABASE_QUERY_TIMEOUT = 0.5