# Config
# -*- coding: utf-8 -*-

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
TEST_GMAIL_ID = os.environ.get('test_gmail_id_2')
TEST_GMAIL_PWD = os.environ.get('test_gmailpwd_2')
PSQL_USER = os.environ.get('psql_user')


twilio_client = TwilioRestClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Connect to postgres
conn = None

try:
     
    conn = psycopg2.connect(database='jeeves_db', user=PSQL_USER) 
    db = conn.cursor()
    db.execute('SELECT version()')          
    ver = db.fetchone()
    print ver    
    

except psycopg2.DatabaseError, e:
    print 'Error %s' % e    
    sys.exit(1)
    
    
# finally:
    
#     if conn:
#         conn.close()


# Pulled in example code I can use if I want to use sqlalchemy...
# # Code to setup a postgres database
# if os.environ.get('DATABASE_URL') is None:
#     SQLALCHEMY_DATABASE_URI = 'postgresql://localhost/jeeves_db'
# else:
#     SQLALCHEMY_DATABASE_URI = os.environ['DATABASE_URL']

# # Stores migrate data files
# SQLALCHEMY_MIGRATE_REPO = os.path.join(basedir, 'jeeves_db_repository')

# # Threashold for slow loading (in seconds)
# DATABASE_QUERY_TIMEOUT = 0.5