# Config
# -*- coding: utf-8 -*-

import os

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

twilio_client = TwilioRestClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
