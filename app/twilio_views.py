from flask import request
from app import app
from config import twilio_client, TWILIO_NUM, MY_NUM
import fetch_gmail

from datetime import datetime
from email_class_model import *


def check_email():
# set script to run at intervals (time.sleep?) - morn, mid-day, afternoon

    current_time = datetime.now()
    previous_check = unpickle_stuff('./model_pkl/last_check_time.pkl')
    
    emails = fetch_gmail.get_emails('INBOX', previous_check)
    email_model = unpickle_stuff('./model_pkl/final.pkl') # confirm location
    
    for email in emails:
        if email.sent_at > previous_check:
            model_result = email_model(email)
            eval_email(model_result)
    pickle_stuff(current_time, './model_pkl/last_check_time.pkl')

def eval_email(model_result):
    if result == True:
        message = create_message(email)
        jeeves_notifications(message)

def create_message(email):
    # add date for meeting into message...
    fr_list = email.fr.split()
    fr = fr_list[0]
#   verify full name?
#    if not fr[1].startswith('<'):
#       fr += fr_list[1] 

    return "%s wants to meet. Where do you want to meet?" % fr

@app.route("/jeeves_notification")
def jeeves_notifications(message='No message submitted'):
    '''
    Jeeves provides updates
    '''
    
    twilio_client.messages.create(
        from_ = TWILIO_NUM,
        to = MY_NUM,
        body = message
    )
    return ''