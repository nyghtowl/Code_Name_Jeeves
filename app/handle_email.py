from twilio_views import jeeves_notifications
from datetime import datetime
import app.fetch_gmail as fg
import app.feature_model as fm
import gmail


import cPickle 
import re

#from email_class_model import *

def pickle_stuff(stuff, filename):
    # use pkl for the filename and write in binary
    with open(filename, 'wb') as f:
        cPickle.dump(stuff, f)

def unpickle_stuff(filename):
    with open(filename, 'rb') as f:
        return cPickle.load(f)


def clean_body(body):
    # sample string literal issues found 0xc2, 0x20, \xa9, 0x80
    if body:
        body = body.replace('\r', '').replace('\n', ' ')
        body = re.sub(r'[\x80-\xff]', r'', body)
        body = re.sub(r'\s+', ' ', body)
    return body

def check_email():
# set script to run at intervals (time.sleep?) - morn, mid-day, afternoon

    current_time = datetime.now()
#    previous_check = unpickle_stuff('./model_pkl/last_check_time.pkl')
    previous_check = current_time   
    emails = fg.get_emails('INBOX', previous_check)
    
    email_model = unpickle_stuff('./app/svc_322_2.pkl') # confirm location

    for email in emails:
        # TO DO flip the less than symbol to greater
        if email.sent_at < previous_check:
            print email.body
            clean_b = clean_body(email.body)
            email_bag = fm.model_in_action(clean_b)
            print 'email_bag', email_bag.shape
    #         model_result = email_model.predict(email_bag)
    #         print "model result", model_result
    #         eval_email(model_result)
    # pickle_stuff(current_time, '../model_pkl/last_check_time.pkl')

def eval_email(model_result):
    print "in eval", model_result == True
    if model_result == True:
        message = create_message(email)
        print "message", message
        jeeves_notifications(message)

def create_message(email):
    # add date for meeting into message...
    fr_list = email.fr.split()
    fr = fr_list[0]
#   verify full name?
#    if not fr[1].startswith('<'):
#       fr += fr_list[1] 

    return "%s wants to meet. Where do you want to meet?" % fr

if __name__ == '__main__':
    main()