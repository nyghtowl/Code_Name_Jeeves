'''
Jeeves Project Pipeline

Get emails, vectorize, classify and text if true
'''

from twilio_views import jeeves_notifications
from datetime import datetime
from config import pkl_dir
import app.common as cpm
import gmail


import cPickle 
import os

def check_email():
# set script to run at intervals (time.sleep?) - morn, mid-day, afternoon

    current_time = datetime.now()
    previous_check = cpm.unpickle(os.path.join(pkl_dir, 'last_check_time.pkl'))
    print previous_check
    emails = cpm.get_emails('INBOX', previous_check)

    if not emails: # avoid unpickling if empty inbox
        return

    feature_model = cpm.unpickle(os.path.join(pkl_dir, 'final_vec.pkl'))
    classifier_model = cpm.unpickle(os.path.join(pkl_dir, 'final_model.pkl')) # confirm location

    for email in emails:
        if email.sent_at > previous_check:
            clean_b = cpm.clean_raw_txt(email.body)
            # print 'clean_email', email.body
            email_features = feature_model.transform(clean_b)
            # print 'email_bag', email_features.shape
            classifier_result = classifier_model.predict(email_features)
            #classifier_result
            eval_email(classifier_result.mean(), email)
    
    cpm.pickle(current_time, os.path.join(pkl_dir,'last_check_time.pkl'))

def eval_email(model_result, email):
    print "in eval", model_result == True
    print 'model_result', email
    if model_result:
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