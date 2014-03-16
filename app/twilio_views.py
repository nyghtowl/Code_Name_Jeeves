from flask import request
from app import app
from config import twilio_client, TWILIO_NUM, MY_NUM
# from twilio import twiml
# from twilio.rest import TwilioRestClient


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