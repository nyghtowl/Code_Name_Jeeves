from flask import render_template, flash, redirect, session, url_for, request, jsonify, g
from flask import Response
from app import app
from config import twilio_client, TWILIO_NUM, MY_NUM
from twilio import twiml
from twilio.rest import TwilioRestClient


@app.route("/jeeves_notification")
def jeeves_notification():
    '''
    Jeeves provides updates
    '''
    response = twiml.Response()

    # Change these vars - from number will be my num
    from_number = str(request.values.get('From', None))
    body = request.values.get('Body', None)
    simplify_body = simplify_txt(body)

    client.messages.create(
        from_ = TWILIO_NUM,
        to = from_number,
        body = message
    )
    return ''