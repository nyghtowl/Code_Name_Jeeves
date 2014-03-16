import os
from flask import Flask


# Initialize Flask app
app = Flask(__name__)

app.config.from_object('config') 



from app import views