# Code Name Jeeves 
Zipfian Final Project Spring 2014

What is it?
--------

Jeeves is an email natural language classifier that finds messages that need a meeting location defined.

I built this as my Zipfian final project because I want my computer to do more things for me. Why not when it has all this data on me. I made a long wish list of items and then focused on getting my computer to read emails and classify the ones that meeting location defined. Then it sends me a text if an email is classified as true. 

If there was enough time I wanted the program to take the next step of finding a couple recommendations on locations. Just getting the classifier working was plenty for the two weeks we had for the project. 

This project is similar to spam filters where a false positive (getting texts on email incorrectly classified as true) is more acceptible than a false negative (missing an email that needs a location)


Main Structure:
--------

- Handle_email.py, common.py & twilio_views.py in the app folder are the main files that run the application. 
    - Handle_email:
        - Main file that runs the full project pipeline from getting the email to sending a text 
        - Call check_email function to get the program to check for new gmails
        - There is a pickle file of the last time that the email was checked that will be upated after new email is found
        - If there are new emails then the data is parsed out and cleaned
        - The modified tf-idf vectorizer is unpickled and applied to the email message to generate features
        - The modified logistic regression model is unpickled and the email features are run through the logistic regression for a prediction
        - If the prediction is true then a message is created
        - The message is sent through the jeeves_notifications function under the Twilio views
        - Also the message is passed to the unix say command so my computer says that a new email needs a meeting location defined
            - [See a screen capture of the project in action](http://nyghtowl.io/2014/03/30/jeeves-is-talking/)

    - Common:
        - Holds common functions that are used for the project pipeline and for building out and testing the vectorizer and classifier
        - Pickle functions
        - API call to get new gmails
        - Function to clean data (strip whitespace and characters that are not UTF8, etc.)
        - Store data in Postgres
        - Put data into Pandas dataframe
        - Generate dataset splits for X, y and cross validation

    - Twilio_views: 
        - Passes the message through the Twilio API to turn it into a text message on my phone

More Information:
--------

Check out [nyghtowl.io](http://nyghtowl.io) for my blog posts on my progress with developing Jeeves:
- [Starting the project](http://nyghtowl.io/2014/03/16/begin-with-the-end/)
- [Completing the project pipeline](http://nyghtowl.io/2014/03/23/zipfian-project-week-1-closing-the-loop/)
- [Fine tuning](http://nyghtowl.io/2014/03/30/jeeves-is-talking/)

Future Plans:
--------

- Try Porter for stemming

- Continue to code for producing the model:
    - Grid Search again to improve parameters
    - plot error rate and learning curves (regularization sprint)
    - explore other ways to apply k-fold

- Other feature ideas / customization
    - Make date in body of text more informatitve
    - Python NLP - Regex library
    - length of thread (3+)
    - # email from an email address / new or not
    - email address in contacts or on Linkedin...

- Closing loop with a cronjob to automate email checking

- Use Ec2, picloud(install package - pass function and arguments and it goes up to EC2) to run models / esp grid search

- Continue to look at dimensionality reduction approaches

- Turn pipeline sections into classes/objects?

- Take model to another level
    - use partial predict so can take in feedback on new results and if they are correct (Adam gave this idea)
    - make the vectorizer/classifier result one feature and then add in other engineered features and feed through another classifer



Data Storage Structure:
--------

Raw data stored in Postgres DB

~1000+ emails total
~120 emails meet target conditions (may be smaller based on thread emails)

Raw Data:
- Message ID (message_id) = string / primary key
- Thread ID (thread_id) = string
- To (to_email) = string
- From (from_email) = string
- CC (cc) = string
- Date Sent (date) = timestamp
- Subject (subject) = string
- Starred Email (starred) = boolean (true meanss its starred)
- Message Body (body) = text
- Message Subject & Body (sub_body) = text
- Email Owner (email_owner) = string (email source)
- Box (email box) = string
- Needs Location (target) = boolean (true needs a meeting location and based on labels 'Jeeves' & 'Starred')


Support:
--------
- Any bugs about Markdown Preview please feel free to report [here](https://github.com/nyghtowl/Code_Name_Jeeves/issues)
- And you are welcome to fork and submit pullrequests


Copyright:
--------

Copyright (c) 2014 Nyghtowl