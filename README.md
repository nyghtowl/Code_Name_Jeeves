# Code Name Jeeves 
Zipfian Final Project

General idea idea is to get my computer to do something smart. Narrowed down to have my computer classify incoming emails that need a meeting location defined.

------------

To Do:

- Improve features
    - Has date in text body
    - combine subject with body words (also run with separate models)
    - Remove certain words that are skewing the results - enable idf : inverse doc frequency reweighting? but does this impact meeting
        - my name (so it is more abstract)


- Closing loop - check into adding time.sleep and make sure gmail object working with script abstraction


- Improve models
    - Improve script on how multiple models are run and results shown
        - Look at EC2 to help run
    - Work on tuning parameters - look at other ways to optimize alpha for naive bayes
    - Find an output for comparison of models
        - plot the error (or accuracy) of the classifier as the number of training examples increases (learning curves) (use regularization sprint)
    - Work on train/test split and applying k-folds / stratified


As Time Permits:
- Look at pulling in other data sources
- Update raw data structure to track where emails are from

- Other feature ideas / customization
    - Python NLP / Regex library
    - length of thread (3+)
    - # email from an email address / new or not
    - email address in contacts or on Linkedin...
- Run more EDA


- Continue to look at dimensionality reduction approaches
- Turn pipeline sections into classes/objects

- Setup code for my computer to "say" the classification outcome whent its true
- Find way to stream in new email



Visualization Ideas:
- map words and features to coefficients
- visualize dominant coeef - % positive emails has this dominant cooef

---------------
Issues:
- Limited data for significant enough results (potential for overfitting and/or low accuracy)
- Duplication of data where there is a row per email no matter if its in a thread and emails typically contain duplications of the content from previous emails
    - should just pick one email out of  thread and one that has the latest date
    - Granted this is allowing resampling of the data
- Facet Plot seems to extreme for words. Too many features and what value is there plotting features against features


- Resolved Anaconda, virtualenv and ipython issues
- Resolved science packages work in virtualenv

---------------
Data:

Stored in Postgres DB

~800+ emails total
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
- Needs Location (target) = boolean (true needs a meeting location and based on labels 'Jeeves' & 'Starred')

---------------
Completed:
- Setup Twilio text message structure for results
- Picked Gmail package to pull gmail and applied
- Labeled messages that meet conditions / classify true & determined how to designate results in raw data database (target column)
- Built & rebuilt Postgres table
- Data munged and loaded raw data into Postgres
- Pulled in science packages for virtualenv (resolved matplotlib issues after some experimenting with creating symbolic links to anaconda)
- Figured out how to pull postgres into pandas
- Ran initial EDA and confirmed dataset small as well as ways to consider subsetting. Also found a few issues with the data like empty cells.
- Started vectorization and began dimensionality reduction 
- Realized dimensionality not as valuable as thought but will keep it as back pocket item
- Built initial classification models (review and leverage work from nlp project)
    - Logistic Regression & Naive Bayes
    - Built classification model analysis with accuracy, roc plot, etc.
- Built out code to fetch new email, check email and send text if true condition met - need to test
- Built out alternative classification models: Random Forest, Gradient Boost, Ada Boost and SVC. Gradient Boost gave best results with plain out of the box
- Applied grid search which improved Logistic Regression and Multinomial NB. 
- Applied grid search to Random Forest and SVC. SVC had the best result. Gradient Boost is taking too long to run right now.
- Closed the loop and proved to take in an email classify and send a text
- Cleaned up code / streamlined functions and connections between scrips.
- Improved feature set with:
    - stemming
    - n-grams up to 3
    - lowercase
    - use_idf
    - normalizing the data
    - word shape adjustment



---------------
Stuff:
- Think of this project similar to spam filters where a false positive (email sent that needs a location) is more acceptible than a false negative (email not sent that needs a location)

- To start ngrok use:
    ./ngrok 80

- Applied cPickle to save models because fast and commone solution. Sklearn has joblib which is similar and good for big data

- python sendmail - send an email
