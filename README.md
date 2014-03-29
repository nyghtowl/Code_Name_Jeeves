# Code Name Jeeves 
Zipfian Final Project

General idea idea is to get my computer to do something smart. Narrowed down to have my computer classify incoming emails that need a meeting location defined.

------------

To Do:

- Improve features
    - Pass subject and body through model
    - Has date in text body
    - combine subject with body words (also run with separate models)

- get new data to test the validity of the classifier

- Improve code for producing the model and plot results to compare
    - Grid Search again to improve parameters

- Improve models
    - Find an output for comparison of models
        - plot the error (or accuracy) of the classifier as the number of training examples increases (learning curves) (use regularization sprint)
    - Work on train/test split and applying k-folds / stratified


As Time Permits:

- Other feature ideas / customization
    - Python NLP / Regex library
    - length of thread (3+)
    - # email from an email address / new or not
    - email address in contacts or on Linkedin...

- Closing loop - check into cronjobs

- Setup code for my computer to "say" the classification outcome whent its true

- Use Ec2, picloud(install package - pass function and arguments and it goes up to EC2) to run models / esp grid search

- Continue to look at dimensionality reduction approaches

- Turn pipeline sections into classes/objects?



Visualization Ideas:
- map words and features to coefficients
- visualize dominant coeef - % positive emails has this dominant cooef

---------------
Issues:
- Limited data for significant enough results (potential for overfitting and/or low accuracy)
- Duplication of data where there is a row per email no matter if its in a thread and emails typically contain duplications of the content from previous emails
    - should just pick one email out of  thread and one that has the latest date
    - Granted this is allowing resampling of the data
- Facet Plot seems too extreme for words. Too many features and what value is there plotting features against features


- Resolved Anaconda, virtualenv and ipython issues
- Resolved science packages work in virtualenv

---------------
Data:

Stored in Postgres DB

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
    - Using use_idf drops the values on words like my name that show up across the corpus
- Updated raw data structure to track where emails are from
- Confirmed the model results and how to handle the classification - fixed it basically
- Reloaded the raw data to pull out dates and repeated script
- Reworked the script to make more encapsulated and abstracted. Consolidated certain functions where possible and fixed variable names to improve code clarity
- Built out grid search script to run pipeline
- Set stage if I want to run multiple models with grid search
- Used random parameter to maintain split and changed test train split to take place after vectorized
- Tried different approaches with vectorizer
    - Utilized subject and body text in vectorizer
    - Expanded the num of vectorizer features and pulled strip out of word adjustment because seemed to be collapsing words incorrectly 
- Added in cross validation scores for output

---------------
Stuff:
- Think of this project similar to spam filters where a false positive (email sent that needs a location) is more acceptible than a false negative (email not sent that needs a location)

- To start ngrok use:
    ./ngrok 80

- Applied cPickle to save models because fast and commone solution. Sklearn has joblib which is similar and good for big data

- python sendmail - send an email


---------------
Points to Note:

Tfidf - very min impact if do all features vs max 10000
