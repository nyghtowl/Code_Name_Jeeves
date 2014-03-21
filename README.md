# Code Name Jeeves 
Zipfian FInal Project

Project idea is to get my computer to do something smart. Have my computer classify incoming emails that need a meeting location defined.

------------

To Do:

- Continue to work on models
    - Work on adjusting parameters
    - Find an output for comparison of models
        - plot the error (or accuracy) of the classifier as the number of training examples increases (learning curves) (use regularization sprint)
    - Try SVM, Random Forest, Ensemble, GradienBoost? and others... use gridesearch
- Work on train/test split and applying k-folds / stratified
- Add feature binary (has a date or not)
- Remove certain words that are skewing the results
    - my name (so it is more abstract)
- Close loop to bring in new email, apply classification model, use prediction to trigger whether to push data to text


As Time Permits:
- Other feature ideas / customization
    - Python NLP / Regex library
    - length of thread
    - # email from an email address / new or not
    - email address in contacts or on Linkedin...
- Run more EDA

- Look at pulling in other data sources
- Update raw data structure to track where emails are from
- Continue to look at dimensionality reduction approaches
- Turn my pipeline sections into classes/objects

Nice to haves - will work on as time permits or when need a break:

- Setup Gmail API to receive new mail
- Setup code for my computer to "say" the classification outcome whent its true


Visualization Ideas:
- map words and features to coefficients
- visualize dominant coeef - % positive emails has this dominant cooef

---------------
Issues:
- Limited data for significant enough results (potential for overfitting and/or low accuracy)
- Duplication of data where there is a row per email no matter if its in a thread and emails typically contain duplications of the content from previous emails
    - should just pick one email out of  thread and one that has the latest date
    - Granted this is allowing resampling of the data


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



