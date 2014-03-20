# Code Name Jeeves 
Zipfian FInal Project

Project idea is to get my computer to do something smart. Have my computer classify incoming emails that need a meeting location defined.

------------

To Do:
- Develop features
- Continue to look at dimensionality reduction approaches
- Build classification model (review and leverage work from nlp project)
 

As Time Permits:
- Run more EDA

- Look at pulling in other data sources
- Update raw data structure to track where emails are from


Nice to haves - will work on as time permits or when need a break:

- Setup Gmail API to receive new mail
- Setup code for my computer to "say" the classification outcome whent its true

---------------
Issues:
- Limited data for significant enough results (potential for overfitting and/or low accuracy)
- Resolved Anaconda, virtualenv and ipython issues
- Still tbd if science packages have issues with virtualenv

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
- Started vectorization and began dimensionality reduction functions


