import csv

import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


#import stuff here
og_train = pd.read_csv('https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv')

#og_test = pd.read_csv('https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/test.csv')
og_test = pd.read_csv('new_tweets.csv')

train = og_train.copy()
test = og_test.copy()

#0 is positive
#1 is negative

print("The closer the number to zero, the more positively the tweets speak of them")
print("The average score seems to be at around 0.09, with more popular things scoring at around 0.01")
print("The internet is a negative place, please do not take these results as gospel.")

#Pre-Processing
################################################################################
#combines train and test to scrub certain things from datasets
combine = train.append(test,ignore_index=True,sort=True)

def remove_pattern(text,pattern):
    
    r = re.findall(pattern,text)
    
    for i in r:
        text = re.sub(i,"",text)
    
    return text

combine['Tidy_Tweets'] = np.vectorize(remove_pattern)(combine['tweet'], "@[\w]*")
combine['Tidy_Tweets'] = combine['Tidy_Tweets'].str.replace("[^a-zA-Z#]", " ")

combine['Tidy_Tweets'] = combine['Tidy_Tweets'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

tokenized_tweet = combine['Tidy_Tweets'].apply(lambda x: x.split())



from nltk import PorterStemmer

ps = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [ps.stem(i) for i in x])

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

combine['Tidy_Tweets'] = tokenized_tweet

#Bag of Words
#############################################################################

from sklearn.feature_extraction.text import CountVectorizer

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

bow = bow_vectorizer.fit_transform(combine['Tidy_Tweets'])

df_bow = pd.DataFrame(bow.todense())

#TDIF vectorizer
#################################################################################

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer(max_df=0.90, min_df=2,max_features=1000,stop_words='english')

tfidf_matrix=tfidf.fit_transform(combine['Tidy_Tweets'])

df_tfidf = pd.DataFrame(tfidf_matrix.todense())

train_bow = bow[:31962]

train_bow.todense()

train_tfidf_matrix = tfidf_matrix[:31962]

train_tfidf_matrix.todense()

#training data
###########################################################################################

from sklearn.model_selection import train_test_split

x_train_bow, x_valid_bow, y_train_bow, y_valid_bow = train_test_split(train_bow,train['label'],test_size=0.3,random_state=2)

x_train_tfidf, x_valid_tfidf, y_train_tfidf, y_valid_tfidf = train_test_split(train_tfidf_matrix,train['label'],test_size=0.3,random_state=17)

from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
Log_Reg = LogisticRegression(random_state=0,solver='lbfgs')

Log_Reg.fit(x_train_bow,y_train_bow)

prediction_bow = Log_Reg.predict_proba(x_valid_bow)

##############################################################################################
prediction_int = prediction_bow[:,1]>=0.3

# converting the results to integer type
prediction_int = prediction_int.astype(np.int)
prediction_int

# calculating f1 score
#log_bow = f1_score(y_valid_bow, prediction_int)

############################################################################################3

Log_Reg.fit(x_train_tfidf,y_train_tfidf)

prediction_tfidf = Log_Reg.predict_proba(x_valid_tfidf)

#####################################################

prediction_int = prediction_tfidf[:,1]>=0.3

prediction_int = prediction_int.astype(np.int)


# calculating f1 score
log_tfidf = f1_score(y_valid_tfidf, prediction_int)


test_tfidf = tfidf_matrix[31962:]

test_pred = Log_Reg.predict_proba(test_tfidf)

test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)

test['label'] = test_pred_int

submission = test[['label']]
submission.to_csv('result.csv', index=False)

res = pd.read_csv('result.csv')

mean = res['label'].mean()

print("Result: " + str(mean))

with open('new_tweets_test.txt', 'w', encoding='UTF8') as f:
    # create the csv writer
    f.write(str(mean))
