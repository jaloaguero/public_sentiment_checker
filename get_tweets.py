import tweepy
from tweepy import OAuthHandler, API

import csv

import sys


auth = tweepy.OAuth2BearerHandler("AAAAAAAAAAAAAAAAAAAAAKgKjQEAAAAAomhGNF2%2BtkXZbc%2Fdbc1%2Byn7fiCY%3DsxHaIkSh9BY8DDzXLk9aO4l1RDt0kZHQpXUmqX2YTUGU8AlumI")

api = API(auth)
user_input = input("Enter a name of a person, or company: ")
GIVEN_WORD = user_input
#comment above 2 lines & uncomment line below when ur ready to hook this up
#GIVEN_WORD = str(sys.argv[1])

#retweets are messy so we get rid of them, should be ok
QUERY = GIVEN_WORD+" -filter:retweets"

tweets = api.search_tweets(QUERY, lang="en",tweet_mode="extended", count=1000, result_type="recent")


# open the file in the write mode
with open('new_tweets.csv', 'w', encoding='UTF8') as f:
    # create the csv writer
    writer = csv.writer(f)

    # write a row to the csv file


    writer.writerow(['tweet']+['label'])


    for i in tweets:
        writer.writerow([i.full_text]+[1])

