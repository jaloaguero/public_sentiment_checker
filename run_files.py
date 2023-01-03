#Hopefully I come back and make some front end javascript that handles this but for right now
#this will have to do
import time

exec(open('get_tweets.py').read())
#to give get_tweets.py time to write someting 
time.sleep(3)
exec(open('machine_learning.py').read())