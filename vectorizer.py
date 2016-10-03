import re
import os
import pickle
from sklearn.feature_extraction.text import HashingVectorizer
cur_dir = os.path.dirname(__file__)
stop = pickle.load(open(os.path.join(cur_dir,
                                  'pkl_obj/stopwords.pkl'), 'rb'))
def tweetTokenizer(tweet):
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Replace # with HASH_TAG
    tweet = re.sub('#', 'HASH_TAG ', tweet)
    #Temporarily store emoticons
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', tweet)
    #Remove all non-word charactersand add the emoticons to 
    # the end of the processed document string
    # remove emoticon nose to be consistent
    tweet = re.sub('[\W]+', ' ', tweet) + ' ' + \
            ' '.join(emoticons).replace('-', '')
    #trim
    tweet = tweet.strip('\'"')
    tw = [w for w in tweet.split() if w not in stop]
    return tw
vect = HashingVectorizer(decode_error='ignore', n_features=2**21, preprocessor=None, tokenizer=tweetTokenizer)
