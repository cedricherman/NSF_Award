# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 03:10:50 2017

@author: herma
"""


# customize CountVectorizer to add stemmer or lemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import re
# lemmatization to convert plurals words to singular word
from nltk.stem.wordnet import WordNetLemmatizer
Lem = WordNetLemmatizer()
#Lem.lemmatize('tries')
# stemming via snowball
#from nltk.stem.snowball import SnowballStemmer
#stem = SnowballStemmer('english')
#stem.stem('information')



# create a class that superseed CountVectorizer
class CustomVectorizer(CountVectorizer):
  # override build_tokenizer() that is part of CountVectorizer.fit_transform()
  def build_tokenizer(self):
    # tokenize is the return value of build_tokenizer() from CountVectorizer
    # tokenize is a lambda function that intends to receive a string
    tokenize = super(CustomVectorizer, self).build_tokenizer()
    # tokenize(doc) is a list of words (=tokens)
    return lambda doc: list(custom_stemming(tokenize(doc)))

# generator function for enhanced tokenizer
def custom_stemming(tokens):
  for t in tokens:
    t = Lem.lemmatize(t)
#    t = stem.stem(t)
    # return one word each time custom_stemming() gets called
    # it gets called by list iteratively
    yield t

# custom preprocessor:
def remove_Tag_Http(x):
  # remove all <anything> occurence
  tagFree = re.sub(r'<.+?>', '', x).lower()
  # remove all http web link, lots of unique words
  return re.sub(r'http.+?', '', tagFree)


#### The Main program, can be used as a script or as a module
if __name__ == "__main__":
	pass

