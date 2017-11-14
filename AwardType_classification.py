# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 23:52:41 2017

@author: herma
"""

# import custom vectorizer and associated function
from utils import utilsvectorizer

# don't use use stopwords from nltk
#from nltk.corpus import stopwords
#sw = stopwords.words('english')
## sklearn stopword list is more extensive, ENGLISH_STOP_WORDS is the same
## as stop_words='english' for CountVectorizer
from sklearn.feature_extraction import stop_words
# add list of first names from nltk, ATTENTION names has duplicates!!! use union()
# and each name starts with a CAPITAL LETTER
from nltk.corpus import names
firstname_corp = [na.lower() for na in names.words()]
sw = stop_words.ENGLISH_STOP_WORDS.union(firstname_corp)


# GET DATA
#######-----------------------------------------------------------------#######
from utils import Abstract_transformation as abt
# get data set
df_corpus = abt.get_Abstract('Abstract_full_Startdate.csv')
#######-----------------------------------------------------------------#######
from utils import AwardInstr_transformation as awt
# get Target
df_Award_Instr_target = awt.get_Award_Instrument('DB_1960_to_2017.csv')
#######-----------------------------------------------------------------#######

# MERGE
#######-----------------------------------------------------------------#######
import pandas as pd
# merge corpus and target on AwardID. AwardID is conserved
df = pd.merge(df_corpus, df_Award_Instr_target, how='inner', on=['AwardID'])
#######-----------------------------------------------------------------#######
# temporary downsizez of data
df = df.iloc[:10]

# LABEL
#######-----------------------------------------------------------------#######
# label those categories
target = df.AwardInstrument
from sklearn.preprocessing import LabelEncoder
Award_Instr_encoder = LabelEncoder()
Award_Instr_coded = Award_Instr_encoder.fit_transform(target)
#######-----------------------------------------------------------------#######


###############################################################################
# divide data into train and test set
from sklearn.model_selection import train_test_split
# split data/target in train and test sets
corpus_train, corpus_test, target_train, target_test = train_test_split(
													df.Raw_Abstract, Award_Instr_coded,\
													 test_size=0.3, random_state=42)
# retrieve target names
target_train_names =  Award_Instr_encoder.inverse_transform(target_train)
target_test_names =  Award_Instr_encoder.inverse_transform(target_test)
target_names_list = Award_Instr_encoder.classes_

###############################################################################
# Define pipeline
# CountVectorizer--->tf-idf--->Support Vector Machine Classifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

text_clf = Pipeline([('CustomVect', utilsvectorizer.CustomVectorizer( min_df = 1,
							  max_df = 1.0,\
							  analyzer = 'word',\
							  stop_words = sw,\
							  strip_accents = 'unicode',\
							  token_pattern = r'(?u)\b[a-zA-Z][a-zA-Z]+\b',\
                        preprocessor = utilsvectorizer.remove_Tag_Http )),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, random_state=42,
                                            max_iter=5, tol=None)),
							])

## train, get model named text_clf
#text_clf.fit(corpus_train, target_train) 
## test
#predicted = text_clf.predict(corpus_test)
#print( 'Accuracy = {:.2f}'.format( np.mean(predicted == target_test) ) )
## got 0.65
## more metrics
## precision, recall, f1 score
#print(metrics.classification_report(target_test,\
#									 predicted,\
#									 target_names = target_names_list))
# confusion matrix
#mat = metrics.confusion_matrix(target_test, predicted)
#print(mat)
#sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
#            xticklabels=target_names_list,
#            yticklabels=target_names_list)
#plt.xlabel('true label')
#plt.ylabel('predicted label');
#
#
## save model
#from sklearn.externals import joblib
#joblib.dump(text_clf, 'SVM_default_Model.pkl')
##text_clf = joblib.load('SVM_default_Model.pkl') 

###############################################################################
# Grid search
from sklearn.model_selection import GridSearchCV

# ngram: used unigram (bag of word) or bigrams
# use_idf: Enable inverse-document-frequency reweighting.
# SGD alpha is the regularization constant
# pick 3 param out of 2 choices each: 2^3 = 8 possibilities
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
               'tfidf__use_idf': (True, False),
               'clf__alpha': (1e-2, 1e-3),
}

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(corpus_train, target_train)

print(gs_clf.best_score_)
for param_name in sorted(parameters.keys()):
     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
# import to pandas to see it
gs_clf.cv_results_

# save model
from sklearn.externals import joblib
joblib.dump(text_clf, 'SVM_default_Model.pkl')
#text_clf = joblib.load('SVM_default_Model.pkl') 





################################################################################
## Define pipeline
## CountVectorizer--->tf-idf--->Naive Bayes
#import numpy as np
#from sklearn.pipeline import Pipeline
#from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.naive_bayes import MultinomialNB
#import matplotlib.pyplot as plt
#import seaborn as sns
#
#text_clf = Pipeline([('CustomVect', CustomVectorizer( min_df = 1,
#							  max_df = 1.0,\
#							  analyzer = 'word',\
#							  stop_words = sw,\
#							  strip_accents = 'unicode',\
#							  token_pattern = r'(?u)\b[a-zA-Z][a-zA-Z]+\b',\
#                        preprocessor = remove_Tag_Http )),
#                      ('tfidf', TfidfTransformer()),
#                      ('clf', MultinomialNB()),
#					     ])
#	
## train, get model named text_clf
#text_clf.fit(corpus_train, target_train) 
## test
#predicted = text_clf.predict(corpus_test)
#print( 'Accuracy = {:.2f}'.format( np.mean(predicted == target_test) ) )
## 69 %
## more metrics
#from sklearn import metrics
## precision, recall, f1 score
#print(metrics.classification_report(target_test,\
#									 predicted,\
#									 target_names = target_names))
## confusion matrix
#mat = metrics.confusion_matrix(target_test, predicted)
#print(mat)
#sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
#            xticklabels=target_names,
#            yticklabels=target_names)
#plt.xlabel('true label')
#plt.ylabel('predicted label');
#
## save model
#from sklearn.externals import joblib
#joblib.dump(text_clf, 'NB_default_Model.pkl')
##text_clf = joblib.load('NB_default_Model.pkl') 


#sw = stop_words.ENGLISH_STOP_WORDS.union(firstname_corp)
## equivalent to sw = set(names.words() + list(stop_words.ENGLISH_STOP_WORDS))
## or sw = set(y.extend(list(stop_words.ENGLISH_STOP_WORDS)))  y=names.words()
## BUT sw = names.words().extend(list(stop_words.ENGLISH_STOP_WORDS)) returns None!


## use scikit-learn to derive bag of words features
## instantiate vetorizer
#vectorizer = CustomVectorizer( min_df = 1,
#							  max_df = 1.0,\
#							  analyzer = 'word',\
#							  stop_words = sw,\
#							  strip_accents = 'unicode',\
#							  token_pattern = r'(?u)\b[a-zA-Z][a-zA-Z]+\b',\
#                        preprocessor = remove_Tag_Http )
#The default regexp select tokens of 2 or more alphanumeric characters
# (?u) sets flag Unicode dependent, can be done with flag= argument
# \w\w+ any 2 or more alphanumeric characters
# \b matches empty string only at end/beginning of word
# use token_pattern to replace default pattern
# preprocessor is None by default, when specified override lowercase!!!
# tokenizer is None by default, custom callable tokenization

# Get matrix of token counts, this requires list of text as input
#Abstract_features_array = vectorizer.fit_transform(df.Raw_Abstract)
#print(Abstract_features_array.shape)
#vectorizer.get_feature_names()
#vectorizer.vocabulary_.get(u'algorithm')

# There is 327,825 non-empty abstract
# ALL ABSTRACT (438,352) with custom sw and lem:    355,506 features
# ALL ABSTRACT (438,352) with custom sw and stem:   303,323 features

# TEST FOR LAST 10,000 abstract, stopwords, max_df/min_df, stem/lem
# without stopwords:                  39,843
# with 'english' stopwords:           39,550
# with custom sw (stopwords+names):   38,294
# without spotwords but max_df = 0.8: 39,835
# without spotwords but max_df = 0.8: 39,828
# without spotwords but max_df = 0.7: 39,824
# custom sw and max_df = 0.8:         38,292
# without spotwords but stem:         29,756
# stem and sw:                        28,437

# NOTE:
# sometimes there is missing space like zylstrahighly is zylstra highly, and
# zylstra is a last name



################################################################################
## TFID vectorizer
#from sklearn.feature_extraction.text import TfidfTransformer
#
## instantiate tfidf, normalize counts and lower weights of high frequencies
#tfidf_transformer = TfidfTransformer()
#Abstract_features_array_tfidf = tfidf_transformer.fit_transform(Abstract_features_array)
#print(Abstract_features_array_tfidf.shape)

################################################################################
## Naive bayes classifier
#from sklearn.naive_bayes import MultinomialNB
#target= []
#clf = MultinomialNB().fit(Abstract_features_array_tfidf, target)
#
## need test set
#X_new_tfidf = []
#predicted = clf.predict(X_new_tfidf)
#





############# IDEAS, NOTES, etc...  ###########################################

#def to_british(tokens):
#  for t in tokens:
#    t = re.sub(r"(...)our$", r"\1or", t)
#    t = re.sub(r"([bt])re$", r"\1er", t)
#    t = re.sub(r"([iy])s(e$|ing|ation)", r"\1z\2", t)
#    t = re.sub(r"ogue$", "og", t)
#    yield t
# 
#class CustomVectorizer(CountVectorizer):
#  
#  def build_tokenizer(self):
#    tokenize = super(CustomVectorizer, self).build_tokenizer()
#    return lambda doc: list(to_british(tokenize(doc)))
#
##print(CustomVectorizer().build_analyzer()(u"color colour"))
## doctest: +NORMALIZE_WHITESPACE +ELLIPSIS [...'color', ...'color']
##cu = CustomVectorizer()
##feat = cu.fit_transform(["color colour"]).toarray()



# could have an initial Dataframe, it needs to be done in Process2struct
# and append multiple word count dictionaries to it
#df1.append(measurements, ignore_index=True)
# so that it will add columns for new words and consolidate counts
# over the whole vocabulary
# can convert final Dataframe to scipy sparse matrix
# scipy.sparse.csr_matrix(df1.values)


##TODO: load dictionaries from file
## difficult to save dictionaries of varying length to text
#from sklearn.feature_extraction import DictVectorizer
## list of dictionaries, one per record
#measurements = [{}, {}]
## create vectorizer
#vec = DictVectorizer()
## create feature array based on bag of words
## each row represent a record (Award)
#Abstract_features_array = vec.fit_transform(measurements).toarray()