# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 02:02:58 2017

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

#######-----------------------------------------------------------------#######
from utils import Abstract_transformation as abt
# get data set
df_corpus = abt.get_Abstract('Abstract_full_Startdate.csv')
#######-----------------------------------------------------------------#######
from utils import Directorate_transformation as dit
# get data set
df_direct = dit.get_Directorate('DB_1960_to_2017.csv')
#######-----------------------------------------------------------------#######


# MERGE
#######-----------------------------------------------------------------#######
import pandas as pd
# merge corpus and target on AwardID. AwardID is conserved
df = pd.merge(df_corpus, df_direct, how='inner', on=['AwardID'])
#######-----------------------------------------------------------------#######
# temporary downsizez of data
df = df.iloc[:10]

# LABEL
#######-----------------------------------------------------------------#######
target = df.Directorate_Name
# label those categories
from sklearn.preprocessing import LabelEncoder
Directorate_encoder = LabelEncoder()
Directorate_coded = Directorate_encoder.fit_transform(target)
target_names_list = Directorate_encoder.classes_
#######-----------------------------------------------------------------#######


## #############################################################################
# Do the actual clustering
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import accuracy_score
from time import time
from scipy.stats import mode
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# need a vectorizer
vectorizer = utilsvectorizer.CustomVectorizer( min_df = 1,
							  max_df = 1.0,\
							  analyzer = 'word',\
							  stop_words = sw,\
							  strip_accents = 'unicode',\
							  token_pattern = r'(?u)\b[a-zA-Z][a-zA-Z]+\b',\
                        preprocessor = utilsvectorizer.remove_Tag_Http )
# proceed with Count vectorizer first
Abstract_features_array = vectorizer.fit_transform(df.Raw_Abstract)
# TFID vectorizer
# instantiate tfidf, normalize counts and lower weights of high frequencies
tfidf_transformer = TfidfTransformer()
X = tfidf_transformer.fit_transform(Abstract_features_array)
print(X.shape)

# number of label representing directorate
labels = df_direct.Directorate_Name.unique()
# number of expected cluster
true_k = df_direct.Directorate_Name.unique().shape[0]
# use Kmeans unsupervised learning
# verbose produces looging info
km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                verbose=0, n_jobs = -1)
#km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
#                         init_size=1000, batch_size=1000, verbose=opts.verbose)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))
print()

from sklearn.externals import joblib
joblib.dump(km, 'Kmeans_default_Model.pkl')
#km = joblib.load('Kmeans_default_Model.pkl') 

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))

print()

print("How well does it match our ")
# does it match our Directorate?
pred_clusters = km.predict(X)
Pred_DirectName_coded = np.zeros_like(pred_clusters)
Clusters_names = np.zeros((1, true_k))
for i in range(true_k):
    mask = (pred_clusters == i)
    Pred_DirectName_coded[mask] = mode(Directorate_coded[mask])[0]
    Clusters_names[i] = mode(Directorate_coded[mask])[0]
# recover names of directorate
Pred_DirectName = Directorate_encoder.inverse_transform(Pred_DirectName_coded)
# print cluster number and most frequent directorate name
print('Cluster #{}, Associated most frequent Directorate: {}'.format(\
	  np.arange(true_k),Clusters_names))
# accuracy score
score = accuracy_score(Directorate_coded, Pred_DirectName_coded)
print('Accuracy score = {:.2f}'.format(score))
# confusion matrix
mat = metrics.confusion_matrix(Directorate_coded, Pred_DirectName_coded)
print(mat)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=target_names_list,
            yticklabels=target_names_list)
plt.xlabel('true label')
plt.ylabel('predicted label');


print("Top terms per cluster:")
# clusters center should be an array of true_k * vocab
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = tfidf_transformer.get_feature_names()
for i in range(true_k):
        print("Cluster %d:" % i, end='')
		  # print the top 10 words for each cluster
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()
		
		
		
		
		