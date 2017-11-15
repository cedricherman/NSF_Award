# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 21:33:23 2017

@author: herma
"""

# import to read and filter data
import pandas as pd

def get_Abstract(filename, nrecord=None):
	"""
	This function loads a csv file in memory,
	removes records/lines/samples which are missing Abstract
	returns two columns named 'AwardID' and 'Raw_Abstract' as pandas dataframe
	
	Required argument: 
		file name with relative or full path
		
	Optional argument:
		number of non-empty Abstract records desired
		
	"""
	# get abstract data
	# number of abstract present = 327,825 over 438,352
	df = pd.read_csv(filename,\
					 header=0,\
					  encoding = 'utf-8',\
					  nrows = None)
	# replace nan values by empty string
	# read_csv() replaces empty string by nan automatically, force it back to ''
	#df.fillna(value='', inplace=True)
	# discard records that have no abstract
	df.dropna(subset=['Raw_Abstract'], inplace=True )
	# only get a portion of data
	if nrecord is not None:
		return df.iloc[:nrecord, df.columns.get_indexer(['AwardID', 'Raw_Abstract'])] 
	else:
		return df.iloc[:, df.columns.get_indexer(['AwardID', 'Raw_Abstract'])] 



#### The Main program, can be used as a script or as a module
if __name__ == "__main__":
	# get entire corpus
	corpus = get_Abstract('Abstract_full_Startdate.csv', nrecord=int(1e5))
	print( corpus.count() )
