# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 18:03:30 2017

@author: herma

Standard Grant 
NSF provides a specific level of support for a specified period of time with
no statement of NSF intent to provide additional future support without
submission of another proposal.

Continuing grant
NSF provides a specific level of support for an initial specified period of time,
usually a year, with a statement of intent to provide additional support of the
project for additional periods, provided funds are available and the results
achieved warrant further support.

Fixed Price Award
Fellowship  (similar to standard grant)
The purpose of the NSF Graduate Research Fellowship Program (GRFP) is to help
ensure the vitality and diversity of the scientific and engineering workforce
of the United States. The program recognizes and supports outstanding graduate
students who are pursuing research-based master's and doctoral degrees in science,
technology, engineering, and mathematics (STEM) or in STEM education. The GRFP
provides three years of support for the graduate education of individuals who 
have demonstrated their potential for significant research achievements in STEM 
or STEM education.

Cooperative Agreement
Contract
A type of assistance award which should be used when substantial agency
involvement is anticipated during the project performance period. Substantial
agency involvement may be necessary when an activity is technically and/or
managerially complex and requires extensive or close coordination between NSF
and the awardee.
Interagency Agreement
Contract Interagency Agreement
Industry–University Cooperative Research Centers (IUCRC)
IUCRC grants are awarded via a competitive peer review process at NSF which
eliminates the need for other federal agencies to hold a competitive procurement
process. NSF cannot change the terms and conditions of the grant as a result of
the IAA. Because NSF grants are not contracts, but are financial assistance awards,
IAAs cannot be accepted that have acquisition or contractual requirements outlined
or embedded in them, such as deliverables or reporting at intervals other than
those of the original award.
Basic Ordering Agreement (BOA)/Task Order

Dicarded categories:
	GAA
	Intergovernmental Personnel Award
	Personnel Agreement ---> reimbursement

"""

# import to read and filter data
import pandas as pd


def get_Award_Instrument(filename):
	"""
	Read Award Instrument and merge or discard categorical samples
	based on reserach from NSF website.
	returns two columns named 'AwardID' and 'AwardInstrument' as pandas dataframe
	"""
	# read database (csv file)
	df_core = pd.read_csv(filename,header=0, encoding = 'utf-8')
	
	# Transform Award Instrument
	# filter out discarded Award Instrument categories
	AInstr2reject = ['GAA',\
				   'Intergovernmental Personnel Award',\
				    'Personnel Agreement']
	df_filt = df_core[~df_core.AwardInstrument.isin(AInstr2reject) ]
	
	
	# merge Fixed Price Award and Fellowship
	df_filt.AwardInstrument.replace(to_replace=['Fixed Price Award'],\
									value=['Fellowship'],\
									inplace=True, method='pad')
	
	
	# merge Cooperative Agreement, Contract, Interagency Agreement,
	# Contract Interagency Agreement (BOA)/Task Order
	Coop_list = ['Contract',\
			  'Interagency Agreement',\
			  'Contract Interagency Agreement',\
			  'BOA/Task Order']
	df_filt.AwardInstrument.replace(to_replace=Coop_list,\
									value='Cooperative Agreement',\
									inplace=True, method='pad')
	# remove nan if any
	df_filt.dropna(subset=['AwardInstrument'], inplace=True )
	# Award Instrument is down to 4 categories
	return df_filt.iloc[:, df_filt.columns.get_indexer(\
											['AwardID', 'AwardInstrument'])]


#### The Main program, can be used as a script or as a module
if __name__ == "__main__":
	# get Award Instrument
	target = get_Award_Instrument('../DB_1960_to_2017.csv')
	print(target.AwardInstrument.value_counts())
	print(target.count())









