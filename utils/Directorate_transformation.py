# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 23:52:41 2017

@author: herma

There are 7 directorates in the NSF Organization
Each directorate has multiple division.

There is also a number of offices which we will group together 
and treat them at the same level as directorate

"""

# import to read and filter data
import pandas as pd
import re
import itertools
from sklearn.feature_extraction import stop_words

    
def clean_text(text):
    """
    remove punctuations and stop words for further processing
    """
    if type(text) is str:
        # split text on alphanumeric characters only
        text_tokens = re.findall('\w+', text)
        # remove stopwords
        text_tokens_sw = [s for s in text_tokens if s not in stop_words.ENGLISH_STOP_WORDS]
        # reunite text
        return' '.join(text_tokens_sw)
    else:
        return text
    
def are_letters_common(abbr, full_word):
    """
    returns true if all letters in abbreaviation are present in full word
    """
    # check if all letters are in full_word
    for l in list(abbr):
        if l not in full_word:
            return False
    # true if loop completed (all letters in full)
    return True
    
def find_abbreviations(ValCount_dict):
    """
    make pairs of abbreviated, non-abbreviated names
    """
    # make two lists: abbrevation list and replacement list
    abbreviation = []
    replacement = []
    # group dict by value
    for w_cnt in range(min(ValCount_dict.values()), max(ValCount_dict.values())+1):
        # make a list of keys which have the same count
        list4pairs = [kp.split() for kp,vp in ValCount_dict.items() if vp == w_cnt]
        # make a list of pair combinations
        pairs = list(itertools.combinations(list4pairs , 2))
        # compare pairs
        for p in pairs:
            abbre_list =[]
            repl_list = []
            # compare word by word
            for w in range(w_cnt):
                # get abbreviated word and longer word (full)
                if len(p[0][w]) >=  len(p[1][w]):
#                   wlen = len(p[1][w])
                    abbre_word = p[1][w]
                    full_word = p[0][w]
                else:
#                   wlen = len(p[0][w])
                    abbre_word = p[0][w]
                    full_word = p[1][w]
                # do they have the same root?
    #           if abbre_word == full_word[:wlen]:
                # test if all letters in abbre_word are in full_word
                if are_letters_common(abbre_word,full_word):
                    abbre_list.append(abbre_word)
                    repl_list.append(full_word)
                else:
                    # root is different, move on
                    # decrement w to indicate loop ended with a break statement
                    w -= 1
                    break
            # if for loop complete, concatenate word list
            if w == w_cnt-1:
                abbreviation.append(' '.join(abbre_list))
                replacement.append(' '.join(repl_list))
    # return two list
    return abbreviation,replacement


def get_Directorate(filename):
    """
    return consolidated directorate names records
    """
    # read database (csv file)
    df_core = pd.read_csv(filename,header=0, encoding = 'utf-8')
    # Take care of directorate
    # set string to lower case, NOTE: lambda must have if AND else statement
    # HAVE TO LOWER TEXT FIRST TO REMOVE STOP WORDS LATER!!!!!
    df_core.Directorate_Name= df_core.Directorate_Name.str.lower()
    df_core.Directorate_Name = df_core.Directorate_Name.apply(clean_text)
    # merge abbreviations
    # get all possible directorate name
    df_directValCount = df_core.Directorate_Name.value_counts()
    # create directorate dict, value is name length
    Directorate_lendict = {u: len(u.split()) for u in df_directValCount.index}
    # figure out directorate name that matches non abrreviated name
    abbreviation, replacement = find_abbreviations(Directorate_lendict)
    # replace each abbreaviation by full name
    df_core.Directorate_Name.replace(to_replace=abbreviation, value=replacement,\
                                     inplace=True, method='pad')
    # merge all offices together and treat it as one directorate
#    df_core.Directorate_Name = \
#        df_core.Directorate_Name.apply(lambda x: 'office'\
#                                        if type(x) is str and 'office' in x else x )
    # remove all nan records
    df_core.dropna(subset=['Directorate_Name'], inplace=True )
	
    # removes all office!
    df_core = df_core[ ~df_core.Directorate_Name.str.contains('office',\
													 case=False)]
	
    # returns pandas data frame
    return df_core.iloc[:, df_core.columns.get_indexer(\
                                                    ['AwardID', 'Directorate_Name'])] 

#### The Main program, can be used as a script or as a module
if __name__ == "__main__":
    # get entire corpus
    directorate = get_Directorate('../DB_1960_to_2017.csv')
    print( directorate.count() )


