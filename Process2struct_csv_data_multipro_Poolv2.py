# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 21:59:19 2017

@author: herma

FROM 1967 to 2017, total size zipped is 666-699 MB
Unzipped it is 1.42 GB
"""

#import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
import glob
import os.path

import multiprocessing
import time
import csv

# check for each tag and tag string
def checkTag(tag):
    # tag should be an element.Tag object
    if tag is None or tag.string is None:
        return None
    else:
        return tag.string.__str__()

# have to cast to unicode via str() otherwise get recursion depth error
# need to convert unicode back to beautifulSoup object
        
# sort out input_soup tags and return list of values
def extract_Xml_Tag(input_xml):
    
    # make soup and extract tags
    input_soup = BeautifulSoup(input_xml, 'lxml-xml')
    # fetch tags
    Title = checkTag(input_soup.AwardTitle)
    Eff_date = checkTag(input_soup.AwardEffectiveDate)
    Exp_date = checkTag(input_soup.AwardExpirationDate)
    # Error checking on tags that needs casting
    Amount = checkTag(input_soup.AwardAmount)
    if input_soup.AwardInstrument is not None:
        AwardInstr = checkTag(input_soup.AwardInstrument.Value)
    else:
        AwardInstr = None
    if input_soup.Organization is not None:
#        Org_code = checkTag(input_soup.Organization.Code)
#        if Org_code is not None: Org_code = int(Org_code)
        if input_soup.Organization.Division is not None:
            Org_div = checkTag(input_soup.Organization.Division.LongName)
        else:
            Org_div = None
        if input_soup.Organization.Directorate is not None:
            Org_dir = checkTag(input_soup.Organization.Directorate.LongName)
        else:
            Org_dir = None
    else:
        Org_dir = None
        Org_div = None
    if input_soup.ProgramOfficer is not None:
        NSF_officer = checkTag(input_soup.ProgramOfficer.SignBlockName)
    else:
        NSF_officer = None
    Award_ID = checkTag(input_soup.AwardID)
    if Award_ID is not None: Award_ID = int(Award_ID)
    if input_soup.Investigator is not None:
        PI_firstname = checkTag(input_soup.Investigator.FirstName)
        PI_lastname = checkTag(input_soup.Investigator.LastName)
#        if PI_firstname is not None and PI_lastname is not None:
#            PI_Fullname = PI_firstname + ' ' + PI_lastname
#        elif PI_firstname is None and PI_lastname is not None:
#            PI_Fullname = PI_lastname
#        else:
#            PI_Fullname = None
    else:
#        PI_Fullname = None
        PI_firstname = None
        PI_lastname = None
    if input_soup.Institution is not None:
        Institution = checkTag(input_soup.Institution.Name)
        Institution_state = checkTag(input_soup.Institution.StateCode)
    else:
        Institution = None
        Institution_state = None
    if input_soup.ProgramElement is not None:
        Program_code = checkTag(input_soup.ProgramElement.Code)
        Program_text = checkTag(input_soup.ProgramElement.Text)
    else:
        Program_code = None
        Program_text = None
    # compile all tags recipient into one list
    tags = [ Title, Eff_date, Exp_date, Amount, AwardInstr,\
                Org_dir, Org_div, NSF_officer, Award_ID, PI_firstname, PI_lastname,\
               Institution, Institution_state, Program_code, Program_text ]
    
    # ABSTRACT ANALYSIS
    # get abstract description
    Abstract = checkTag(input_soup.AbstractNarration)
    if Abstract is not None:
        # Create tokens (create a list of words, ignores ponctuation)
        tokens = tokenizer.tokenize(Abstract)
        # set all words to lower case and remove stopwords and do stemming
        sw = nltk.corpus.stopwords.words('english')
        words = [stemmer.stem(w.lower()) for w in tokens if w.lower() not in sw]
        # Create dictionary where key=word and value=count
        dict_word_freq = nltk.FreqDist(words)
        
        # sort dict by values using get(), keep top 3 words in dataframe
        # sorted makes a list of iterable from dict
        Top3words = sorted(dict_word_freq, key=dict_word_freq.get,\
                           reverse=True)[:3]
#        # add to tags
#        tags.append('-'.join(Top3words))
        
        # appends tags and entire dictionary too
        tags.extend(['-'.join(Top3words), Abstract])
    else:
        # append empty string if not available
#        tags.append('')
        tags.extend(['', ''])
        
    # make a list of information
    return tags


# read and file and extract info
def readExtract(file_list):

    Tag_listOflist = []
    # read data in each xml file
    for thisfname in file_list:
        with open(thisfname, encoding='utf-8') as f:
            xml_text = f.read()
        
        # extract info from xml
        tag_list = extract_Xml_Tag( xml_text )
        
        # append list
        Tag_listOflist.append(tag_list)
        
    return Tag_listOflist

# Create tokenizer to use in loop
tokenizer = RegexpTokenizer('\w+')
# use stemmer for abstract
stemmer = SnowballStemmer("english")

#### The Main program
if __name__ == "__main__":
    # counter to update print statement showing progression
    ListOfInfo = ['AwardTitle', 'AwardEffectiveDate', 'AwardExpirationDate', 'AwardAmount',\
                  'AwardInstrument',\
                'Directorate_Name', 'Division_Name', 'NSF_Officer_FullName', \
               'AwardID', 'PI_FirstName', 'PI_LastName',\
               'Institution_Name', 'Institution_State', \
                'ProgramElement_Code', 'ProgramElement_Text', 'Top3words']
    
    # year range for url, REMINDER: start at 1960
    years = range(1960,2017+1)
    # make sure csv file does not exist, otherwise delete it
    CSV_DB_file = 'test.csv'
    if os.path.isfile(CSV_DB_file): os.remove(CSV_DB_file)
    CSV_Abstract_file = 'Abstract.csv'
    if os.path.isfile(CSV_Abstract_file): os.remove(CSV_Abstract_file)
    # create an empty dataframe
#    df = pd.DataFrame(columns=ListOfInfo)
    # number of processes (quad cores have 8 CPU, 1 CPU = 1 process at most)
    NUM_PROCESS = 8
    # cumulative number of files read
    cumind=0
    filecnt=0
#    nck_counter=0
    # get start time of timer for processing time
    start_time = time.time()
    # create pool
    # sys.getrecursionlimit() returns max recursion (=2000 on my system)
    # readExtract() exceeds that limit between 14 and 15 tasks
    # Error is maximum recursion depth exceeded while calling a Python object
    # sometimes error is maximum recursion depth exceeded while getting the str of an object
    # another error maximum recursion depth exceeded in comparison
    # is beautiful soup to blame for recursion? Yes! recursion is in BeautifulSoup
    pool = multiprocessing.Pool(processes=NUM_PROCESS, maxtasksperchild=None)
    for ny,y in enumerate(years):
        # number of files read for current year
        ind=0
        # folders are organized by year
        year_folder = '{}'.format(y)
        listIn = glob.glob(os.path.join('NSF_data', year_folder, '*.xml'))
        # use listIn as stack for multi-threading
#        listIn = listIn[:1000]
        # break down list in chunk of files, keep length of last list in listIn_ck
        Nchunk = 200
        listIn_ck = [ listIn[i:i+Nchunk] for i in range(0,len(listIn), Nchunk) ]
        ck_adj = len(listIn_ck[-1])
        tot_chunk = len(listIn_ck)
        
        
        # feed pool with all files from current year
        pool_outputs = pool.map(readExtract, listIn_ck)
        # add tag list to dataframe
        Unnested_listoflist = [ h[:-1] for l in pool_outputs for h in l ]
        
		  # write select list to file 
        with open(CSV_DB_file, "a", newline='',  encoding='utf-8') as f:
            writer = csv.writer(f)
            if ny == 0: writer.writerow(ListOfInfo)
            writer.writerows(Unnested_listoflist)
            
        # take care of the abstract, keep award ID (index 7)
        Abst_list = [ [ h[ListOfInfo.index('AwardID')] ,\
                       h[ListOfInfo.index('AwardEffectiveDate')], h[-1]] \
                     for l in pool_outputs for h in l ]
        with open(CSV_Abstract_file, "a", newline='',  encoding='utf-8') as f:
            writer = csv.writer(f)
            if ny == 0: writer.writerow(['AwardID', 'AwardEffectiveDate', 'Raw_Abstract'])
            writer.writerows(Abst_list)
        
        # file counters
        cumind += tot_chunk*Nchunk + ck_adj - Nchunk
        ind += tot_chunk*Nchunk + ck_adj - Nchunk
        
        print('\rYear {}, File #{:6d},Total File {:6d}'.format\
              (y,ind,cumind) ,end='', flush=True)
        
    # close pool
    pool.close()
    # make sure all processes are fisnished, map() does it too!
    pool.join()
    # closing print statement
    print('\rYear {}, File #{:6d},Total File {:6d}'.format(y,ind,cumind),\
                                                          end='\n', flush=True)
    print("--- %s seconds ---" % (time.time() - start_time))
    # First 1000 files in 2016 and 2017: ~ 6.3s, more than 2x than no pool





#        # make a list of list of list <=> list of listIn_ck
#        Nckmap = 5
##        listIn_ckmap = [ listIn_ck[i:i+Nckmap] for i in range(0,len(listIn_ck), Nckmap) ]
##        ckmap_adj = len(listIn_ckmap[-1])
#        # feed worker by chunk
##        while(listIn_ckmap):
##        for nck,listckmap in enumerate(listIn_ckmap):
#        for nck in range(0,tot_chunk, Nckmap):
#        
##            pool_outputs = pool.map(readExtract, listIn_ckmap.pop())
#            # distribute work via map() or imap(), map() is just a bit faster
##            pool_outputs = pool.map(readExtract, listckmap)
#            pool_outputs = pool.map(readExtract, listIn_ck[nck:nck+Nckmap])
##            nck_counter+=len(listckmap)
##            print('Task #{:6d}'.format(nck_counter) ,end='\n', flush=False)
#            
#            # add tag list to dataframe
#            Unnested_listoflist = [ h for l in pool_outputs for h in l ]
#            df_temp = pd.DataFrame(Unnested_listoflist, columns=ListOfInfo)
#            # TODO try concat()
#            df = df.append(df_temp, ignore_index=True)
#            
#            # file counters
#            cumind += Nckmap*Nchunk
#            ind += Nckmap*Nchunk
##            if listIn_ckmap:
#            if nck == tot_chunk - tot_chunk%Nckmap:
##                cumind += ckmap_adj*Nchunk - Nckmap*Nchunk + ck_adj - Nchunk
##                ind += ckmap_adj*Nchunk - Nckmap*Nchunk + ck_adj - Nchunk
#                cumind += len(listIn_ck[nck:nck+Nckmap])*Nchunk\
#                              - Nckmap*Nchunk + ck_adj - Nchunk
#                ind += len(listIn_ck[nck:nck+Nckmap])*Nchunk\
#                              - Nckmap*Nchunk + ck_adj - Nchunk
#            if cumind - filecnt > 100:
#                filecnt = cumind
#                # end='' prevents automatic carriage return, default is \n
#                # \r goes back to the beginning of the line
#                print('\rYear {}, File #{:6d},Total File {:6d}'.format\
#                      (y,ind,cumind) ,end='', flush=True)
                
                
#        # 1 year done, dump data into csv
#        if ny == 0:
#            df.to_csv(CSV_DB_file, header=True, index=False, mode='a', encoding='utf-8')
#        else:
#            df.to_csv(CSV_DB_file, header=False, index=False, mode='a', encoding='utf-8')
#        # clear dataframe
#        df = pd.DataFrame(columns=ListOfInfo)
#        #could empty dataframe instead: df.iloc[0:0]



# Extract useful information in this order:
# Title of research,                       AwardTitle
# Start Date,                          AwardEffectiveDate
# Stop Date,                         AwardExpirationDate
# Amount in USD,                           AwardAmount
# NSF Directorate, NSF Division  Organization.Directorate.LongName and\
#  Organization.Division.LongName
# name of NSF officer who approved award   ProgramOfficer.SignBlockName
# Award ID,                                AwardID
# Firstname and lastname of PI             Investigator.FirstName and Investigator.LastName
# Name and State of Institution awarded    Institution.Name and Institution.StateCode
# info on NSF Program                     ProgramElement.Code and ProgramElement.Text
# top 3 words 
#Year 2017, File #  8679,Total File 438352
#--- 1312.2511858940125 seconds ---, that's 21min



    