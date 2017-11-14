# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 21:49:38 2017

@author: herma

This is a script to download zipfiles from each year on the NSF website.
After dowload of a file is completed, extract it so that it will create
a folder named after its year. Each year folder has a one xml file per award
"""

# imports necessary for zip file download
import requests
from zipfile import ZipFile
import re


start_year = 1960
stop_year = 2017
# total size of data (over all zipped files)
Tot_size=0
# year range for url
years = range(start_year,stop_year+1)
for ny,y in enumerate(years):
    url = 'https://www.nsf.gov/awardsearch/download?DownloadFileName={}&All=true'.format(y)
    request = requests.get(url)
    content_name = request.headers.get('Content-Disposition')
    zip_name = re.findall('filename="(\w+.\w+)"', content_name)[0]
    # download zip files one by one
    with open(zip_name,'ab') as f:
        f.write(request.content)
    # extract zip file
    thiszip = ZipFile(zip_name, 'r')
    # HAVE to filter list of file in zip file
    # In 2002 data, there is one named '0225630.xml\r' and '0225630.xml'
    # so use strip() to remove line separator (return carriage)
    # Even if there are duplicate of xml file, it will overwrite and keep one only
    thiszip.extractall(path=zip_name.split('.')[0], \
                       members=[ pin.strip() for pin in thiszip.namelist() ])
    thiszip.close()