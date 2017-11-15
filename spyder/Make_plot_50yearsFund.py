# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 21:41:26 2017

@author: herma
"""

# import to read and filter data
import pandas as pd
# imports for plots
import numpy as np
import matplotlib.pyplot as plt



# read database (csv file)
df = pd.read_csv('test_1967_to_2017.csv',header=0, encoding = 'utf-8')


# Plot total founding per year
# add a year column
date_col = pd.to_datetime(df['AwardEffectiveDate'],format='%m/%d/%Y')
df['StartYear'] = date_col.dt.strftime('%Y')
# group by year and sum amount for each year
totalUSD_PerYear = df.groupby(['StartYear'])['AwardAmount'].sum()
# only keep result between 1967 and 2017 included
totalUSD_PerYear = totalUSD_PerYear[ \
                            np.logical_and(totalUSD_PerYear.index >= '1967', \
                            totalUSD_PerYear.index <= '2017')]
# convert index to DateTimeIndex and group every 10 years
totalUSD_PerYear.index = pd.to_datetime(totalUSD_PerYear.index,format='%Y')
totalUSD_PerDecade = totalUSD_PerYear.groupby([pd.TimeGrouper(freq='10AS')]).sum()
# sort series by descending year
totalUSD_PerDecade.sort_index( ascending=False, inplace=True)
# make a horizontal bar plot
fig1 = plt.figure(figsize=(10, 6) )
ax1 = fig1.add_axes([0.15, 0.15, 0.7, 0.7])
# get string year back
yearStrList = totalUSD_PerDecade.index.to_pydatetime()
yearStrList = [ ys.strftime('%Y')+' to \n'+(ys+pd.DateOffset(years=10)).strftime('%Y') \
               if ys+pd.DateOffset(years=10) < totalUSD_PerYear.index[-1] \
               else ys.strftime('%Y')+' to \n'+totalUSD_PerYear.index[-1].strftime('%Y') \
               for ys in yearStrList]
y_pos = np.arange(len(yearStrList))
spendingList = totalUSD_PerDecade.values
ax1.bar(y_pos, spendingList, align='center', color='green', ecolor='black')
ax1.set_xticks(y_pos)
ax1.set_xticklabels(yearStrList, fontsize=16)
#ax1.invert_xaxis()  # labels read top-to-bottom
ax1.set_xlabel('Ten Calendar Years Intervals (Except for 2017)', fontsize=20)
ax1.set_ylabel('Amount in Billion USD', fontsize=20)
ax1.set_title('NSF Total Award Amount per decade', fontsize=24)
# adjust for x tick label spaced by $10M
# determine next highest power of 10 number for values
powMaxVal = len(str(int(max(spendingList))))
maxtick = np.ceil(max(spendingList)/10**powMaxVal)*10**powMaxVal
xtickloc = np.ceil(np.linspace(0, maxtick-maxtick/10, num=10))
# only keep 1 tick above max
xtick_filt = xtickloc <= \
        np.ceil(max(spendingList)/10**(powMaxVal-1))*10**(powMaxVal-1)
xtickloc = xtickloc[xtick_filt]
ax1.set_yticks(xtickloc)
ax1.set_yticklabels( ['${0:d}B'.format(int(ti/1e9))  \
                      for ti in xtickloc], fontsize=16 )
