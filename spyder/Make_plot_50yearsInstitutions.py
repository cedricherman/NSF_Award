# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 21:42:45 2017

@author: herma
"""

# import to read and filter data
import pandas as pd
# imports for plots
import numpy as np
import matplotlib.pyplot as plt


# read database (csv file)
df = pd.read_csv('test_1967_to_2017.csv',header=0, encoding = 'utf-8')


# Institution: number of grants over 50 Years
NumGrants_PerInst = df.groupby(['Institution_Name'])['AwardAmount'].count()
# sort using values by descending order
NumGrants_PerInst.sort_values( ascending=False, inplace=True)
# plot top 10 
Top10_Inst = NumGrants_PerInst.iloc[:10]
fig2 = plt.figure(figsize=(12, 6) )
ax2 = fig2.add_axes([0.45, 0.1, 0.45, 0.8])
InstStrList = Top10_Inst.index
y_pos_Inst = np.arange(len(InstStrList))
countList = Top10_Inst.values
ax2.barh(y_pos_Inst, countList, align='center', color='blue', ecolor='black')
ax2.set_yticks(y_pos_Inst)
ax2.set_yticklabels(InstStrList , fontsize=16)
ax2.invert_yaxis()  # labels read top-to-bottom
ax2.set_ylabel('Institution Name', fontsize=20)
ax2.set_xlabel('Number of Grants awarded between 1967 and 2017', \
               horizontalalignment='center', fontsize=20)
ax2.set_title('Awards over 50 years', \
              horizontalalignment='center', fontsize=24)
ax2.set_xticklabels([])
ax2.set_xticks([])
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)
for tx in range(len(countList)):
    ax2.text(countList[tx]+225*2, tx, '%d' % countList[tx], fontsize=16,
    horizontalalignment='center', verticalalignment='center', color='black')
