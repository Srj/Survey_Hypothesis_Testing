import numpy as np
import pandas as pd
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

DATA = 'data/'
params = ['AgeBin','Education','Profession','Sex']

def generate_freq():
    df = pd.read_csv(os.path.join(DATA,'Cleaned.csv'))
    df = df[~df['Profession'].isin(['Others'])]
    bins = pd.IntervalIndex.from_tuples([(0,25),(25,50),(50,np.inf)])
    df['AgeBin'] = pd.cut(df.Age,bins)
    df.AgeBin = df.AgeBin.astype(str)
    freq_df = freq_generate_helper(df,params).T
    freq_df = freq_df.fillna(0).astype(int)
    freq_df.to_csv(os.path.join(DATA,'Freq_Table.csv'))


def freq_generate_helper(df,params):
    freq = df.groupby(['Country',params[0]])[params[0]].agg(['count'])
    freq.columns = ['f']
    freq = freq.reset_index()
    freq = freq.pivot(index=params[0],columns="Country")
    for i in params[1:]:
        temp = df.groupby(['Country',i])[params[0]].agg(['count'])
        temp.columns = ['f']
        temp = temp.reset_index()
        temp = temp.pivot(index=i,columns="Country")
        freq = pd.concat([freq,temp])
    return freq

generate_freq()
        
