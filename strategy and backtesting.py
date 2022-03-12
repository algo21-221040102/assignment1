from ast import In
from jqdata import *                                        # The database of JoinQuant
import jqfactor
from pandas import DataFrame,Series
import pandas as pd
import numpy as np
from IPython.display import display
from datetime import datetime

#%%
indus=get_industries(name='sw_l1', date=None)               # Acquire the industry data of Shenwan Level1
code=indus.index
df=pd.DataFrame()
for i in list(code):
    dfx=finance.run_query(query(
                               finance.SW1_DAILY_PRICE.date,
                               finance.SW1_DAILY_PRICE.close,
                               #finance.SW1_DAILY_PRICE.change_pct,
                              ).filter(finance.SW1_DAILY_PRICE.code==i
                                      ).order_by(finance.SW1_DAILY_PRICE.date.desc()).limit(500))['close']    # Acquire the close data
    
    df=pd.concat([df,dfx],axis=1)

#%%
df.columns=code
df.index=finance.run_query(query(
                               finance.SW1_DAILY_PRICE.date,
                               finance.SW1_DAILY_PRICE.close,
    #                            finance.SW1_DAILY_PRICE.change_pct,
                              ).filter(finance.SW1_DAILY_PRICE.code==code[1]
                                      ).order_by(finance.SW1_DAILY_PRICE.date.desc()).limit(500))['date']
    
print(df)

#%%
data1=df.sort_index()
data1['hs300']=get_price('000300.XSHG',start_date=list(data1.index)[0],end_date=list(data1.index)[-1],
                         frequency='daily', fields='close', fq='pre', panel=True)['close']
data1.head()

#%%
data=data1.pct_change()
# beta sequence
# def Pearson(x,y):
# beta1=rets[['上证综指','贵州茅台']].cov().iat[0,1]/rets['上证综指'].var()
a=-1
date=[]
b=[]
for j in range(0,len(data.index),5):
    date.append(list(data.index)[j])
new_data=pd.DataFrame(index=date,columns=list(data.columns)[:-1])
for k in range(0,len(data.index),5):
    a=a+1
    for i in list(data.columns)[:-1]:
        new_data[i][a]=data[[i,'hs300']][k:k+5].cov().iat[0,1]/data['hs300'][k:k+5].var()

new_data

#%%
# calculate rate of return by week
c=-1
date1=[]
d=[]
for j in range(0,len(data1.index),5):
    date1.append(list(data1.index)[j])
new_data_ret=pd.DataFrame(index=date1,columns=list(data1.columns)[:-1])
print(len(data1.index))
for k in range(0,len(data1.index),5):
    #   print(k)
    c=c+1
    for i in list(data1.columns)[:-1]:
        if k<495:
    #   print(data1[i][k+5])
            new_data_ret[i][c]=data1[i][k+5]/data1[i][k]-1
        else:
            new_data_ret[i][c]=data1[i][k+4]/data1[i][k]-1
new_data_ret

#%%
# calculate Spearman coefficient
spear=[]
for i in list(new_data.T.columns):   
    spear.append(new_data.T[i].corr(new_data_ret.T[i],method='spearman'))
spear_data=pd.DataFrame(index=new_data.T.columns)
spear_data['spearman_corr']=spear
spear_data['hs300']=data1['hs300']
spear_data.head()

#%%
# Beta goes up with rate of return, then the emotion of investors turns optimistic
# Beta goes down together with rate of return, then the emotion of investors turn pessimistic
spear_data['short']=0
spear_data['long']=0
spear_data['spearman_corr']=spear_data['spearman_corr']*100                        # 0.317*100=31.7
for i in range(1,len(spear_data.index)):
    spear_data['short'][i]=np.where((spear_data['spearman_corr'][i-1]>31.7) and 
                                   (spear_data['spearman_corr'][i]<=31.7) and 
                                   (spear_data['spearman_corr'][i+1]<=31.7),spear_data['hs300'][i],0)
    spear_data['long'][i]=np.where((spear_data['spearman_corr'][i-1]<31.7) and 
                                   (spear_data['spearman_corr'][i]>=31.7) and 
                                   (spear_data['spearman_corr'][i+1]>=31.7),spear_data['hs300'][i],0)
spear_data

#%%
# show the long and short list between the testing period
spear_data['str']=np.where(spear_data['short']>0,-1,np.where(spear_data['long']>0,1,0))
spear_data.head()

spear_data['px']=np.arange(len(spear_data))
long=list(spear_data[spear_data['str']==1].index)
short=['2019-12-19','2020-02-10','2020-04-07','2020-05-08','2020-05-29','2020-07-07']
short_n=[]
long_n=[]
for i in short:
    short_n.append(datetime.strptime(i,'%Y-%m-%d'))
for j in long:
    long_n.append(datetime.strptime(i,'%Y-%m-%d'))
display(long_n,short_n)
display(spear_data[spear_data['str']==1])
display(spear_data[spear_data['str']==-1])

#%%
spear_data['mak']=0
spear_data
for i in range(len(long)):
    date_long=long[i]   
    date_short=short[i]
    #spear_data['mak'][date_long: date_short]=1
spear_data['mak']=np.where(spear_data['str']==-1,0,spear_data['mak'])
spear_data

spear_data['ret']=spear_data['hs300'].pct_change()*spear_data['mak']
spear_data['ret'][0]=spear_data['hs300'][0]
spear_data['str_ret']=0
spear_data['str_ret'][0]=spear_data['hs300'][0]
for i in range(1,len(spear_data)):
    spear_data['str_ret'][i]=spear_data['str_ret'][i-1]*(1+spear_data['ret'][i])
spear_data

#%%
#Backtesting
import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))
plt.plot(spear_data[['hs300','str_ret']])
plt.plot(spear_data['short'],'*g')
plt.plot(spear_data['long'],'*r')
plt.ylim(2500,5000)