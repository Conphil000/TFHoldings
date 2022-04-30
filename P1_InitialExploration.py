# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 13:51:34 2022

@author: Conor
"""


import os
import pandas as pd

import DataHandling

# Centralized Data Structure
ds = \
    DataHandling.importPKL('xlsxData','.\\pklSupport')

# Sample1 Exploration & Problem Statement

"""
    It's April 1st, 2020 and the Coronavirus has already had a severe negative impact 
    on the economy of United States of America, resulting in millions of people losing their job.
    The Federal and State goverments have decided to provide unempolyment benefits to support
    the citizens financially. While state benefits and duration of benefits vary by state,
    the Federal government has decided to pay $600 per week to all unemployed (who recently lost their job)
    for the next 12 weeks on top of the state benefits. 
    
State benefits are as follows:
CA- 50% of the income per week up to a max of $450 for 26 weeks
ID- 45% of the income per week up to a max of $543 for 21 weeks
SC- 50% of the income per week up to a max of $326 for 20 weeks
TN- 52.5% of the income per week up to a max of $375 for 26 weeks
UT- 50% of the income per week up to a max of $405 for 26 weeks

Neither Federal nor State government is going to give any monetary benefit to self-employed."																	
																
1. How is the increase in unemployment and the government stimulus going to impact
    the portfolio risk in next 12 weeks (April-June 2020)

Credit Line Balance and Chargeoff will decline to a steady state between week 1 and week 12
Ending of the Federal Benefit will cause a slight increase in both line usage and chargeoff but remain under 
under pre pandemic normals.

2. If one-third of our customers are laid-off and are unable to find a job after 
    12 weeks, what is the expected increase in risk in the 13th week?
Since 86.4% of customers eligible for unemployment would actually make more per week if laid-off

"""													

ds.keys()

sample = ds['sample1'].copy()

#Need to calculate income/week and then

# Dictionary of Benefits by state
print(sample.state.unique())

beneDict = \
    {
     'CA':0.5,
     'ID':0.45,
     'SC':0.50,
     'TN':0.525,
     'UT':0.5
}
beneEnd = \
    {
     'CA':26,
     'ID':21,
     'SC':20,
     'TN':26,
     'UT':26
}
beneMax = \
    {
     'CA':450,
     'ID':543,
     'SC':326,
     'TN':375,
     'UT':405
}
print(sample.App_Income_PaymentCycle.unique())

# Dictionary of # of paychecks per year based on cycle      
cycleDict = \
    {
     'Every2Weeks':52/2,
     'Monthly':12,
     'TwiceMonthly':12*2,
     'Weekly':52
}
# Dictionary of employment types and if they will recieve benefits

print(sample.App_Income_Type.unique())

# Assumption: SelfEmployed, RetirementPension, and SocialSecurityDisability can not be laid off/given unemployment benefits
# Assumption: Credit Line
# Assumption: Portfolio remains constant

willRecieveDict = \
    {
     'JobEmployment':1,
     'SelfEmployed':0,
     'RetirementPension':0,
     'SocialSecurityDisability':0
}
# Some employment types can't be laid off so don't consider them in the 1/3


# Annual income is # checks/year * $ per check
sample['App_Annual_Income'] = \
    sample.apply(lambda x: x.App_Income_Amount*cycleDict[x.App_Income_PaymentCycle],axis = 1)
# Weekly income is Annual income / 52 weeks
sample['Weekly_Income'] = \
    sample.apply(lambda x: x.App_Annual_Income/52,axis = 1)
# Weekly benefits is Weekly Income * % of weekly income per state
sample['Weekly_Benefits'] = \
    sample.apply(lambda x: willRecieveDict[x.App_Income_Type]*(600+(x.Weekly_Income*beneDict[x.state] if x.Weekly_Income*beneDict[x.state] < beneMax[x.state] else beneMax[x.state])),axis = 1)
sample['Can_Lose_Job'] = \
    sample.apply(lambda x: willRecieveDict[x.App_Income_Type],axis = 1)
    
# FIRST 12 WEEKS
# How many people will make more unemployed?
sample['Makes_More_Unemployed'] = \
    sample.apply(lambda x: 1 if x.Weekly_Benefits > x.Weekly_Income else 0, axis = 1)
# How many people will make the same unemployed?
sample['Makes_Same_Unemployed'] = \
    sample.apply(lambda x: 1 if x.Weekly_Benefits == x.Weekly_Income else 0, axis = 1)
# If can be unemployed
sample['Makes_Less_Unemployed'] = \
    sample.apply(lambda x: 1 if x.Weekly_Benefits < x.Weekly_Income else 0, axis = 1)

# Simulate Unemployment and Normal Weekly Income

import numpy as np
for w in np.arange(0,30):
    sample[f'Week_{w}_Benefits'] = \
        sample.apply(
            lambda x: 
                0 if x['Can_Lose_Job'] == 0 else \
                (-600 if w > 12 else 0) + 
                (x['Weekly_Benefits'] if w > 0 and w <= beneEnd[x['state']] else 0)
            ,
            axis = 1
        )
    sample[f'Week_{w}_Normal'] = \
        sample['Weekly_Income']
        
eligible = sample[sample['Can_Lose_Job']==1].copy()


print(f'{eligible.shape[0]}/{sample.shape[0]} ({round(eligible.shape[0]/sample.shape[0],3)*100}%) are eligible for benefits if job lost.')

print(f'{eligible["Makes_More_Unemployed"].sum()}/{eligible.shape[0]} ({round(eligible["Makes_More_Unemployed"].sum()*100/eligible.shape[0],1)}%) will make more if unemployed.')
print(f'{eligible["Makes_Same_Unemployed"].sum()}/{eligible.shape[0]} ({round(eligible["Makes_Same_Unemployed"].sum()*100/eligible.shape[0],1)}%) will make the same if unemployed.')
print(f'{eligible["Makes_Less_Unemployed"].sum()}/{eligible.shape[0]} ({round(eligible["Makes_Less_Unemployed"].sum()*100/eligible.shape[0],1)}%) will make less if unemployed.')

eCopy = eligible.copy()
Laid_Off = {}
Not_Laid_Off = {}
t = {}
j = 0

# 100 random 33% laid off, what is the average income per week?
for i in np.arange(0,100):
    eCopy['Laid_Off'] = 0
    eCopy.loc[eCopy.sample(frac=1/3).index,'Laid_Off'] = 1
    eCopy[[f'Week_{w}' for w in np.arange(0,25)]] = 0
    
    j += 1
    Laid_Off[j] = eCopy[eCopy['Laid_Off'] == 1].copy()[[f'Week_{w}_Benefits' for w in np.arange(1,25)]].mean()
    Not_Laid_Off[j] = eCopy[eCopy['Laid_Off'] == 0].copy()[[f'Week_{w}_Normal' for w in np.arange(1,25)]].mean()
    
    for w in np.arange(1,25):
        eCopy[f'Week_{w}'] = eCopy.apply(lambda x: x[f'Week_{w}_Benefits'] if x['Laid_Off'] == 1 else x[f'Week_{w}_Normal'], axis = 1)
        
    t[j] = eCopy[[f'Week_{w}' for w in np.arange(1,25)]].mean()
    
Laid_Off_STD = pd.DataFrame.from_dict(Laid_Off,orient='index').std()
Laid_Off_MEAN = pd.DataFrame.from_dict(Laid_Off,orient='index').mean()

import matplotlib.pyplot as plt

plt.figure(figsize=(10,3),dpi=80)
plt.errorbar(np.array(Laid_Off_MEAN.index.tolist()), np.array(Laid_Off_MEAN.tolist()), np.array(Laid_Off_STD.tolist()), linestyle='None', marker='^')
plt.yticks(np.arange(0,1200,150))
plt.xticks(rotation = 90)
plt.title('100 Simlulations of Income per Week for random 33% Laid-off')
plt.show()

Not_Laid_Off_STD = pd.DataFrame.from_dict(Not_Laid_Off,orient='index').std()
Not_Laid_Off_MEAN = pd.DataFrame.from_dict(Not_Laid_Off,orient='index').mean()

Both_STD = pd.DataFrame.from_dict(t,orient='index').std()
Both_MEAN = pd.DataFrame.from_dict(t,orient='index').mean()

print('pre-covid income ',sample['Weekly_Income'].mean())
