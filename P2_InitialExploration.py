# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 19:08:33 2022

@author: Conor
"""


# Build a couple models,

# 1. Model using only Internal Features
# 2. Model using only Income Validation Vendor
# 3. Model Combining these 2 Data Sources



def main():
    
    import DataHandling
    import numpy as np
    import pandas as pd
    import math
    from scipy.stats import linregress
    
    random_state = 123
    uid = 'ID'
    
    
    # Centralized DataStructure
    ds = DataHandling.importPKL('xlsxData','.//pklSupport')
    
    new_product = ds['sample2_new'].copy()
    new_product['Coverage'] = 1
    # EMPLOYEE_STATUS_CODE is has the least amount of rows null
    new_product.loc[new_product['EMPLOYEE_STATUS_CODE'].isnull(),'Coverage'] = 0
    # How much coverage does this product have for applications?
    new_product['Coverage'].mean()
    # How much coverage does this product have for applications and funded
    new_product.groupby(by='Funded')['Coverage'].mean()
    
    internal = ds['sample2_internal'].copy()
    # Fix a weird spelling issue from excel
    internal['CHARGE_OFF_PRINCIPAL_LIFETIME'] = internal[ 'CHARGE_OFF_PRINCIPAL _LIFETIME'] 
    # funded is covered by new_product df
    del internal['Funded']
    del internal['CHARGE_OFF_PRINCIPAL _LIFETIME']
    
    # No Application Date?
    # Why is Internal shorter than the product?
    # Only have booked for internal? Might be a change to reject inference if internal  full sample
    
    # Use New_Product as Base
    print(new_product.shape[0])
    
    sample = new_product.merge(internal, on = uid, how = 'left')
    # make sure that we didn't add dupes
    print(sample.shape[0])
    # how many apps have new_data
    sample['Coverage'].sum()
    
    
    # notVar = \
    #     ['ID','FUNDED_DOLLARS','TOTAL_CASH_REVENUE_LIFETIME','CHARGEOFF_4M','CHARGE_OFF_PRINCIPAL_LIFETIME']
        
    # [i for i in internal.columns if i not in notVar]
    
    
    
    # Define which applications are booked
    sample['population'] = 'application'
    sample.loc[sample['Funded']==1,'population'] = 'booked'
    
    # Number of funded accounts that have new data product
    sample[(sample['population']=='booked')&(~sample['EMPLOYEE_STATUS_CODE'].isnull())].shape[0]
    
    class customerAppModifier:
        # Class to do some feature engineering
        def __init__(self):
            self.paychecks_per_year = \
                {
                'Monthly':12,
                'Weekly':52,
                'TwiceMonthly':12*2,
                'Every2Weeks':52/2
            }
        def modify(
                self,
                x
            ):
            x = self.annualComp(x)
            x = self.incomeDifference(x)
            x = self.hasTermination(x)
            return x
        def hasTermination(
                self,
                x
            ):
            if pd.isnull(x['EMPLOYEE_TERMINATION_DATE']):
                x['TERMINATED'] = 0
            else:
                x['TERMINATED'] = 1
                
            del x['EMPLOYEE_TERMINATION_DATE']
            return x
        def annualComp(
                self,
                x
            ):
            # This function expands the 3 general columns of comp1,comp2, etc
            x['MAX_COMP_YEAR'] = max(x[f'ANNUAL_COMP_YEAR1'],x[f'ANNUAL_COMP_YEAR2'],x[f'ANNUAL_COMP_YEAR3'])
            x['MIN_COMP_YEAR'] = min(x[f'ANNUAL_COMP_YEAR1'],x[f'ANNUAL_COMP_YEAR2'],x[f'ANNUAL_COMP_YEAR3'])
            for i in [1,2,3]:
                if math.isnan(x[f'ANNUAL_COMP_YEAR{i}']):
                    # del x[f'ANNUAL_COMP_YEAR{i}']
                    for j in ['BASE','BONUS','OTHER','TOTAL','COMMISSION','OVERTIME']:
                        # del x[f'ANNUAL_{j}_COMP{i}']
                        pass
                else:
                    YEAR = int(x[f'ANNUAL_COMP_YEAR{i}'])
                    # del x[f'ANNUAL_COMP_YEAR{i}']
                    for j in ['BASE','BONUS','OTHER','TOTAL','COMMISSION','OVERTIME']:
                        if math.isnan(x[f'ANNUAL_{j}_COMP{i}']):
                            pass
                        else:
                            x[f'{YEAR}_ANNUAL_{j}'] = x[f'ANNUAL_{j}_COMP{i}']
                        # del x[f'ANNUAL_{j}_COMP{i}']
            # Could do like a slope of annual_comp_year incomes dft rate etc
            sorts = []
            nonNull = 0
            for i in [1,2,3]:
                sorts.append((i,x[f'ANNUAL_COMP_YEAR{i}']))
                if math.isnan(x[f'ANNUAL_COMP_YEAR{i}']):
                    pass
                else:
                    nonNull += 1
         
            x['NON_NULL_COMP'] = nonNull  

            if nonNull > 2:
                sorts.sort(key = lambda x: x[1])
                xN = [1,2,3]
                xYear = [x[f'ANNUAL_COMP_YEAR{i[0]}'] for i in sorts]
                y = [x[f'ANNUAL_BASE_COMP{i[0]}'] for i in sorts]
                x['SLOPE_N'] = linregress(xN,y)[0]
                x['SLOPE_Y'] = linregress(xYear,y)[0]
                x['SLOPE_DIFF'] = x['SLOPE_N']-x['SLOPE_Y'] 
                x['SLOPE_RATIO'] = x['SLOPE_N']/x['SLOPE_Y'] 
            for i in [1,2,3]:
                del x[f'ANNUAL_COMP_YEAR{i}']
                for j in ['BASE','BONUS','OTHER','TOTAL','COMMISSION','OVERTIME']:
                    del x[f'ANNUAL_{j}_COMP{i}']
            return x
        def incomeDifference(
                self,
                x
            ):
            # This function finds the difference between app and projected, people lie or guess if they are stealing info
            vendor_projection = x['ANNUAL_PROJECTED_INCOME']
            if math.isnan(x['App_Income_Amount']):
                pass
            else:
                annual_income = \
                    x['App_Income_Amount'] * self.paychecks_per_year[x['App_Income_PaymentCycle']]
                    
                x['APP_ANNUAL_INCOME'] = annual_income
                x['PROJ_TO_APP_RATIO'] = annual_income/vendor_projection
            return x
    
    x = sample[~(sample['ANNUAL_COMP_YEAR1'].isnull())&~(sample['ANNUAL_COMP_YEAR2'].isnull())].sample(1,random_state = 35463456).squeeze().to_dict()
    
    x = customerAppModifier().modify(x)
    
    # Little feature engineering, i know ratio of actual to applciation income is usually ranks risk well
    sample['dict'] = sample.apply(lambda x: customerAppModifier().modify(x.to_dict()),axis = 1)
    
    sample2 = pd.DataFrame(sample['dict'].tolist())
    sample_model = sample2[sample2['population']=='booked'].copy()
    covered = sample_model[sample_model['Coverage']==1]
    # want to remove missing based on the ones with coverage
    missingCutoff = 0.75# Must have 75% non-missing
    # Good is the columns meeting missing and freq cutoffs
    good = covered.loc[:, covered.isin([np.nan]).mean() < 1-missingCutoff].columns.tolist()
    
    # sample_model = sample_model.fillna(-999)
    
    notVar = \
        ['ID','Funded','Funded_Internal','FUNDED_DOLLARS','EMPLOYEE_INFO_EFFECTIVE_DATE',
          'EMPLOYEE_MOST_RECENT_HIRE_DATE','EMPLOYEE_STATUS_CODE','PAY_FREQ_CODE','population','dict',
          'EMPLOYEE_INFO_EFFECTIVE_DATE','EMPLOYEE_ORIG_HIREDATE']
    
    targets = \
        ['TOTAL_CASH_REVENUE_LIFETIME','CHARGE_OFF_PRINCIPAL_LIFETIME','CHARGEOFF_4M','FUNDED_DOLLARS']
        
    isVar = [i for i in good if i not in notVar+targets]
    
    # DecisionTree should be able to handle using Coverage as a split,
    view = sample_model[isVar]
    # Converting all to Numeric and then changing back to char if seen fit
    sample_model[isVar] = sample_model[isVar].apply(pd.to_numeric,errors = 'ignore')
    view = sample_model[isVar]
    
    might_be_cat = []
    for i in isVar:
        print(i,': ',len(sample_model[i].unique()))
        if len(sample_model[i].unique()) < 50:
            might_be_cat.append((i,len(sample_model[i].unique())))
            
    temp = covered[[i[0] for i in might_be_cat]]
    
    # Need to combine Annual_Base_comp1 and annual_comp_year1,ANNUAL_COMP_YEAR2,ANNUAL_COMP_YEAR3
    isCategorical = \
        ['EMPLOYEE_STATUS_MESSAGE','EMPLOYEE_STATUS_TYPE','PAY_FREQ_MESSAGE','Hourly_Pay_Bkt',
         'App_Income_PaymentCycle','App_Income_Type','NON_NULL_COMP','MAX_COMP_YEAR','MIN_COMP_YEAR']
    
    def OHE(
            df,
            varList
        ):
        new_varList = varList.copy()
        for i in varList:
            ohe = pd.get_dummies(df[i])
            ohe.columns = [str(i)+'_'+str(c) for c in ohe.columns]
            df = df.join(ohe)
            del df[i]
            new_varList+=ohe.columns.tolist()
            new_varList.remove(i)
        return df,new_varList
    
    sample_model,varList = OHE(covered.copy(),isCategorical)
    
    modeldata = sample_model[[uid,'population']+targets+[i for i in isVar if i not in isCategorical] + varList].copy()
    from sklearn.tree import DecisionTreeClassifier
    
    def buildCLF(
            sample,
            predictors
        ):
        maxDepth = 2
        target = 'CHARGEOFF_4M'
        
        summary = pd.DataFrame(columns = ['seed','node_tar','dev_tar','val_tar','gap_tar','dev_size','val_size'])
        
        for var in predictors:
            sample[var] = sample[var].fillna(-999)
        for seed in range(100):
            
            sample['sample'] = 'dev'
            sample.loc[sample.sample(frac = 0.3,random_state = seed).index,'sample'] = 'val'
            sample_train = sample[sample['sample']=='dev']
            
            clf = DecisionTreeClassifier(random_state=1,min_samples_leaf=0.05,max_depth = maxDepth)
            clf.fit(sample_train[predictors],sample_train[target])
            
            sample['estimate'] = clf.predict_proba(sample[predictors])[:,1]
            
            sample2 = sample[['sample','estimate',target]]
            sample2['count']= 1
            
            summaryData = sample2.groupby(['sample','estimate'],as_index=False).sum()
            summaryData['target_pct'] = summaryData[target]/summaryData['count']
            
            node_tar = summaryData[summaryData['sample']=='dev']['estimate'].max()
            dev_tar = summaryData[(summaryData['sample']=='dev')&(summaryData['estimate']==node_tar)]['target_pct'].max()
            val_tar = summaryData[(summaryData['sample']=='val')&(summaryData['estimate']==node_tar)]['target_pct'].max()
            gap_tar = dev_tar - val_tar
            dev_size = summaryData[(summaryData['sample']=='dev')&(summaryData['estimate']==node_tar)]['count'].sum()
            val_size = summaryData[(summaryData['sample']=='val')&(summaryData['estimate']==node_tar)]['count'].sum()
            
            row_df = pd.DataFrame(
                            {'seed':seed,
                             'node_tar':node_tar,
                             'dev_tar':dev_tar,
                             'val_tar':val_tar,
                             'gap_tar':gap_tar,
                             'dev_size':dev_size,
                             'val_size':val_size},index = [0])
            summary = pd.concat([summary,row_df],ignore_index = True)
        
        summary['size'] = summary['dev_size'] + summary['val_size']
        summary2 = summary[(summary['gap_tar']>-0.03)&(summary['gap_tar']<0.03)]
        
        summary3 = summary2[summary2['size']>100].sort_values(by = 'val_tar',ascending = False)
        
        seed = summary3.iloc[0]['seed']
        
        sample['sample'] = 'dev'
        sample.loc[sample.sample(frac = 0.3,random_state = seed).index,'sample'] = 'val'
        sample_train = sample[sample['sample']=='dev']
        
        clf = DecisionTreeClassifier(random_state=1,min_samples_leaf=0.05,max_depth = maxDepth)
        clf.fit(sample_train[predictors],sample_train[target])
        
        return clf
    fVar = [i for i in isVar if i not in isCategorical] + varList
    
    clf = buildCLF(modeldata,fVar)

    import matplotlib.pyplot as plt
    from sklearn import tree

    fig = plt.figure(figsize=(10,5),dpi = 90)
    _ = tree.plot_tree(clf, 
                   feature_names=fVar,
                   filled=True)
    
    modeldata['predict'] = clf.predict_proba(modeldata[fVar])[:,1]
    
    modeldata.groupby(by='predict')[targets].mean().to_clipboard()
    modeldata[targets].mean()
    
    modeldata[modeldata['predict']<0.339].groupby(by=['predict','sample'])[targets].mean().to_clipboard()
    
    modeldata[modeldata['predict']<0.339][targets].mean()
    

if __name__ == '__main__':
    main()































