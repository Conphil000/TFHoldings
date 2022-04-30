# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 13:53:37 2022

@author: Conor
"""
import os
import pandas as pd
import pickle

def getXLSX(
        file,
        page,
        folder = ''
    ):
    return \
        pd.read_excel(
            io = open(os.path.join(folder,file)+'.xlsx','rb'),
            sheet_name = page,
            engine='openpyxl'
        )
def exportPKL(
        obj,
        file,
        folder = '',
        schema = 'pkl'
    ):
    pickle.dump(
        obj,
        open(os.path.join(folder,file)+f'.{schema}','wb')
    )

def importPKL(
        file,
        folder = '',
        schema = 'pkl'
    ):
    with open(os.path.join(folder,file)+f'.{schema}','rb') as f:
        return pickle.load(f)
    
def main():
    print('Convert Excel file to PKL.')
    file = 'TF_Holdings_Data_Challenge'
    folder = '.\\xlsxSupport'

    ds = {}
    file = 'TF_Holdings_Data_Challenge'
    ds['sample1'] = \
        getXLSX(file,'1. Data',folder)
        
    ds['sample1_dict'] = \
        getXLSX(file,'1. Data- dictionary',folder)
    
    ds['sample2_new'] = \
        getXLSX(file,'2. New Data Source',folder)
       
    ds['sample2_new'] = \
        getXLSX(file,'2. Verified income_employment',folder)
        
    ds['sample2_internal'] = \
        getXLSX(file,'2. Internal Data',folder)
    
    ds['sample2_new_dict'] = \
        getXLSX(file,'2. Verified inc_emp- dictionary',folder)
    ds['sample2_internal_dict'] = \
        getXLSX(file,'2. Internal Data- dictionary',folder)
    
    exportPKL(ds, 'xlsxData',folder = './/pklSupport')

if __name__ == '__main__':
    main()
























