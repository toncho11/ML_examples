# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:19:20 2022

@author: antona
"""

from moabb.datasets import bi2013a, bi2014a, bi2014b, bi2015a, bi2015b, BNCI2014008, BNCI2014009, BNCI2015003, EPFLP300, DemonsP300 
from moabb.paradigms import P300

def GetDataSetInfo(ds):
    #print("Parameters: ", ds.ds.__dict__)
    print("Dataset name: ", ds.__class__.__name__)
    print("Subjects: ", ds.subject_list)
    print("Subjects count: ", len(ds.subject_list))
    
    X, y, metadata = paradigm.get_data(dataset=ds, subjects=[ds.subject_list[0]])
    
    print("Electrodes count (inferred): ", X.shape[1])
    print("Epoch length (inferred)    : ", X.shape[2])
    #print("Description:    : ", ds.__doc__)

paradigm = P300()

#not available: bi2013a(), Lee2019_ERP()
for d in [bi2013a(), DemonsP300(), bi2014a(), bi2014b(), bi2015a(), bi2015b(), BNCI2014008(), BNCI2014009(), BNCI2015003(), EPFLP300()]:
    GetDataSetInfo(d)
    print("======================================================================================")