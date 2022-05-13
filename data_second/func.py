import pandas as pd
import sklearn.neighbors._base
import sys
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

def load():
    #df = pd.read_excel("kamir_5.xlsx", engine = 'openpyxl')
    df = pd.read_csv("./data/kamir_5_utf.csv",encoding='utf-8')
    return df

def lvef_type_change(data):
    for k in data.index:
        if str(type(data[k])) != "<class 'float'>":
            data[k]=data[k].split()[0]
            if data[k] == '|':
                data[k] = str(0)
        data[k] = float(data[k])
    
    return data

def le(data):
    LE = LabelEncoder()
    LE.fit(data)
    return LE.transform(data)

def ohe(data):
    data = data.reshape(-1, 1)
    OHE = OneHotEncoder()
    OHE.fit(data)
    return OHE.transform(data)