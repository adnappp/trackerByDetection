import pandas as pd
import os

filePath = 'ids.txt'
outPath = 'log.csv'
col= ['Time']

f = open(filePath,'r')
lines = f.readlines()
for line in lines:
    line = line.strip()
    col.append(line)
col = pd.DataFrame(col)
col = col.T

col.to_csv(outPath,index=0,header=0)
