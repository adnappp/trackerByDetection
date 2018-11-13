import numpy as np
import pandas as pd

def distance(str1,str2):
    if not str1 or not str2:
        return 0
    list1 = str1.split(',')
    list2 = str2.split(',')
    dis = pow(int(list1[0])-int(list2[0]),2)+pow(int(list1[1])-int(list2[1]),2)
    dis = np.sqrt(dis)
    return dis


log = pd.read_csv('log.csv')
