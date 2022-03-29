# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 16:09:36 2021

@author: Administrator
"""

import numpy as np
import random

S=200
N=4

def cpzh(S,N):#生成产品组合
    cpzh_list=[]
    while True:
        b=[]
        for i in range(N):
            a=int(np.random.normal(100,20))
            #print('a',a)
            if a>0:
                b.append(a)
        if len(b)==N and sum(b)==400:
            cpzh_list.append(b)
        if len(cpzh_list)==S:
            break
    return cpzh_list
print('cpzh(S,N)',cpzh(S,N))

cpzh_list=cpzh(S,N)
#print('cpzh_list',cpzh_list)

cpzh_list=np.array(cpzh_list) 
print('矩阵',cpzh_list)

#np.savetxt('10cp_zuhe.txt',cpzh_list)
 
#np.savetxt('5cp_zuhe(sumis500).txt',cpzh_list, delimiter=" ", fmt='%s')
