# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 20:35:58 2021

@author: Administrator
"""
import time
import gurobipy as gp
from gurobipy import GRB
import numpy as np


def serutest(demand,numOfWorker,numOfSeru,TC,FC,sc,ec,A):
    # 保存行列标签
    numOfScenario=len(demand)
    numOfProduct=len(demand[0])
   
    D=np.array(demand)
    
    R=1000 #一个足够大的数
    M=500
    
    # Create a new model
    m = gp.Model("seru1")
    
    # Create variables 创建变量
    z = m.addVars(range(numOfSeru),lb=1,vtype=gp.GRB.INTEGER, name="z") #决策变量，单元内人数，取值为大于等于1小于等于工人数量
    y = m.addVars(range(numOfProduct),range(numOfSeru),vtype=gp.GRB.BINARY, name="y")#决策变量，产品-SERU分配，0-1变量
    x = m.addVars(range(numOfScenario),range(numOfProduct),range(numOfSeru),lb=0,vtype=gp.GRB.INTEGER, name="x")#决策变量，SERU生产产品的数量，大于等于0
    v = m.addVars(range(numOfProduct),range(numOfSeru),vtype=gp.GRB.INTEGER, name="v")#为了线性化加入的变量
    
    # 更新变量环境
    m.update()
    
    # 建立一个表达式计算成本
    shortage_cost=gp.quicksum((D[s,i]-gp.quicksum(x[s,i,j] for j in range(numOfSeru))) for s in range(numOfScenario) for i in range(numOfProduct)) * sc
    excess_cost=gp.quicksum((z[j] * A -gp.quicksum(x[s,i,j] for i in range(numOfProduct))) for s in range(numOfScenario) for j in range(numOfSeru)) * ec
    total_cost=(shortage_cost + excess_cost) / numOfScenario
    expr=numOfSeru*FC + gp.quicksum(TC*v[i,j] for i in range(numOfProduct) for j in range(numOfSeru)) + total_cost
    
    # Set objective设置目标函数
    m.setObjective(expr,sense=gp.GRB.MINIMIZE)
    
    # 添加约束条件
    m.addConstr(gp.quicksum(z[j] for j in range(numOfSeru)) == numOfWorker)#系统内人数不变
    m.addConstrs(gp.quicksum(y[i,j] for i in range(numOfProduct)) >=1 for j in range(numOfSeru))#每个单元分配的产品种类数量至少为1
    m.addConstrs(gp.quicksum(y[i,j] for i in range(numOfProduct)) <=numOfProduct for j in range(numOfSeru))#每个单元分配的产品种类数量至多为全部产品类型
    m.addConstrs(gp.quicksum(y[i,j] for j in range(numOfSeru)) >=1 for i in range(numOfProduct))#每种产品都至少有1个单元能够装配
    m.addConstrs(v[i,j] <= (z[j]+M*(1-y[i,j])) for i in range(numOfProduct) for j in range(numOfSeru))
    m.addConstrs((z[j]-M*(1-y[i,j])) <= v[i,j] for i in range(numOfProduct) for j in range(numOfSeru))
    m.addConstrs(v[i,j] <= M*y[i,j] for i in range(numOfProduct) for j in range(numOfSeru))
    m.addConstrs(-1*M*y[i,j] <= v[i,j] for i in range(numOfProduct) for j in range(numOfSeru))
    m.addConstrs(x[s,i,j] <= R*y[i,j] for s in range(numOfScenario) for i in range(numOfProduct) for j in range(numOfSeru))#如果单元不能装配产品，则二者之间不会发生产品流，即如果，则；如果，则，小于需求量与单元生产能力中较大的数，大于较小的那个数
    m.addConstrs(gp.quicksum(x[s,i,j] for i in range(numOfProduct)) <= z[j]*A for s in range(numOfScenario) for j in range(numOfSeru))#单元生产量不能超过其生产能力
    m.addConstrs(gp.quicksum(x[s,i,j] for j in range(numOfSeru)) <= D[s,i] for s in range(numOfScenario) for i in range(numOfProduct))#单元生产量不能超过其生产能力
    
    # Optimize model
    m.optimize()
    
    # 输出信息
    if m.status == gp.GRB.Status.OPTIMAL:
        print("\nGlobal optimal solution found. 得到了全局最优解")
    
    Z= m.getAttr('x',z)
    #print('Z',Z)
    workerToseru= []
    for j in range(numOfSeru):
        workerToseru.append(Z[j])
    print('workerToseru',workerToseru)

    Y= m.getAttr('x',y)
    #print('Y',Y)
    productToseru=[]
    for i in range(numOfProduct):
        product=[]
        for j in range(numOfSeru):
            product.append(Y[i,j]) 
        productToseru.append(product)
    #print('0productToseru',productToseru)
    productToseru=np.array(productToseru)
    print('productToseru',productToseru)
    
    X= m.getAttr('x',x)
    #print('X',X)
    loading=[]
    for s in range(numOfScenario):
        scenario=[]
        for i in range(numOfProduct):
            pro=[]
            for j in range(numOfSeru):
                pro.append(X[s,i,j])
            scenario.append(pro)
        #print('scenario',scenario)
        loading.append(scenario)
    #print('loading',loading)
    loading= np.array(loading)
    loading=loading.tolist()
    #print('1loading',loading) 
    #print('loading', loading)
    TotalCost= m.objVal
    print("总成本为：",TotalCost)
    return workerToseru,productToseru,TotalCost,loading

demand=[[39, 68, 92, 201], [135, 171, 66, 28], [121, 85, 149, 45], [111, 54, 133, 102], [86, 117, 107, 90], [120, 46, 167, 67], [80, 132, 121, 67], [123, 68, 83, 126], [107, 60, 105, 128], [159, 83, 115, 43], [103, 133, 78, 86], [79, 106, 135, 80], [84, 134, 131, 51], [101, 126, 116, 57], [123, 106, 87, 84], [137, 80, 88, 95], [104, 168, 54, 74], [111, 68, 69, 152], [35, 125, 93, 147], [41, 108, 162, 89], [95, 109, 135, 61], [66, 49, 140, 145], [113, 75, 73, 139], [60, 65, 178, 97], [55, 99, 121, 125], [131, 39, 133, 97], [55, 66, 198, 81], [39, 110, 152, 99], [89, 84, 158, 69], [114, 56, 156, 74], [106, 47, 69, 178], [88, 135, 130, 47], [90, 22, 138, 150], [147, 63, 61, 129], [128, 132, 78, 62], [144, 61, 85, 110], [107, 89, 86, 118], [114, 96, 134, 56], [125, 103, 71, 101], [106, 143, 63, 88], [134, 30, 112, 124], [109, 44, 121, 126], [141, 99, 100, 60], [136, 34, 98, 132], [130, 119, 103, 48], [184, 26, 124, 66], [120, 87, 97, 96], [86, 118, 117, 79], [94, 89, 89, 128], [126, 106, 36, 132], [138, 99, 27, 136], [65, 68, 180, 87], [115, 158, 51, 76], [96, 23, 139, 142], [31, 125, 209, 35], [116, 69, 39, 176], [114, 70, 110, 106], [34, 97, 53, 216], [78, 116, 81, 125], [137, 10, 145, 108], [124, 29, 190, 57], [66, 159, 112, 63], [94, 125, 57, 124], [167, 125, 47, 61], [88, 108, 118, 86], [108, 80, 90, 122], [85, 79, 80, 156], [85, 168, 84, 63], [144, 159, 33, 64], [127, 64, 128, 81], [136, 96, 81, 87], [43, 77, 158, 122], [207, 47, 13, 133], [65, 60, 103, 172], [66, 69, 148, 117], [159, 49, 58, 134], [155, 141, 81, 23], [130, 114, 112, 44], [66, 97, 126, 111], [92, 104, 146, 58], [76, 126, 40, 158], [141, 99, 108, 52], [123, 103, 157, 17], [84, 191, 75, 50], [67, 102, 180, 51], [110, 154, 66, 70], [81, 107, 125, 87], [19, 136, 96, 149], [65, 155, 54, 126], [109, 76, 137, 78], [104, 90, 59, 147], [132, 90, 88, 90], [107, 100, 110, 83], [50, 122, 136, 92], [92, 58, 125, 125], [14, 136, 134, 116], [79, 119, 88, 114], [19, 69, 124, 188], [93, 85, 131, 91], [127, 72, 101, 100], [176, 4, 130, 90], [85, 35, 123, 157], [179, 27, 100, 94], [139, 44, 116, 101], [116, 156, 57, 71], [127, 35, 138, 100], [89, 68, 60, 183], [92, 117, 69, 122], [128, 123, 78, 71], [88, 134, 91, 87], [101, 113, 106, 80], [74, 60, 120, 146], [96, 37, 91, 176], [170, 89, 100, 41], [85, 75, 122, 118], [102, 77, 112, 109], [102, 74, 121, 103], [66, 177, 88, 69], [75, 166, 117, 42], [96, 124, 88, 92], [32, 97, 87, 184], [23, 42, 149, 186], [94, 113, 167, 26], [150, 80, 84, 86], [149, 49, 114, 88], [112, 125, 62, 101], [27, 56, 102, 215], [76, 112, 125, 87], [111, 49, 162, 78], [68, 86, 149, 97], [119, 76, 63, 142], [118, 73, 162, 47], [192, 83, 77, 48], [143, 120, 92, 45], [25, 139, 183, 53], [107, 100, 142, 51], [69, 111, 103, 117], [172, 67, 107, 54], [122, 108, 57, 113], [165, 103, 66, 66], [110, 141, 93, 56], [56, 84, 191, 69], [111, 85, 102, 102], [125, 122, 47, 106], [21, 173, 153, 53], [69, 122, 139, 70], [92, 39, 146, 123], [116, 31, 113, 140], [19, 103, 74, 204], [51, 77, 184, 88], [26, 207, 82, 85], [144, 72, 101, 83], [104, 137, 128, 31], [70, 86, 110, 134], [125, 91, 128, 56], [88, 136, 94, 82], [99, 131, 50, 120], [30, 134, 125, 111], [54, 155, 134, 57], [108, 163, 28, 101], [99, 16, 75, 210], [164, 120, 32, 84], [163, 46, 127, 64], [63, 112, 109, 116], [94, 77, 74, 155], [89, 32, 95, 184], [63, 115, 139, 83], [22, 131, 120, 127], [95, 152, 124, 29], [129, 37, 109, 125], [71, 49, 93, 187], [114, 101, 56, 129], [106, 78, 35, 181], [147, 99, 50, 104], [203, 165, 11, 21], [137, 115, 120, 28], [52, 86, 144, 118], [99, 104, 116, 81], [101, 150, 108, 41], [157, 42, 95, 106], [111, 80, 99, 110], [145, 79, 118, 58], [95, 69, 130, 106], [116, 180, 47, 57], [81, 55, 183, 81], [119, 113, 82, 86], [110, 84, 101, 105], [137, 62, 63, 138], [161, 65, 124, 50], [84, 82, 120, 114], [131, 66, 81, 122], [68, 86, 70, 176], [80, 95, 139, 86], [12, 103, 146, 139], [101, 59, 168, 72], [107, 60, 72, 161], [111, 93, 121, 75], [53, 215, 42, 90], [51, 94, 169, 86], [106, 123, 59, 112]]

print(len(demand))
numOfWorker=4
numOfSeru=4
TC=1
FC=1
sc=3
ec=1
A=100

start =time.time()
def main():
    Z_list=[]#用于存放不同单元数量下的最优工人配置
    Y_list=[]#用于存放不同单元数量下的最优产品分配
    Cost_list=[]#用于存放不同单元数量下的最优目标值
    solution=[]
    solutionSet=[]
    xLoadingSet=[]
    bestZ,bestY,bestCost,XLoading=serutest(demand,numOfWorker,1,TC,FC,sc,ec,A)
    Z_list.append(bestZ)
    Y_list.append(bestY)
    Yi=bestY.tolist()
    Asolution=[]
    Asolution.append(bestZ)
    Asolution.append(Yi)
    solutionSet.append(Asolution)
    xLoadingSet.append(XLoading)
    print('###############Asolution',Asolution)
    Cost_list.append(bestCost)
    for i in range(2,numOfSeru+1):
        Ysolution=[]
        workerToseru,productToseru,TotalCost,XLoading=serutest(demand,numOfWorker,i,TC,FC,sc,ec,A)
        Z_list.append(workerToseru)
        Y_list.append(productToseru)
        Cost_list.append(TotalCost)
        
        Yi=productToseru.tolist()
        Ysolution.append(workerToseru)
        Ysolution.append(Yi)
        solutionSet.append(Ysolution)
        xLoadingSet.append(XLoading)
        if TotalCost < bestCost:  
            bestZ=workerToseru
            bestY=productToseru
            bestCost=TotalCost
        
    print('solutionSet',solutionSet)    
    print('Z_list',Z_list)
    print('Y_list',Y_list)
    print('Cost_list',Cost_list)
    print('bestZ',bestZ)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    print('bestY',bestY)
    Y=bestY.tolist()
    solution.append(bestZ)
    solution.append(Y)
    print('solution',solution)
    print('bestCost',bestCost)
    print('Ysolution',Ysolution)
    return solutionSet,bestZ,bestY,bestCost

if __name__=="__main__":
    main()
end = time.time()
print('Running time: %s Seconds'%(end-start))   
