# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 07:28:25 2021
此版本是按照李晓龙的意见将父代与子代种群合并取消了，不合并了!选择时所有个体都被遍历的版本！
@author: Administrator
"""

import time,copy,random,os
import numpy as np
import gurobipy as gp
import multiprocessing
from multiprocessing import Pool,Queue,Manager

print(os.cpu_count())#计算你的计算机有多少cpu

def workerAllocation(numOfWorker,maxSeru):#生成单元人数分配
    worker_allocation=[1 for i in range(maxSeru)]
    while sum(worker_allocation)<numOfWorker:
        index=random.randint(0,len(worker_allocation)-1)
        worker_allocation[index]+=1
    return worker_allocation

def productDistribution(numOfPoduct,maxSeru):#随机生成产品-单元的分配方案，且方案满足每个单元都有产品分配
    count1=0
    count2=0
    while (count1<maxSeru or count2<numOfPoduct):
        distribution=[[np.random.randint(0,2)for j in range(maxSeru)]for i in range(numOfPoduct)]
        column_sum=[sum(i) for i in zip(*distribution)]
        for i in range(len(column_sum)):#计算每行的和，确保每种产品都有单元能够生产
            if column_sum[i]>0:
                count1+=1
        for i in range(numOfPoduct):#计算每列的和，确保每个单元都分配了产品
            if sum(distribution[i])>0:
                count2+=1
        if (count1 == maxSeru and count2==numOfPoduct):
            return distribution
        else:
            count1=0
            count2=0
          
def initPopulation(popsize,numOfWorker,numOfProduct,maxSeru):#初始化种群，要求生成的个体没有重复个体(目标值相同个体)
    pop1=[]
    pop1.append(workerAllocation(numOfWorker,maxSeru))
    pop1.append(productDistribution(numOfProduct,maxSeru))
    solution=[]
    pop1_cost=computeTotalcost(pop1,demand,TC,FC,A,sc,ec)
    solution.append([pop1,pop1_cost])
    while(len(solution)<popsize):
        pop=[]
        pop.append(workerAllocation(numOfWorker,maxSeru))
        pop.append(productDistribution(numOfProduct,maxSeru))
        pop_solution=[pop,computeTotalcost(pop,demand,TC,FC,A,sc,ec)]
        if pop_solution not in solution:
            solution.append(pop_solution)
    solution.sort(key= lambda x: x[1])#按成本从小到大排序,带成本的
    return solution

def computeFixcost(pop,TC,FC):#计算固定成本，这里指与SERU调度无关的成本，包括构建成本和培训成本
    workerallocation=copy.deepcopy(pop[0])
    productdistribution=copy.deepcopy(pop[1])
    column_sum=[sum(i) for i in zip(*productdistribution)]
    #计算构建成本
    formationCost=len(workerallocation)*FC
    #print('formationCost',formationCost)
    #计算培训成本
    trainingCost=0
    for i in range(len(workerallocation)):#遍历每个单元，用单元人数*单元分配的产品种类数
        everyserutraining_cost=workerallocation[i]*column_sum[i]*TC
        trainingCost+=everyserutraining_cost   
    Fixcost=formationCost+trainingCost
    return Fixcost


def seruLoading(pop,demand,TC,A,sc,ec):#定义函数，输入一个SERU构造和需求，调用gurobi求解最优SERU调度和损失成本
    # 相关数据和类型的简单处理
    demand_copy = []
    for item in demand:
        demand_copy.append(item)
    D=np.array(demand_copy)
    #由SERU构造获得Z和Y的值
    workerallocation=copy.deepcopy(pop[0])
    productdistribution=copy.deepcopy(pop[1])
    z=np.array(workerallocation)
    y=np.array(productdistribution)
    numOfSeru = len(workerallocation)
    numOfProduct = len(productdistribution)
    numOfScenario = len(demand_copy)
    R=1000 #一个足够大的数
    # Create a new model
    m = gp.Model("seruloading")
    
    # Create variables 创建变量
    x = m.addVars(range(numOfScenario),range(numOfProduct),range(numOfSeru),lb=0,vtype=gp.GRB.INTEGER, name="x")#决策变量，SERU生产产品的数量，大于等于0
    
    # 更新变量环境
    m.update()
    
    # 建立一个表达式计算损失成本
    shortage_cost=gp.quicksum((D[s,i]-gp.quicksum(x[s,i,j] for j in range(numOfSeru))) for s in range(numOfScenario) for i in range(numOfProduct)) * TC * sc
    excess_cost=gp.quicksum((z[j] * A -gp.quicksum(x[s,i,j] for i in range(numOfProduct))) for s in range(numOfScenario) for j in range(numOfSeru)) * TC * ec
    total_cost=(shortage_cost + excess_cost) / numOfScenario
    # Set objective设置目标函数
    m.setObjective(total_cost,sense=gp.GRB.MINIMIZE)
    
    # 添加约束条件
    m.addConstrs(x[s,i,j] <= R*y[i,j] for s in range(numOfScenario) for i in range(numOfProduct) for j in range(numOfSeru))#如果单元不能装配产品，则二者之间不会发生产品流，即如果，则；如果，则，小于需求量与单元生产能力中较大的数，大于较小的那个数
    m.addConstrs(gp.quicksum(x[s,i,j] for i in range(numOfProduct)) <= z[j]*A for s in range(numOfScenario) for j in range(numOfSeru))#单元生产量不能超过其生产能力
    m.addConstrs(gp.quicksum(x[s,i,j] for j in range(numOfSeru)) <= D[s,i] for s in range(numOfScenario) for i in range(numOfProduct))#单元生产量不能超过其生产能力
    
    # Optimize model
    m.optimize()
    
    # 输出信息
    if m.status == gp.GRB.Status.OPTIMAL:
        print("\nGlobal optimal solution found. 得到了全局最优解")
        
    loading = []
    for v in m.getVars():
        #print(v.varName, v.X)
        loading.append(v.X)
    #print('loading', loading) 
    loss_cost= m.objVal
    #print("损失成本为：")
    #print(loss_cost)
    loading= np.array(loading)
    loading = loading.reshape((numOfScenario, numOfProduct, numOfSeru))
    seruloading=loading.tolist()
    return loss_cost,seruloading

def computeTotalcost(pop,demand,TC,FC,A,sc,ec):
    fixcost=computeFixcost(pop,TC,FC)
    losscost=seruLoading(pop,demand,TC,A,sc,ec)[0]
    totalcost=fixcost+losscost
    return totalcost

#交叉算子，两点交叉，不改变工人-seru分配，只交换产品-seru分配,交叉后个体有可能是不可行解
def crossoverOperator(pop1,pop2):
    workerallocation1=copy.deepcopy(pop1[0])
    dpop1=copy.deepcopy(pop1[1])
    #print('dpop1',dpop1)
    workerallocation2=copy.deepcopy(pop2[0])
    dpop2=copy.deepcopy(pop2[1])
    #print('dpop2',dpop2)    
    index1=np.random.randint(0,len(dpop1))#随机产生2个交叉点,可以说全部产品的分配都改变，也可以改变一个或者多个
    index2=np.random.randint(index1,len(dpop1))
    #print('交叉点1',index1)
    #print('交叉点2',index2)
    new_p1=dpop1[:index1]+dpop2[index1:index2+1]+dpop1[(index2+1):]
    new_p2=dpop2[:index1]+dpop1[index1:index2+1]+dpop2[index2+1:]
    newpop1=[]
    newpop1.append(workerallocation1)
    newpop1.append(new_p1)
    newpop2=[]
    newpop2.append(workerallocation2)
    newpop2.append(new_p2)
    newpops=[]
    newpops.append(newpop1)
    newpops.append(newpop2)
    #print('交叉个体',newpops)
    return newpops

def mutationOperator(pop):#突变函数，同时突变工人分配和产品-单元分配，变异后的个体仍是可行解
    new_pop=[]
    workerallocation=copy.deepcopy(pop[0])
    productdistribution=copy.deepcopy(pop[1])
    index1=np.random.randint(0,len(workerallocation)-1)#针对工人分配，随机产生两个突变点
    index2=np.random.randint(index1+1,len(workerallocation))
    min_worker=min(workerallocation[index1],workerallocation[index2])
    a=np.random.randint(0,min_worker+1)#作用是随机的生成介于[low,high)
    if a==0:#工人分配两种变异方式：交换两个单元人数；两个单元人数分别调增调减
        #print('工人数不同交换工人数')
        workerallocation[index1],workerallocation[index2]=workerallocation[index2],workerallocation[index1]
    elif a!=0 and workerallocation[index1]==workerallocation[index2]:#两个单元人数相同
        #print('工人数相同不变')
        workerallocation[index1],workerallocation[index2]=workerallocation[index1],workerallocation[index1]
    else:
        #print('工人数调增调减')
        if workerallocation[index1]==min_worker:
            workerallocation[index1]=workerallocation[index1]+a
            workerallocation[index2]=workerallocation[index2]-a
        else:
            workerallocation[index1]=workerallocation[index1]-a
            workerallocation[index2]=workerallocation[index2]+a
    #突变产品-SERU分配，采用两点变异法,交换两个产品的分配
    index3=np.random.randint(0,len(productdistribution)-1)#随机产生两个突变点
    index4=np.random.randint(index3+1,len(productdistribution))
    #print('交换两个产品分配',index3)
    #print('交换两个产品分配',index4)
    productdistribution[index3],productdistribution[index4]=productdistribution[index4],productdistribution[index3]
    new_pop.append(workerallocation)
    new_pop.append(productdistribution)
    return new_pop

def compute_rowsum_columnsum(pop):#计算一个个体的行和和列和
    row_sum=[]
    for i in pop[1]:#求行和，其意义是产品分配单元的数量
        row_sum.append(sum(i))
    column_sum=[sum(i) for i in zip(*pop[1])]#求列和，其意义是单元分配产品的数量
    return row_sum,column_sum

def isfeasibleSolution(pop):#计算各单元生产产品类型数量，便于后期判断是否是可行解
    column_sum=[sum(i) for i in zip(*pop[1])]
    t1=0
    for i in range(len(column_sum)):#计算列的和，确保每个单元都有产品分配
        if column_sum[i]>0:
            t1+=1
            if t1==len(column_sum):#需要继续判断行和是否大于0
                break
        else:
            return False#不是可行解
    t2=0        
    for i in range(len(pop[1])):#计算每行的和，确保每个单元都分配了产品
        if sum(pop[1][i])>0:
            t2+=1
            if t2==len(pop[1]):#判断行和是否大于0
                return True
        else:
            return False#不是可行解
        
def repair(pop):#修复函数，交叉后只可能会出现列和为0的不可行解
    #找到需要修复的位置,可能存在多个列和为0
    clone_pop=copy.deepcopy(pop)
    column_sum=[sum(i) for i in zip(*clone_pop[1])]#求列和，其意义是单元分配产品的数量
    #print('column_sum',column_sum)
    columniszero=[k for k, e in enumerate(column_sum) if e == 0]#找到未分配产品的单元下标
    #print('columniszero',columniszero)
    for i in columniszero:#遍历和为0的列，每一个都需要修复
        repair_pd=np.random.randint(0,len(clone_pop[1]))#随机选一个产品进行修复
        clone_pop[1][repair_pd][i]=1
    #print('clone_pop',clone_pop)
    repair_pop=clone_pop
    return repair_pop

def off_neighbour(pop,pop_cost,max_num):#搜索变异后个体的邻域，返回个体及领域个体中最好的个体，改善子代种群  
    row_sum,column_sum=compute_rowsum_columnsum(pop)
    #找到不能调整的关键位置
    a=[k for k, e in enumerate(row_sum) if e == 1]#找到只分配1个单元的产品下标
    b=[k for k, e in enumerate(column_sum) if e == 1]#找到只分配1个产品的单元下标
    aList=[]
    for i in a:
        aList.append([i,pop[1][i].index(1)])
    bList=[]
    for i in range(len(pop[1])):
        for j in b:
            if pop[1][i][j]==1:
                bList.append([i,j])
    index=[]
    for i in range(len(pop[1])):
        for j in range(len(pop[1][i])):
            index.append([i,j])
    #可以发生变化的位置
    possible_list=[]
    for i in index:
        if (i not in aList) and (i not in bList):
            possible_list.append(i)
    #print('possible_list',possible_list)
    nbs=[]
    neighbours = []
    neighbours.append([pop,pop_cost])
    k=0
    while k < max_num:#生成要求数量的邻域，一定都是可行解
        temp_pop=copy.deepcopy(pop)
        position=random.sample(possible_list,1)#随机选择一个需要调整的位置
        #print('position',position)
        position1=position[0][0]
        position2=position[0][1]
        #print('位置1的值',temp_pop[1][position1][position2])
        
        if temp_pop[1][position1][position2]==0:
            temp_pop[1][position1][position2]=1
        else:
            temp_pop[1][position1][position2]=0
        #print('temp_pop[1]',temp_pop[1])
        
        if temp_pop not in nbs:#产生的邻域个体不允许有重复
            cost=computeTotalcost(temp_pop,demand,TC,FC,A,sc,ec)#计算领域个体的目标函数
            nbs.append(temp_pop)
            neighbours.append([temp_pop, cost])
            k+=1
        else:
            continue
    neighbours.sort(key= lambda x: x[1])#为对前面的对象中的第二维数据（即value）的值进行排序。
    #print('neighbours',neighbours)
    best_nb=neighbours[0]
    return best_nb  


def tournament_selection(solution):#二元锦标赛选择算子，每次选择1个个体
    #个体索引列表
    index=range(len(solution)) 
    #选择1个个体
    tempList=[]
    tempList.extend(random.sample(index,2))#从染色体下表中随机选两个数，选择更优的
    #print('tempList.',tempList)
    #print(' solution[tempList[0]]', solution[tempList[0]])
    #print(' solution[tempList[1]]', solution[tempList[1]])
    if solution[tempList[0]][1]<=solution[tempList[1]][1]:
        select=solution[tempList[0]]
    else:
        select=solution[tempList[1]]
    return select #select=[个体，个体成本]

def roulettewheel_selection(solution):#每次选择1个个体,solution是带有成本的种群
    cumulativeProbability=[]
    sumFitness=sum(1/x[1] for x in solution)#计算适应度值倒数之和
    #print('sumFitness',sumFitness)
    probability=[]
    #计算每个个体被选中概率
    for i in range(len(solution)):
        probability.append((1/solution[i][1])/sumFitness)#成本小的被选中的概率高
    #print('probability',probability)
    #计算累计概率
    for j in range(len(solution)):
        if j == 0:
            cumulativeProbability.append(probability[j])
        else:
            cumulativeProbability.append(cumulativeProbability[j-1]+probability[j])
    #print('cumulativeProbability',cumulativeProbability)
    #选择1个个体
    r=np.random.uniform(0,1)#在（0，1）之间随机取一个值
    #print('r',r)
    for j in range(len(cumulativeProbability)):#用r和cumulativeProbability中的每个值比较
        if j==0:
            if r>=0 and r<= cumulativeProbability[j]:
                select=solution[j]
        else:
            if r>=cumulativeProbability[j-1] and r<= cumulativeProbability[j]:
                select=solution[j]
    return select

def traversalSampling_selection(solution):   #随机遍历抽样 
    sumFitness=sum(x[1] for x in solution)#计算适应度值之和
    #print('sumFitness',sumFitness)
    cumulativeFitness=[]
    #计算累计适应度
    for j in range(len(solution)):
        if j == 0:
            cumulativeFitness.append(solution[j][1])
        else:
            cumulativeFitness.append(cumulativeFitness[j-1]+solution[j][1])
    #print('cumulativeFitness',cumulativeFitness)
    #计算指针的间距
    spacing=sumFitness/1
    #print('spacing',spacing)
    #随机生成起点指针位置
    start=np.random.uniform(0,spacing)#在（0，spacing）之间随机取一个数
    #print('start',start)
    #计算各指针的位置Pointers
    Pointers=[start+0*spacing]#[start+i*spacing for i in [0,1]]
    #print('Pointers',Pointers)
    #根据各指针位置，选择出2个个体
    for j in range(len(solution)):#用Pointers里的每个值和solution中的适应度值比较
        if j==0:
            if Pointers[0]>=0 and Pointers[0]<= cumulativeFitness[j]:
                select=solution[j]
        else:
            if Pointers[0]>=cumulativeFitness[j-1] and Pointers[0]<= cumulativeFitness[j]:
                select=solution[j]
    return select

def random_selection(solution):#随机选择
    solution_copy=copy.deepcopy(solution)
    number=np.random.randint(0,len(solution_copy))
    #print('当前子种群',solution)
    #print('number',number)
    select=solution_copy[number]
    #print('select',select)
    return select

def selectOperator(selectProbability):#输入策略选择概率，返回选择策略
    cumulativeP=[]
    sumP=sum(selectProbability)#计算被选择概率之和
    method=[i for i in range(len(selectProbability))]
    #print('method',method)
    probability=[]
    #计算每个个体被选中的相对概率
    for i in range(len(selectProbability)):
        probability.append(selectProbability[i]/sumP)
    #print('probability',probability)
    #计算累计概率
    for j in range(len(selectProbability)):
        if j == 0:
            cumulativeP.append(probability[j])
        else:
            cumulativeP.append(cumulativeP[j-1]+probability[j])
    #print('cumulativeP',cumulativeP)
    #选择1个选择策略，选择两个个体
    r=np.random.uniform(0,1)#在（0，1）之间随机取一个值
    #print('r',r)
    for j in range(len(cumulativeP)):#用r和cumulativeP中的每个值比较
        if j==0:
            if r>=0 and r<= cumulativeP[j]:#如果选择第一种，则用锦标赛选择策略选择2个个体
                selectmethod=method[0]
                #print('锦标赛',selectmethod)
        elif j==1:
            if r>=cumulativeP[j-1] and r<= cumulativeP[j]:
                selectmethod=method[1]
                #print('轮盘赌',selectmethod)
        else:
            if r>=cumulativeP[j-1] and r<= cumulativeP[j]:
                selectmethod=method[2]
                #print('随机遍历抽样',selectmethod)
    return selectmethod

def getPop(method,solution):#输入选择策略和种群，输出选择的一个个体
    if method==0:
        pop=tournament_selection(solution)
    elif method==1:
        pop=roulettewheel_selection(solution)
    else:
        pop=random_selection(solution)
    return pop

'''
demand=[[92, 116, 92], [63, 177, 60], [89, 126, 85], [114, 93, 93], [173, 118, 9], [66, 149, 85], [104, 85, 111], [107, 90, 103], [51, 105, 144], [90, 114, 96], [28, 89, 183], [89, 123, 88], [83, 128, 89], [174, 37, 89], [84, 107, 109], [144, 51, 105], [139, 26, 135], [51, 118, 131], [53, 160, 87], [108, 159, 33], [153, 90, 57], [141, 103, 56], [85, 51, 164], [101, 103, 96], [149, 41, 110], [108, 126, 66], [115, 87, 98], [35, 131, 134], [103, 75, 122], [162, 39, 99], [75, 145, 80], [134, 36, 130], [86, 118, 96], [11, 142, 147], [160, 103, 37], [134, 94, 72], [95, 105, 100], [120, 95, 85], [109, 50, 141], [102, 113, 85], [154, 75, 71], [139, 79, 82], [118, 88, 94], [15, 137, 148], [101, 89, 110], [164, 85, 51], [96, 68, 136], [106, 87, 107], [129, 77, 94], [50, 118, 132], [136, 108, 56], [131, 71, 98], [95, 157, 48], [108, 86, 106], [158, 27, 115], [92, 121, 87], [92, 134, 74], [107, 62, 131], [154, 66, 80], [122, 163, 15], [112, 145, 43], [145, 23, 132], [139, 98, 63], [142, 30, 128], [138, 73, 89], [71, 119, 110], [45, 171, 84], [102, 123, 75], [72, 176, 52], [132, 117, 51], [104, 133, 63], [93, 55, 152], [98, 69, 133], [86, 108, 106], [79, 109, 112], [72, 130, 98], [127, 128, 45], [79, 65, 156], [72, 107, 121], [128, 101, 71], [108, 75, 117], [111, 84, 105], [43, 141, 116], [73, 145, 82], [114, 128, 58], [140, 56, 104], [89, 107, 104], [53, 44, 203], [99, 30, 171], [144, 47, 109], [116, 169, 15], [110, 155, 35], [85, 94, 121], [66, 141, 93], [100, 144, 56], [107, 101, 92], [123, 55, 122], [129, 110, 61], [111, 84, 105], [70, 47, 183], [128, 46, 126], [133, 65, 102], [114, 122, 64], [55, 189, 56], [127, 23, 150], [160, 115, 25], [115, 83, 102], [49, 82, 169], [139, 94, 67], [95, 115, 90], [151, 54, 95], [103, 69, 128], [22, 200, 78], [155, 80, 65], [77, 130, 93], [42, 159, 99], [65, 112, 123], [135, 84, 81], [129, 92, 79], [232, 35, 33], [104, 47, 149], [104, 100, 96], [114, 113, 73], [48, 105, 147], [98, 132, 70], [34, 130, 136], [166, 52, 82], [75, 76, 149], [134, 55, 111], [174, 32, 94], [43, 112, 145], [165, 38, 97], [136, 58, 106], [19, 207, 74], [147, 54, 99], [103, 151, 46], [123, 118, 59], [86, 122, 92], [38, 136, 126], [125, 80, 95], [102, 71, 127], [101, 58, 141], [139, 28, 133], [58, 79, 163], [102, 60, 138], [93, 115, 92], [113, 145, 42], [142, 66, 92], [136, 73, 91], [137, 40, 123], [122, 114, 64], [125, 123, 52], [96, 133, 71], [72, 122, 106], [106, 125, 69], [105, 120, 75], [86, 75, 139], [137, 132, 31], [195, 102, 3], [97, 148, 55], [80, 92, 128], [15, 165, 120], [125, 86, 89], [83, 88, 129], [103, 168, 29], [127, 49, 124], [123, 98, 79], [21, 163, 116], [131, 48, 121], [109, 122, 69], [152, 47, 101], [141, 89, 70], [139, 72, 89], [72, 93, 135], [107, 119, 74], [101, 69, 130], [99, 91, 110], [134, 59, 107], [104, 118, 78], [102, 72, 126], [106, 104, 90], [87, 105, 108], [58, 119, 123], [82, 130, 88], [105, 146, 49], [90, 148, 62], [84, 106, 110], [154, 66, 80], [133, 115, 52], [99, 112, 89], [181, 56, 63], [131, 92, 77], [55, 98, 147], [142, 93, 65], [47, 119, 134], [83, 103, 114], [22, 119, 159], [41, 109, 150], [46, 125, 129], [96, 92, 112]]
#demand=[[90, 93, 94, 102, 121], [104, 98, 103, 95, 100], [107, 82, 97, 111, 103], [91, 115, 81, 106, 107], [99, 97, 107, 90, 107], [90, 88, 106, 95, 121], [102, 98, 103, 102, 95], [101, 85, 115, 86, 113], [100, 94, 105, 111, 90], [92, 117, 97, 100, 94], [110, 105, 117, 92, 76], [90, 105, 101, 99, 105], [97, 122, 85, 97, 99], [103, 116, 99, 116, 66], [93, 99, 113, 86, 109], [98, 89, 95, 110, 108], [111, 118, 74, 99, 98], [98, 85, 106, 110, 101], [97, 94, 107, 103, 99], [81, 95, 101, 110, 113], [107, 106, 103, 95, 89], [88, 112, 107, 111, 82], [89, 93, 106, 101, 111], [105, 102, 94, 92, 107], [98, 104, 95, 94, 109], [86, 102, 108, 95, 109], [113, 90, 79, 107, 111], [100, 97, 100, 112, 91], [99, 99, 97, 104, 101], [104, 108, 102, 92, 94], [102, 102, 103, 91, 102], [98, 85, 90, 122, 105], [100, 114, 90, 110, 86], [102, 95, 104, 96, 103], [102, 102, 106, 96, 94], [97, 81, 96, 107, 119], [100, 100, 103, 98, 99], [80, 102, 102, 110, 106], [96, 113, 107, 90, 94], [99, 106, 108, 99, 88], [97, 101, 85, 96, 121], [104, 99, 95, 117, 85], [108, 106, 93, 91, 102], [110, 96, 102, 91, 101], [99, 99, 98, 101, 103], [94, 106, 101, 103, 96], [105, 98, 96, 101, 100], [92, 94, 100, 127, 87], [91, 86, 94, 121, 108], [108, 91, 103, 88, 110], [90, 97, 92, 109, 112], [81, 106, 93, 119, 101], [96, 109, 112, 96, 87], [109, 101, 109, 81, 100], [96, 104, 105, 94, 101], [102, 100, 94, 97, 107], [96, 94, 98, 110, 102], [81, 100, 114, 90, 115], [93, 96, 97, 107, 107], [101, 98, 100, 104, 97], [105, 91, 110, 104, 90], [115, 76, 86, 112, 111], [97, 102, 105, 98, 98], [112, 102, 104, 94, 88], [109, 91, 102, 98, 100], [112, 94, 104, 98, 92], [93, 79, 108, 107, 113], [102, 98, 96, 96, 108], [81, 105, 99, 100, 115], [82, 106, 115, 91, 106], [101, 101, 101, 97, 100], [120, 108, 102, 86, 84], [80, 102, 95, 105, 118], [106, 93, 97, 106, 98], [96, 90, 92, 111, 111], [102, 105, 103, 99, 91], [113, 105, 76, 101, 105], [97, 103, 105, 90, 105], [103, 93, 98, 102, 104], [112, 88, 108, 95, 97], [91, 110, 97, 109, 93], [109, 98, 100, 92, 101], [98, 80, 105, 107, 110], [98, 101, 86, 124, 91], [88, 91, 102, 111, 108], [106, 99, 99, 93, 103], [97, 97, 109, 100, 97], [103, 88, 99, 107, 103], [99, 90, 98, 109, 104], [100, 101, 115, 102, 82], [86, 102, 95, 95, 122], [100, 107, 101, 98, 94], [106, 93, 108, 110, 83], [97, 91, 106, 117, 89], [101, 102, 95, 105, 97], [95, 98, 113, 105, 89], [98, 109, 90, 104, 99], [102, 99, 93, 101, 105], [108, 105, 88, 102, 97], [100, 103, 109, 99, 89], [84, 108, 95, 108, 105], [88, 114, 92, 100, 106], [96, 103, 106, 99, 96], [95, 95, 116, 106, 88], [95, 106, 102, 97, 100], [104, 92, 96, 110, 98], [111, 92, 98, 104, 95], [98, 119, 105, 94, 84], [96, 99, 110, 85, 110], [114, 86, 86, 111, 103], [109, 96, 112, 86, 97], [112, 94, 108, 91, 95], [81, 104, 108, 100, 107], [87, 119, 100, 94, 100], [100, 102, 94, 100, 104], [87, 102, 83, 106, 122], [100, 103, 107, 92, 98], [91, 92, 91, 102, 124], [88, 93, 116, 112, 91], [105, 105, 93, 101, 96], [104, 81, 98, 93, 124], [95, 101, 105, 109, 90], [114, 106, 101, 84, 95], [105, 88, 105, 115, 87], [114, 104, 81, 101, 100], [101, 94, 104, 106, 95], [94, 99, 94, 119, 94], [92, 102, 113, 96, 97], [105, 106, 104, 95, 90], [91, 99, 105, 105, 100], [94, 103, 71, 115, 117], [110, 100, 98, 85, 107], [97, 109, 84, 100, 110], [118, 92, 98, 90, 102], [97, 117, 92, 111, 83], [86, 103, 102, 97, 112], [105, 89, 101, 109, 96], [93, 90, 111, 99, 107], [91, 109, 98, 116, 86], [113, 87, 96, 97, 107], [100, 110, 117, 95, 78], [129, 89, 94, 97, 91], [106, 92, 92, 106, 104], [108, 104, 91, 99, 98], [92, 108, 97, 91, 112], [110, 95, 97, 106, 92], [110, 109, 90, 91, 100], [95, 109, 91, 112, 93], [90, 101, 100, 92, 117], [106, 107, 88, 96, 103], [91, 95, 101, 112, 101], [93, 113, 92, 105, 97], [105, 97, 93, 100, 105], [105, 114, 92, 91, 98], [101, 86, 110, 107, 96], [112, 94, 111, 101, 82], [96, 103, 93, 97, 111], [96, 90, 111, 92, 111], [110, 99, 102, 104, 85], [111, 100, 95, 95, 99], [109, 115, 99, 83, 94], [103, 109, 87, 101, 100], [111, 97, 102, 92, 98], [100, 107, 96, 95, 102], [88, 109, 106, 102, 95], [100, 105, 110, 86, 99], [102, 105, 95, 112, 86], [101, 107, 99, 100, 93], [99, 98, 101, 112, 90], [94, 105, 108, 83, 110], [99, 106, 97, 105, 93], [88, 104, 107, 95, 106], [105, 121, 98, 91, 85], [103, 77, 111, 99, 110], [98, 113, 96, 97, 96], [92, 106, 107, 107, 88], [99, 98, 118, 85, 100], [96, 95, 103, 114, 92], [92, 110, 105, 86, 107], [105, 95, 107, 93, 100], [90, 108, 92, 101, 109], [98, 102, 101, 107, 92], [95, 103, 94, 104, 104], [97, 103, 87, 113, 100], [92, 115, 97, 94, 102], [94, 114, 89, 103, 100], [90, 92, 97, 115, 106], [120, 97, 88, 97, 98], [104, 95, 95, 103, 103], [96, 91, 108, 87, 118], [108, 107, 93, 105, 87], [110, 106, 95, 90, 99], [106, 98, 92, 108, 96], [87, 94, 101, 108, 110], [96, 87, 109, 105, 103], [109, 93, 103, 117, 78], [100, 96, 91, 99, 114], [106, 96, 106, 97, 95], [93, 99, 109, 103, 96], [92, 96, 119, 94, 99]]
numOfProduct=3
numOfWorker=3
maxSeru=3
TC=1
FC=0
sc=0.8 
ec=0.6
A=100
max_num=1
popsize=10#种群数量
crossRate=0.8
mutationRate=0.4
maxIterator=10#最优解保持不变的最大代数
'''
slave=[[[[1, 1, 1], [[0, 1, 1], [1, 1, 1], [1, 1, 0]]], 7.244999999999834],
 [[[1, 1, 1], [[0, 1, 1], [1, 1, 0], [1, 1, 1]]], 7.272999999999797],
 [[[1, 1, 1], [[1, 1, 0], [1, 1, 0], [0, 1, 1]]], 28.14099999999985],
 [[[1, 1, 1], [[1, 0, 1], [1, 1, 1], [1, 0, 1]]], 29.280999999999835],
 [[[1, 1, 1], [[1, 0, 0], [1, 1, 1], [1, 1, 1]]], 30.736999999999853],
 [[[1, 1, 1], [[1, 0, 1], [1, 1, 0], [0, 1, 0]]], 32.49599999999981],
 [[[1, 1, 1], [[0, 0, 1], [1, 1, 1], [0, 1, 1]]], 38.941999999999894],
 [[[1, 1, 1], [[1, 0, 0], [1, 0, 1], [1, 1, 1]]], 41.1959999999998],
 [[[1, 1, 1], [[0, 0, 1], [1, 1, 0], [0, 1, 0]]], 55.008999999999844],
 [[[1, 1, 1], [[0, 0, 1], [1, 0, 0], [1, 1, 0]]], 57.262999999999806]]
'''
slave=[[[[1, 1, 1], [[0, 1, 1], [1, 1, 1], [1, 1, 0]]], 7.244999999999834],
 [[[1, 1, 1], [[0, 1, 1], [1, 1, 0], [1, 1, 1]]], 7.272999999999797],
 [[[1, 1, 1], [[1, 1, 0], [1, 1, 0], [0, 1, 1]]], 28.14099999999985]]


q = Queue(5)  
q.put([[[1, 1, 1], [[1, 1, 0], [0, 1, 1], [1, 0, 1]]], 6.293999999999869])
q.put([[[1, 1, 1], [[1, 1, 1], [1, 1, 0], [1, 0, 1]]], 7.069999999999936])
#print(q.get())
#print(q.get())
selectProbability=[1/3,1/3,1/3]

'''

def son_process(slave,selectProbability,q):#子进程，输入从种群、选择策略概率和主种群的队列，输出下一代从种群和更新选择概率
    selectProbability_copy=copy.deepcopy(selectProbability)
    newslave=[] 
    selectmethod=selectOperator(selectProbability_copy) 
    #print('selectmethod',selectmethod)
    while len(newslave)<len(slave):
        '''选择是从子种群选择一个个体，从主种群选择一个个体'''
        #print('############################进入下一循环###################')
        parents=[]
        #从主种群取出一个个体
        parent1=q.get()#从队列里取值.如果q里没有值，会一直处于等待状态
        #print('parent1',parent1)
        masterPop=copy.deepcopy(parent1)
        q.put(masterPop)#将主种群中取出来的个体复制后再放回去，#往队列里放入数据；队列满，就等待
        #从从种群中随机选择一个个体（自适应选择算子）
        parent2=getPop(selectmethod,slave)
        parents.append(parent1)
        parents.append(parent2)
        parents.sort(key= lambda x: x[1])#根据目标值从大到小排序
        #print('###########parents',parents)
        
        cross=[]
        #交叉
        if parents[0]!=parents[1]:
            rc= random.random()#随机产生一个数
            if rc<=crossRate:#交叉
                cp= crossoverOperator(parents[0][0], parents[1][0])#cp中有两个个体
                for i in cp: #遍历新个体，先判断变异后个体是否是可行解，如果是再判断其是否优于原个体
                    if i ==parents[0][0]:#产生的不是新解
                        cross.append(parents[0])
                    elif i==parents[1][0]:
                        cross.append(parents[1])
                    else:#产生的是新解，则需计算目标值
                        if isfeasibleSolution(i) is True:
                            cp_pop=i
                            cp_cost=computeTotalcost(cp_pop,demand,TC,FC,A,sc,ec)
                        else:#不是可行解需要修复
                            cp_pop=repair(i)
                            cp_cost=computeTotalcost(cp_pop,demand,TC,FC,A,sc,ec)
                        if cp_cost<parents[0][1]:#交叉后代优于两个父代个体
                            cross.append([cp_pop,cp_cost])
                        else:
                            if parents[0] not in cross:
                                cross.append(parents[0])
                            else:
                                cross.append(parents[1])       
            else:#没有发生交叉，添加父代
                cross=parents
                #print('############没有发生交叉')
        else:
            cross=parents
            #cross.append(min_parent)#此时只添加一个个体
        #print('###########交叉后代',cross)    
        #变异
        mutation=[]
        for i in cross:#遍历交叉后解（包含目标值），进行变异
            rm = random.random()#随机产生一个数
            if rm<=mutationRate:#变异
                mp=mutationOperator(i[0])
                mp_cost=computeTotalcost(i[0],demand,TC,FC,A,sc,ec)
                if mp_cost<i[1]:#如果变异个体优于交叉后个体，则添加到变异后代中
                    mutation.append([mp,mp_cost])
                    #print('@@@@@发生变异')
                else:
                    mutation.append(i)#否则将交叉后个体添加到变异后代中
                    #print('@@@@@没有发生变异')
            else:
                mutation.append(i)
        #print('###########变异后代',mutation)  
        #邻域搜索
        for m in mutation:#一定包含两个个体,分别进行邻域搜索
            newslave.append(off_neighbour(m[0],m[1],max_num))
        #print('#########################邻域',newslave)
    newslave.sort(key= lambda x: x[1])#按成本从小到大排序,带成本的
    newslave=newslave[:len(slave)]#选出最优的种群数量个个体，作为新一代种群
    #print('newslave',newslave)
    #更新选择概率
    newselect=selectProbability_copy
    #print('@@@@@@@@@@@@@@@@@@@@@更新前newselect',newselect)
    if newslave[0][1]<slave[0][1]:#比较父代与子代最优解
        newselect[selectmethod] += 0.1
    else:
        if  newselect[selectmethod] >= 0.1:
            newselect[selectmethod] -= 0.1
        else:
            newselect[selectmethod] -= 0
    #print('@@@@@@@@@@@@@@@@@@@@更新后newselect',newselect)
    return newslave,newselect

def subset(alist, idxs):
    '''
        用法：根据下标idxs取出列表alist的子集
        alist: list
        idxs: list
    '''
    sub_list = []
    for idx in idxs:
        sub_list.append(alist[idx])

    return sub_list

def split_list(alist, group_num=4, shuffle=True, retain_left=False):
    '''
        用法：将alist切分成group个子列表，每个子列表里面有len(alist)//group个元素
        shuffle: 表示是否要随机切分列表，默认为True
        retain_left: 若将列表alist分成group_num个子列表后还要剩余，是否将剩余的元素单独作为一组
    '''

    index = list(range(len(alist))) # 保留下标
    #print('初始index',index)

    # 是否打乱列表
    if shuffle: 
        random.shuffle(index) 
    #print('打乱index',index)    
    
    elem_num = len(alist) // group_num # 每一个子列表所含有的元素数量
    sub_lists = []
    
    # 取出每一个子列表所包含的元素，存入列表中
    for idx in range(group_num):
        start, end = idx*elem_num, (idx+1)*elem_num
        #print('start',start)
        #print('end',end)
        sub_lists.append(subset(alist, index[start:end]))
    
    # 是否将最后剩余的元素作为单独的一组
    if retain_left and group_num * elem_num != len(index): # 列表元素数量未能整除子列表数，需要将最后那一部分元素单独作为新的列表
        sub_lists.append(subset(alist, index[end:]))
    return sub_lists

demand=[[92, 116, 92], [63, 177, 60], [89, 126, 85], [114, 93, 93], [173, 118, 9], [66, 149, 85], [104, 85, 111], [107, 90, 103], [51, 105, 144], [90, 114, 96], [28, 89, 183], [89, 123, 88], [83, 128, 89], [174, 37, 89], [84, 107, 109], [144, 51, 105], [139, 26, 135], [51, 118, 131], [53, 160, 87], [108, 159, 33], [153, 90, 57], [141, 103, 56], [85, 51, 164], [101, 103, 96], [149, 41, 110], [108, 126, 66], [115, 87, 98], [35, 131, 134], [103, 75, 122], [162, 39, 99], [75, 145, 80], [134, 36, 130], [86, 118, 96], [11, 142, 147], [160, 103, 37], [134, 94, 72], [95, 105, 100], [120, 95, 85], [109, 50, 141], [102, 113, 85], [154, 75, 71], [139, 79, 82], [118, 88, 94], [15, 137, 148], [101, 89, 110], [164, 85, 51], [96, 68, 136], [106, 87, 107], [129, 77, 94], [50, 118, 132], [136, 108, 56], [131, 71, 98], [95, 157, 48], [108, 86, 106], [158, 27, 115], [92, 121, 87], [92, 134, 74], [107, 62, 131], [154, 66, 80], [122, 163, 15], [112, 145, 43], [145, 23, 132], [139, 98, 63], [142, 30, 128], [138, 73, 89], [71, 119, 110], [45, 171, 84], [102, 123, 75], [72, 176, 52], [132, 117, 51], [104, 133, 63], [93, 55, 152], [98, 69, 133], [86, 108, 106], [79, 109, 112], [72, 130, 98], [127, 128, 45], [79, 65, 156], [72, 107, 121], [128, 101, 71], [108, 75, 117], [111, 84, 105], [43, 141, 116], [73, 145, 82], [114, 128, 58], [140, 56, 104], [89, 107, 104], [53, 44, 203], [99, 30, 171], [144, 47, 109], [116, 169, 15], [110, 155, 35], [85, 94, 121], [66, 141, 93], [100, 144, 56], [107, 101, 92], [123, 55, 122], [129, 110, 61], [111, 84, 105], [70, 47, 183], [128, 46, 126], [133, 65, 102], [114, 122, 64], [55, 189, 56], [127, 23, 150], [160, 115, 25], [115, 83, 102], [49, 82, 169], [139, 94, 67], [95, 115, 90], [151, 54, 95], [103, 69, 128], [22, 200, 78], [155, 80, 65], [77, 130, 93], [42, 159, 99], [65, 112, 123], [135, 84, 81], [129, 92, 79], [232, 35, 33], [104, 47, 149], [104, 100, 96], [114, 113, 73], [48, 105, 147], [98, 132, 70], [34, 130, 136], [166, 52, 82], [75, 76, 149], [134, 55, 111], [174, 32, 94], [43, 112, 145], [165, 38, 97], [136, 58, 106], [19, 207, 74], [147, 54, 99], [103, 151, 46], [123, 118, 59], [86, 122, 92], [38, 136, 126], [125, 80, 95], [102, 71, 127], [101, 58, 141], [139, 28, 133], [58, 79, 163], [102, 60, 138], [93, 115, 92], [113, 145, 42], [142, 66, 92], [136, 73, 91], [137, 40, 123], [122, 114, 64], [125, 123, 52], [96, 133, 71], [72, 122, 106], [106, 125, 69], [105, 120, 75], [86, 75, 139], [137, 132, 31], [195, 102, 3], [97, 148, 55], [80, 92, 128], [15, 165, 120], [125, 86, 89], [83, 88, 129], [103, 168, 29], [127, 49, 124], [123, 98, 79], [21, 163, 116], [131, 48, 121], [109, 122, 69], [152, 47, 101], [141, 89, 70], [139, 72, 89], [72, 93, 135], [107, 119, 74], [101, 69, 130], [99, 91, 110], [134, 59, 107], [104, 118, 78], [102, 72, 126], [106, 104, 90], [87, 105, 108], [58, 119, 123], [82, 130, 88], [105, 146, 49], [90, 148, 62], [84, 106, 110], [154, 66, 80], [133, 115, 52], [99, 112, 89], [181, 56, 63], [131, 92, 77], [55, 98, 147], [142, 93, 65], [47, 119, 134], [83, 103, 114], [22, 119, 159], [41, 109, 150], [46, 125, 129], [96, 92, 112]]
#print(len(demand))
#demand=[[90, 93, 94, 102, 121], [104, 98, 103, 95, 100], [107, 82, 97, 111, 103], [91, 115, 81, 106, 107], [99, 97, 107, 90, 107], [90, 88, 106, 95, 121], [102, 98, 103, 102, 95], [101, 85, 115, 86, 113], [100, 94, 105, 111, 90], [92, 117, 97, 100, 94], [110, 105, 117, 92, 76], [90, 105, 101, 99, 105], [97, 122, 85, 97, 99], [103, 116, 99, 116, 66], [93, 99, 113, 86, 109], [98, 89, 95, 110, 108], [111, 118, 74, 99, 98], [98, 85, 106, 110, 101], [97, 94, 107, 103, 99], [81, 95, 101, 110, 113], [107, 106, 103, 95, 89], [88, 112, 107, 111, 82], [89, 93, 106, 101, 111], [105, 102, 94, 92, 107], [98, 104, 95, 94, 109], [86, 102, 108, 95, 109], [113, 90, 79, 107, 111], [100, 97, 100, 112, 91], [99, 99, 97, 104, 101], [104, 108, 102, 92, 94], [102, 102, 103, 91, 102], [98, 85, 90, 122, 105], [100, 114, 90, 110, 86], [102, 95, 104, 96, 103], [102, 102, 106, 96, 94], [97, 81, 96, 107, 119], [100, 100, 103, 98, 99], [80, 102, 102, 110, 106], [96, 113, 107, 90, 94], [99, 106, 108, 99, 88], [97, 101, 85, 96, 121], [104, 99, 95, 117, 85], [108, 106, 93, 91, 102], [110, 96, 102, 91, 101], [99, 99, 98, 101, 103], [94, 106, 101, 103, 96], [105, 98, 96, 101, 100], [92, 94, 100, 127, 87], [91, 86, 94, 121, 108], [108, 91, 103, 88, 110], [90, 97, 92, 109, 112], [81, 106, 93, 119, 101], [96, 109, 112, 96, 87], [109, 101, 109, 81, 100], [96, 104, 105, 94, 101], [102, 100, 94, 97, 107], [96, 94, 98, 110, 102], [81, 100, 114, 90, 115], [93, 96, 97, 107, 107], [101, 98, 100, 104, 97], [105, 91, 110, 104, 90], [115, 76, 86, 112, 111], [97, 102, 105, 98, 98], [112, 102, 104, 94, 88], [109, 91, 102, 98, 100], [112, 94, 104, 98, 92], [93, 79, 108, 107, 113], [102, 98, 96, 96, 108], [81, 105, 99, 100, 115], [82, 106, 115, 91, 106], [101, 101, 101, 97, 100], [120, 108, 102, 86, 84], [80, 102, 95, 105, 118], [106, 93, 97, 106, 98], [96, 90, 92, 111, 111], [102, 105, 103, 99, 91], [113, 105, 76, 101, 105], [97, 103, 105, 90, 105], [103, 93, 98, 102, 104], [112, 88, 108, 95, 97], [91, 110, 97, 109, 93], [109, 98, 100, 92, 101], [98, 80, 105, 107, 110], [98, 101, 86, 124, 91], [88, 91, 102, 111, 108], [106, 99, 99, 93, 103], [97, 97, 109, 100, 97], [103, 88, 99, 107, 103], [99, 90, 98, 109, 104], [100, 101, 115, 102, 82], [86, 102, 95, 95, 122], [100, 107, 101, 98, 94], [106, 93, 108, 110, 83], [97, 91, 106, 117, 89], [101, 102, 95, 105, 97], [95, 98, 113, 105, 89], [98, 109, 90, 104, 99], [102, 99, 93, 101, 105], [108, 105, 88, 102, 97], [100, 103, 109, 99, 89], [84, 108, 95, 108, 105], [88, 114, 92, 100, 106], [96, 103, 106, 99, 96], [95, 95, 116, 106, 88], [95, 106, 102, 97, 100], [104, 92, 96, 110, 98], [111, 92, 98, 104, 95], [98, 119, 105, 94, 84], [96, 99, 110, 85, 110], [114, 86, 86, 111, 103], [109, 96, 112, 86, 97], [112, 94, 108, 91, 95], [81, 104, 108, 100, 107], [87, 119, 100, 94, 100], [100, 102, 94, 100, 104], [87, 102, 83, 106, 122], [100, 103, 107, 92, 98], [91, 92, 91, 102, 124], [88, 93, 116, 112, 91], [105, 105, 93, 101, 96], [104, 81, 98, 93, 124], [95, 101, 105, 109, 90], [114, 106, 101, 84, 95], [105, 88, 105, 115, 87], [114, 104, 81, 101, 100], [101, 94, 104, 106, 95], [94, 99, 94, 119, 94], [92, 102, 113, 96, 97], [105, 106, 104, 95, 90], [91, 99, 105, 105, 100], [94, 103, 71, 115, 117], [110, 100, 98, 85, 107], [97, 109, 84, 100, 110], [118, 92, 98, 90, 102], [97, 117, 92, 111, 83], [86, 103, 102, 97, 112], [105, 89, 101, 109, 96], [93, 90, 111, 99, 107], [91, 109, 98, 116, 86], [113, 87, 96, 97, 107], [100, 110, 117, 95, 78], [129, 89, 94, 97, 91], [106, 92, 92, 106, 104], [108, 104, 91, 99, 98], [92, 108, 97, 91, 112], [110, 95, 97, 106, 92], [110, 109, 90, 91, 100], [95, 109, 91, 112, 93], [90, 101, 100, 92, 117], [106, 107, 88, 96, 103], [91, 95, 101, 112, 101], [93, 113, 92, 105, 97], [105, 97, 93, 100, 105], [105, 114, 92, 91, 98], [101, 86, 110, 107, 96], [112, 94, 111, 101, 82], [96, 103, 93, 97, 111], [96, 90, 111, 92, 111], [110, 99, 102, 104, 85], [111, 100, 95, 95, 99], [109, 115, 99, 83, 94], [103, 109, 87, 101, 100], [111, 97, 102, 92, 98], [100, 107, 96, 95, 102], [88, 109, 106, 102, 95], [100, 105, 110, 86, 99], [102, 105, 95, 112, 86], [101, 107, 99, 100, 93], [99, 98, 101, 112, 90], [94, 105, 108, 83, 110], [99, 106, 97, 105, 93], [88, 104, 107, 95, 106], [105, 121, 98, 91, 85], [103, 77, 111, 99, 110], [98, 113, 96, 97, 96], [92, 106, 107, 107, 88], [99, 98, 118, 85, 100], [96, 95, 103, 114, 92], [92, 110, 105, 86, 107], [105, 95, 107, 93, 100], [90, 108, 92, 101, 109], [98, 102, 101, 107, 92], [95, 103, 94, 104, 104], [97, 103, 87, 113, 100], [92, 115, 97, 94, 102], [94, 114, 89, 103, 100], [90, 92, 97, 115, 106], [120, 97, 88, 97, 98], [104, 95, 95, 103, 103], [96, 91, 108, 87, 118], [108, 107, 93, 105, 87], [110, 106, 95, 90, 99], [106, 98, 92, 108, 96], [87, 94, 101, 108, 110], [96, 87, 109, 105, 103], [109, 93, 103, 117, 78], [100, 96, 91, 99, 114], [106, 96, 106, 97, 95], [93, 99, 109, 103, 96], [92, 96, 119, 94, 99]]
numOfProduct=3
numOfWorker=3
maxSeru=3
TC=1
FC=0
sc=0.8 
ec=0.6
A=100
max_num=2#邻域个数
popsize=10#从种群规模
crossRate=0.8
mutationRate=0.4
maxIterator=2#最优解保持不变的最大代数


def main():#主进程,join()的作用是是在进程中可以阻塞主进程的执行，指导全部的子进程全部完成之后，才继续运行主进程后面的代码
    print("………………主进程启动………………")
    bestsolution_list=[]#用于存放不同单元数下的最优解
    select=[[1/3,1/3,1/3],[1/3,1/3,1/3],[1/3,1/3,1/3],[1/3,1/3,1/3]]#初始化选择概率
    for i in range(2,maxSeru+1):
        print('开始运行单元数为：',i) 
        #初始化从种群，生成四个从种群
        slaveList=initPopulation(4*popsize,numOfWorker,numOfProduct,i)
        sub_slave=split_list(slaveList, group_num=4, shuffle=True, retain_left=False)
        print('sub_slave',sub_slave)
        #初始化主种群
        q = Manager().Queue()#队列(容器)，用以存储主种群
        slaveList.sort(key= lambda x: x[1])#按成本从小到大排序,带成本的
        intMaster=slaveList[:popsize]#选出最优的种群数量个个体，初始化主种群
        for pop in range(len(intMaster)):
            q.put(intMaster[pop])
        print('初始化主种群个体数',q.qsize())
        #种群进化
        iterator=0
        print("@@@@@@@@@@@@@@@@@@@@进入进化循环")
        while iterator < maxIterator:#最大迭代次数
            print('…………………………第几代……………………',iterator)
            #定义一个进程池，最大进程数4，使四个从种群分别进化
            pool= multiprocessing.Pool(4)
            process = []
            for i in range(4):#循环创建子进程
                process.append(pool.apply_async(son_process, (sub_slave[i],select[i],q,)))
            pool.close()
            pool.join()
            #获取返回值，返回值格式[(new_slave1,selectP1),(new_slave2,selectP2),(new_slave3,selectP3),(new_slave4,selectP4)]
            returnList=[]#返回值列表
            for j in process:
               returnList.append(j.get())#得到子从种群集合，获得子进程得到的子种群列表，通过使用.get() 才能获取到数据
            print('returnListreturnList',returnList)
            #拆分返回值列表，分离新从种群和新选择概率
            new_subslave=[]#新从种群集合[newslave1,newslave2,newslave3,newslave4]
            new_select=[]#新选择概率[newselect1,newselect2,newselect3,newselect4]
            for i in range(len(returnList)):
                new_subslave.append(returnList[i][0])
                new_select.append(returnList[i][1])
            print('new_subslave',new_subslave)    
            print('new_select',new_select)
            #合并从种群，选出最好的popsize个最优个体
            new_slaveList=[]#从种群[]
            for i in range(len(new_subslave)):
                for j in new_subslave[i]:
                    new_slaveList.append(j)
            print('new_slaveList',new_slaveList)
            new_slaveList.sort(key= lambda x: x[1])#按成本从小到大排序,带成本的
            currentsolution=new_slaveList[:popsize]#选出最优的种群数量个个体 
            #更新主种群：将新从种群最优解与当前主种群合并，筛选popsize个个体作为新主种群，然后放进队列
            master=[]
            for i in range(q.qsize()):
                master.append(q.get())
            print('前一代master',master)
            master.extend(currentsolution)
            print('前一代master+子代SP最优解',master)
            master.sort(key= lambda x: x[1])#按成本从小到大排序,带成本的
            master=master[:popsize]
            print('更新后的master',master)
            for i in range(popsize):#将处理的主种群放进队列
                q.put(master[i])
            #print('#更新后主种群个体数',q.qsize())
            #更新
            new_subslave=sub_slave
            new_select=select
            iterator += 1
        print("@@@@@@@@@@@@@@@@@@@@退出进化循环")    
        finalMaster=[]#最终主种群
        for i in range(q.qsize()):
            finalMaster.append(q.get())
        finalMaster.sort(key= lambda x: x[1])#按成本从小到大排序,带成本的
        print('最终的finalMaster',finalMaster)
        
        bestsolution_list.append(finalMaster[0])#存储不同单元数下的最优解
    print('bestsolution_list',bestsolution_list)
    bestsolution_list.sort(key= lambda x: x[1])#按成本从小到大排序,带成本的
    bestsolution=bestsolution_list[0]
    print('bestsolution',bestsolution)
    return bestsolution
                       
          
if __name__=="__main__":
    start=time.time()
    main()    
    end=time.time()
    print('运行时间',end-start)
 
        
        
    
    
