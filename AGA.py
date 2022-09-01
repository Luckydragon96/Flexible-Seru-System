# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 07:28:25 2021
此版本是按照李晓龙的意见将父代与子代种群合并取消了，不合并了！选择还是按照之前的方法选择，其余都修改了！
@author: Administrator
"""

import time
import copy
import random
import numpy as np
import gurobipy as gp
from gurobipy import GRB

'''编码（初始化种群）'''

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
    #print(workerallocation)
    productdistribution=copy.deepcopy(pop[1])
    column_sum=[sum(i) for i in zip(*productdistribution)]
    #计算构建成本
    formationCost=len(workerallocation)*FC
    print('formationCost',formationCost)
    #计算培训成本
    trainingCost=0
    for i in range(len(workerallocation)):#遍历每个单元，用单元人数*单元分配的产品种类数
        everyserutraining_cost=workerallocation[i]*column_sum[i]*TC
        trainingCost+=everyserutraining_cost   
    Fixcost=formationCost+trainingCost
    return Fixcost

#demand=[[24, 127, 149], [82, 47, 171], [57, 141, 102], [11, 100, 189], [87, 109, 104], [84, 125, 91], [113, 121, 66], [121, 114, 65], [83, 88, 129], [86, 123, 91], [68, 149, 83], [89, 121, 90], [95, 115, 90], [55, 85, 160], [96, 96, 108], [100, 78, 122], [132, 91, 77], [92, 41, 167], [93, 64, 143], [113, 143, 44], [47, 145, 108], [176, 22, 102], [105, 109, 86], [77, 142, 81], [65, 99, 136], [97, 137, 66], [105, 27, 168], [108, 57, 135], [172, 33, 95], [58, 136, 106], [156, 104, 40], [75, 143, 82], [72, 55, 173], [81, 135, 84], [126, 62, 112], [153, 66, 81], [153, 132, 15], [66, 135, 99], [104, 115, 81], [169, 66, 65], [69, 129, 102], [9, 135, 156], [29, 131, 140], [113, 124, 63], [131, 52, 117], [57, 167, 76], [142, 111, 47], [107, 75, 118], [150, 69, 81], [100, 85, 115]]

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


def tournament_selection(solution):#二元锦标赛选择算子，每次选择2个个体
    #个体索引列表
    index=range(len(solution)) 
    #选择两个个体
    select=[]
    for i in range(2):#选择两个个体
        tempList=[]
        tempList.extend(random.sample(index,2))#从染色体下表中随机选两个数，选择更优的
        if solution[tempList[0]][1]<=solution[tempList[1]][1]:
            select.append(solution[tempList[0]])
        else:
            select.append(solution[tempList[1]])
    return select #select=[个体，个体成本]

def roulettewheel_selection(solution):#每次选择两个个体,solution是带有成本的种群
    cumulativeProbability=[]
    sumFitness=sum(1/x[1] for x in solution)#计算适应度值倒数之和
    #print('sumFitness',sumFitness)
    probability=[]
    select=[]
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
    #选择2个个体
    for i in range(2):
        r=np.random.uniform(0,1)#在（0，1）之间随机取一个值
        #print('r',r)
        for j in range(len(cumulativeProbability)):#用r和cumulativeProbability中的每个值比较
            if j==0:
                if r>=0 and r<= cumulativeProbability[j]:
                    select.append(solution[j])
            else:
                if r>=cumulativeProbability[j-1] and r<= cumulativeProbability[j]:
                    select.append(solution[j])
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
    spacing=sumFitness/2
    #print('spacing',spacing)
    #随机生成起点指针位置
    start=np.random.uniform(0,spacing)#在（0，spacing）之间随机取一个数
    #print('start',start)
    #计算各指针的位置Pointers
    Pointers=[start+i*spacing for i in [0,1]]
    #print('Pointers',Pointers)
    #根据各指针位置，选择出2个个体
    select=[]
    for i in range(2):
        for j in range(len(solution)):#用Pointers里的每个值和solution中的适应度值比较
            if j==0:
                if Pointers[i]>=0 and Pointers[i]<= cumulativeFitness[j]:
                    select.append(solution[j])
            else:
                if Pointers[i]>=cumulativeFitness[j-1] and Pointers[i]<= cumulativeFitness[j]:
                    select.append(solution[j])
    return select

def random_selection(solution):#随机选择
    solution_copy=copy.deepcopy(solution)
    select=[]
    for i in range(2):
        number=np.random.randint(0,len(solution_copy))
        #print('当前子种群',solution)
        #print('number',number)
        select.append(solution_copy[number])
    #print('select',select)
    return select

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
    #print('new_p1',new_p1)
    #print('new_p2',new_p2)
    newpop1=[]
    newpop1.append(workerallocation1)
    newpop1.append(new_p1)
    newpop2=[]
    newpop2.append(workerallocation2)
    newpop2.append(new_p2)
    newpops=[]
    newpops.append(newpop1)
    newpops.append(newpop2)
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
        workerallocation[index1],workerallocation[index2]=workerallocation[index2],workerallocation[index1]
    elif a!=0 and workerallocation[index1]==workerallocation[index2]:#两个单元人数相同
        workerallocation[index1],workerallocation[index2]=workerallocation[index1],workerallocation[index1]
    else:
        if workerallocation[index1]==min_worker:
            workerallocation[index1]=workerallocation[index1]+a
            workerallocation[index2]=workerallocation[index2]-a
        else:
            workerallocation[index1]=workerallocation[index1]-a
            workerallocation[index2]=workerallocation[index2]+a
    #突变产品-SERU分配，采用两点变异法,交换两个产品的分配
    index3=np.random.randint(0,len(productdistribution)-1)#随机产生两个突变点
    index4=np.random.randint(index3+1,len(productdistribution))
    productdistribution[index3],productdistribution[index4]=productdistribution[index4],productdistribution[index3]
    new_pop.append(workerallocation)
    new_pop.append(productdistribution)
    return new_pop

def isfeasibleSolution(pop):#计算各单元生产产品类型数量，便于后期判断是否是可行解
    #print('判断是否是可行解')
    column_sum=[sum(i) for i in zip(*pop[1])]
    #print('column_sum',column_sum)
    t1=0
    for i in range(len(column_sum)):#计算列的和，确保每个单元都有产品分配
        if column_sum[i]>0:
            t1+=1
            if t1==len(column_sum):#需要继续判断行和是否大于0
                #print('列和均大于0')
                break
        else:
            #print('列和小于0')
            return False#不是可行解
    t2=0        
    for i in range(len(pop[1])):#计算每行的和，确保每个单元都分配了产品
        if sum(pop[1][i])>0:
            t2+=1
            if t2==len(pop[1]):#判断行和是否大于0
               # print('行和均大于0')
                return True
        else:
            #print('行和小于0')
            return False#不是可行解
     
def compute_rowsum_columnsum(pop):#计算一个个体的行和和列和
    row_sum=[]
    for i in pop[1]:#求行和，其意义是产品分配单元的数量
        row_sum.append(sum(i))
    #print('row_sum',row_sum)
    column_sum=[sum(i) for i in zip(*pop[1])]#求列和，其意义是单元分配产品的数量
    #print('column_sum',column_sum)
    return row_sum,column_sum

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

def getPops(method,solution):#输入选择策略和种群，输出选择的个体
    if method==0:
        pops=tournament_selection(solution)
    elif method==1:
        pops=roulettewheel_selection(solution)
    else:
        pops=random_selection(solution)
    return pops

def next_Population(solution,selectProbability):#输入父代，选择概率，返回下一代及新选择概率
    selectProbability_copy=copy.deepcopy(selectProbability)
    newsolution=[]
    selectmethod=selectOperator(selectProbability_copy) 
    print('selectmethod',selectmethod)
    while len(newsolution)<len(solution):#产生子代
        #选择两个个体
        parents=getPops(selectmethod,solution)
        #print('parents',parents)
        parents.sort(key= lambda x: x[1])#根据目标值从大到小排序
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
        for m in mutation:#一定包含两个个体,分别进行邻域搜索
            newsolution.append(off_neighbour(m[0],m[1],max_num))
    newsolution.sort(key= lambda x: x[1])#按成本从小到大排序,带成本的
    newsolution=newsolution[:len(solution)]#选出最优的种群数量个个体，作为新一代种群
    #print('###########邻域后代',newsolution)
    #更新选择概率
    newselectP=selectProbability_copy
    print('selectmethod',selectmethod)
    print('@@@@@@@@@@@@@@@@@@@@@更新前newselectP',newselectP)
    if newsolution[0][1]<solution[0][1]:#比较父代与子代最优解
        newselectP[selectmethod] += 0.1
    else:
        if  newselectP[selectmethod] >= 0.1:
            newselectP[selectmethod] -= 0.1
        else:
            newselectP[selectmethod] -= 0
       #newselectP[selectmethod] = selectProbability_copy[selectmethod]
    print('@@@@@@@@@@@@@@@@@@@@更新后newselectP',newselectP)
    return newsolution,newselectP

#模型参数
#demand=[[24, 127, 149], [82, 47, 171], [57, 141, 102], [11, 100, 189], [87, 109, 104], [84, 125, 91], [113, 121, 66], [121, 114, 65], [83, 88, 129], [86, 123, 91], [68, 149, 83], [89, 121, 90], [95, 115, 90], [55, 85, 160], [96, 96, 108], [100, 78, 122], [132, 91, 77], [92, 41, 167], [93, 64, 143], [113, 143, 44], [47, 145, 108], [176, 22, 102], [105, 109, 86], [77, 142, 81], [65, 99, 136], [97, 137, 66], [105, 27, 168], [108, 57, 135], [172, 33, 95], [58, 136, 106], [156, 104, 40], [75, 143, 82], [72, 55, 173], [81, 135, 84], [126, 62, 112], [153, 66, 81], [153, 132, 15], [66, 135, 99], [104, 115, 81], [169, 66, 65], [69, 129, 102], [9, 135, 156], [29, 131, 140], [113, 124, 63], [131, 52, 117], [57, 167, 76], [142, 111, 47], [107, 75, 118], [150, 69, 81], [100, 85, 115]]
demand=[[92, 116, 92], [63, 177, 60], [89, 126, 85], [114, 93, 93], [173, 118, 9], [66, 149, 85], [104, 85, 111], [107, 90, 103], [51, 105, 144], [90, 114, 96], [28, 89, 183], [89, 123, 88], [83, 128, 89], [174, 37, 89], [84, 107, 109], [144, 51, 105], [139, 26, 135], [51, 118, 131], [53, 160, 87], [108, 159, 33], [153, 90, 57], [141, 103, 56], [85, 51, 164], [101, 103, 96], [149, 41, 110], [108, 126, 66], [115, 87, 98], [35, 131, 134], [103, 75, 122], [162, 39, 99], [75, 145, 80], [134, 36, 130], [86, 118, 96], [11, 142, 147], [160, 103, 37], [134, 94, 72], [95, 105, 100], [120, 95, 85], [109, 50, 141], [102, 113, 85], [154, 75, 71], [139, 79, 82], [118, 88, 94], [15, 137, 148], [101, 89, 110], [164, 85, 51], [96, 68, 136], [106, 87, 107], [129, 77, 94], [50, 118, 132], [136, 108, 56], [131, 71, 98], [95, 157, 48], [108, 86, 106], [158, 27, 115], [92, 121, 87], [92, 134, 74], [107, 62, 131], [154, 66, 80], [122, 163, 15], [112, 145, 43], [145, 23, 132], [139, 98, 63], [142, 30, 128], [138, 73, 89], [71, 119, 110], [45, 171, 84], [102, 123, 75], [72, 176, 52], [132, 117, 51], [104, 133, 63], [93, 55, 152], [98, 69, 133], [86, 108, 106], [79, 109, 112], [72, 130, 98], [127, 128, 45], [79, 65, 156], [72, 107, 121], [128, 101, 71], [108, 75, 117], [111, 84, 105], [43, 141, 116], [73, 145, 82], [114, 128, 58], [140, 56, 104], [89, 107, 104], [53, 44, 203], [99, 30, 171], [144, 47, 109], [116, 169, 15], [110, 155, 35], [85, 94, 121], [66, 141, 93], [100, 144, 56], [107, 101, 92], [123, 55, 122], [129, 110, 61], [111, 84, 105], [70, 47, 183], [128, 46, 126], [133, 65, 102], [114, 122, 64], [55, 189, 56], [127, 23, 150], [160, 115, 25], [115, 83, 102], [49, 82, 169], [139, 94, 67], [95, 115, 90], [151, 54, 95], [103, 69, 128], [22, 200, 78], [155, 80, 65], [77, 130, 93], [42, 159, 99], [65, 112, 123], [135, 84, 81], [129, 92, 79], [232, 35, 33], [104, 47, 149], [104, 100, 96], [114, 113, 73], [48, 105, 147], [98, 132, 70], [34, 130, 136], [166, 52, 82], [75, 76, 149], [134, 55, 111], [174, 32, 94], [43, 112, 145], [165, 38, 97], [136, 58, 106], [19, 207, 74], [147, 54, 99], [103, 151, 46], [123, 118, 59], [86, 122, 92], [38, 136, 126], [125, 80, 95], [102, 71, 127], [101, 58, 141], [139, 28, 133], [58, 79, 163], [102, 60, 138], [93, 115, 92], [113, 145, 42], [142, 66, 92], [136, 73, 91], [137, 40, 123], [122, 114, 64], [125, 123, 52], [96, 133, 71], [72, 122, 106], [106, 125, 69], [105, 120, 75], [86, 75, 139], [137, 132, 31], [195, 102, 3], [97, 148, 55], [80, 92, 128], [15, 165, 120], [125, 86, 89], [83, 88, 129], [103, 168, 29], [127, 49, 124], [123, 98, 79], [21, 163, 116], [131, 48, 121], [109, 122, 69], [152, 47, 101], [141, 89, 70], [139, 72, 89], [72, 93, 135], [107, 119, 74], [101, 69, 130], [99, 91, 110], [134, 59, 107], [104, 118, 78], [102, 72, 126], [106, 104, 90], [87, 105, 108], [58, 119, 123], [82, 130, 88], [105, 146, 49], [90, 148, 62], [84, 106, 110], [154, 66, 80], [133, 115, 52], [99, 112, 89], [181, 56, 63], [131, 92, 77], [55, 98, 147], [142, 93, 65], [47, 119, 134], [83, 103, 114], [22, 119, 159], [41, 109, 150], [46, 125, 129], [96, 92, 112]]
len(demand)
#demand=[[89, 162, 99, 82, 68], [156, 116, 58, 44, 126], [68, 79, 123, 153, 77], [53, 134, 82, 138, 93], [85, 37, 95, 119, 164], [132, 97, 119, 94, 58], [110, 119, 136, 42, 93], [110, 101, 44, 110, 135], [128, 12, 135, 183, 42], [106, 82, 79, 190, 43], [97, 135, 161, 14, 93], [114, 86, 94, 83, 123], [62, 142, 92, 82, 122], [69, 81, 149, 63, 138], [113, 186, 60, 29, 112], [45, 95, 54, 199, 107], [114, 120, 99, 107, 60], [104, 103, 119, 94, 80], [71, 43, 137, 47, 202], [31, 177, 115, 94, 83], [112, 107, 46, 85, 150], [82, 79, 151, 118, 70], [69, 47, 178, 129, 77], [141, 76, 134, 95, 54], [137, 85, 66, 100, 112], [122, 64, 76, 148, 90], [97, 151, 77, 51, 124], [105, 90, 140, 26, 139], [156, 60, 145, 106, 33], [81, 106, 149, 24, 140], [48, 145, 133, 46, 128], [96, 97, 79, 165, 63], [57, 148, 113, 84, 98], [115, 107, 99, 75, 104], [145, 45, 143, 54, 113], [102, 30, 115, 96, 157], [36, 105, 105, 93, 161], [133, 113, 83, 121, 50], [88, 87, 110, 71, 144], [89, 72, 53, 154, 132], [99, 78, 129, 86, 108], [28, 131, 165, 94, 82], [92, 74, 125, 122, 87], [111, 134, 42, 71, 142]]
numOfProduct=3
numOfWorker=3
maxSeru=3
TC=1
FC=0
sc=0.8 
ec=0.6
A=100

max_num=2#领域个体数量
popsize=40#种群数量
crossRate=0.8
mutationRate=0.4
maxIterator=5#最优解保持不变的最大代数


time_start = time.perf_counter()
def main():
    result=[]#记录不同单元数量的最优解
    selectProbability=[1/3,1/3,1/3]#初始化选择概率
    solution=initPopulation(1,numOfWorker,numOfProduct,1)
    result.append([solution[0][0],solution[0][1]])
    for i in range(2,maxSeru+1):
        bestcost_list=[]#用于存放该单元数下每代最优个体
        solution=initPopulation(popsize,numOfWorker,numOfProduct,i)
        print('………………………………………………初始种群………………………………………………',solution)
        bestcost_list.append(solution[0][1])
        iterator=1
        while iterator<maxIterator:#解保持不变的代数
            print('##########进入循环',iterator)
            new_solution,new_selectProbability=next_Population(solution,selectProbability)#按成本从小到大排好序的种群
            #print('………………………………………………下一代………………………………………………',new_solution)
            print('newselectProbability',new_selectProbability)
            if new_solution[0][1]==bestcost_list[-1]:
                iterator+=1
            else:
                iterator=0
            bestcost_list.append(new_solution[0][1])
            print('bestcost_list',bestcost_list)
            solution=new_solution
            selectProbability=new_selectProbability
        result.append([new_solution[0][0],new_solution[0][1]])
    #print('不同单元数的最优SERU构造和成本分别为',result)
    result.sort(key= lambda x: x[1])#按成本从小到大排序,带成本的
    #print('结果排序',result)
    best_pop=result[0][0]
    best_cost=result[0][1]
    print('best_pop',best_pop)
    print('best_cost',best_cost)
    return best_pop,best_cost
 
if __name__=="__main__":
    main()
    
time_end = time.perf_counter()

time_consumed = time_end - time_start
print("耗费的时间: {} s".format(time_consumed))
   

    
        
        
    
    
