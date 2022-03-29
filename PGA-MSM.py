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
    print('formationCost',formationCost)
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
    print("损失成本为：")
    print(loss_cost)
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
    print('交叉点1',index1)
    print('交叉点2',index2)
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
    print('交叉个体',newpops)
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
        print('工人数不同交换工人数')
        workerallocation[index1],workerallocation[index2]=workerallocation[index2],workerallocation[index1]
    elif a!=0 and workerallocation[index1]==workerallocation[index2]:#两个单元人数相同
        print('工人数相同不变')
        workerallocation[index1],workerallocation[index2]=workerallocation[index1],workerallocation[index1]
    else:
        print('工人数调增调减')
        if workerallocation[index1]==min_worker:
            workerallocation[index1]=workerallocation[index1]+a
            workerallocation[index2]=workerallocation[index2]-a
        else:
            workerallocation[index1]=workerallocation[index1]-a
            workerallocation[index2]=workerallocation[index2]+a
    #突变产品-SERU分配，采用两点变异法,交换两个产品的分配
    index3=np.random.randint(0,len(productdistribution)-1)#随机产生两个突变点
    index4=np.random.randint(index3+1,len(productdistribution))
    print('交换两个产品分配',index3)
    print('交换两个产品分配',index4)
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
    neighbours = []
    neighbours.append([pop,pop_cost])
    k=0
    while k < max_num:#生成要求数量的邻域，一定都是可行解
        temp_pop=copy.deepcopy(pop)
        if len(possible_list) > 0:
            number=np.random.randint(0,len(possible_list))#无放回地选择一个需要调整的位置
            position=possible_list[number]
            
            possible_list.remove(possible_list[number])
            
            position1=position[0]
            position2=position[1]
            
            if temp_pop[1][position1][position2]==0:
                temp_pop[1][position1][position2]=1
            else:
                temp_pop[1][position1][position2]=0
                
            cost=computeTotalcost(temp_pop,demand,TC,FC,A,sc,ec)#计算领域个体的目标函数
            neighbours.append([temp_pop, cost])
            k+=1
        else:
            k=max_num
    print('neighbours',neighbours)
    neighbours.sort(key= lambda x: x[1])#为对前面的对象中的第二维数据（即value）的值进行排序。
    best_nb=neighbours[0]
    return best_nb  

def son_process(slave,q):#子进程，需要输入上一代从种群和主种群
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    newslave=[] 
    clone_slave=copy.deepcopy(slave)
    while len(clone_slave)>0:#只要父代个体没有遍历完就执行 
        #选择是从子种群选择一个个体，从主种群选择一个个体
        parents=[]
        parent1=q.get()
        clone_master=copy.deepcopy(parent1)
        print('clone_master',clone_master)
        q.put(clone_master)#将主种群中取出来的个体复制后再放回去
        
        number=np.random.randint(0,len(clone_slave))
        print('当前子种群',clone_slave)
        print('number',number)
        parent2=clone_slave[number]
        
        clone_slave.remove(clone_slave[number])
        print('更新子种群',clone_slave)
        
        parents.append(parent1)
        parents.append(parent2)
        parents.sort(key= lambda x: x[1])#根据目标值从大到小排序
        print('###########parents',parents)
        
        cross=[]
        rc= random.random()#随机产生一个数
        if rc<=crossRate:#发生交叉
            print('…………………………发生交叉……………………')
            cp= crossoverOperator(parents[0][0], parents[1][0])#cp中有两个个体
            for i in cp: #遍历新个体，先判断变异后个体是否是可行解，如果是再判断其是否优于原个体
                if i == parents[0][0]:#虽然交叉但未产生新个体，这样做的目的是为了不再次计算目标值
                    cross.append(parents[0])
                    print('交叉个体与父代较小个体一样',cross)
                elif i == parents[1][0]:
                    cross.append(parents[1])
                    print('交叉个体与父代较大个体一样',cross)
                else:#产生的是新解，则需计算目标值
                    if isfeasibleSolution(i) is True:
                        cp_pop=i
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
                    print('………………………………3cross',cross)
        else:#没有发生交叉，添加父代
            print('…………………………没有交叉……………………')
            cross=parents
        print('交叉后 ',cross)
        
        mutation=[]
        for i in cross:#遍历交叉后解（包含目标值），进行变异
            rm = random.random()#随机产生一个数
            if rm<=mutationRate:#变异
                print('@@@@@发生变异')
                mp=mutationOperator(i[0])
                mp_cost=computeTotalcost(i[0],demand,TC,FC,A,sc,ec)
                if mp_cost<i[1]:#如果变异个体优于交叉后个体，则添加到变异后代中
                    mutation.append([mp,mp_cost])
                    
                else:
                    mutation.append(i)#否则将交叉后个体添加到变异后代中
            else:
                mutation.append(i)
                print('@@@@@没有发生变异')
        print('###########变异后代',mutation)  
        
        for m in mutation:#一定包含两个个体,分别进行邻域搜索
            newslave.append(off_neighbour(m[0],m[1],max_num))
        print('###########邻域后代',newslave)
        
    newslave.sort(key= lambda x: x[1])#按成本从小到大排序,带成本的
    newslave=newslave[:len(slave)]
    print('newslave',newslave)
    
    q.put(newslave[0])#将每一进程的结果保存到queue中 
    return newslave

def son_initpopulation(q,numSeru):
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    slave=initPopulation(popsize,numOfWorker,numOfProduct,numSeru)#初始种群
    q.put(slave[0])#将子种群中最好的1个个体放入队列中，队列用于存储主种群
    q.put(slave[1])
    q.put(slave[2])
    q.put(slave[3])
    q.put(slave[4])
    q.put(slave[5])
    q.put(slave[6])
    q.put(slave[7])
    return slave

demand=[[65, 21, 196, 138, 108, 70, 18, 184], [124, 66, 62, 146, 72, 55, 89, 186], [76, 74, 227, 29, 52, 117, 71, 154], [93, 124, 88, 35, 134, 105, 139, 82], [76, 93, 151, 74, 43, 88, 122, 153], [62, 94, 123, 166, 116, 94, 58, 87], [73, 108, 49, 60, 154, 129, 110, 117], [17, 161, 117, 69, 127, 185, 92, 32], [144, 94, 45, 84, 136, 72, 106, 119], [124, 25, 145, 141, 95, 31, 102, 137], [84, 38, 169, 113, 44, 156, 65, 131], [112, 44, 70, 109, 70, 221, 122, 52], [132, 188, 17, 65, 131, 84, 37, 146], [86, 78, 140, 81, 137, 175, 42, 61], [36, 101, 200, 97, 37, 105, 81, 143], [102, 100, 159, 35, 93, 133, 90, 88], [35, 72, 72, 70, 117, 101, 144, 189], [89, 63, 44, 101, 176, 88, 168, 71], [120, 120, 128, 120, 131, 31, 59, 91], [161, 89, 186, 110, 71, 6, 61, 116], [132, 33, 75, 161, 61, 133, 178, 27], [138, 52, 33, 74, 137, 93, 127, 146], [142, 38, 78, 74, 142, 65, 245, 16], [53, 90, 152, 86, 89, 44, 44, 242], [25, 126, 173, 60, 143, 73, 135, 65], [38, 137, 83, 88, 161, 102, 50, 141], [122, 117, 136, 88, 96, 48, 26, 167], [59, 166, 171, 44, 76, 86, 146, 52], [118, 80, 40, 35, 138, 186, 81, 122], [33, 11, 101, 103, 247, 160, 7, 138], [183, 82, 99, 61, 137, 109, 92, 37], [43, 115, 115, 106, 231, 66, 72, 52], [99, 106, 163, 108, 26, 83, 3, 212], [131, 90, 52, 105, 137, 101, 86, 98], [116, 52, 88, 186, 94, 179, 17, 68], [190, 63, 124, 30, 83, 145, 96, 69], [76, 114, 90, 149, 8, 159, 97, 107], [100, 116, 76, 137, 146, 69, 73, 83], [93, 95, 84, 186, 91, 46, 154, 51], [110, 190, 66, 94, 26, 109, 57, 148], [73, 19, 32, 243, 130, 61, 124, 118], [92, 159, 37, 114, 121, 35, 71, 171], [133, 63, 149, 104, 90, 146, 36, 79], [106, 19, 30, 116, 141, 86, 188, 114], [67, 39, 123, 60, 144, 118, 223, 26], [132, 161, 102, 29, 91, 89, 113, 83], [70, 123, 130, 19, 108, 15, 219, 116], [103, 133, 105, 91, 45, 125, 119, 79], [134, 13, 71, 183, 104, 54, 74, 167], [84, 89, 81, 121, 141, 104, 103, 77]]

numOfProduct=8
numOfWorker=8
maxSeru=8
TC=1
FC=0
sc=0.8 
ec=0.6
A=100
max_num=7
popsize=30#种群数量
crossRate=0.8
mutationRate=0.4
#maxIterator=10#最优解保持不变的最大代数

def main():#主进程,join()的作用是是在进程中可以阻塞主进程的执行，指导全部的子进程全部完成之后，才继续运行主进程后面的代码
    print("………………主进程启动………………")
    bestsolution_list=[]#用于存放不同单元数下的最优解
    for numseru in range(2,maxSeru+1):
        q = Manager().Queue()#队列(容器)，用以存储主种群
        print('开始运行单元数为：',numseru)
        pool0 = multiprocessing.Pool(4)
        process0= []
        print ("初始化种群")
        for i in range(4):#循环创建子进程
            process0.append(pool0.apply_async(son_initpopulation,(q,numseru)))#由于apply_async的返回结果是一个结果对象，要通过使用.get() 才能获取到数据
        pool0.close()
        pool0.join()
        
        slavelist=[]#初始种群
        for i in process0:
            slavelist.append(i.get())#获得子进程得到的子种群列表，通过使用.get() 才能获取到数据
            print ("初始子种群集合", slavelist)
        print('初始主种群个体数',q.qsize())
        
        pool1 = multiprocessing.Pool(4)#定义一个进程池，最大进程数为8，如果不填写，默认CPU的核数
        process1= []
        print ("……………………开始进化第1代子种群……………………")
        for i in range(4):#循环创建子进程
            process1.append(pool1.apply_async(son_process, (slavelist[i],q)))#由于apply_async的返回结果是一个结果对象，要通过使用.get() 才能获取到数据
        pool1.close()#关闭进程池，表示不能再往进程池中添加进程，需要在join之前调用
        pool1.join()# 等待进程池中的所有进程执行完毕
        
        new_slavelist1=[]#第一代子种群
        for i in process1:
            new_slavelist1.append(i.get())#获得子进程得到的子种群列表，通过使用.get() 才能获取到数据
        print('第1代后主种群个体数',q.qsize())
        print ("第1代子进程执行结束")
        
        master=[]#第一代后对主种群进行处理
        for i in range(q.qsize()):#将队列中的主种群全部取出来，处理后再放回去
            master.append(q.get())
        master.sort(key= lambda x: x[1])#按成本从小到大排序,带成本的
        master=master[:popsize]
        print('去掉多于的master',master)
        
        for i in range(popsize):#将处理的主种群放进队列
            q.put(master[i])
        print('##第1代更新后主种群个体数',q.qsize())

        n=1    
        while master[0][1] != master[-1][1]:#主种群个体全部相同时停止
            print('…………………………第几代……………………',n)
            pool= multiprocessing.Pool(4)
            process = []
            for i in range(4):#循环创建子进程
                process.append(pool.apply_async(son_process, (new_slavelist1[i],q)))
            pool.close()
            pool.join()
            
            new_slavelist=[]
            for j in process:
                new_slavelist.append(j.get())#获得子进程得到的子种群列表，通过使用.get() 才能获取到数据
            print('##子种群列表##',new_slavelist)
            
            print('更新前主种群个体数',q.qsize())
            master=[]
            for i in range(q.qsize()):
                master.append(q.get())
            master.sort(key= lambda x: x[1])#按成本从小到大排序,带成本的
            master=master[:popsize]
            print('去掉多于的master',master)
            
            for i in range(popsize):#将处理的主种群放进队列
                q.put(master[i])
            print('##更新后主种群个体数',q.qsize())
            print ("子进程执行结束")
            
            new_slavelist1=new_slavelist
            n+=1
        
        master=[]
        for i in range(q.qsize()):
            master.append(q.get())
        master.sort(key= lambda x: x[1])#按成本从小到大排序,带成本的
        print('最终的master',master)
        bestsolution_list.append(master[0])
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
 
    