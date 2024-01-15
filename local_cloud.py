import sys
import math
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from functools import cmp_to_key
from collections import OrderedDict, deque

###################################################################################
'''一.任务排序策略(针对的是等待队列)'''
# 1.执行时间:小->大
def time_become_bigger(exec_list_of_i):
    exec_list_of_i.sort(key=cmp_to_key(lambda x, y: x[0][2] - y[0][2]))
    return exec_list_of_i

# 2.执行时间:大->小
def time_become_smaller(exec_list_of_i):
    exec_list_of_i.sort(key=cmp_to_key(lambda x, y: y[0][2] - x[0][2]))
    return exec_list_of_i

# 3.数据量:小->大
def data_become_bigger(exec_list_of_i):
    exec_list_of_i.sort(key=cmp_to_key(lambda x, y: x[0][1] - y[0][1]))
    return exec_list_of_i

# 4.数据量:大->小
def data_become_smaller(exec_list_of_i):
    exec_list_of_i.sort(key=cmp_to_key(lambda x, y: y[0][1] - x[0][1]))
    return exec_list_of_i

# 5.执行时间/数据量:小->大
def ratio_become_bigger(exec_list_of_i):
    exec_list_of_i.sort(key=cmp_to_key(lambda x, y: (x[0][2] / x[0][1]) - (y[0][2] / y[0][1])))
    return exec_list_of_i

# 6.执行时间/数据量:大->小
def ratio_become_smaller(exec_list_of_i):
    exec_list_of_i.sort(key=cmp_to_key(lambda x, y: (y[0][2] / y[0][1]) - (x[0][2] / x[0][1])))
    return exec_list_of_i

# 7.截止期:小->大
def deadline_become_bigger(exec_list_of_i, t):
    exec_list_of_i.sort(key=cmp_to_key(lambda x, y: (t - x[0][3]) - (t - y[0][3])))
    return exec_list_of_i


###################################################################################
'''二.使用Dijkstra算法求解最短路径'''
def get_count_by_dijkstra(G, s, e):
    # print(nx.dijkstra_path(G, source=0, target=7))
    return nx.dijkstra_path_length(G, source=s, target=e)


###################################################################################
'''三.函数置换策略'''
# 1.LRU最近最久未使用页面置换算法
class LRUCache:
    def __init__(self, Capacity):
        self.size = Capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return -1
        val = self.cache[key]
        self.cache.move_to_end(key)
        return val

    def put(self, key, val):
        if key in self.cache:
            del self.cache[key]
        self.cache[key] = val
        if len(self.cache) > self.size:
            self.cache.popitem(last=False)

    def inCache(self, key):
        if key in self.cache:
            return True
        else:
            return False

# 2.LFU最近最少使用页面置换算法
class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.f2kv = {}
        self.k2f = {}
        self.min_f = -1
        self.nums = 0

    def get(self, key: int) -> int:
        if key in self.k2f:
            freq = self.k2f[key]
            val = self.f2kv[freq][key]
            self.increaseFreq(key)
            return val
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if self.capacity <= 0:
            return None
        if key in self.k2f:
            freq = self.k2f[key]
            self.increaseFreq(key, value)
        else:
            self.nums += 1
            self.f2kv[1] = self.f2kv.get(1, OrderedDict())
            self.f2kv[1][key] = value
            self.k2f[key] = 1
            self.removeMinFreq()
            self.min_f = 1

    def increaseFreq(self, key, value=None):
        freq = self.k2f[key]
        self.k2f[key] += 1
        self.f2kv[freq + 1] = self.f2kv.get(freq + 1, OrderedDict())
        if value is None:
            self.f2kv[freq + 1][key] = self.f2kv[freq][key]
        else:
            self.f2kv[freq + 1][key] = value
        del self.f2kv[freq][key]
        if not self.f2kv[freq]:
            del self.f2kv[freq]
        if self.min_f == freq and not self.f2kv.get(freq, None):
            self.min_f += 1

    def removeMinFreq(self):
        if self.nums > self.capacity:
            k, v = self.f2kv[self.min_f].popitem(last=False)
            if not self.f2kv[self.min_f]:
                del self.f2kv[self.min_f]
            del self.k2f[k]
            self.nums = self.capacity

    def inCache(self, key):
        if key in self.f2kv:
            return True
        else:
            return False

# 3.FIFO先进先出页面置换算法
class FIFOCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.queue = deque()

    def get(self, key):
        if key in self.cache:
            return self.cache[key]
        else:
            return -1

    def put(self, key, value):
        if key in self.cache:
            self.cache[key] = value
        else:
            if len(self.queue) == self.capacity:
                oldest = self.queue.popleft()
                del self.cache[oldest]
            self.queue.append(key)
            self.cache[key] = value
    
    def inCache(self, key):
        if key in self.cache:
            return True
        else:
            return False

###################################################################################
'''
func_n:函数集中的函数个数
func_cache:每个边缘节点可以缓存的函数个数
func_deal_media:每个边缘节点异构处理能力的中位数
cloud_hop:该区域距云计算中心的跳数
_lambda:泊松分布平均每秒到达的请求个数
func_tran:固定带宽
func_repl:函数置换策略(0:LRU|1:LFU)
task_sort:任务排序策略
'''
def TFHO(func_n, func_cache, func_deal_media, cloud_hop, _lambda, func_tran, func_repl, task_sort):
    # 1.读取三个数据集中的数据
    with open(r"data/data.pk", "rb") as file:
        result = pickle.load(file)

    # 2.定义各类超参数
    # 边缘节点个数
    n = 125
    # 从FaasNet数据集中随机挑选func_n个函数组成函数集
    func_i = np.random.choice([i for i in np.arange(100)], size=func_n, replace=False)
    # 最终选定的函数集
    func = []
    for i in func_i:
        func.append(funcs[i])
    '''------[函数置换策略]------'''
    # 每个节点的函数集合(0:LRU|1:LFU|2:FIFO)
    if func_repl == 0:
        func_each = [LRUCache(func_cache) for i in np.arange(n)]
    elif func_repl == 1:
        func_each = [LFUCache(func_cache) for i in np.arange(n)]
    else:
        func_each = [FIFOCache(func_cache) for i in np.arange(n)]
    # 每个节点当前的处理能力
    func_deal = [np.random.randint(func_deal_media - 100, func_deal_media + 101) for i in np.arange(n)]
    # 每个节点当前的执行队列
    exec_list = [[] for i in np.arange(n)]
    # 每个节点当前的传输队列
    tran_list = [[] for i in np.arange(n)]
    # 每个节点当前的等待队列
    wait_list = [[] for i in np.arange(n)]
    # 每个请求的延迟时间(结束时间-(开始时间+预期执行时间))
    time = {}
    complete = {}
    # 当前请求id
    index = 0
    # 当前时间id
    t = 0

    req_all_num = 0

    while True:

        # print("当前时间:---", t, "---")
        # result=[instance_num|plan_mem|deal_time|req_deadline|req_id]
        # 每个边缘节点请求的到达服从泊松分布系统运行200s
        if t < 1000:
            '''------[请求到达方式]------'''
            poisson = np.random.poisson(_lambda, n)
            req_num = 0
            req_all_num += sum(poisson)
            # 第i个边缘节点
            for i in np.arange(n):
                # 泊松分布在该秒到来的请求数量
                req_num = poisson[i]
                # 当前正在处理到来的第j个请求
                j = 0
                while j < req_num:
                    # 本次请求请求哪个函数
                    select_func = np.random.randint(0, func_n)
                    # 将请求加入延迟字典(新的平均延迟计算方式)
                    time[result[index][4]] = t
                    # 更新请求的截止期字段
                    result[index][3] = t
                    '''------[边缘节点卸载决策]------
                        1.存在函数&&处理能力够->本地直接处理
                        2.本地处理能力不足->卸载到云中心
                        3.不存在函数&&处理能力够->卸载到云中心+拉取函数冷启动
                    '''
                    # ---情况1---
                    if (func_each[i].inCache(select_func)) and (func_deal[i] >= result[index][0]):
                        wait_list[i].append([result[index], select_func])
                    # ---情况2---
                    elif (func_deal[i] < result[index][0]):
                        complete[result[index][4]] = math.ceil(result[index][1] / func_tran) * cloud_hop
                    # ---情况3---
                    else: 
                        tmp = math.ceil(result[index][1] / func_tran) * cloud_hop
                        # 卸载到云中心更优
                        if tmp < func[select_func][2]:
                            complete[result[index][4]] = tmp
                        # 冷启动更优
                        else:
                            wait_list[i].append([result[index], select_func])
                    j += 1
                    index += 1
        else:
            break

        # ------传输|拉取队列------
        # 0.[某条请求,传输剩余时间,请求函数编号]
        # 1.从边缘节点决策处添加(其它边缘卸载而来|冷启动拉取函数而来)
        # 2.传输结束后添加到等待队列中
        #print("------传输队列------")
        for i in np.arange(n):
            end = []
            for j in np.arange(len(tran_list[i])):
                tran_list[i][j][1] -= 1
                # 传输结束放入等待队列
                if tran_list[i][j][1] == 0:
                    end.append(j)
                    wait_list[i].append([tran_list[i][j][0], tran_list[i][j][2]])
            tran_list[i] = [x for num, x in enumerate(tran_list[i]) if num not in end]
        # print(tran_list)

        # ------等待队列------
        # 0.[某条请求,请求函数编号]
        # 1.从边缘节点决策处添加
        # 2.从传输队列中添加
        # 3.等待到足够资源后添加到执行队列中
        #print("------等待队列------")
        for i in np.arange(n):
            end = []
            '''------[任务排序策略]------'''
            if task_sort == 1:
                wait_list[i] = time_become_bigger(wait_list[i])
            elif task_sort == 2:
                wait_list[i] = time_become_smaller(wait_list[i])
            elif task_sort == 3:
                wait_list[i] = data_become_bigger(wait_list[i])
            elif task_sort == 4:
                wait_list[i] = data_become_smaller(wait_list[i])
            elif task_sort == 5:
                wait_list[i] = ratio_become_bigger(wait_list[i])
            elif task_sort == 6:
                wait_list[i] = ratio_become_smaller(wait_list[i])
            else:
                wait_list[i] = deadline_become_bigger(wait_list[i], t)
            for j in np.arange(len(wait_list[i])):
                # 该边缘节点当前可用算力能够执行多少等待队列中的任务
                if func_deal[i] >= wait_list[i][j][0][0]:
                    # 占用节点处理能力
                    func_deal[i] -= wait_list[i][j][0][0]
                    end.append(j)
                    # 计算延迟时间
                    tmp = wait_list[i][j][1]
                    # 判断缓存中是否有这个函数
                    if func_each[i].inCache(tmp):
                        func_each[i].get(tmp)
                        complete[wait_list[i][j][0][4]] = t - time[wait_list[i][j][0][4]]# +1
                    else:
                        func_each[i].put(tmp, tmp)
                        wait_list[i][j][0][2] += func[wait_list[i][j][1]][2]
                        complete[wait_list[i][j][0][4]] = t - time[wait_list[i][j][0][4]] + func[wait_list[i][j][1]][2]# +1
                    # 加入执行队列
                    exec_list[i].append(wait_list[i][j][0])
                else:
                    break
            wait_list[i] = [x for num, x in enumerate(wait_list[i]) if num not in end]
        # print(wait_list)

        # ------执行队列------
        # 0.[某条请求]
        # 1.从等待队列中获取
        # 2.执行完毕后删除
        #print("------执行队列------")
        for i in np.arange(n):
            # 已执行完task编号
            end = []
            for j in np.arange(len(exec_list[i])):
                exec_list[i][j][2] -= 1
                # 该task执行完毕
                if exec_list[i][j][2] == 0:
                    # 从执行队列中删除
                    end.append(j)
                    # 释放节点处理能力
                    func_deal[i] += exec_list[i][j][0]
            exec_list[i] = [x for num, x in enumerate(exec_list[i]) if num not in end]
        # print(exec_list)

        t += 1

        if t == 200:
            return np.mean(list(complete.values()))


with open(r"data/eua-bsc500.pk", "rb") as file:
    graph = pickle.load(file)

with open(r"data/eua-bsc500-hop.pk", "rb") as file:
    hops = pickle.load(file)

with open(r"data/function.pk", "rb") as file:
    funcs = pickle.load(file)

G = nx.Graph()
for i in np.arange(125):
    G.add_node(i)
for i in np.arange(125):
    for j in np.arange(i + 1, 125):
        if graph[i][j] == 1:
            G.add_weighted_edges_from([(i, j, 1)])


print(TFHO(40, 5, 200, 10, 8, 0.1, 0, 1))
print(TFHO(40, 5, 200, 10, 8, 0.1, 1, 1))
print(TFHO(40, 5, 200, 10, 8, 0.1, 2, 1))